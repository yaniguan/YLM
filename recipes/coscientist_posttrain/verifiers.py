"""Sealed verifiers and the OPSD spec-leak guard.

verify(record, completion) -> {"reward": float in [0, 1], "passed": bool, "kind": str, "detail": str}
Each check runs in a child process with a timeout. The model never sees the verifier `spec`:
it is not rendered into prompts, not given to the OPSD teacher, and not kept in student data.
"""
import json
import multiprocessing as mp
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile


class VerifierUnavailable(RuntimeError):
    """The external checker (labbench protocol envs, Lean) is not installed here."""


class SpecLeak(ValueError):
    pass


def norm(text):
    return " ".join(re.findall(r"\w+", str(text).lower()))


def check_exact(spec, text):
    if "answer_key" in spec:
        ok = f" {norm(spec['answer_key'])} " in f" {norm(text)} "
    else:
        ok = norm(text) == norm(spec["answer"])
    return ok, "match" if ok else "no match"


def check_numeric(spec, text):
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", text)
    if not nums:
        return False, "no number"
    want = float(spec["value"])
    ok = abs(float(nums[-1]) - want) <= spec.get("rel_tol", 1e-2) * abs(want) + spec.get("abs_tol", 0.0)
    return ok, "within tolerance" if ok else "out of tolerance"


def check_sql(spec, text):
    """Run the query on an in-memory sqlite fixture (read-only after setup); compare result sets."""
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    query = (m.group(1) if m else text).strip()
    db = sqlite3.connect(":memory:")
    db.executescript(spec["setup"])
    db.execute("PRAGMA query_only = ON")
    want = db.execute(spec["expected_sql"]).fetchall()
    try:
        got = db.execute(query).fetchall()
    except sqlite3.Error as e:
        return False, f"sql error: {e}"
    if not spec.get("ordered", False):
        got, want = sorted(got, key=repr), sorted(want, key=repr)
    return got == want, "same rows" if got == want else "different rows"


def check_protocol(spec, text):
    t = norm(text)
    if "error_step" in spec:  # seeded twin: name the wrong step and give the right value
        ok = re.search(rf"\bstep {spec['error_step']}\b", t) is not None and f" {norm(spec['correct'])} " in f" {t} "
        return ok, "error found and fixed" if ok else "wrong step or missing fix"
    if "order" in spec:
        ok = [int(x) for x in re.findall(r"\d+", text)] == list(spec["order"])
        return ok, "order ok" if ok else "wrong order"
    raise VerifierUnavailable("this protocol spec needs the labbench-posttrain bioenvs checker")


def check_lean(spec, text):
    """Typecheck with the Lean 4 protocol DSL project (lean/). Pass = no errors and no `sorry`."""
    if shutil.which("lake") is None:
        raise VerifierUnavailable("lake/lean not on PATH")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "Check.lean")
        with open(path, "w") as f:
            f.write(spec.get("header", "") + "\n" + text)
        r = subprocess.run(["lake", "env", "lean", path], cwd=spec.get("project", "lean"),
                           capture_output=True, text=True, timeout=spec.get("timeout", 60))
    cheated = re.search(r"\b(sorry|admit|axiom|unsafe)\b|implemented_by|extern", text) is not None
    ok = r.returncode == 0 and not cheated
    return ok, "typechecks" if ok else "lean error"


# No plain fork: the trainer is multi-threaded, and forking it can deadlock the child.
START_METHOD = "forkserver" if "forkserver" in mp.get_all_start_methods() else "spawn"

CHECKS = {"exact": check_exact, "numeric": check_numeric, "sql": check_sql,
          "protocol": check_protocol, "lean": check_lean}


def _child(kind, spec, text, queue):
    try:
        queue.put(CHECKS[kind](spec, text))
    except VerifierUnavailable as e:
        queue.put((None, str(e)))
    except Exception as e:  # a crashing check is a failed answer, not a crashed trainer
        queue.put((False, f"{type(e).__name__}: {e}"))


def verify(record, completion, timeout=5.0):
    kind = record["verifier"]["kind"]
    if kind == "none":
        return {"reward": 0.0, "passed": False, "kind": kind, "detail": "no verifier"}
    ctx = mp.get_context(START_METHOD)
    queue = ctx.Queue()
    proc = ctx.Process(target=_child, args=(kind, record["verifier"]["spec"], completion, queue), daemon=True)
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"reward": 0.0, "passed": False, "kind": kind, "detail": f"timeout after {timeout}s"}
    try:
        ok, detail = queue.get(timeout=1.0)
    except Exception:
        ok, detail = False, f"checker exited with code {proc.exitcode}"
    if ok is None:
        raise VerifierUnavailable(detail)
    return {"reward": float(ok), "passed": bool(ok), "kind": kind, "detail": detail}


def _strings(x, skip_keys=("field",)):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in skip_keys:
                continue
            yield from _strings(v, skip_keys)
    elif isinstance(x, list):
        for v in x:
            yield from _strings(v, skip_keys)
    elif isinstance(x, str):
        yield x


def spec_leaks(spec, messages, check_values=True):
    """Spec text found in messages: the spec JSON or a `"key":` fragment in any turn, and (if
    check_values) any spec string value of >= 4 chars in a system/user turn. Short values and
    bare numbers are not checked (too many false hits)."""
    every = "\n".join(m["content"] for m in messages)
    found = [f for f in [json.dumps(spec), json.dumps(spec, sort_keys=True)] + [f'"{k}":' for k in spec]
             if f in every]
    if check_values:
        prompt = f" {norm(' '.join(m['content'] for m in messages if m['role'] != 'assistant'))} "
        found += [v for v in _strings(spec) if len(v.strip()) >= 4 and f" {norm(v)} " in prompt]
    return found


def assert_no_spec_leak(spec, messages, check_values=True):
    leaks = spec_leaks(spec, messages, check_values)
    if leaks:
        raise SpecLeak(f"verifier spec text in messages: {leaks}")


def opsd_teacher_messages(record, feedback):
    """OPSD teacher context = prompt + gold answer + verifier pass/fail and detail. Never the spec."""
    prompt = [m for m in record["messages"] if m["role"] != "assistant"]
    gold = next((m["content"] for m in reversed(record["messages"]) if m["role"] == "assistant"), "")
    note = f"Reference answer: {gold}\nVerifier: passed={feedback['passed']}; {feedback['detail']}"
    msgs = [{"role": "system", "content": note}] + prompt
    assert_no_spec_leak(record["verifier"]["spec"], msgs, check_values=False)  # gold may share words
    return msgs


def opsd_student_record(record, teacher_completion):
    """Student data = original prompt + teacher completion. No spec, no verifier, no reference answer."""
    msgs = [m for m in record["messages"] if m["role"] != "assistant"]
    msgs.append({"role": "assistant", "content": teacher_completion})
    assert_no_spec_leak(record["verifier"]["spec"], msgs)
    out = {k: record.get(k, [] if k in ("geoms", "images", "sims", "audios") else None)
           for k in ("schema", "task", "geoms", "images", "sims", "audios", "source")}
    out.update(id=record["id"] + "-opsd", messages=msgs, verifier={"kind": "none", "spec": {}}, pair=None,
               provenance={"generator": "opsd_teacher", "verified_by": "none", "twin_of": record["id"]})
    return out
