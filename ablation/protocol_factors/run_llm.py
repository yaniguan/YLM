"""Run a CLI LLM over items x conditions in batches (one condition per call, items from different
twin groups, fixed batch size). Resumable. Output: results/<tag>_<bench>.jsonl"""
import argparse
import hashlib
import json
import random
import re
import shlex
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import reps
from adapters import permute, prompt_err, prompt_mcq
from paths import CACHE, DATA, RESULTS
from render import factorial_conditions, facts, make_smap

HEAD = (
    "Answer each of the following {n} independent items. Treat each item separately: information given for "
    "one item does not apply to any other item.\n\n"
)
TAIL = (
    "\n\nReturn ONLY a JSON list with one object per item, no other text. Multiple-choice item: "
    "{{\"item\": k, \"answer\": \"<letter>\"}}. Protocol-error item: {{\"item\": k, \"p_error\": <0-100, your "
    "probability that it contains an error>, \"step\": <number of the wrong step, or 0 if none>}}."
)


def call(model, prompt, timeout=600, run_dir=None):
    fam, name = model.split(":", 1) if ":" in model else ("llm", model)
    cmd = shlex.split(fam) + ["-p", "--model", name, prompt]
    cwd = str(run_dir or Path(tempfile.gettempdir()) / "llmrun")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
    return r.stdout, r.stderr


def item_prompt(it, cond, fm, smap):
    if cond.get("kind") == "rep":
        block = facts(it, cond if cond.get("with_layers") else {"layers": set(), "identity": True}, fm, smap)
        rep_text = reps.rep_text(it, cond)
    else:
        block = facts(it, cond, fm, smap)
        rep_text = None
    if it["task"] == "error":
        return "[Protocol-error item]\n" + prompt_err(it, block, rep_text)
    return "[Multiple-choice item]\n" + prompt_mcq(it, block, rep_text) + "\nAnswer with the letter."


def batches(items, size, seed):
    rng = random.Random(seed)
    pool = items[:]
    rng.shuffle(pool)
    out = []
    while pool:
        b, groups, rest = [], set(), []
        for it in pool:
            if len(b) < size and it.get("group", it["id"]) not in groups:
                b.append(it)
                groups.add(it.get("group", it["id"]))
            else:
                rest.append(it)
        out.append(b)
        pool = rest
    return out


def _det_seed(key: str) -> int:
    return int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--items", required=True)
    ap.add_argument("--conds", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--rerun", default="")
    a = ap.parse_args()
    run_dir = Path(tempfile.gettempdir()) / "llmrun"
    run_dir.mkdir(parents=True, exist_ok=True)
    items = [json.loads(line) for line in open(a.items, encoding="utf-8")]
    bench = items[0]["bench"]
    fm_path = (CACHE / "fm_pred.json") if (CACHE / "fm_pred.json").exists() else (DATA / "fm_pred.json")
    fm = json.load(open(fm_path, encoding="utf-8"))
    smap = make_smap(items)
    allc = factorial_conditions()
    allc.update(reps.rep_conditions())
    out = RESULTS / f"{a.tag}_{bench}.jsonl"
    raw = RESULTS / f"raw_{a.tag}_{bench}.jsonl"
    done = set()
    if out.exists():
        for line in open(out, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["id"], r["cond"]))
    lock = threading.Lock()
    stop = {"quota": False}
    jobs = []
    for cn in a.conds.split(","):
        cname = cn.split("@")[0]
        cond = allc[cname]
        todo = [it for it in items if (it["id"], cn) not in done]
        for b in batches(todo, a.batch, seed=_det_seed(cn)):
            jobs.append((cn, cond, b))
    print(len(jobs), "calls", flush=True)

    def work(job):
        cn, cond, b = job
        if stop["quota"]:
            return
        prompt = (
            HEAD.format(n=len(b))
            + "\n\n".join(f"=== Item {k + 1} ===\n" + item_prompt(it, cond, fm, smap) for k, it in enumerate(b))
            + TAIL
        )
        ans = None
        so = ""
        for attempt in range(4):
            try:
                so, se = call(a.model, prompt, run_dir=run_dir)
            except Exception as e:
                so, se = "", str(e)
            if "quota" in (so + se).lower() or "session limit" in (so + se).lower() or "RESOURCE_EXHAUSTED" in se:
                print("QUOTA", (so + se)[-300:], flush=True)
                stop["quota"] = True
                return
            m = re.search(r"\[.*\]", so, re.S)
            try:
                ans = {int(x["item"]): x for x in json.loads(m.group(0))} if m else None
            except Exception:
                ans = None
            if ans and len(ans) >= len(b) * 0.75:
                break
            time.sleep(3 + 5 * attempt)
        if not ans:
            return
        with lock:
            with open(raw, "a", encoding="utf-8") as f:
                f.write(json.dumps({"cond": cn, "ids": [it["id"] for it in b], "out": so}) + "\n")
            with open(out, "a", encoding="utf-8") as f:
                for k, it in enumerate(b):
                    x = ans.get(k + 1)
                    if x is None:
                        continue
                    rec = {"id": it["id"], "cond": cn, "pos": k}
                    if it["task"] == "error":
                        try:
                            pe = float(x.get("p_error", 50)) / 100.0
                        except Exception:
                            pe = 0.5
                        try:
                            st = int(x.get("step", 0) or 0)
                        except Exception:
                            st = 0
                        rec.update(
                            p_error=pe,
                            pred_error=pe > 0.5,
                            is_error=it["is_error"],
                            step=st,
                            broken_step=it.get("broken_step"),
                        )
                        rec["correct"] = rec["pred_error"] == it["is_error"]
                        rec["correct_loc"] = rec["correct"] and (
                            not it["is_error"] or it.get("broken_step") in (None, st)
                        )
                    else:
                        _, ai = permute(it)
                        letter = str(x.get("answer", "")).strip().upper()[:1]
                        pred = ord(letter) - 65 if letter and letter.isalpha() else -1
                        rec.update(pred=pred, answer=ai, correct=pred == ai)
                    f.write(json.dumps(rec) + "\n")

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))
    print("finished; quota stop" if stop["quota"] else "finished", flush=True)


if __name__ == "__main__":
    main()
