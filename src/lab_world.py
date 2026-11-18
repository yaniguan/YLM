"""A small lab world that mirrors LabLean (YLM PR #92, lean/LabLean/Semantics.lean).

Used by the lab token factory (src/lab_token_factory.py), and by the token-quality test
added in the stacked PR #99.

* `check(state, steps)` is a Python copy of LabLean's `run`: same rules, same order of
  checks, same integer arithmetic. It returns ("OK", final volumes) or ("FAIL", step, error).
* `lean_check(cases, lablean_dir)` writes the same protocols as a Lean file, runs it with
  `lake env lean`, and parses LabLean's own verdicts. Lean is the authority; the Python
  copy is only a second, independent check (the two must agree).

Volumes are in 0.1 uL units (LabLean `Vol.t`), so 2.5 uL is 25. Only the steps the
generator uses are mirrored: pick_tip, drop_tip, transfer, mix, incubate, seal, thaw.
"""
import copy
import os
import re
import shutil
import subprocess
import tempfile

PIPETTES = {"P2": (2, 20), "P20": (20, 200), "P200": (200, 2000), "P1000": (1000, 10000)}
STEP_KINDS = ("pick_tip", "drop_tip", "transfer", "mix", "incubate", "seal", "thaw")


def fmt_vol(t):
    """0.1 uL units -> '12.5' (same digits as LabLean's Repr)."""
    return f"{t // 10}.{t % 10}"


# ---------- state ----------

def container(cid, cap, vol=0, reagent=None, plate=None):
    return {"id": cid, "plate": plate, "cap": cap, "vol": vol,
            "contents": [[reagent, vol]] if reagent else []}


def make_state(containers, reagents):
    """reagents: list of dict(name, state in frozen/thawed/onIce/roomTemp, keep_cold)."""
    return {"cs": containers, "reagents": reagents, "tip": None, "sealed": []}


def find(s, cid):
    for c in s["cs"]:
        if c["id"] == cid:
            return c
    return None


def names(c):
    return [r for r, _ in c["contents"]]


def tip_names(s):
    return s["tip"] if isinstance(s["tip"], list) else []


def union(a, b):
    return a + [x for x in b if x not in a]


def is_sealed(s, c):
    return c["plate"] is not None and c["plate"] in s["sealed"]


def weight(s, cid):
    c = find(s, cid)
    if c is not None:
        return c["vol"]
    wells = [x for x in s["cs"] if x["plate"] == cid]
    return sum(x["vol"] for x in wells) if wells else None


# ---------- checks (same order as LabLean `checks`) ----------

def _source_checks(s, pip, cid, a, v):
    lo, hi = PIPETTES[pip]
    if s["tip"] is None:
        return ("noTip",)
    if not lo <= v <= hi:
        return ("outOfRange", pip, v)
    if is_sealed(s, a):
        return ("sealed", cid)
    if not all(r in names(a) for r in tip_names(s)):
        return ("contaminated", cid)
    if v > a["vol"]:
        return ("notEnough", cid, a["vol"], v)
    for r in s["reagents"]:
        if r["name"] in names(a) and r["state"] == "frozen":
            return ("notThawed", r["name"])
    return None


def first_error(s, st):
    k = st[0]
    if k == "pick_tip":
        return ("tipAlreadyOn",) if s["tip"] is not None else None
    if k == "drop_tip":
        return ("noTip",) if s["tip"] is None else None
    if k == "transfer":
        _, pip, src, dst, v = st
        a, b = find(s, src), find(s, dst)
        if a is None:
            return ("unknown", src)
        if b is None:
            return ("unknown", dst)
        e = _source_checks(s, pip, src, a, v)
        if e:
            return e
        if src == dst:
            return ("sameContainer", src)
        if is_sealed(s, b):
            return ("sealed", dst)
        if b["vol"] + v > b["cap"]:
            return ("overflow", dst, b["vol"], v, b["cap"])
        return None
    if k == "mix":
        _, pip, cid, v, _times = st
        a = find(s, cid)
        if a is None:
            return ("unknown", cid)
        return _source_checks(s, pip, cid, a, v)
    if k == "incubate":
        _, target, temp, _secs = st
        if weight(s, target) is None:
            return ("unknown", target)
        if not 4 <= temp <= 98:
            return ("badTemp", temp)
        return None
    if k == "seal":
        p = st[1]
        if not any(c["plate"] == p for c in s["cs"]):
            return ("unknown", p)
        if p in s["sealed"]:
            return ("alreadySealed", p)
        return None
    if k == "thaw":
        if not any(r["name"] == st[1] for r in s["reagents"]):
            return ("unknown", st[1])
        return None
    raise ValueError(f"unknown step {k}")


# ---------- effects (same integer arithmetic as LabLean `apply`) ----------

def apply(s, st):
    s = copy.deepcopy(s)
    k = st[0]
    if k == "pick_tip":
        s["tip"] = []          # fresh tip = used with nothing (tipNames is [] for both)
    elif k == "drop_tip":
        s["tip"] = None
    elif k == "transfer":
        _, _pip, src, dst, v = st
        a, b = copy.deepcopy(find(s, src)), find(s, dst)
        touched = union(names(a), names(b))
        sa = find(s, src)
        old = sa["vol"]
        sa["contents"] = [[r, x - x * v // old] for r, x in sa["contents"]]
        sa["vol"] = old - v
        b = find(s, dst)
        for r, x in a["contents"]:
            add = x * v // a["vol"]
            for item in b["contents"]:
                if item[0] == r:
                    item[1] += add
                    break
            else:
                b["contents"].append([r, add])
        b["vol"] += v
        s["tip"] = union(tip_names(s), touched)
    elif k == "mix":
        s["tip"] = union(tip_names(s), names(find(s, st[2])))
    elif k == "seal":
        s["sealed"] = [st[1]] + s["sealed"]
    elif k == "thaw":
        for r in s["reagents"]:
            if r["name"] == st[1] and r["state"] == "frozen":
                r["state"] = "onIce" if r["keep_cold"] else "thawed"
    return s


def check(s, steps):
    """Python copy of LabLean `run`. Returns dict(ok, fail_step (1-based), error, volumes)."""
    for i, st in enumerate(steps, 1):
        e = first_error(s, st)
        if e is not None:
            return {"ok": False, "fail_step": i, "error": err_tag(e), "volumes": None}
        s = apply(s, st)
    return {"ok": True, "fail_step": None, "error": None,
            "volumes": {c["id"]: c["vol"] for c in s["cs"]}}


def trace(s, steps):
    """Per-step states for record text: list of (step, error or None, changed volumes)."""
    out = []
    for st in steps:
        e = first_error(s, st)
        if e is not None:
            out.append((st, err_tag(e), {}))
            break
        before = {c["id"]: c["vol"] for c in s["cs"]}
        s = apply(s, st)
        out.append((st, None, {c["id"]: c["vol"] for c in s["cs"] if c["vol"] != before[c["id"]]}))
    return out


_VOL_ARGS = {"outOfRange": (2,), "notEnough": (2, 3), "overflow": (2, 3, 4), "unbalanced": (3, 4)}


def err_tag(e):
    """Compact error text with volumes in uL; the Lean prelude below prints the same text."""
    vol = _VOL_ARGS.get(e[0], ())
    return " ".join(fmt_vol(x) if i in vol else str(x) for i, x in enumerate(e))


# ---------- text ----------

def step_text(st):
    k = st[0]
    if k == "transfer":
        return f"transfer {st[1]} {st[2]} {st[3]} {fmt_vol(st[4])}"
    if k == "mix":
        return f"mix {st[1]} {st[2]} {fmt_vol(st[3])} x{st[4]}"
    if k == "incubate":
        return f"incubate {st[1]} {st[2]}C {st[3]}s"
    if k in ("seal", "thaw"):
        return f"{k} {st[1]}"
    return k.replace("_", " ")


# ---------- Lean ----------

_LEAN_PRELUDE = r"""import LabLean
open LabLean

namespace TF
def v (e : Vol) : String := s!"{e.t / 10}.{e.t % 10}"
def tag : Error → String
  | .unknown i => s!"unknown {i}"
  | .noTip => "noTip"
  | .tipAlreadyOn => "tipAlreadyOn"
  | .outOfRange p x => s!"outOfRange {p} {v x}"
  | .sameContainer i => s!"sameContainer {i}"
  | .sealed i => s!"sealed {i}"
  | .contaminated i => s!"contaminated {i}"
  | .notEnough i h w => s!"notEnough {i} {v h} {v w}"
  | .overflow i h a c => s!"overflow {i} {v h} {v a} {v c}"
  | .notThawed r => s!"notThawed {r}"
  | .badTemp t => s!"badTemp {t.c}"
  | .badRotor n => s!"badRotor {n}"
  | .tooFast n => s!"tooFast {n}"
  | .notSealed p => s!"notSealed {p}"
  | .unbalanced a b x y => s!"unbalanced {a} {b} {v x} {v y}"
  | .alreadySealed p => s!"alreadySealed {p}"
/-- LabLean's own `step`, with the 1-based index of the failing step. -/
def runIdx (s : LabState) : Nat → List Step → String
  | _, [] => "OK|" ++ ",".intercalate (s.cs.map fun c => s!"{c.id}={c.vol.t}")
  | i, st :: rest =>
    match LabLean.step s st with
    | .error e => s!"FAIL|{i}|{tag e}"
    | .ok s' => runIdx s' (i + 1) rest
end TF
"""

_STATE = {"frozen": ".frozen", "thawed": ".thawed", "onIce": ".onIce", "roomTemp": ".roomTemp"}


def _lean_str(x):
    assert re.fullmatch(r"[A-Za-z0-9_]+", x), x
    return f'"{x}"'


def lean_state(s):
    cs = []
    for c in s["cs"]:
        plate = f"some {_lean_str(c['plate'])}" if c["plate"] else "none"
        contents = ", ".join(f"({_lean_str(r)}, {x})" for r, x in c["contents"])
        cs.append(f"{{ id := {_lean_str(c['id'])}, plate := {plate}, cap := ⟨{c['cap']}⟩, "
                  f"vol := ⟨{c['vol']}⟩, contents := [{contents}] }}")
    rs = [f"{{ name := {_lean_str(r['name'])}, state := {_STATE[r['state']]}, "
          f"keepCold := {'true' if r['keep_cold'] else 'false'} }}" for r in s["reagents"]]
    assert s["tip"] is None and not s["sealed"]
    return "{ cs := [" + ", ".join(cs) + "], reagents := [" + ", ".join(rs) + "] }"


def lean_step(st):
    k = st[0]
    if k == "pick_tip":
        return ".pickTip"
    if k == "drop_tip":
        return ".dropTip"
    if k == "transfer":
        return f".transfer {st[1]} {_lean_str(st[2])} {_lean_str(st[3])} ⟨{st[4]}⟩"
    if k == "mix":
        return f".mix {st[1]} {_lean_str(st[2])} ⟨{st[3]}⟩ {st[4]}"
    if k == "incubate":
        return f".incubate {_lean_str(st[1])} ⟨{st[2]}⟩ ⟨{st[3]}⟩"
    if k == "seal":
        return f".seal {_lean_str(st[1])}"
    if k == "thaw":
        return f".thaw {_lean_str(st[1])}"
    raise ValueError(k)


def lean_source(cases):
    """cases: list of (state, steps). One `#eval` per case keeps elaboration small."""
    parts = [_LEAN_PRELUDE]
    for i, (s, steps) in enumerate(cases):
        parts.append(f"def s{i} : LabState := {lean_state(s)}")
        parts.append(f"def p{i} : List Step := [{', '.join(lean_step(x) for x in steps)}]")
        parts.append(f'#eval IO.println ("C{i}|" ++ TF.runIdx s{i} 1 p{i})')
    return "\n".join(parts) + "\n"


def find_lake():
    for p in (shutil.which("lake"), os.path.expanduser("~/.elan/bin/lake")):
        if p and os.path.exists(p):
            return p
    return None


def lean_available(lablean_dir):
    return bool(lablean_dir) and find_lake() is not None and \
        os.path.exists(os.path.join(lablean_dir, ".lake", "build", "lib", "lean", "LabLean.olean"))


def parse_lean_line(rest):
    parts = rest.split("|")
    if parts[0] == "OK":
        vols = dict(kv.split("=") for kv in parts[1].split(",") if kv)
        return {"ok": True, "fail_step": None, "error": None,
                "volumes": {k: int(x) for k, x in vols.items()}}
    return {"ok": False, "fail_step": int(parts[1]), "error": parts[2], "volumes": None}


def lean_check(cases, lablean_dir, batch=150, timeout=900):
    """Run LabLean on every case. Returns a list of verdict dicts (same shape as `check`)."""
    lake = find_lake()
    out = []
    for start in range(0, len(cases), batch):
        chunk = cases[start:start + batch]
        with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False, encoding="utf-8") as f:
            f.write(lean_source(chunk))
            path = f.name
        try:
            r = subprocess.run([lake, "env", "lean", path], cwd=lablean_dir, capture_output=True,
                               text=True, timeout=timeout)
        finally:
            os.unlink(path)
        got = {}
        for line in r.stdout.splitlines():
            m = re.match(r"C(\d+)\|(.*)$", line)
            if m:
                got[int(m.group(1))] = parse_lean_line(m.group(2))
        if r.returncode != 0 or len(got) != len(chunk):
            raise RuntimeError(f"lean failed (code {r.returncode}): {(r.stderr or r.stdout)[-2000:]}")
        out.extend(got[i] for i in range(len(chunk)))
    return out
