"""A small lab world that mirrors LabLean (YLM PR #92, commit a30571a, lean/LabLean/Semantics.lean
and Checker.lean).

Used by the lab token factory (src/lab_token_factory.py), and by the token-quality test
added in the stacked PR #99.

* `check(state, steps)` is a Python copy of LabLean's `run`: the start state is checked first
  (`initChecks`), then every step, with the same rules, the same order of checks and the same
  integer arithmetic. A start state error is reported at step 0.
* `lean_check(cases, lablean_dir)` writes the same protocols as a Lean file, runs it with
  `lake env lean`, and parses LabLean's own verdicts. Lean is the authority; the Python
  copy is only a second, independent check (the two must agree).

Units are LabLean's: volumes in 0.1 uL (`Vol.t`, so 2.5 uL is 25); reagent amounts inside a
container in 1e-5 uL (volume x AMT_SCALE); mass in mg; density in mg/mL. All steps are
mirrored: pick_tip, drop_tip, transfer, mix, incubate, centrifuge, seal, thaw.
"""
import copy
import os
import re
import shutil
import subprocess
import tempfile

PIPETTES = {"P2": (2, 20), "P20": (20, 200), "P200": (200, 2000), "P1000": (1000, 10000)}
STEP_KINDS = ("pick_tip", "drop_tip", "transfer", "mix", "incubate", "centrifuge", "seal", "thaw")
AMT_SCALE = 10000          # LabLean amtScale
ROTOR_TOL = 100            # mg, LabLean rotorTol
MAX_RPM = 15000
CONC_TOL_PERMILLE = 1


def fmt_vol(t):
    """0.1 uL units -> '12.5' (same digits as LabLean's Repr)."""
    return f"{t // 10}.{t % 10}"


def _div(a, b):
    """Lean `Nat` division: x / 0 = 0."""
    return a // b if b else 0


# ---------- state ----------

def container(cid, cap, vol=0, reagent=None, plate=None, tare=0):
    """A tube (plate=None) or a well of `plate`, holding `vol` of one reagent (or empty)."""
    return {"id": cid, "plate": plate, "cap": cap, "vol": vol, "tare": tare,
            "contents": [[reagent, vol * AMT_SCALE]] if reagent and vol else []}


def reagent(name, state="roomTemp", keep_cold=False, diluent=False, density=1000):
    return {"name": name, "state": state, "keep_cold": keep_cold, "diluent": diluent,
            "density": density}


def make_state(containers, reagents, plates=()):
    """reagents: dicts from `reagent()` (state in frozen/thawed/onIce/roomTemp).
    plates: (plate id, plastic mass in mg) pairs. The tip is off and nothing is sealed."""
    rs = [{**reagent(r["name"]), **r} for r in reagents]
    return {"cs": containers, "reagents": rs, "plates": [list(p) for p in plates],
            "tip": None, "sealed": []}


def find(s, cid):
    for c in s["cs"]:
        if c["id"] == cid:
            return c
    return None


def names(c):
    return [r for r, _ in c["contents"]]


def liquid(c):
    """LabLean `Container.liquid`: (amounts, volume), as remembered by a used tip."""
    return (tuple((r, x) for r, x in c["contents"]), c["vol"])


def amt(amts, r):
    for q, x in amts:
        if q == r:
            return x
    return 0


def touched(s):
    return s["tip"] if s["tip"] is not None else []


def plate_ids(s):
    return [p for p, _ in s["plates"]]


def is_sealed(s, c):
    return c["plate"] is not None and c["plate"] in s["sealed"]


def nodup(xs):
    return len(xs) == len(set(xs))


# ---------- mass (rotor balance) ----------

def density_of(s, r):
    for x in s["reagents"]:
        if x["name"] == r:
            return x["density"]
    return 1000


def mass_of(s, c):
    """mg = tare + (sum of amount (1e-5 uL) * density (mg/mL)) / 1e8, rounded down once."""
    return c["tare"] + sum(x * density_of(s, r) for r, x in c["contents"]) // 100000000


def item_mass(s, cid):
    """Mass of a tube, or of a plate (plastic + all its wells). None if unknown."""
    c = find(s, cid)
    if c is not None:
        return mass_of(s, c)
    for p, tare in s["plates"]:
        if p == cid:
            return tare + sum(mass_of(s, x) for x in s["cs"] if x["plate"] == cid)
    return None


def slot_mass(s, rotor, i):
    if i < len(rotor) and rotor[i] is not None:
        m = item_mass(s, rotor[i])
        return 0 if m is None else m
    return 0


# ---------- checks (same order as LabLean `checks` and `initChecks`) ----------

def is_diluent(s, r):
    return any(x["name"] == r and x["diluent"] for x in s["reagents"])


def may_enter(s, L, c):
    """A tip that touched liquid L may enter liquid c (both are (amounts, volume))."""
    la, lv = L
    ca, cv = c
    cnames = [r for r, _ in ca]
    return all(r in cnames and (is_diluent(s, r) or
                                1000 * amt(ca, r) * lv <= (1000 + CONC_TOL_PERMILLE) * amt(la, r) * cv)
               for r, _ in la)


def tip_clean(s, c):
    return all(may_enter(s, L, liquid(c)) for L in touched(s))


def is_item(s, x):
    """A tube (container not on a plate) or a declared plate."""
    return any(c["id"] == x and c["plate"] is None for c in s["cs"]) or x in plate_ids(s)


def _enter_checks(s, pip, cid, a, v):
    lo, hi = PIPETTES[pip]
    if s["tip"] is None:
        return ("noTip",)
    if not lo <= v <= hi:
        return ("outOfRange", pip, v)
    if is_sealed(s, a):
        return ("sealed", cid)
    if not tip_clean(s, a):
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
        if src == dst:
            return ("sameContainer", src)
        e = _enter_checks(s, pip, src, a, v)
        if e:
            return e
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
        return _enter_checks(s, pip, cid, a, v)
    if k == "incubate":
        _, target, temp, _secs = st
        if not is_item(s, target):
            return ("notTubeOrPlate", target)
        if not 4 <= temp <= 98:
            return ("badTemp", temp)
        return None
    if k == "centrifuge":
        _, rotor, rpm, _secs = st
        n = len(rotor)
        items = [x for x in rotor if x is not None]
        if not (n % 2 == 0 and n > 0):
            return ("badRotor", n)
        if rpm > MAX_RPM:
            return ("tooFast", rpm)
        if not nodup(items):
            return ("twiceInRotor",)
        for x in items:
            if not is_item(s, x):
                return ("notTubeOrPlate", x)
        for x in items:
            if x in plate_ids(s) and x not in s["sealed"]:
                return ("notSealed", x)
        for i in range(n // 2):
            a, b = slot_mass(s, rotor, i), slot_mass(s, rotor, i + n // 2)
            if not (a <= b + ROTOR_TOL and b <= a + ROTOR_TOL):
                return ("unbalanced", i, i + n // 2, a, b)
        return None
    if k == "seal":
        p = st[1]
        if p not in plate_ids(s):
            return ("unknown", p)
        if p in s["sealed"]:
            return ("alreadySealed", p)
        return None
    if k == "thaw":
        if not any(r["name"] == st[1] for r in s["reagents"]):
            return ("unknown", st[1])
        return None
    raise ValueError(f"unknown step {k}")


def start_error(s):
    """LabLean `initChecks`: the first problem with the start state, or None."""
    cs, pids = s["cs"], plate_ids(s)
    if not nodup([c["id"] for c in cs]):
        return ("duplicateIds",)
    for c in cs:
        if c["vol"] > c["cap"]:
            return ("overfull", c["id"])
    for c in cs:
        if c["id"] in pids:
            return ("idIsPlate", c["id"])
    for c in cs:
        if c["plate"] is not None and c["plate"] not in pids:
            return ("unknownPlate", c["plate"])
    if not nodup([r["name"] for r in s["reagents"]]):
        return ("duplicateReagents",)
    declared = {r["name"] for r in s["reagents"]}
    for c in cs:
        if not all(r in declared for r in names(c)):
            return ("undeclaredReagent", c["id"])
    for c in cs:
        if not (nodup(names(c)) and sum(x for _, x in c["contents"]) == c["vol"] * AMT_SCALE):
            return ("badContents", c["id"])
    return None


# ---------- effects (same integer arithmetic as LabLean `apply`) ----------

def _take_out(c, v):
    old = c["vol"]
    new = max(old - v, 0)                       # Nat subtraction
    c["contents"] = [[r, _div(x * new, old)] for r, x in c["contents"]]
    c["vol"] = new


def _put_in(a, v, c):
    """Put `v` of the liquid of `a` into `c` (LabLean `putIn`, with `addAmt`)."""
    for r, x in a["contents"]:
        add = _div(x * v, a["vol"])
        for item in c["contents"]:
            if item[0] == r:
                item[1] += add
                break
        else:
            c["contents"].append([r, add])
    c["vol"] += v


def _touch(s, ls):
    acc = list(touched(s))
    for L in ls:
        if L not in acc:
            acc.insert(0, L)
    return acc


def apply(s, st):
    s = copy.deepcopy(s)
    k = st[0]
    if k == "pick_tip":
        s["tip"] = []          # a fresh tip has touched nothing
    elif k == "drop_tip":
        s["tip"] = None
    elif k == "transfer":
        _, _pip, src, dst, v = st
        a, b = find(s, src), find(s, dst)
        if a is None or b is None:
            return s
        a0, b_after = copy.deepcopy(a), copy.deepcopy(b)
        _put_in(a0, v, b_after)
        # the tip touches the source, then the destination after the dispense
        s["tip"] = _touch(s, [liquid(a0), liquid(b_after)])
        _take_out(a, v)        # modifyC src first, then modifyC dst (same order as Lean)
        _put_in(a0, v, find(s, dst))
    elif k == "mix":
        a = find(s, st[2])
        if a is not None:
            s["tip"] = _touch(s, [liquid(a)])
    elif k == "seal":
        s["sealed"] = [st[1]] + s["sealed"]
    elif k == "thaw":
        for r in s["reagents"]:
            if r["name"] == st[1] and r["state"] == "frozen":
                r["state"] = "onIce" if r["keep_cold"] else "thawed"
    return s


def check(s, steps):
    """Python copy of LabLean `run`. Returns dict(ok, fail_step (1-based; 0 = start state),
    error, volumes)."""
    e = start_error(s)
    if e is not None:
        return {"ok": False, "fail_step": 0, "error": err_tag(e), "volumes": None}
    for i, st in enumerate(steps, 1):
        e = first_error(s, st)
        if e is not None:
            return {"ok": False, "fail_step": i, "error": err_tag(e), "volumes": None}
        s = apply(s, st)
    return {"ok": True, "fail_step": None, "error": None,
            "volumes": {c["id"]: c["vol"] for c in s["cs"]}}


def trace(s, steps):
    """Per-step states for record text: list of (step, error or None, changed volumes).
    Empty if the start state itself is not well formed."""
    out = []
    if start_error(s) is not None:
        return out
    for st in steps:
        e = first_error(s, st)
        if e is not None:
            out.append((st, err_tag(e), {}))
            break
        before = {c["id"]: c["vol"] for c in s["cs"]}
        s = apply(s, st)
        out.append((st, None, {c["id"]: c["vol"] for c in s["cs"] if c["vol"] != before[c["id"]]}))
    return out


_VOL_ARGS = {"outOfRange": (2,), "notEnough": (2, 3), "overflow": (2, 3, 4)}


def err_tag(e):
    """Compact error text with volumes in uL (masses in mg); the Lean prelude below prints the
    same text."""
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
    if k == "centrifuge":
        return f"centrifuge [{' '.join(x or '-' for x in st[1])}] {st[2]}rpm {st[3]}s"
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
  | .notTubeOrPlate i => s!"notTubeOrPlate {i}"
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
  | .twiceInRotor => "twiceInRotor"
  | .notSealed p => s!"notSealed {p}"
  | .unbalanced a b x y => s!"unbalanced {a} {b} {x} {y}"
  | .alreadySealed p => s!"alreadySealed {p}"
  | .duplicateIds => "duplicateIds"
  | .overfull i => s!"overfull {i}"
  | .idIsPlate i => s!"idIsPlate {i}"
  | .unknownPlate p => s!"unknownPlate {p}"
  | .duplicateReagents => "duplicateReagents"
  | .undeclaredReagent i => s!"undeclaredReagent {i}"
  | .badContents i => s!"badContents {i}"
/-- LabLean's own `step`, with the 1-based index of the failing step. -/
def runIdx (s : LabState) : Nat → List Step → String
  | _, [] => "OK|" ++ ",".intercalate (s.cs.map fun c => s!"{c.id}={c.vol.t}")
  | i, st :: rest =>
    match LabLean.step s st with
    | .error e => s!"FAIL|{i}|{tag e}"
    | .ok s' => runIdx s' (i + 1) rest
/-- LabLean's `run`: the start state (`initChecks`, reported as step 0), then every step. -/
def runAll (s : LabState) (l : List Step) : String :=
  match firstErr (initChecks s) with
  | some e => s!"FAIL|0|{tag e}"
  | none => runIdx s 1 l
end TF
"""

_STATE = {"frozen": ".frozen", "thawed": ".thawed", "onIce": ".onIce", "roomTemp": ".roomTemp"}


def _lean_str(x):
    assert re.fullmatch(r"[A-Za-z0-9_:]+", x), x
    return f'"{x}"'


def _lean_bool(b):
    return "true" if b else "false"


def _lean_nat(x):
    assert isinstance(x, int) and x >= 0, x
    return str(x)


def lean_state(s):
    cs = []
    for c in s["cs"]:
        plate = f"some {_lean_str(c['plate'])}" if c["plate"] is not None else "none"
        contents = ", ".join(f"({_lean_str(r)}, {_lean_nat(x)})" for r, x in c["contents"])
        cs.append(f"{{ id := {_lean_str(c['id'])}, plate := {plate}, cap := ⟨{_lean_nat(c['cap'])}⟩, "
                  f"vol := ⟨{_lean_nat(c['vol'])}⟩, contents := [{contents}], "
                  f"tare := {_lean_nat(c['tare'])} }}")
    rs = [f"{{ name := {_lean_str(r['name'])}, state := {_STATE[r['state']]}, "
          f"keepCold := {_lean_bool(r['keep_cold'])}, diluent := {_lean_bool(r['diluent'])}, "
          f"density := {_lean_nat(r['density'])} }}" for r in s["reagents"]]
    ps = [f"({_lean_str(p)}, {_lean_nat(m)})" for p, m in s["plates"]]
    assert s["tip"] is None and not s["sealed"]
    return ("{ cs := [" + ", ".join(cs) + "], reagents := [" + ", ".join(rs) + "], plates := ["
            + ", ".join(ps) + "] }")


def lean_step(st):
    k = st[0]
    if k == "pick_tip":
        return ".pickTip"
    if k == "drop_tip":
        return ".dropTip"
    if k == "transfer":
        return f".transfer {st[1]} {_lean_str(st[2])} {_lean_str(st[3])} ⟨{_lean_nat(st[4])}⟩"
    if k == "mix":
        return f".mix {st[1]} {_lean_str(st[2])} ⟨{_lean_nat(st[3])}⟩ {_lean_nat(st[4])}"
    if k == "incubate":
        assert isinstance(st[2], int), st[2]
        return f".incubate {_lean_str(st[1])} ⟨({st[2]} : Int)⟩ ⟨{_lean_nat(st[3])}⟩"
    if k == "centrifuge":
        rotor = ", ".join("none" if x is None else f"some {_lean_str(x)}" for x in st[1])
        return f".centrifuge [{rotor}] {_lean_nat(st[2])} ⟨{_lean_nat(st[3])}⟩"
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
        parts.append(f'#eval IO.println ("C{i}|" ++ TF.runAll s{i} p{i})')
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
