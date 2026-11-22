"""Auto-convert item text to (R2) structured scientific form and (R3) LabLean code; run the Lean checker (R4).
The converter is told to transcribe faithfully and never to fix or flag errors."""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from paths import DATA, ROOT

LEAN_DIR = os.environ.get("LABLEAN_DIR", str(ROOT.parents[1] / "lean"))
TMP = Path(os.environ.get("LEAN_TMP", str(ROOT / "leantmp"))).resolve()
TMP.mkdir(exist_ok=True)
MODEL = os.environ.get("CONVERT_MODEL", "")
LLM_CMD = os.environ.get("LLM_CMD", "llm")

R2_PROMPT = """Rewrite the following lab text in a structured scientific form. Use these sections:
CONTAINERS: each container with capacity and initial contents (units).
STEPS: each step as a state transition with explicit quantities and SI-style units.
BALANCE: running volume (and mass or moles where given) per container after each step.
REACTIONS: balanced chemical equations for any chemical reaction (or "none").
STATE: temperature, pH, atmosphere, and other state variables per step where stated.
Transcribe faithfully. Do NOT correct, judge, warn about, or flag anything, even if it looks wrong. No commentary.
If the text is a question with options, structure the givens and each option the same way instead.

TEXT:
"""

R3_PROMPT = """Translate the lab protocol below into LabLean, a Lean 4 DSL. Output ONLY Lean code (no fences) defining
`def init : LabState := ...` and `def steps : List Step := [...]`, or the single word NOT_EXPRESSIBLE if no step is liquid handling.
Transcribe faithfully: do NOT correct or flag anything, even if it looks wrong. Map amounts exactly.
LabLean API (namespace opened):
- Vol literals are microliters: (200 : Vol) = 200 uL, (2.5 : Vol) = 2.5 uL. mL must be converted (1.5 mL = 1500).
- Temp literals are whole degrees C. Time: (sec 60), (mins 10).
- Reagent: { name := "PBS", state := .roomTemp }  -- states: .frozen | .thawed | .onIce | .roomTemp
- tube (id : String) (reagent : String) (v : Vol) (cap : Vol := 1500) : Container   -- a tube holding v of one reagent
- empty container: { id := "W1", cap := 1500, vol := 0 } ; plate well: { id := "A1", plate := some "P1", cap := 300, vol := 0 }
- LabState: { reagents := [...], cs := [...], plates := [("P1", 14000)] }
- Pipettes: P2 (0.2-2 uL), P20 (2-20), P200 (20-200), P1000 (100-1000)
- Steps: .pickTip | .dropTip | .transfer P200 "SRC" "DST" 100 | .mix P200 "A1" 100 3 | .incubate "A1" 37 (mins 10)
  | .centrifuge [some "T1", none, some "T2", none] 5000 (sec 60) | .seal "P1" | .thaw "enzyme"
- A "fresh tip" means .pickTip before and .dropTip after. Steps that are not liquid handling (weighing, stirring,
  heating outside 4-98 C, vortexing) are omitted with a Lean comment `-- not modeled: ...`.
Example:
def init : LabState :=
  { reagents := [ { name := "r1", state := .roomTemp }, { name := "enzyme", state := .frozen } ],
    cs := [ tube "R1" "r1" 500, tube "ENZ" "enzyme" 50, { id := "A1", plate := some "P", cap := 200, vol := 0 } ],
    plates := [("P", 14000)] }
def steps : List Step := [.thaw "enzyme", .pickTip, .transfer P20 "ENZ" "A1" 5, .dropTip, .seal "P"]

PROTOCOL:
"""


def llm(prompt):
    base = shlex.split(LLM_CMD)
    cmd = base + (["-m", MODEL] if MODEL else []) + [prompt]
    for _ in range(3):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=tempfile.gettempdir())
            if r.stdout.strip() and "RESOURCE_EXHAUSTED" not in r.stderr:
                return r.stdout.strip()
        except Exception:
            pass
    return None


def lean_check(code):
    code = re.sub(r"^```\w*|```$", "", code.strip(), flags=re.M).strip()
    h = hashlib.md5(code.encode()).hexdigest()[:12]
    f = TMP / f"c_{h}.lean"
    f.write_text("import LabLean\nopen LabLean\n\n" + code + "\n\n#eval report init steps\n", encoding="utf-8")
    r = subprocess.run(
        ["lake", "env", "lean", str(f)],
        cwd=LEAN_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )
    out = (r.stdout + r.stderr).strip()
    m = re.search(r'"(OK|REJECTED[^"]*(?:\\"[^"]*)*)"', out)
    if r.returncode == 0 and m:
        return m.group(1).replace('\\"', '"'), code
    return "Lean compile error: " + (out.splitlines()[0][:200] if out else "unknown"), code


def convert(it, do_r3):
    text = (
        it["protocol"]
        if it["task"] == "error"
        else (it["stem"] + "\n" + "\n".join(f"- {c}" for c in it["choices"]))
    )
    rec = {"structured": llm(R2_PROMPT + text)}
    if do_r3:
        code = llm(R3_PROMPT + text)
        if code and "NOT_EXPRESSIBLE" not in code[:40]:
            rec["verdict"], rec["lean"] = lean_check(code)
    return it["id"], rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--r2", action="store_true", help="only generate R2 structured text (skip R3/R4 Lean)")
    a = ap.parse_args()
    items = [json.loads(line) for line in open(a.items, encoding="utf-8")]
    bench = items[0]["bench"]
    out = DATA / f"reps_{bench}.json"
    res = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    todo = [it for it in items if it["id"] not in res or not res[it["id"]].get("structured")]
    with ThreadPoolExecutor(a.workers) as ex:
        for k, (iid, rec) in enumerate(
            ex.map(lambda it: convert(it, (not a.r2) and bool(it.get("lablean_expressible"))), todo)
        ):
            res[iid] = rec
            if k % 10 == 0:
                out.write_text(json.dumps(res, indent=0), encoding="utf-8")
                print(k, flush=True)
    out.write_text(json.dumps(res, indent=0), encoding="utf-8")


if __name__ == "__main__":
    main()
