"""Extract small-molecule names from public benchmark texts with an LLM (names only, no answers),
then resolve names to SMILES with PubChem PUG-REST. Cached."""
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from paths import CACHE

PC = CACHE / "pubchem.json"
pc = json.loads(PC.read_text(encoding="utf-8")) if PC.exists() else {}
MODEL = os.environ.get("EXTRACT_MODEL", "")
LLM_CMD = os.environ.get("LLM_CMD", "llm")


def llm(prompt):
    base = shlex.split(LLM_CMD)
    cmd = base + (["-m", MODEL] if MODEL else []) + [prompt]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=tempfile.gettempdir())
        return r.stdout
    except subprocess.TimeoutExpired:
        return ""


def extract(texts):
    body = "\n".join(f"<<{k}>> {t[:3000]}" for k, t in texts.items())
    p = (
        "For each text below, list the small-molecule chemicals named in it (reagents, salts, acids, bases, buffers, "
        "solvents, detergents, dyes, drugs). Exclude proteins, enzymes, antibodies, cells, tissues, nucleic acids, kits, "
        "media brand names, and generic words. Use the common chemical name. Output ONLY a JSON object mapping each key "
        "to a list of names.\n\n" + body
    )
    for _ in range(3):
        out = llm(p)
        m = re.search(r"\{.*\}", out, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        time.sleep(5)
    return {}


def resolve(name):
    key = name.lower().strip()
    if key in pc:
        return pc[key]
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        + urllib.parse.quote(key)
        + "/property/SMILES,MolecularWeight/JSON"
    )
    smi = None
    for a in range(3):
        try:
            d = json.load(urllib.request.urlopen(url, timeout=20))
            p = d["PropertyTable"]["Properties"][0]
            smi = p.get("SMILES") or p.get("IsomericSMILES")
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                break
            time.sleep(1 + a)
        except Exception:
            time.sleep(1 + a)
    pc[key] = smi
    time.sleep(0.25)
    return smi


def run(texts, out_path, batch=20, workers=4):
    out_path = Path(out_path)
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    todo = {k: v for k, v in texts.items() if k not in done}
    keys = list(todo)
    batches = [{k: todo[k] for k in keys[i : i + batch]} for i in range(0, len(keys), batch)]
    with ThreadPoolExecutor(workers) as ex:
        for res in ex.map(extract, batches):
            for k, names in res.items():
                if k in todo:
                    done[k] = {"names": names}
            out_path.write_text(json.dumps(done, indent=0), encoding="utf-8")
    for k, v in done.items():
        if "mols" in v:
            continue
        mols = []
        for n in v["names"]:
            s = resolve(n)
            if s:
                mols.append({"name": n, "smiles": s})
        v["mols"] = mols
    out_path.write_text(json.dumps(done, indent=0), encoding="utf-8")
    CACHE.mkdir(parents=True, exist_ok=True)
    PC.write_text(json.dumps(pc), encoding="utf-8")
    return done
