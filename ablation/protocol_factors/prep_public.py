"""Build ID lists and resolved chemical names for public protocol benches.

LAB-Bench ProtocolQA public split (CC-BY-SA 4.0) and BioProBench ERR test
(CC-BY-NC 4.0, research use only) are read from local files set in the
environment. This repo stores only item IDs and resolved molecule names,
never the benchmark text or answers.
"""
import json, os, random
from pathlib import Path
from extract_chems import run
from paths import DATA

def main():
    pq = [json.loads(l) for l in open(os.environ["PROTOCOLQA_FILE"])]
    err = json.load(open(os.environ["BIOPROBENCH_ERR_FILE"]))
    rng = random.Random(0)
    err_c = [x for x in err if x["is_correct"]]; err_w = [x for x in err if not x["is_correct"]]
    rng.shuffle(err_c); rng.shuffle(err_w)
    pool = err_c[:300] + err_w[:300]
    texts = {x["id"]: x["protocol"] + "\n" + x["question"] for x in pq}
    texts.update({x["id"]: (x["corrupted_text"] or x["corrected_text"]) for x in pool})
    done = run(texts, DATA / "public_chems.json")
    # Drop any residual source text; keep ids, names, and SMILES only.
    slim = {k: {"names": v.get("names", []), "mols": v.get("mols", [])} for k, v in done.items()}
    (DATA / "public_chems.json").write_text(json.dumps(slim, indent=0))
    n_mol = sum(1 for v in slim.values() if v.get("mols"))
    print(len(slim), "ids;", n_mol, "with at least one resolved molecule")

if __name__ == "__main__":
    main()
