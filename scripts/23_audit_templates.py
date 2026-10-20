"""Independent audit of the LLM-generated templates with a different, stronger model than the generator.

Each template is judged against the exact definition of its task. Verdicts: ok / ambiguous / wrong (+ reason).
This is NOT a substitute for a human check: it produces data/audit_templates.json and a CSV for human review
(data/audit_sample_200.csv: 200 filled-in questions, stratified by task and style, to be marked ok/wrong/ambiguous).

  ANTHROPIC_API_KEY=... python scripts/23_audit_templates.py
"""
import csv
import json
import os
import random
import re

import requests

MODEL = "claude-sonnet-5-5"
DEF = {
    "force": "the 3D force vector on the atom {a}",
    "diff": "the 3D vector F_a - F_b (force on {a} minus force on {b}); a reader must be able to tell which atom is subtracted",
    "proj": "the signed vector projection of the force on {a} onto the straight line through {a} and {b}: a vector parallel to "
            "that line (points toward {b} if the force has a positive component that way, away otherwise). {a} and {b} are "
            "ANY two atoms, not necessarily bonded to each other",
    "quad": "the traceless electric quadrupole tensor of the whole molecule",
    "pol_iso": "the isotropic (orientation-averaged) static polarizability of the molecule, one scalar",
    "pol_aniso": "the traceless anisotropic part of the static polarizability tensor (symmetric traceless 3x3), not the isotropic average",
    "pol_dir": "the SCALAR u^T alpha u: the polarizability along the direction from atom {a} to atom {b} ({a}, {b} are any two atoms, not necessarily bonded)",
    "pol_ind": "the 3D VECTOR alpha u: the dipole induced by a unit electric field pointing from atom {a} toward atom {b} ({a}, {b} any two atoms, not necessarily bonded)",
}


def call(prompt):
    r = requests.post("https://api.anthropic.com/v1/messages",
                      headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                               "content-type": "application/json"},
                      json={"model": MODEL, "max_tokens": 16000, "messages": [{"role": "user", "content": prompt}]},
                      timeout=300)
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"] if b["type"] == "text")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/paraphrases.json")
    ap.add_argument("--tasks", default=None, help="comma list; default all")
    ap.add_argument("--apply", action="store_true", help="rewrite --file keeping only templates judged ok (and without 'bond' wording)")
    args = ap.parse_args()
    P = json.load(open(args.file))
    verdict = json.load(open("data/audit_templates.json")) if os.path.exists("data/audit_templates.json") else {}
    for task, styles in P["templates"].items():
        if args.tasks and task not in args.tasks.split(","):
            continue
        for style, temps in styles.items():
            for i in range(0, len(temps), 30):
                chunk = temps[i:i + 30]
                listing = "\n".join(f"{j}: {t}" for j, t in enumerate(chunk))
                prompt = (f"Task definition: the question must ask for {DEF[task]}.\n\nJudge every template strictly. "
                          f"'wrong' = asks for something else, reverses roles/sign, asks for a scalar/magnitude, or implies a "
                          f"false physical fact (e.g. that the two atoms are bonded). 'ambiguous' = a careful reader could "
                          f"reasonably take it another way (e.g. unclear which atom is subtracted, or 'difference' with no "
                          f"order). Otherwise 'ok'.\n\n{listing}\n\nReturn only a JSON list of "
                          f"[index, verdict, short reason] triples, one per template.")
                text = call(prompt)
                try:
                    triples = json.JSONDecoder().raw_decode(text[text.find("["):])[0]
                except Exception:
                    triples = []
                for j, v, why in triples:
                    verdict[chunk[j]] = dict(task=task, style=style, verdict=v, why=why)
        print(task, "audited", sum(1 for v in verdict.values() if v["task"] == task), flush=True)
    json.dump(verdict, open("data/audit_templates.json", "w"), indent=1)
    from collections import Counter
    print(Counter((v["task"], v["verdict"]) for v in verdict.values()))
    if args.apply:
        bad = re.compile(r"\bbond(?!ed to)\w*|\bbonding\b", re.I)
        removed = []
        for task, styles in P["templates"].items():
            if args.tasks and task not in args.tasks.split(","):
                continue
            for style, temps in styles.items():
                keep = [t for t in temps if verdict.get(t, {}).get("verdict") == "ok" and not bad.search(t)]
                removed += [dict(task=task, style=style, template=t) for t in temps if t not in keep]
                styles[style] = keep
        json.dump(P, open(args.file, "w"), indent=1)
        print("removed", len(removed))


if __name__ == "__main__":
    main()
