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
    P = json.load(open("data/paraphrases.json"))
    verdict = {}
    for task, styles in P["templates"].items():
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


if __name__ == "__main__":
    main()
