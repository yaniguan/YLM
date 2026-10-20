"""Generate free-form question templates with an LLM (one-off; the output is committed, nobody needs a key to reproduce).

For every task and every style family the LLM writes templates with {a}/{b} placeholders. A second, independent
call then checks each template's meaning against the exact definition; only templates judged faithful are kept.
Style families are the unit of the held-out split (a family is either all train or all test_heldout).

  ANTHROPIC_API_KEY=... python scripts/19_paraphrase_templates.py          # -> data/paraphrases.json (+ prints token use)
"""
import argparse
import json
import os
import re

import requests

MODEL = "claude-haiku-5-5"
URL = "https://api.anthropic.com/v1/messages"
STYLES = {
    "textbook": "formal physics-textbook phrasing",
    "terse": "very short imperative commands, a few words",
    "chatty": "casual, conversational, as a chemist asking a colleague",
    "technical": "precise technical phrasing using vector and tensor terminology",
    "indirect": "indirect or polite requests ('I would like to know...', 'Could you tell me...')",
    "inverted": "unusual word order, e.g. the target of the question comes first or the sentence is a statement to complete",
}
HELDOUT_STYLES = ["chatty", "inverted"]
TASKS = {
    "force": dict(ph=["a"], meaning="the 3D force VECTOR on atom {a}",
                  base="What is the force on {a}?"),
    "diff": dict(ph=["a", "b"], meaning="the 3D VECTOR F_a - F_b: the force on atom {a} minus the force on atom {b} (order matters)",
                 base="What is the difference between the force on {a} and the force on {b}?"),
    "proj": dict(ph=["a", "b"], meaning="the component of the force on atom {a} along the direction pointing from atom {a} toward atom {b}, as a 3D VECTOR parallel to that direction (not a scalar length)",
                 base="What is the part of the force on {a} that points toward {b}?"),
    "quad": dict(ph=[], meaning="the traceless electric quadrupole tensor of the whole molecule",
                 base="What is the quadrupole moment of the molecule?"),
}
USE = {"in": 0, "out": 0}


def call(prompt, max_tokens=8000):
    r = requests.post(URL, headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                                    "content-type": "application/json"},
                      json={"model": MODEL, "max_tokens": max_tokens, "temperature": 1.0,
                            "messages": [{"role": "user", "content": prompt}]}, timeout=180)
    r.raise_for_status()
    j = r.json()
    USE["in"] += j["usage"]["input_tokens"]
    USE["out"] += j["usage"]["output_tokens"]
    return "".join(b.get("text", "") for b in j["content"] if b["type"] == "text")


def json_list(text):
    i = text.find("[")
    if i < 0:
        return []
    try:
        return json.JSONDecoder().raw_decode(text[i:])[0]
    except json.JSONDecodeError:
        return []


def valid_shape(t, ph):
    found = re.findall(r"\{(\w+)\}", t)
    return sorted(found) == sorted(ph) and "\n" not in t.strip() and len(t) > 8


def generate(task, style, n):
    spec = TASKS[task]
    ph = ", ".join("{%s}" % p for p in spec["ph"]) or "(no placeholders)"
    prompt = (f"Write {n} different question or request templates asking for: {spec['meaning']}.\n"
              f"Reference wording: \"{spec['base']}\"\nStyle: {STYLES[style]}.\n"
              f"Each template must contain the placeholders {ph} exactly once each (they stand for a description of an atom, "
              f"such as 'the carbon atom bonded to two hydrogens'), keep the exact meaning, and be a single sentence. "
              f"Vary vocabulary and structure. Output only a JSON list of strings.")
    return [t.strip() for t in json_list(call(prompt)) if isinstance(t, str)]  # prompt asks for {n} candidates


def faithful(task, templates):
    spec = TASKS[task]
    meaning = spec["meaning"]
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(templates))
    prompt = (f"Target quantity: {meaning}.\n\nFor each template below, answer whether a physicist would read it as asking for "
              f"exactly that quantity, with the same sign and the same roles for the placeholders. Reject anything ambiguous, "
              f"reversed, or asking for something different (e.g. a magnitude, length or scalar, a different atom, or a sign flip). "
              f"Also reject any template that explicitly asks for a scalar.\n\n{numbered}\n\n"
              f"Output only a JSON list of the integer indices of the faithful templates.")
    keep = json_list(call(prompt, 8000))
    return [templates[i] for i in keep if isinstance(i, int) and 0 <= i < len(templates)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=24, help="templates requested per (task, style)")
    p.add_argument("--out", default="data/paraphrases.json")
    args = p.parse_args()
    out = {"model": MODEL, "heldout_styles": HELDOUT_STYLES, "templates": {}}
    for task, spec in TASKS.items():
        out["templates"][task] = {}
        for style in STYLES:
            raw = generate(task, style, args.n)
            shaped = [t for t in dict.fromkeys(raw) if valid_shape(t, spec["ph"])]
            kept = faithful(task, shaped) if shaped else []
            out["templates"][task][style] = kept
            print(f"{task:6s} {style:10s} requested {args.n:3d} generated {len(raw):3d} well-formed {len(shaped):3d} faithful {len(kept):3d}", flush=True)
    out["usage"] = dict(USE)
    json.dump(out, open(args.out, "w"), indent=1)
    print("tokens", USE)


if __name__ == "__main__":
    main()
