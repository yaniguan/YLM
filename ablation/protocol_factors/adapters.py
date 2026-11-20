"""Convert every benchmark into one item format and build prompts that differ only in the facts block."""
import json, random, string
from paths import DATA
L = string.ascii_uppercase

def calisol_items(split):
    out = []
    for l in open(DATA / f"calisol_{split}.jsonl"):
        r = json.loads(l); mols = {}
        for f in r["formulations"]:
            mols[f["salt"]] = f["salt_smiles"]
            for s in f["solvents"]: mols[s["name"]] = s["smiles"]
        out.append({"id": r["id"], "bench": "calisol", "task": "reasoning", "label": "solvation", "dependent": True,
                    "stem": r["question"].split("\n")[0], "choices": [r["question"].split("\n")[1][3:], r["question"].split("\n")[2][3:]],
                    "answer_index": r["answer_index"], "molecules": [{"name": k, "smiles": v} for k, v in mols.items()],
                    "context": {"T_C": round(r["formulations"][0]["T_K"] - 273.15)}, "split": r["split"], "group": r["id"]})
    return out

def permute(item, seed=0):
    """Fixed cyclic shift of MCQ options per item (same for all conditions)."""
    n = len(item["choices"]); k = random.Random(item["id"] + str(seed)).randrange(n)
    ch = item["choices"][k:] + item["choices"][:k]
    return ch, (item["answer_index"] - k) % n

INSTR = "You are an experienced lab scientist. Use any provided information if it helps."

def prompt_mcq(item, block, rep_text=None):
    ch, _ = permute(item)
    body = rep_text if rep_text is not None else item.get("context_text", "")
    s = INSTR + "\n\n"
    if block: s += "Information:\n" + block + "\n\n"
    if body: s += body + "\n\n"
    s += "Question: " + item["stem"] + "\n" + "\n".join(f"{L[i]}) {c}" for i, c in enumerate(ch))
    return s

def prompt_err(item, block, rep_text=None):
    body = rep_text if rep_text is not None else item["protocol"]
    s = INSTR + "\n\n"
    if block: s += "Information:\n" + block + "\n\n"
    s += item.get("err_preamble", "Protocol:") + "\n" + body + "\n\n"
    s += item.get("err_question", "Does this protocol contain an error that would make it fail, give a wrong result, or be unsafe?")
    return s
