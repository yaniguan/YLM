"""Run Qwen2.5-0.5B over items x conditions. Resumable. Output: results/qwen_<bench>.jsonl"""
import json, sys, math, os
from pathlib import Path
from render import facts, factorial_conditions, make_smap
from adapters import prompt_mcq, prompt_err, permute
import qwen_score as Q
import reps
from paths import CACHE

def conds_for(names):
    c = factorial_conditions(); c.update(reps.rep_conditions())
    return {n: c[n] for n in names}

def main(item_file, out_file, cond_names):
    items = [json.loads(l) for l in open(item_file)]
    fm = json.load(open(CACHE / "fm_pred.json")); smap = make_smap(items)
    done = set()
    if Path(out_file).exists():
        for l in open(out_file):
            r = json.loads(l); done.add((r["id"], r["cond"]))
    f = open(out_file, "a")
    for cn, cond in conds_for(cond_names).items():
        for it in items:
            if (it["id"], cn) in done: continue
            block = facts(it, cond, fm, smap) if cond.get("kind") != "rep" or cond.get("with_layers") else facts(it, {"layers": set(), "identity": True}, fm, smap)
            rep_text = reps.rep_text(it, cond) if cond.get("kind") == "rep" else None
            if it["task"] == "error":
                p = prompt_err(it, block, rep_text) + "\nAnswer Yes or No."
                lp = Q.option_logprobs(p, ["Yes", "No"])
                pe = 1 / (1 + math.exp(lp[1] - lp[0]))
                rec = {"id": it["id"], "cond": cn, "p_error": pe, "pred_error": pe > 0.5, "is_error": it["is_error"]}
                rec["correct"] = rec["pred_error"] == it["is_error"]
            else:
                p = prompt_mcq(it, block, rep_text) + "\nAnswer with the letter."
                ch, a = permute(it)
                lp = Q.option_logprobs(p, [chr(65 + i) for i in range(len(ch))])
                pred = max(range(len(ch)), key=lambda i: lp[i])
                rec = {"id": it["id"], "cond": cn, "pred": pred, "answer": a, "correct": pred == a}
            f.write(json.dumps(rec) + "\n"); f.flush()

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(","))
