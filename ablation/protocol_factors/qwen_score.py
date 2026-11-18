"""Score items with a small open LLM (Qwen2.5-0.5B-Instruct, CPU) by answer log-likelihood."""
import torch, math, os
from transformers import AutoModelForCausalLM, AutoTokenizer
NAME = "Qwen/Qwen2.5-0.5B-Instruct"
torch.set_num_threads(4)
_tok = _m = None

def load():
    global _tok, _m
    if _m is None:
        _tok = AutoTokenizer.from_pretrained(NAME)
        _m = AutoModelForCausalLM.from_pretrained(NAME, dtype=getattr(torch, os.environ.get("QWEN_DTYPE", "bfloat16"))).eval()
    return _tok, _m

def chat_prefix(user):
    tok, _ = load()
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

@torch.no_grad()
def option_logprobs(user, options, prefix_answer="Answer: "):
    """Return log P(option | prompt) for each option string (one forward over prompt, then continuations)."""
    tok, m = load()
    p = chat_prefix(user) + prefix_answer
    ids = tok(p, return_tensors="pt").input_ids[:, -int(os.environ.get("QWEN_MAX_TOKENS", "2048")):]
    out = m(ids, use_cache=True); past = out.past_key_values; last = out.logits[0, -1].float().log_softmax(-1)
    res = []
    for o in options:
        oid = tok(o, add_special_tokens=False).input_ids
        lp = last[oid[0]].item()
        if len(oid) > 1:
            import copy
            o2 = m(torch.tensor([oid[:-1]]), past_key_values=copy.deepcopy(past), use_cache=True)
            lps = o2.logits[0].float().log_softmax(-1)
            lp += sum(lps[i, oid[i + 1]].item() for i in range(len(oid) - 1))
        res.append(lp)
    return res
