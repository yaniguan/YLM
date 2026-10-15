"""Sanity check for the LLM baseline: can it memorise 16 force questions (atom hint on)? Usage: lr head_lr steps."""
import sys, json, math, importlib.util, torch, numpy as np
spec = importlib.util.spec_from_file_location("m", "scripts/05_train_llm_baseline.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
lr, head_lr, steps = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(0)
labels, splits = m.load("data/qm9_xtb.pkl", "data/tensorqa_v0.json")
qs = [q for q in splits["train"] if q["task"] == "force"][:16]
scale = torch.tensor([0.8147, 0.0604], device="cuda")
tok = m.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B"); tok.padding_side = "right"
model = m.LLMRegressor("Qwen/Qwen2.5-0.5B").cuda()
opt = torch.optim.AdamW([{"params": model.llm.parameters(), "lr": lr}, {"params": model.head.parameters(), "lr": head_lr}], weight_decay=0)
for step in range(steps):
    chunk = qs[(step % 2) * 8:(step % 2) * 8 + 8]
    ids, am, tgt, task = m.get_batch(labels, chunk, tok, "cuda", hint=True)
    y = model(ids, am)
    loss = ((y - tgt / scale[task][:, None]) ** 2).mean()
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 5 == 0: print(f"lr {lr} step {step} loss {loss.item():.4f}", flush=True)
