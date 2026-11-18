"""Token-quality test: are checked agent tokens worth more than web text, token for token?

Three training arms get the same number of tokens (characters):
  (a) web       real protocol text written by people (Wet Lab Protocols corpus, MIT license)
  (b) agent     every record the lab agent wrote (src/lab_token_factory.py), unchecked
  (c) verified  only the records that passed the checkers (LabLean, Python copy, volumes, dedup)
plus (o) oracle, text in the eval format itself (a positive control), and an untrained model.

The eval is held out (no protocol in training) and scored by the checker: LabLean gives the
true outcome of every eval protocol. Two kinds of 4-option items:
  verdict  "what happens?"  pass, or fail at step k with a rule name
  volume   "final volume of the target well?" (only protocols that pass)
A model picks the option with the best calibrated score:
  log p(option | question) - log p(option | "A: ").

Tokens are characters (vocab = newline + printable ASCII), so every arm is cut to exactly D
characters. The model is a small GPT trained from scratch on CPU.
"""
import glob
import hashlib
import math
import os
import random
import unicodedata

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from src import lab_token_factory as F
from src import lab_world as W

VOCAB = ["\n"] + [chr(i) for i in range(32, 127)]
STOI = {c: i for i, c in enumerate(VOCAB)}
CTX = 512


def to_ascii(text):
    text = text.replace("µ", "u").replace("μ", "u").replace("°", " deg ").replace("\t", " ")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if c == "\n" or 32 <= ord(c) < 127)


def encode(text):
    return [STOI[c] for c in text]


# ---------- training pools ----------

def web_docs(web_dir):
    """Every protocol .txt in the Wet Lab Protocols corpus, as ASCII documents (sorted, fixed)."""
    paths = sorted(glob.glob(os.path.join(web_dir, "**", "*.txt"), recursive=True))
    docs = [to_ascii(open(p, encoding="utf-8", errors="ignore").read()).strip() for p in paths]
    return [d for d in docs if d]


def agent_records(seed, n_tasks, lablean_dir, p_blind=0.5):
    """A fixed agent (no learning) writes n_tasks protocol records at levels 1-4; all are checked."""
    rng = random.Random(seed)
    agent = F.Agent(rng, p_blind=p_blind, learn=False)
    recs = []
    for i in range(n_tasks):
        recs += F.make_protocol_records(rng, 1, 1 + i % 4, agent)
    used_lean = F.verify_protocols(recs, lablean_dir)
    seen = set()
    for r in recs:                                   # the gate also drops exact duplicates
        h = F.sha(r["text"])
        if r["trust"] == "verified" and h in seen:
            r["trust"] = "rejected"
            r["first_fail"] = "dedup"
        seen.add(h)
    return recs, used_lean


def protocol_hash(state, steps):
    return hashlib.sha256(repr((state, steps)).encode()).hexdigest()


def stream(docs, budget, rng):
    """Shuffle documents, join with newlines, cut to exactly `budget` characters."""
    docs = list(docs)
    rng.shuffle(docs)
    out, n = [], 0
    for d in docs:
        out.append(d)
        n += len(d) + 1
        if n >= budget:
            break
    text = "\n".join(out)
    if len(text) < budget:
        raise ValueError(f"only {len(text)} characters available, need {budget}")
    return text[:budget]


# ---------- eval (checker-scored) ----------

RULE_WORDS = ["overflow", "notThawed", "outOfRange", "noTip", "contaminated", "notEnough",
              "badTemp", "sealed"]


def question_text(task, steps):
    body = "; ".join(f"{i} {W.step_text(s)}" for i, s in enumerate(steps, 1))
    return f"Q: start {F.start_text(task)}. steps: {body}."


def outcome_text(truth):
    return "pass" if truth["ok"] else f"fail at step {truth['fail_step']}: {truth['error'].split()[0]}"


def pad(options):
    n = max(len(o) for o in options)
    return [o + "." * (n - len(o)) for o in options]


def make_items(rng, task, steps, truth):
    """Return a list of eval items for one protocol (1 verdict item, maybe 1 volume item)."""
    items = []
    q = question_text(task, steps)
    gold = outcome_text(truth)
    opts = {gold}
    while len(opts) < 4:
        k = rng.randint(1, len(steps))
        cand = rng.choice(["pass", f"fail at step {k}: {rng.choice(RULE_WORDS)}"])
        if not truth["ok"] and rng.random() < 0.5:     # near misses: same step or same rule
            cand = rng.choice([f"fail at step {truth['fail_step']}: {rng.choice(RULE_WORDS)}",
                               f"fail at step {k}: {truth['error'].split()[0]}"])
        opts.add(cand)
    opts = sorted(opts)
    rng.shuffle(opts)
    items.append({"kind": "verdict", "q": q + " what happens? A: ", "options": pad(opts),
                  "answer": opts.index(gold)})
    if truth["ok"]:
        v = truth["volumes"][task["target"]]
        vals = {v}
        while len(vals) < 4:
            d = rng.choice([-1, 1]) * rng.choice([50, 100, 200, 500, 1000])
            if v + d > 0:
                vals.add(v + d)
        vals = sorted(vals)
        rng.shuffle(vals)
        opts = [f"{W.fmt_vol(x)} uL" for x in vals]
        items.append({"kind": "volume", "q": q + f" final volume in {task['target']}? A: ",
                      "options": pad(opts), "answer": vals.index(v)})
    return items


def build_eval(seed, n_protocols, lablean_dir, exclude_hashes, p_err=0.15):
    """Held-out protocols with outcomes from LabLean (or the Python copy without Lean)."""
    rng = random.Random(seed)
    tasks = []
    while len(tasks) < n_protocols:
        task = F.propose_task(rng, 1 + len(tasks) % 4)
        steps, _ = F.propose_protocol(rng, task, p_err)
        h = protocol_hash(task["state"], steps)
        if h not in exclude_hashes:
            exclude_hashes.add(h)
            tasks.append((task, steps))
    cases = [(t["state"], s) for t, s in tasks]
    truths = W.lean_check(cases, lablean_dir) if W.lean_available(lablean_dir) else \
        [W.check(*c) for c in cases]
    items = []
    for (task, steps), truth in zip(tasks, truths):
        items += make_items(rng, task, steps, truth)
    return items


def oracle_docs(seed, n_protocols, lablean_dir, exclude_hashes):
    """Positive control: eval-format text (question + right answer) on other protocols."""
    items = build_eval(seed, n_protocols, lablean_dir, exclude_hashes)
    return [it["q"] + it["options"][it["answer"]].rstrip(".") for it in items]


# ---------- model ----------

class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.h, self.drop = h, drop
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        y = Fn.scaled_dot_product_attention(q, k, v, is_causal=True,
                                            dropout_p=self.drop if self.training else 0.0)
        x = x + Fn.dropout(self.proj(y.transpose(1, 2).reshape(b, t, d)), self.drop, self.training)
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """4 layers, d=128, 4 heads, context 512, tied embeddings (about 0.85M parameters)."""

    def __init__(self, n_layer=4, d=128, h=4, drop=0.1):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), d)
        self.pos = nn.Embedding(CTX, d)
        self.blocks = nn.ModuleList(Block(d, h, drop) for _ in range(n_layer))
        self.ln = nn.LayerNorm(d)
        self.drop = drop
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1]))
        x = Fn.dropout(x, self.drop, self.training)
        for b in self.blocks:
            x = b(x)
        return self.ln(x) @ self.tok.weight.T


def lr_at(step, total, base, warmup=50):
    if step < warmup:
        return base * (step + 1) / warmup
    return 0.5 * base * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


def train(text, seed, steps=500, batch=8, lr=1e-3):
    """Train from scratch on random 513-character windows of `text`. Returns (model, losses)."""
    torch.manual_seed(seed)
    model = TinyGPT()
    data = torch.tensor(encode(text), dtype=torch.long)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for s in range(steps):
        ix = torch.randint(0, len(data) - CTX - 1, (batch,), generator=g)
        x = torch.stack([data[i:i + CTX] for i in ix])
        y = torch.stack([data[i + 1:i + CTX + 1] for i in ix])
        for group in opt.param_groups:
            group["lr"] = lr_at(s, steps, lr)
        model.train()
        loss = Fn.cross_entropy(model(x).reshape(-1, len(VOCAB)), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    return model, losses


def untrained(seed):
    torch.manual_seed(seed)
    return TinyGPT()


@torch.no_grad()
def option_logprob(model, prefixes, options, batch=32):
    """Sum of log p(option chars | prefix) for each (prefix, option) pair."""
    model.eval()
    out = []
    for i in range(0, len(prefixes), batch):
        seqs, spans = [], []
        for p, o in zip(prefixes[i:i + batch], options[i:i + batch]):
            s = (p + o)[-CTX - 1:]
            seqs.append(encode(s))
            spans.append(len(o))
        n = max(len(s) for s in seqs)
        x = torch.zeros(len(seqs), n - 1, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s) - 1] = torch.tensor(s[:-1])
        lp = Fn.log_softmax(model(x), -1)
        for j, s in enumerate(seqs):
            L, k = len(s), spans[j]
            tgt = torch.tensor(s[L - k:])
            out.append(lp[j, L - k - 1:L - 1].gather(1, tgt[:, None]).sum().item())
    return out


def evaluate(model, items):
    """Per-item correctness with calibrated and raw scores."""
    pre, opt, neutral = [], [], []
    for it in items:
        for o in it["options"]:
            pre.append(it["q"])
            opt.append(o)
            neutral.append("A: ")
    cond = option_logprob(model, pre, opt)
    base = option_logprob(model, neutral, opt)
    cal, raw, k = [], [], 0
    for it in items:
        n = len(it["options"])
        c = np.array(cond[k:k + n])
        b = np.array(base[k:k + n])
        k += n
        cal.append(int(np.argmax(c - b) == it["answer"]))
        raw.append(int(np.argmax(c) == it["answer"]))
    return np.array(cal), np.array(raw)
