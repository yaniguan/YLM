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

Eval v2 (build_eval). Options are exchangeable: held-out protocols are grouped in blocks of 4
with 4 different true outcomes (verdict: same number of steps; volume: same level, passing
protocols), and the 4 items of a block share one option list in one order. Each option is the
right answer for exactly one item of its block, so any scorer that does not read the question
scores exactly 25%. Protocols whose question plus the longest option would not fit in the
context are rejected, and nothing is ever truncated.

Primary score (declared before any re-scoring, docs/TOKEN_QUALITY.md): the unpadded likelihood
  argmax_option  sum log p(option | question)
with no padding and no content-free prompt. Secondary scores (always reported): the old padded
raw and padded calibrated scores, the per-character-normalized likelihood and the
newline-terminated likelihood. build_eval_v1 / evaluate_v1 keep the first eval and its scores
unchanged, so a re-run can be compared item for item with the committed results.

Tokens are characters (vocab = newline + printable ASCII), so every arm is cut to exactly D
characters. The model is a small GPT trained from scratch on CPU.
"""
import glob
import hashlib
import itertools
import math
import os
import random
import unicodedata
from collections import Counter

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


def stream(docs, budget, rng, return_ids=False):
    """Shuffle documents, join with newlines, cut to exactly `budget` characters.

    With return_ids=True, also returns the indices into `docs` of the documents used, in order.
    The loop stops on the length of the joined text itself, so a document boundary that lands
    exactly on the budget never leaves the text one character short.
    """
    docs = list(docs)
    order = list(range(len(docs)))
    rng.shuffle(order)                 # the same permutation as rng.shuffle(docs)
    ids, joined = [], -1               # joined == len("\n".join(docs[i] for i in ids))
    for i in order:
        ids.append(i)
        joined += len(docs[i]) + 1
        if joined >= budget:
            break
    text = "\n".join(docs[i] for i in ids)
    if len(text) < budget:
        raise ValueError(f"only {len(text)} characters available, need {budget}")
    return (text[:budget], ids) if return_ids else text[:budget]


# ---------- eval (checker-scored) ----------

RULE_WORDS = ["overflow", "notThawed", "outOfRange", "noTip", "contaminated", "notEnough",
              "badTemp", "sealed"]
VERDICT_Q = " what happens? A: "
MAX_TAG_CHARS = 13         # longest error tag in src/lab_world.py (alreadySealed, sameContainer)
MAX_VOLUME_CHARS = 9       # wells hold at most 2000 uL: "2000.0 uL"


def question_text(task, steps):
    body = "; ".join(f"{i} {W.step_text(s)}" for i, s in enumerate(steps, 1))
    return f"Q: start {F.start_text(task)}. steps: {body}."


def volume_q(task):
    return f" final volume in {task['target']}? A: "


def outcome_text(truth):
    return "pass" if truth["ok"] else f"fail at step {truth['fail_step']}: {truth['error'].split()[0]}"


def volume_text(v):
    return f"{W.fmt_vol(v)} uL"


def pad(options):
    n = max(len(o) for o in options)
    return [o + "." * (n - len(o)) for o in options]


def check_cases(cases, lablean_dir):
    """True outcomes from LabLean, or from its Python copy when Lean is not available."""
    return W.lean_check(cases, lablean_dir) if W.lean_available(lablean_dir) else [W.check(*c) for c in cases]


def heldout_protocols(rng, seen, p_err=0.15, fits=None, rejected=None):
    """Endless stream of (task, steps, hash) for new protocols at levels 1-4 in turn.

    Every proposed hash is added to `seen`, and protocols already in `seen` are skipped.
    Protocols for which fits(task, steps) is False are skipped and their hash appended to
    `rejected`.
    """
    k = 0
    while True:
        task = F.propose_task(rng, 1 + k % 4)
        steps, _ = F.propose_protocol(rng, task, p_err)
        h = protocol_hash(task["state"], steps)
        if h in seen:
            continue
        seen.add(h)
        if fits is not None and not fits(task, steps):
            if rejected is not None:
                rejected.append(h)
            continue
        k += 1
        yield task, steps, h


def fits_context(task, steps):
    """True if both questions plus their longest possible option and a newline fit in CTX + 1
    characters (what the model sees), so no scored sequence is ever truncated."""
    q = question_text(task, steps)
    longest_verdict = len(f"fail at step {len(steps)}: ") + MAX_TAG_CHARS
    return (len(q + VERDICT_Q) + longest_verdict + 1 <= CTX + 1 and
            len(q + volume_q(task)) + MAX_VOLUME_CHARS + 1 <= CTX + 1)


def _greedy_blocks(members, key, group):
    """Put members (in order) into blocks of 4 with 4 different keys, within each group.

    Returns the closed blocks in the order they were closed; members left in open blocks are
    dropped.
    """
    open_blocks, closed = {}, []
    for m in members:
        blocks = open_blocks.setdefault(group(m), [])
        b = next((b for b in blocks if key(m) not in {key(x) for x in b}), None)
        if b is None:
            b = []
            blocks.append(b)
        b.append(m)
        if len(b) == 4:
            blocks.remove(b)
            closed.append(b)
    return closed


def no_prefix(options):
    return not any(a != b and b.startswith(a) for a in options for b in options)


def build_eval(seed, n_protocols, lablean_dir, exclude_hashes, p_err=0.15, stats=None):
    """Eval v2: held-out protocols in blocks of 4 with exchangeable options.

    Verdict blocks: 4 protocols with the same number of steps and 4 different true outcomes.
    Volume blocks: 4 passing protocols of the same level with 4 different final target volumes.
    The 4 items of a block share one option list (one shuffled order), and each option is the
    true answer of exactly one of them, so a scorer that ignores the question scores exactly 25%.
    Protocols that would not fit in the context are rejected; nothing is truncated.
    Adds the hashes of the n_protocols chosen protocols to `exclude_hashes`.
    """
    assert n_protocols % 4 == 0, "n_protocols must be a multiple of 4"
    rng = random.Random(seed)
    seen, rejected = set(exclude_hashes), []
    gen = heldout_protocols(rng, seen, p_err, fits=fits_context, rejected=rejected)
    n_cand, pending, verdict_blocks = 0, [], []
    while 4 * len(verdict_blocks) < n_protocols:
        batch = [next(gen) for _ in range(n_protocols)]
        truths = check_cases([(t["state"], s) for t, s, _ in batch], lablean_dir)
        for (task, steps, h), truth in zip(batch, truths):
            pending.append({"i": n_cand, "task": task, "steps": steps, "hash": h, "truth": truth,
                            "gold": outcome_text(truth)})
            n_cand += 1
        verdict_blocks = _greedy_blocks(pending, key=lambda p: p["gold"], group=lambda p: len(p["steps"]))
    verdict_blocks = verdict_blocks[:n_protocols // 4]
    chosen = sorted((p for b in verdict_blocks for p in b), key=lambda p: p["i"])
    passing = [p for p in chosen if p["truth"]["ok"]]
    for p in passing:
        p["vol_gold"] = volume_text(p["truth"]["volumes"][p["task"]["target"]])
    volume_blocks = _greedy_blocks(passing, key=lambda p: p["vol_gold"], group=lambda p: p["task"]["level"])

    where = {}                                       # (protocol index, kind) -> (block id, options)
    for kind, bl, key in (("verdict", verdict_blocks, "gold"), ("volume", volume_blocks, "vol_gold")):
        for j, b in enumerate(bl):
            opts = sorted(p[key] for p in b)
            rng.shuffle(opts)
            assert len(set(opts)) == 4 and no_prefix(opts), opts
            assert max(len(o) for o in opts) <= len("fail at step 99: ") + MAX_TAG_CHARS, opts
            for p in b:
                where[(p["i"], kind)] = (f"{kind}-{j}", opts)
    items = []
    for p in chosen:
        exclude_hashes.add(p["hash"])
        q = question_text(p["task"], p["steps"])
        for kind, suffix, key in (("verdict", VERDICT_Q, "gold"), ("volume", volume_q(p["task"]), "vol_gold")):
            if (p["i"], kind) not in where:
                continue
            block, opts = where[(p["i"], kind)]
            it = {"kind": kind, "q": q + suffix, "options": list(opts), "answer": opts.index(p[key]),
                  "level": p["task"]["level"], "n_steps": len(p["steps"]), "block": block,
                  "protocol": p["hash"][:16]}
            assert all(len(it["q"]) + len(o) + 1 <= CTX + 1 for o in opts)      # no truncation
            items.append(it)
    # evaluate() also scores each item's options after a question from another protocol, whose
    # length limit came from its own step count; fail here, not after training, if one is too long
    for it, qs in zip(items, shuffled_questions(items)):
        if any(len(qs) + len(o) > CTX + 1 for o in it["options"]):
            raise ValueError(f"a shuffled question plus an option of {it['protocol']} exceeds the context")
    if stats is not None:
        stats.update({"protocols": len(chosen), "candidates_checked": n_cand,
                      "rejected_too_long": len(rejected),
                      "candidates_not_in_a_full_block": n_cand - len(chosen),
                      "passing_protocols": len(passing),
                      "passing_without_volume_item": len(passing) - 4 * len(volume_blocks),
                      "verdict_blocks": len(verdict_blocks), "volume_blocks": len(volume_blocks)})
    return items


def shuffled_questions(items, seed=0):
    """For each item, the question of another item of the same kind and level, from another block.

    Used for the shuffled-question control. Within each (kind, level) group the blocks and their
    items are put in random order and every item takes the question m places further on, where
    m is the size of the largest block, so donor and item never share a block. If one block
    holds more than half of a group, those items take questions from other blocks of the same
    kind instead.
    """
    rng = random.Random(seed)
    groups, by_kind = {}, {}
    for i, it in enumerate(items):
        groups.setdefault((it["kind"], it["level"]), []).append(i)
        by_kind.setdefault(it["kind"], []).append(i)
    donor = [None] * len(items)
    for (kind, _level), idx in sorted(groups.items()):
        blocks = {}
        for i in idx:
            blocks.setdefault(items[i]["block"], []).append(i)
        m = max(len(b) for b in blocks.values())
        if 2 * m > len(idx):
            for i in idx:
                pool = [j for j in by_kind[kind] if items[j]["block"] != items[i]["block"]]
                if not pool:
                    raise ValueError(f"no other {kind} block to take questions from")
                donor[i] = rng.choice(pool)
            continue
        order = [list(b) for _, b in sorted(blocks.items())]
        rng.shuffle(order)
        for b in order:
            rng.shuffle(b)
        flat = [i for b in order for i in b]
        for k, i in enumerate(flat):
            donor[i] = flat[(k + m) % len(flat)]
    assert all(items[d]["block"] != it["block"] and items[d]["kind"] == it["kind"]
               for d, it in zip(donor, items))
    return [items[d]["q"] for d in donor]


def answer_of(doc):
    """The answer text of an eval-format document ('... A: answer')."""
    return doc.rsplit(" A: ", 1)[1]


def prior_only_correct(items, docs):
    """Question-free baseline: pick the option seen most often as an answer in `docs`.

    Ties go to the first option. Trailing padding dots are ignored (eval v1 options are padded).
    """
    counts = Counter(answer_of(d) for d in docs)
    return np.array([int(int(np.argmax([counts.get(o.rstrip("."), 0) for o in it["options"]])) == it["answer"])
                     for it in items])


def oracle_docs(seed, n_protocols, lablean_dir, exclude_hashes, p_err=0.15):
    """Positive control: eval-format text (question + right answer) on other protocols.

    One verdict document per protocol and one volume document per passing protocol. The text is
    the same as in the first run (no length filter: training windows may cut a document).
    """
    rng = random.Random(seed)
    protos = list(itertools.islice(heldout_protocols(rng, exclude_hashes, p_err), n_protocols))
    truths = check_cases([(t["state"], s) for t, s, _ in protos], lablean_dir)
    docs = []
    for (task, steps, _), truth in zip(protos, truths):
        q = question_text(task, steps)
        docs.append(q + VERDICT_Q + outcome_text(truth))
        if truth["ok"]:
            docs.append(q + volume_q(task) + volume_text(truth["volumes"][task["target"]]))
    return docs


# ---------- eval v1 (the first run; kept unchanged for item-for-item comparison) ----------

def make_items_v1(rng, task, steps, truth):
    """Eval v1 items for one protocol. Known flaws: distractors are drawn uniformly (the answer
    prior gives the answer away) and options are padded with scored dots."""
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
    items.append({"kind": "verdict", "q": q + VERDICT_Q, "options": pad(opts),
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
        opts = [volume_text(x) for x in vals]
        items.append({"kind": "volume", "q": q + volume_q(task), "options": pad(opts),
                      "answer": vals.index(v)})
    return items


def build_eval_v1(seed, n_protocols, lablean_dir, exclude_hashes, p_err=0.15):
    """Eval v1 exactly as in the first run (results/token_quality). Not used for conclusions."""
    rng = random.Random(seed)
    protos = list(itertools.islice(heldout_protocols(rng, exclude_hashes, p_err), n_protocols))
    truths = check_cases([(t["state"], s) for t, s, _ in protos], lablean_dir)
    items = []
    for (task, steps, _), truth in zip(protos, truths):
        items += make_items_v1(rng, task, steps, truth)
    return items


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
    """4 layers, d=128, 4 heads, context 512, tied embeddings (871,168 parameters)."""

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
def option_logprob(model, prefixes, options, batch=32, truncate=False, per_char=False):
    """Sum of log p(option chars | prefix) for each (prefix, option) pair.

    The model sees at most CTX + 1 characters of prefix + option. Longer pairs raise ValueError,
    unless truncate=True (eval v1 only), which cuts characters off the start of the prefix.
    With per_char=True, returns one array of per-character log-probs per pair instead of sums.
    """
    model.eval()
    out = []
    for i in range(0, len(prefixes), batch):
        seqs, spans = [], []
        for p, o in zip(prefixes[i:i + batch], options[i:i + batch]):
            s = p + o
            if len(s) > CTX + 1:
                if not truncate:
                    raise ValueError(f"prefix + option is {len(s)} characters, the model sees {CTX + 1}")
                s = s[-CTX - 1:]
            if len(o) < 1 or len(s) - len(o) < 1:
                raise ValueError("need a non-empty prefix and option")
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
            v = lp[j, L - k - 1:L - 1].gather(1, tgt[:, None])[:, 0]
            out.append(v.double().numpy() if per_char else v.sum().item())
    return out


LOGPROB_KEYS = ("lp", "lp_nl", "lp_pad", "lp_pad_neutral", "lp_shuf")
SCORES = ("primary", "per_char", "newline", "padded_raw", "padded_calibrated", "shuffled_q")


def evaluate(model, items, shuffle_seed=0):
    """Per-option log-probs on eval v2 items. Returns {key: array (items x 4)}:

      lp              sum log p(option | question)              (primary)
      lp_nl           sum log p(option + newline | question)
      lp_pad          sum log p(option padded with dots | question)   (old raw)
      lp_pad_neutral  sum log p(option padded with dots | "A: ")      (old calibration term)
      lp_shuf         sum log p(option | question of another item of the same kind and level)
    """
    shuf = shuffled_questions(items, shuffle_seed)
    pre, opt, padded, pre_shuf = [], [], [], []
    for it, qs in zip(items, shuf):
        for o, po in zip(it["options"], pad(it["options"])):
            pre.append(it["q"])
            opt.append(o)
            padded.append(po)
            pre_shuf.append(qs)
    nl = option_logprob(model, pre, [o + "\n" for o in opt], per_char=True)
    out = {"lp": [c[:-1].sum() for c in nl], "lp_nl": [c.sum() for c in nl],
           "lp_pad": option_logprob(model, pre, padded),
           "lp_pad_neutral": option_logprob(model, ["A: "] * len(padded), padded),
           "lp_shuf": option_logprob(model, pre_shuf, opt)}
    return {k: np.array(v, dtype=float).reshape(len(items), -1) for k, v in out.items()}


def choices(lps, items):
    """The option each score picks, per item. `lps` is the output of evaluate()."""
    lens = np.array([[len(o) for o in it["options"]] for it in items], dtype=float)
    return {"primary": lps["lp"].argmax(1),
            "per_char": (lps["lp"] / lens).argmax(1),
            "newline": lps["lp_nl"].argmax(1),
            "padded_raw": lps["lp_pad"].argmax(1),
            "padded_calibrated": (lps["lp_pad"] - lps["lp_pad_neutral"]).argmax(1),
            "shuffled_q": lps["lp_shuf"].argmax(1)}


def correct(lps, items):
    """Per-item 0/1 correctness for every score in SCORES."""
    ans = np.array([it["answer"] for it in items])
    return {k: (v == ans).astype(int) for k, v in choices(lps, items).items()}


def evaluate_v1(model, items):
    """Eval v1 scoring exactly as in the first run: (calibrated, raw) per-item correctness."""
    pre, opt, neutral = [], [], []
    for it in items:
        for o in it["options"]:
            pre.append(it["q"])
            opt.append(o)
            neutral.append("A: ")
    cond = option_logprob(model, pre, opt, truncate=True)
    base = option_logprob(model, neutral, opt, truncate=True)
    cal, raw, k = [], [], 0
    for it in items:
        n = len(it["options"])
        c = np.array(cond[k:k + n])
        b = np.array(base[k:k + n])
        k += n
        cal.append(int(np.argmax(c - b) == it["answer"]))
        raw.append(int(np.argmax(c) == it["answer"]))
    return np.array(cal), np.array(raw)
