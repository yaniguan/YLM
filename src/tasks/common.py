"""Shared pieces for scripts 20-23: examples, collation, decoding, splits, checkpoints, training loop."""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
import torch

from src.ylm_lm import YLMCausalLM, YLMConfig, collate_geometry

BOHR_TO_ANGSTROM = 0.529177210903

# arm -> (model mode, xyz coordinates as text?)
ARMS = {
    "smiles": ("text", False),
    "smiles_xyz": ("text", True),
    "prefixA": ("prefix", False),
    "xattnA": ("xattnA", False),
    "xattnB": ("xattnB", False),
    "xattnC": ("xattnC", False),
}
KINDS = ("other", "smiles", "num", "xyz", "answer")
KIND_ID = {k: i for i, k in enumerate(KINDS)}


class Seg(NamedTuple):
    text: str
    kind: str = "other"
    prop: Optional[str] = None


@dataclass
class Example:
    segs: list
    geom: Optional[dict] = None
    meta: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segs)


def fmt_num(x: float, nd: int = 3) -> str:
    """Fixed decimals with trailing zeros removed but at least one decimal: 1.0, 0.05, -12.345."""
    s = f"{x:.{nd}f}".rstrip("0")
    s = s + "0" if s.endswith(".") else s
    return "0.0" if s in ("-0.0", "-0") else s


def element_symbol(z: int) -> str:
    from rdkit import Chem
    return Chem.GetPeriodicTable().GetElementSymbol(int(z))


def xyz_segs(Z, pos_bohr, max_atoms: Optional[int] = None, label: Optional[str] = None) -> list:
    """' <xyz> C 0.12 -1.30 0.00 H ... </xyz>' with Angstrom coordinates, 2 decimals."""
    pos = np.asarray(pos_bohr) * BOHR_TO_ANGSTROM
    n = len(Z) if max_atoms is None else min(len(Z), max_atoms)
    head = f" <xyz> {label}:" if label else " <xyz>"
    segs = [Seg(head, "other")]
    for z, (x, y, w) in zip(list(Z)[:n], pos[:n]):
        segs.append(Seg(f" {element_symbol(z)} {x:.2f} {y:.2f} {w:.2f}", "xyz"))
    segs.append(Seg(" </xyz>", "other"))
    return segs


def molecule_segs(smiles: str, geom: Optional[dict] = None, with_xyz: bool = False,
                  max_xyz_atoms: Optional[int] = None) -> list:
    segs = [Seg("<mol> ", "other"), Seg(smiles, "smiles"), Seg(" </mol>", "other")]
    if with_xyz and geom is not None:
        segs += xyz_segs(geom["Z"], geom["pos"], max_xyz_atoms)
    return segs


# ----------------------------------------------------------------------------- encoding / batching

def encode_example(tok, ex: Example, max_len: int, add_eos: bool = True):
    ids, kinds, seg_idx = [tok.bos_id], [KIND_ID["other"]], [-1]
    for i, s in enumerate(ex.segs):
        t = tok.encode(s.text, s.kind)
        ids += t
        kinds += [KIND_ID[s.kind]] * len(t)
        seg_idx += [i] * len(t)
    if add_eos:
        ids.append(tok.eos_id)
        kinds.append(KIND_ID["other"])
        seg_idx.append(-1)
    return ids[:max_len], kinds[:max_len], seg_idx[:max_len]


def collate(examples, tok, max_len: int, loss_kinds=None, add_eos: bool = True, need_geom: bool = True,
            device=None, dtype=None) -> dict:
    """Right-padded batch. labels = token ids where the token kind is in loss_kinds (None = all), else -100."""
    enc = [encode_example(tok, ex, max_len, add_eos) for ex in examples]
    B, L = len(enc), max(len(e[0]) for e in enc)
    input_ids = torch.full((B, L), tok.pad_id, dtype=torch.long)
    attn = torch.zeros(B, L, dtype=torch.long)
    labels = torch.full((B, L), -100, dtype=torch.long)
    kinds = torch.full((B, L), -1, dtype=torch.long)
    allowed = None if loss_kinds is None else {KIND_ID[k] for k in loss_kinds}
    for b, (ids, ks, _) in enumerate(enc):
        n = len(ids)
        input_ids[b, :n] = torch.tensor(ids)
        attn[b, :n] = 1
        kinds[b, :n] = torch.tensor(ks)
        keep = torch.ones(n, dtype=torch.bool) if allowed is None else torch.tensor([k in allowed for k in ks])
        labels[b, :n] = torch.where(keep, torch.tensor(ids), torch.full((n,), -100))
    batch = dict(input_ids=input_ids, attention_mask=attn, labels=labels, kinds=kinds)
    if device is not None:
        batch = {k: v.to(device) for k, v in batch.items()}
    batch["geom"] = collate_geometry([ex.geom for ex in examples], device=device, dtype=dtype) if need_geom else None
    return batch


def batch_tokens(batch: dict, model: YLMCausalLM) -> int:
    """Tokens processed by the LM, including soft prefix tokens."""
    n = int(batch["attention_mask"].sum())
    if model.mode == "prefix" and batch.get("geom") is not None:
        n += int(batch["geom"]["atom_mask"].sum())
    return n


def model_inputs(batch: dict) -> dict:
    return dict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"],
                geom=batch["geom"])


_NUM_RE = re.compile(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


def parse_number(s: str) -> Optional[float]:
    m = _NUM_RE.match(s)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v if math.isfinite(v) else None


@torch.no_grad()
def greedy_decode(model: YLMCausalLM, tok, prompts: list, max_new: int = 12, batch_size: int = 32,
                  max_len: int = 1024, device="cpu") -> list:
    """Left-padded greedy decoding (no KV cache, geometry encoded once per batch)."""
    was_training = model.training
    model.eval()
    outs = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = [encode_example(tok, ex, max_len - max_new, add_eos=False)[0] for ex in chunk]
        B, L = len(enc), max(len(e) for e in enc)
        ids = torch.full((B, L), tok.pad_id, dtype=torch.long)
        attn = torch.zeros(B, L, dtype=torch.long)
        for b, e in enumerate(enc):
            ids[b, L - len(e):] = torch.tensor(e)
            attn[b, L - len(e):] = 1
        ids, attn = ids.to(device), attn.to(device)
        feats = None
        if model.needs_geometry:
            geom = collate_geometry([ex.geom for ex in chunk], device=device,
                                    dtype=next(model.parameters()).dtype)
            feats = model.encode_geometry(geom)
        gen = []
        for _ in range(max_new):
            logits = model(ids, attn, geom_feats=feats)["logits"]
            nxt = logits[:, -1].argmax(-1)
            gen.append(nxt.cpu())
            ids = torch.cat([ids, nxt[:, None]], 1)
            attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=device)], 1)
            if bool((nxt == tok.eos_id).all()):
                break
        gen = torch.stack(gen, 1).tolist()
        for g in gen:
            if tok.eos_id in g:
                g = g[:g.index(tok.eos_id)]
            outs.append(tok.decode(g))
    model.train(was_training)
    return outs


# ----------------------------------------------------------------------------- splits

def random_split(n: int, seed: int, fracs=(0.8, 0.1, 0.1)) -> dict:
    perm = np.random.default_rng(seed).permutation(n)
    a, b = int(round(fracs[0] * n)), int(round((fracs[0] + fracs[1]) * n))
    return {"train": sorted(perm[:a].tolist()), "val": sorted(perm[a:b].tolist()), "test": sorted(perm[b:].tolist())}


def group_split(keys: list, seed: int, fracs=(0.8, 0.1, 0.1)) -> dict:
    """Whole groups go to one split. Groups are shuffled (seeded) and assigned until each split is full."""
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(k, []).append(i)
    names = sorted(groups)
    order = [names[i] for i in np.random.default_rng(seed).permutation(len(names))]
    n = len(keys)
    out = {"train": [], "val": [], "test": []}
    bounds = (fracs[0] * n, (fracs[0] + fracs[1]) * n)
    done = 0
    for j, g in enumerate(order):
        remaining = len(order) - j
        if done < bounds[0] and (remaining > 2 or not out["train"]):
            target = "train"
        elif done < bounds[1] and (remaining > 1 or not out["val"]):
            target = "val"
        else:
            target = "test"
        out[target] += groups[g]
        done += len(groups[g])
    return {k: sorted(v) for k, v in out.items()}


def qm9_split(indices, seed: int, fracs=(0.8, 0.1, 0.1)) -> dict:
    """Split QM9 molecules by qm9_index; the same index always lands in the same split for a given seed."""
    idx = np.array(sorted(set(int(i) for i in indices)))
    s = random_split(len(idx), seed, fracs)
    return {k: set(idx[v].tolist()) for k, v in s.items()}


# ----------------------------------------------------------------------------- checkpoints

def rng_state() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"].cpu())
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s["cuda"]])


def save_checkpoint(ckpt_dir: str, model, opt, state: dict, keep: int = 2) -> str:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step{state['step']:08d}.pt")
    tmp = path + ".tmp"
    torch.save({"model": model.state_dict(), "opt": opt.state_dict() if opt is not None else None,
                "state": state, "rng": rng_state()}, tmp)
    os.replace(tmp, path)
    with open(os.path.join(ckpt_dir, "latest.txt.tmp"), "w") as f:
        f.write(os.path.basename(path))
    os.replace(os.path.join(ckpt_dir, "latest.txt.tmp"), os.path.join(ckpt_dir, "latest.txt"))
    old = sorted(p for p in os.listdir(ckpt_dir) if p.startswith("step") and p.endswith(".pt"))
    for p in old[:-keep]:
        os.remove(os.path.join(ckpt_dir, p))
    return path


def latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    marker = os.path.join(ckpt_dir, "latest.txt")
    if not os.path.exists(marker):
        return None
    path = os.path.join(ckpt_dir, open(marker).read().strip())
    return path if os.path.exists(path) else None


def load_checkpoint(path: str, model, opt=None) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)  # our own file: contains RNG states
    model.load_state_dict(ck["model"])
    if opt is not None and ck.get("opt") is not None:
        opt.load_state_dict(ck["opt"])
    set_rng_state(ck["rng"])
    return ck["state"]


# ----------------------------------------------------------------------------- training loop

class NullProfiler:
    from contextlib import nullcontext as _nc

    def phase(self, name):
        return self._nc()

    def step_done(self, tokens, samples):
        pass


def lr_at(progress: float, base_lr: float, warmup: float = 0.05, final_frac: float = 0.1) -> float:
    """Linear warmup then cosine to final_frac * base_lr, as a function of training progress in [0, 1]."""
    progress = min(max(progress, 0.0), 1.0)
    if progress < warmup:
        return base_lr * progress / warmup
    t = (progress - warmup) / max(1e-12, 1 - warmup)
    return base_lr * (final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def epoch_order(n: int, seed: int, epoch: int) -> list:
    g = torch.Generator().manual_seed(seed * 100003 + epoch)
    return torch.randperm(n, generator=g).tolist()


def train_lm(model: YLMCausalLM, examples: list, tok, *, out_dir: str, tokens_budget: int, batch_size: int,
             lr: float, max_len: int, loss_kinds=None, seed: int = 0, device="cpu", ckpt_every: int = 0,
             resume: bool = False, profiler=None, log=print, schedule_tokens: Optional[int] = None,
             stop_after_steps: Optional[int] = None, weight_decay: float = 0.01, grad_clip: float = 1.0,
             amp_dtype=None, log_every: int = 20) -> dict:
    """Train until tokens_budget LM tokens have been processed (prefix tokens included).

    The learning-rate schedule depends only on tokens seen / schedule_tokens (default tokens_budget), so a
    run split over several sessions (stop_after_steps, then --resume) follows the same trajectory."""
    if tokens_budget < 1 or batch_size < 1 or max_len < 2:
        raise ValueError("positive token budget/batch size and max_len >= 2 required")
    profiler = profiler or NullProfiler()
    schedule_tokens = schedule_tokens or tokens_budget
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr, betas=(0.9, 0.95))
    ckpt_dir = os.path.join(out_dir, "ckpt")
    state = {"step": 0, "tokens": 0, "epoch": 0, "pos": 0, "wall": 0.0, "history": []}
    if resume:
        path = latest_checkpoint(ckpt_dir)
        if path is not None:
            state = load_checkpoint(path, model, opt)
            log(f"resumed from {path}: step {state['step']} tokens {state['tokens']}")
    n = len(examples)
    if n == 0:
        raise ValueError("no training examples")
    order = epoch_order(n, seed, state["epoch"])
    t_start = time.time() - state["wall"]
    steps_this_session = 0
    model.train()
    dtype = next(model.parameters()).dtype
    dev_type = torch.device(device).type
    losses = []
    while state["tokens"] < tokens_budget:
        if stop_after_steps is not None and steps_this_session >= stop_after_steps:
            break
        if state["pos"] >= n:
            state["epoch"] += 1
            state["pos"] = 0
            order = epoch_order(n, seed, state["epoch"])
        idx = order[state["pos"]: state["pos"] + batch_size]
        with profiler.phase("data"):
            batch = collate([examples[i] for i in idx], tok, max_len, loss_kinds, need_geom=model.needs_geometry,
                            device=device, dtype=dtype)
            if loss_kinds and not bool((batch["labels"][:, 1:] != -100).any(1).all()):
                raise ValueError("Some supervised answers were truncated; increase max_len")
            batch = fit_token_budget(batch, model, tokens_budget - state["tokens"])
        if batch is None:
            break
        used_samples = batch["input_ids"].shape[0]
        if not bool((batch["labels"][:, 1:] != -100).any()):
            # A truncated answer-only batch has no trainable answer. Do not train on prompts.
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(state["tokens"] / schedule_tokens, lr)
        with profiler.phase("forward"):
            with torch.autocast(device_type=dev_type, dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
                loss = model(**model_inputs(batch))["loss"]
        with profiler.phase("backward"):
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(params, grad_clip)
        with profiler.phase("optimizer"):
            opt.step()
        n_tok = batch_tokens(batch, model)
        state["step"] += 1
        state["tokens"] += n_tok
        state["pos"] += used_samples
        steps_this_session += 1
        losses.append(float(loss))
        profiler.step_done(n_tok, used_samples)
        if state["step"] % log_every == 0:
            rec = {"step": state["step"], "tokens": state["tokens"], "loss": float(np.mean(losses)),
                   "grad_norm": float(gnorm), "lr": opt.param_groups[0]["lr"],
                   "wall": round(time.time() - t_start, 1)}
            gates = model.gates()
            if gates:
                rec["gates"] = [round(g, 4) for g in gates]
            state["history"].append(rec)
            log(json.dumps(rec))
            losses = []
        if ckpt_every and state["step"] % ckpt_every == 0:
            state["wall"] = time.time() - t_start
            save_checkpoint(ckpt_dir, model, opt, state)
    state["wall"] = time.time() - t_start
    save_checkpoint(ckpt_dir, model, opt, state)
    state["finished"] = state["tokens"] >= tokens_budget
    return state


def fit_token_budget(batch, model, remaining):
    """Cap valid processed tokens, including geometry prefix. Never exceed the budget.

    Final text may be truncated. Stop below budget if even one prefix + two text tokens
    cannot fit. Padded slots are excluded, consistently with batch_tokens.
    """
    lengths = batch["attention_mask"].sum(1).tolist()
    prefix = (batch["geom"]["atom_mask"].sum(1).tolist()
              if model.mode == "prefix" and batch.get("geom") is not None else [0] * len(lengths))
    keep = []
    for length, extra in zip(lengths, prefix):
        size = min(int(length), int(remaining - extra))
        if size < 2:
            break
        keep.append(size)
        remaining -= size + extra
    if not keep:
        return None
    n, width = len(keep), max(keep)
    result = {k: v[:n, :width].clone() for k, v in batch.items() if k != "geom"}
    result["geom"] = ({k: v[:n] for k, v in batch["geom"].items()}
                      if batch.get("geom") is not None else None)
    for b, size in enumerate(keep):
        result["attention_mask"][b, size:] = 0
        result["labels"][b, size:] = -100
        result["kinds"][b, size:] = -1
    return result


# ----------------------------------------------------------------------------- models on disk

def save_run_model(out_dir: str, model: YLMCausalLM, config: dict):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt.tmp"))
    os.replace(os.path.join(out_dir, "model.pt.tmp"), os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "config.json.tmp"), "w") as f:
        json.dump(config, f, indent=2)
    os.replace(os.path.join(out_dir, "config.json.tmp"), os.path.join(out_dir, "config.json"))


def load_run(run_dir: str, device="cpu"):
    """-> (model, tokenizer, config) for a run written by scripts 20/21."""
    from src.tasks.tokenizer import get_tokenizer
    config = json.load(open(os.path.join(run_dir, "config.json")))
    cfg = YLMConfig.from_dict(config["model_cfg"])
    model = YLMCausalLM(cfg)
    sd = torch.load(os.path.join(run_dir, "model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(sd)
    return model.to(device), get_tokenizer(cfg.lm), config


def load_encoder_weights(model: YLMCausalLM, path: str, log=print):
    """Initialise the equivariant encoder from a repo YLM checkpoint (keys 'encoder.*').

    The element embedding of old checkpoints has 10 rows (n_elements=10); they are copied into the first
    10 rows of the 100-element table. Irreps must match the chosen --encoder preset."""
    if model.irreps is None:
        log("init_encoder ignored: text arm has no encoder")
        return
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
    if not enc:
        enc = {k[len("irreps.encoder."):]: v for k, v in sd.items() if k.startswith("irreps.encoder.")}
    if not enc:
        raise ValueError(f"No encoder weights found in {path}")
    target = model.irreps.encoder.state_dict()
    if "embed.weight" in enc and enc["embed.weight"].shape != target["embed.weight"].shape:
        old = enc["embed.weight"]
        new = target["embed.weight"].clone()
        new[: old.shape[0]] = old
        enc["embed.weight"] = new
    bad = [k for k, v in enc.items() if k in target and v.shape != target[k].shape]
    if bad:
        raise ValueError(f"encoder shape mismatch for {bad[:5]} - choose the matching --encoder preset")
    missing, unexpected = model.irreps.encoder.load_state_dict(enc, strict=False)
    log(f"loaded encoder from {path}: missing={list(missing)} unexpected={list(unexpected)}")


def pick_device(arg: Optional[str]) -> str:
    if arg:
        return arg
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_logger(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    f = open(path, "a")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        f.write(s + "\n")
        f.flush()
    return say


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
