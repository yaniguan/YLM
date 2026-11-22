"""Render records to token ids, pack them into rows, build labels and geometry batches.

Packing: first-fit decreasing into rows of `max_tokens`. Each row carries position_ids that
restart at 0 per segment and cu_seqlens for varlen attention, so samples never attend to each
other. Labels are -100 on system/user turns, on <|geom|> slots and on the first token of every
segment (so the last token of sample i never learns to predict sample i+1).
"""
import torch

from src.ylm_lm import collate_geometry

from .schema import PLACEHOLDER


class ByteTokenizer:
    """UTF-8 bytes + a few special ids. No download. A real tokenizer only needs
    encode(text) -> list[int], decode(ids) -> str, a `special` dict and `vocab_size`."""
    SPECIALS = ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>", "<|geom|>", "<|pad|>"]

    def __init__(self):
        self.special = {s: 256 + i for i, s in enumerate(self.SPECIALS)}
        self.vocab_size = 256 + len(self.SPECIALS)

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if i < 256).decode("utf-8", "replace")


def inline_text(kind, i, rec):
    """Text for <img:i> (OCSR SMILES + VERDICT flag), <sim:i> (WignerFlow), and <audio:i>."""
    if kind == "img":
        im = rec["images"][i]
        return f"[image {i}: SMILES={im['smiles']}; ocsr_flag={im.get('ocsr_flag', 0.0):.2f}]"
    if kind == "audio":
        a = rec["audios"][i]
        flag = "flagged" if a.get("score", 0.0) > a.get("threshold", 1.0) else "normal"
        inst = a.get("instrument_id", "unknown")
        mic = a.get("mic", "unknown")
        return (
            f"[audio {i}: {inst}, {mic} mic, {a.get('duration_s', 0.0):g} s, "
            f"score={a.get('score', 0.0):.2f} vs threshold={a.get('threshold', 1.0):.2f}, {flag}]"
        )
    s = rec["sims"][i]
    values = ", ".join(f"{k}={v:g}" for k, v in s["values"].items())
    return f"[sim {i}: {s['system']}, {values}]"


def render(rec, tok, k=8):
    """-> {"ids", "train" (per-token bool), "slots" (k positions per geometry)}."""
    ids, train, slots = [], [], [[] for _ in rec["geoms"]]
    for m in rec["messages"]:
        learn = m["role"] == "assistant"

        def add(text):
            t = tok.encode(text)
            ids.extend(t)
            train.extend([learn] * len(t))

        ids.append(tok.special[f"<|{m['role']}|>"])
        train.append(False)
        start = 0
        for match in PLACEHOLDER.finditer(m["content"]):
            add(m["content"][start:match.start()])
            kind, i = match.group(1), int(match.group(2))
            if kind == "geom":
                slots[i].extend(range(len(ids), len(ids) + k))
                ids.extend([tok.special["<|geom|>"]] * k)
                train.extend([False] * k)
            else:
                add(inline_text(kind, i, rec))
            start = match.end()
        add(m["content"][start:])
        ids.append(tok.special["<|end|>"])
        train.append(learn)
    return {"ids": ids, "train": train, "slots": slots}


def pack(lengths, max_tokens):
    """First-fit decreasing. Returns a list of rows, each a list of sample indices."""
    rows, free = [], []
    for j in sorted(range(len(lengths)), key=lambda j: -lengths[j]):
        for r, space in enumerate(free):
            if lengths[j] <= space:
                rows[r].append(j)
                free[r] -= lengths[j]
                break
        else:
            rows.append([j])
            free.append(max_tokens - lengths[j])
    return rows


def collate(records, tok, max_tokens, k=8):
    """Validated records -> list of packed rows (dicts of tensors)."""
    samples = [render(r, tok, k) for r in records]
    too_long = [r["id"] for r, s in zip(records, samples) if len(s["ids"]) > max_tokens]
    if too_long:
        raise ValueError(f"records longer than max_tokens={max_tokens}: {too_long}")
    rows = []
    for row in pack([len(s["ids"]) for s in samples], max_tokens):
        ids, labels, pos, cu, geoms, slots = [], [], [], [0], [], []
        for j in row:
            s, rec = samples[j], records[j]
            offset = len(ids)
            lab = [t if keep else -100 for t, keep in zip(s["ids"], s["train"])]
            lab[0] = -100
            ids += s["ids"]
            labels += lab
            pos += range(len(s["ids"]))
            cu.append(len(ids))
            geoms += [{"Z": g["Z"], "pos": g["pos"]} for g in rec["geoms"]]
            slots += [[offset + p for p in sl] for sl in s["slots"]]
        cu_t = torch.tensor(cu, dtype=torch.int32)
        rows.append({
            "input_ids": torch.tensor([ids]), "labels": torch.tensor([labels]),
            "position_ids": torch.tensor([pos]), "cu_seqlens": cu_t,
            "max_seqlen": int(cu_t.diff().max()), "record_ids": [records[j]["id"] for j in row],
            "geom": collate_geometry(geoms) if geoms else None,
            "geom_slots": torch.tensor(slots, dtype=torch.long).reshape(-1, k),
        })
    return rows


def attention_bias(cu_seqlens, dtype=torch.float32):
    """Causal block-diagonal additive mask (1, 1, T, T) from cu_seqlens: 0 = attend, min = blocked.
    For CPU checks and eager/SDPA; on GPU pass cu_seqlens to varlen FlashAttention instead."""
    seg = torch.repeat_interleave(torch.arange(len(cu_seqlens) - 1), cu_seqlens.diff().long())
    T = seg.numel()
    allow = (seg[:, None] == seg[None, :]) & torch.ones(T, T, dtype=torch.bool).tril()
    return torch.zeros(T, T, dtype=dtype).masked_fill(~allow, torch.finfo(dtype).min)[None, None]
