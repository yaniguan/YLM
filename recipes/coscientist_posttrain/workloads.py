"""Workload factories for profile.py and scripts/profile_gpu.py (PR #79).

Contract: factory(torch=, device=, dtype=, **sizes) -> {"step", "items_per_step", "flops_per_step",
"bytes_per_step"}; items are tokens. The model is a GPT-2 built from a config (no download).
FLOPs count N = matmul weights = 12*L*d^2 + d*V (LM head included):
train 6*N*T, prefill 2*N*T, decode 2*N per new token. The attention-score term is reported
separately as attn_flops_per_step. Decode bytes = weights + KV cache read per step.
"""
import os

from .collate import ByteTokenizer, attention_bias, collate
from .geom_tokens import GeomSoftTokens, scatter_into
from .schema import load_jsonl

VOCAB = ByteTokenizer().vocab_size
TINY = {"n_layer": 2, "n_embd": 64, "n_head": 4, "n_positions": 512}
EXAMPLES = os.path.join(os.path.dirname(__file__), "examples", "records.jsonl")


def tiny_gpt2(**cfg):
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = {**TINY, **cfg}
    return GPT2LMHeadModel(GPT2Config(vocab_size=VOCAB, bos_token_id=VOCAB - 1, eos_token_id=VOCAB - 1, **cfg))


def matmul_params(model):
    c = model.config
    return 12 * c.n_layer * c.n_embd ** 2 + c.n_embd * c.vocab_size


def kv_bytes_per_token(model, elem):
    return 2 * model.config.n_layer * model.config.n_embd * elem


def _build(torch, device, dtype, train, model_kw):
    torch.manual_seed(0)
    model = tiny_gpt2(**model_kw).to(device=device, dtype=dtype)
    model.train(train)
    return model, torch.tensor([], dtype=dtype).element_size()


def packed_forward(model, row, inputs_embeds=None):
    """One packed row; a block-diagonal causal mask keeps samples apart (CPU / SDPA path)."""
    bias = attention_bias(row["cu_seqlens"], model.dtype).to(row["input_ids"].device)
    feed = {"inputs_embeds": inputs_embeds} if inputs_embeds is not None else {"input_ids": row["input_ids"]}
    return model(**feed, position_ids=row["position_ids"], attention_mask=bias, labels=row["labels"])


def sft_step(torch, device, dtype, seqs=4, seq_len=64, **model_kw):
    model, _ = _build(torch, device, dtype, True, model_kw)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ids = torch.randint(0, 256, (seqs, seq_len), device=device)
    T = seqs * seq_len
    L, d = model.config.n_layer, model.config.n_embd

    def step():
        loss = model(ids, labels=ids).loss
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    return {"step": step, "items_per_step": T, "flops_per_step": 6 * matmul_params(model) * T,
            "attn_flops_per_step": 6 * L * d * seqs * seq_len ** 2, "bytes_per_step": None}


def prefill_step(torch, device, dtype, seqs=4, seq_len=64, **model_kw):
    model, elem = _build(torch, device, dtype, False, model_kw)
    ids = torch.randint(0, 256, (seqs, seq_len), device=device)
    T = seqs * seq_len
    L, d = model.config.n_layer, model.config.n_embd

    def step():
        with torch.no_grad():
            return model(ids, use_cache=True).logits

    return {"step": step, "items_per_step": T, "flops_per_step": 2 * matmul_params(model) * T,
            "attn_flops_per_step": 2 * L * d * seqs * seq_len ** 2,
            "bytes_per_step": matmul_params(model) * elem + kv_bytes_per_token(model, elem) * T}


def decode_step(torch, device, dtype, batch=8, ctx=128, **model_kw):
    """One new token for `batch` live sequences over a fixed `ctx`-token KV cache."""
    model, elem = _build(torch, device, dtype, False, model_kw)
    with torch.no_grad():
        cache = model(torch.randint(0, 256, (batch, ctx), device=device), use_cache=True).past_key_values
    tok = torch.randint(0, 256, (batch, 1), device=device)
    L, d = model.config.n_layer, model.config.n_embd

    def step():
        with torch.no_grad():
            out = model(tok, past_key_values=cache, use_cache=True)
        if hasattr(cache, "crop"):  # Cache object grows in place; legacy tuples do not
            cache.crop(-1)
        return out.logits

    return {"step": step, "items_per_step": batch, "flops_per_step": 2 * matmul_params(model) * batch,
            "attn_flops_per_step": 4 * L * d * ctx * batch,
            "bytes_per_step": matmul_params(model) * elem + kv_bytes_per_token(model, elem) * ctx * batch}


def geom_sft_step(torch, device, dtype, k=8, max_tokens=512, path=EXAMPLES, **model_kw):
    """End to end: example records -> packed rows -> geometry soft tokens scattered into the
    input embeddings -> GPT-2 LM loss -> backward into LM and geometry projector/encoder.
    flops_per_step counts the LM only."""
    model, _ = _build(torch, device, dtype, True, model_kw)
    geo = GeomSoftTokens(model.config.n_embd, k=k).to(device=device, dtype=dtype)
    opt = torch.optim.AdamW(list(model.parameters()) + list(geo.parameters()), lr=1e-4)
    rows = collate(load_jsonl(path), ByteTokenizer(), max_tokens, k=k)
    for row in rows:
        for key, v in row.items():
            if hasattr(v, "to"):
                row[key] = v.to(device)
        if row["geom"] is not None:
            row["geom"] = {key: v.to(device) for key, v in row["geom"].items()}
            row["geom"]["pos"] = row["geom"]["pos"].to(dtype)
    T = sum(r["input_ids"].shape[1] for r in rows)

    def step():
        total = 0.0
        for row in rows:
            emb = model.get_input_embeddings()(row["input_ids"])
            if row["geom"] is not None:
                soft, _ = geo(row["geom"])
                emb = scatter_into(emb, row["geom_slots"], soft)
            loss = packed_forward(model, row, inputs_embeds=emb).loss
            loss.backward()
            total += float(loss.detach())
        opt.step()
        opt.zero_grad(set_to_none=True)
        return total

    return {"step": step, "items_per_step": T, "flops_per_step": 6 * matmul_params(model) * T,
            "bytes_per_step": None, "model": model, "geom": geo}


FACTORIES = {f.__name__: f for f in (sft_step, prefill_step, decode_step, geom_sft_step)}
