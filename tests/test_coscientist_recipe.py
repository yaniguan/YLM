"""CPU smoke test for recipes/coscientist_posttrain (no GPU, no downloads)."""
import copy
import glob
import json
import os
import time

import pytest
import torch
import torch.nn.functional as F
from e3nn import o3

from recipes.coscientist_posttrain import verifiers as V
from recipes.coscientist_posttrain.collate import ByteTokenizer, attention_bias, collate, render
from recipes.coscientist_posttrain.geom_tokens import GeomSoftTokens, scatter_into
from recipes.coscientist_posttrain.schema import (BOHR_PER_ANGSTROM, SchemaError, check_config_tags,
                                                  load_jsonl, validate)
from recipes.coscientist_posttrain.workloads import EXAMPLES, FACTORIES, packed_forward, tiny_gpt2
from src.ylm_lm import collate_geometry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS = os.path.join(ROOT, "recipes", "coscientist_posttrain", "configs")
TOK = ByteTokenizer()
K = 8


@pytest.fixture(scope="module")
def records():
    return load_jsonl(EXAMPLES)


def by_id(records, rid):
    return copy.deepcopy(next(r for r in records if r["id"] == rid))


# 1. schema
def test_schema(records):
    assert {r["id"] for r in records} >= {"pq-001", "tw-017", "gq-003", "sim-009"}
    assert by_id(records, "sim-009")["sims"][0]["tag"] == "EXAMPLE"
    gq = by_id(records, "gq-003")
    bad = copy.deepcopy(gq)
    bad["messages"][0]["content"] += " <geom:1>"
    with pytest.raises(SchemaError):
        validate(bad)
    ang = copy.deepcopy(gq)
    ang["geoms"][0].update(unit="angstrom", pos=[[0, 0, 0], [1.0, 0, 0], [0, 2.0, 0]])
    out = validate(ang)["geoms"][0]
    assert out["unit"] == "bohr" and out["pos"][1][0] == pytest.approx(BOHR_PER_ANGSTROM)
    sim = by_id(records, "sim-009")
    sim["sims"][0]["system"] = "Li+"
    with pytest.raises(SchemaError):
        validate(sim)
    sim["sims"][0].update(system="Ar", tag="MEASURED")
    with pytest.raises(SchemaError):  # MEASURED without a source path
        validate(sim)
    kept = {r["id"] for r in load_jsonl(EXAMPLES, commercial=True)}
    assert "pq-001" not in kept and "tw-017" in kept
    assert "gq-003" not in {r["id"] for r in load_jsonl(EXAMPLES, max_ocsr_flag=0.01)}


# 2. packing
def test_packing(records):
    max_tokens = 300
    rows = collate(records, TOK, max_tokens, k=K)
    assert len(rows) > 1 and sorted(i for r in rows for i in r["record_ids"]) == sorted(r["id"] for r in records)
    for row in rows:
        ids, labels, pos = row["input_ids"][0], row["labels"][0], row["position_ids"][0]
        cu = row["cu_seqlens"].tolist()
        assert row["cu_seqlens"].dtype == torch.int32 and cu[-1] == ids.numel() <= max_tokens
        assert row["max_seqlen"] == max(b - a for a, b in zip(cu, cu[1:]))
        n_geoms = 0
        for (a, b), rid in zip(zip(cu, cu[1:]), row["record_ids"]):
            rec = by_id(records, rid)
            s = render(rec, TOK, K)
            assert pos[a:b].tolist() == list(range(b - a)) and ids[a:b].tolist() == s["ids"]
            want = [t if keep else -100 for t, keep in zip(s["ids"], s["train"])]
            want[0] = -100
            assert labels[a:b].tolist() == want and labels[a] == -100
            for sl in s["slots"]:
                assert (labels[a:b][sl] == -100).all()
            n_geoms += len(rec["geoms"])
        assert row["geom_slots"].shape == (n_geoms, K)
        if n_geoms:
            assert (ids[row["geom_slots"]] == TOK.special["<|geom|>"]).all()
            assert row["geom"]["Z"].shape[0] == n_geoms


# 3. no cross-sample attention
def test_no_cross_sample_attention(records):
    torch.manual_seed(0)
    cu = torch.tensor([0, 5, 12, 20], dtype=torch.int32)
    q, k, v = (torch.randn(1, 2, 20, 8) for _ in range(3))
    packed = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias(cu))
    for a, b in zip(cu.tolist(), cu[1:].tolist()):
        alone = F.scaled_dot_product_attention(q[:, :, a:b], k[:, :, a:b], v[:, :, a:b], is_causal=True)
        assert torch.allclose(packed[:, :, a:b], alone, atol=1e-5)

    model = tiny_gpt2().eval()
    row = max(collate(records, TOK, 300, k=K), key=lambda r: len(r["record_ids"]))
    with torch.no_grad():
        logits = packed_forward(model, row).logits[0]
        c = row["cu_seqlens"].tolist()
        for a, b in zip(c, c[1:]):
            ids, lab = row["input_ids"][:, a:b], row["labels"][0, a:b]
            alone = model(ids).logits[0]
            assert torch.allclose(logits[a:b], alone, atol=1e-5)
            if (lab[1:] != -100).any():
                loss_p = F.cross_entropy(logits[a:b - 1], lab[1:], ignore_index=-100)
                loss_a = F.cross_entropy(alone[:-1], lab[1:], ignore_index=-100)
                assert torch.allclose(loss_p, loss_a, atol=1e-5)


# 4. geometry soft tokens
def test_geometry_soft_tokens():
    torch.manual_seed(0)
    geo = GeomSoftTokens(d_llm=32, k=4)
    Z = [6, 1, 8, 7, 1]
    pos = torch.randn(5, 3) * 1.5
    soft = lambda gs: geo(collate_geometry(gs))[0]  # noqa: E731
    base = soft([{"Z": Z, "pos": pos.tolist()}])
    assert base.shape == (1, 4, 32)
    R = o3.rand_matrix()
    perm = torch.randperm(5)
    for p, z in [(pos @ R.T, Z), (-pos @ R.T, Z), (pos + torch.tensor([3.0, -1.0, 2.0]), Z),
                 (pos[perm], [Z[i] for i in perm])]:
        assert torch.allclose(soft([{"Z": z, "pos": p.tolist()}]), base, atol=1e-4)
    big = {"Z": [6] * 9, "pos": (torch.randn(9, 3) * 2).tolist()}
    assert torch.allclose(soft([{"Z": Z, "pos": pos.tolist()}, big])[0:1], base, atol=1e-4)

    model = tiny_gpt2(n_embd=32, n_head=2)
    emb = model.get_input_embeddings()(torch.randint(0, 256, (1, 12)))
    s, _ = geo(collate_geometry([{"Z": Z, "pos": pos.tolist()}]))
    out = model(inputs_embeds=scatter_into(emb, torch.arange(2, 6)[None], s), labels=torch.randint(0, 256, (1, 12)))
    out.loss.backward()
    for p in (geo.proj[0].weight, geo.queries, geo.encoder.embed.weight):
        assert p.grad is not None and p.grad.abs().sum() > 0


# 5. verifiers and the OPSD spec-leak guard
def test_verifiers(records):
    wrong = {"pq-001": "To lyse the cells.", "tw-017": "No error.", "gq-003": "Yes.", "sim-009": "No.",
             "sql-001": "SELECT tube FROM samples", "ord-002": "1, 2, 3"}
    for rec in records:
        right = rec["messages"][-1]["content"]
        assert V.verify(rec, right)["reward"] == 1.0, rec["id"]
        res = V.verify(rec, wrong[rec["id"]])
        assert res["reward"] == 0.0 and not res["passed"], rec["id"]
    sql = by_id(records, "sql-001")
    assert not V.verify(sql, "DELETE FROM samples")["passed"]  # read-only fixture
    assert V.verify({"verifier": {"kind": "numeric", "spec": {"value": 37}}}, "about 37.1 C")["passed"]

    endless = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    t0 = time.time()
    res = V.verify(sql, endless, timeout=1.0)  # a query that never ends must not hang the trainer
    assert not res["passed"] and "timeout" in res["detail"] and time.time() - t0 < 5


def test_spec_never_reaches_prompts_or_student(records):
    for rec in records:
        spec = rec["verifier"]["spec"]
        s = render(rec, TOK, K)
        prompt_text = TOK.decode([t for t, keep in zip(s["ids"], s["train"]) if not keep])
        assert V.spec_leaks(spec, [{"role": "user", "content": prompt_text}]) == [], rec["id"]
        fb = V.verify(rec, rec["messages"][-1]["content"])
        teacher = V.opsd_teacher_messages(rec, fb)
        assert rec["messages"][-1]["content"] in teacher[0]["content"]
        student = validate(V.opsd_student_record(rec, rec["messages"][-1]["content"]))
        assert student["verifier"] == {"kind": "none", "spec": {}}
        assert "Reference answer" not in json.dumps(student)
    tw = by_id(records, "tw-017")
    leaky = copy.deepcopy(tw)
    leaky["messages"][0]["content"] += " Hint: the fix is 37 C."
    with pytest.raises(V.SpecLeak):
        V.opsd_student_record(leaky, "Step 2: use 37 C.")
    with pytest.raises(V.SpecLeak):
        V.opsd_teacher_messages(tw, {"passed": False, "detail": json.dumps(tw["verifier"]["spec"])})


# 6. workloads (profile_gpu.py contract)
def test_workloads():
    N = 12 * 2 * 64 * 64 + 64 * TOK.vocab_size  # tiny config: L=2, d=64
    expect = {"sft_step": 6 * N * 256, "prefill_step": 2 * N * 256, "decode_step": 2 * N * 8}
    for name, fn in FACTORIES.items():
        out = fn(torch=torch, device=torch.device("cpu"), dtype=torch.float32)
        assert {"step", "items_per_step", "flops_per_step", "bytes_per_step"} <= set(out)
        assert out["items_per_step"] > 0
        out["step"]()
        out["step"]()
        if name in expect:
            assert out["flops_per_step"] == expect[name]
        else:
            assert out["flops_per_step"] == 6 * N * out["items_per_step"]
    dec = FACTORIES["decode_step"](torch=torch, device=torch.device("cpu"), dtype=torch.float32)
    assert dec["bytes_per_step"] == N * 4 + 2 * 2 * 64 * 4 * 128 * 8


# 7. configs
def test_configs():
    names = {os.path.basename(p) for p in glob.glob(os.path.join(CONFIGS, "*.json"))}
    assert {"cpu_smoke.json", "colab.json", "h200.json", "b200.json", "stages.json"} <= names
    for name in names:
        with open(os.path.join(CONFIGS, name)) as f:
            cfg = json.load(f)
        check_config_tags(cfg, root=ROOT, where=name)
        if "profiles" in cfg:
            assert cfg["base_model"] == "<set me>" and cfg["profile_default"] in cfg["profiles"]
    with pytest.raises(SchemaError):
        check_config_tags({"x": {"tag": "MEASURED", "source": "no/such/file.jsonl", "v": 1}}, root=ROOT)
    with pytest.raises(SchemaError):
        check_config_tags({"x": {"v": 1}})
