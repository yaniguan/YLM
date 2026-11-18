"""CPU API (src/api.py) with a deterministic stand-in text encoder: no downloads in CI."""
import time

import numpy as np
import pytest
import torch

from src.api import BOHR_PER_ANGSTROM, MAX_ATOMS, InputError, YLMPredictor, infer_task, ROOT
from e3nn import o3

CKPT = ROOT / "checkpoints" / "v0" / "ylm_matscibert_s0.pt"
WATER = dict(atomic_numbers=[8, 1, 1], positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
FORCE_Q = "What is the force on the oxygen atom bonded to two hydrogens?"


def fake_encoder(question):
    g = torch.Generator().manual_seed(len(question))
    return torch.randn(len(question.split()) + 2, 768, generator=g)


@pytest.fixture(scope="module")
def pred():
    return YLMPredictor(CKPT, threads=1, text_encoder=fake_encoder)


def test_dipole_and_force_shapes(pred):
    d = pred.predict("Give the molecular dipole vector.", **WATER)
    assert d["task"] == "dipole" and d["unit"] == "e*bohr" and d["routed_atom"] is None
    f = pred.predict(FORCE_Q, **WATER)
    assert f["task"] == "force" and f["unit"] == "hartree/bohr" and f["routed_atom"] in (0, 1, 2)
    for r in (d, f):
        assert len(r["vector"]) == 3 and np.isfinite(r["vector"]).all()
        factor = 2.541746 if r is d else 51.42208619
        assert np.allclose(r["vector_physical"], np.array(r["vector"]) * factor)


def test_vector_rotates_with_molecule(pred):
    torch.manual_seed(0)
    R = o3.rand_matrix().numpy()
    pos = np.array(WATER["positions"])
    for q in ("Give the molecular dipole vector.", FORCE_Q):
        a = np.array(pred.predict(q, atomic_numbers=WATER["atomic_numbers"], positions=pos)["vector"])
        b = np.array(pred.predict(q, atomic_numbers=WATER["atomic_numbers"], positions=pos @ R.T)["vector"])
        assert np.linalg.norm(b - R @ a) <= 1e-4 * max(np.linalg.norm(a), 1e-6)


def test_units_angstrom_vs_bohr(pred):
    a = pred.predict(FORCE_Q, **WATER)["vector"]
    b = pred.predict(FORCE_Q, atomic_numbers=WATER["atomic_numbers"],
                     positions=(np.array(WATER["positions"]) * BOHR_PER_ANGSTROM).tolist(), unit="bohr")["vector"]
    assert np.allclose(a, b, atol=1e-6)


def test_question_cache_is_bounded_and_exact():
    p = YLMPredictor(CKPT, threads=1, text_encoder=fake_encoder, cache_size=2)
    first = p.predict(FORCE_Q, **WATER)["vector"]
    for q in ("Give the molecular dipole vector.", "Predict the electric dipole moment.", FORCE_Q):
        p.predict(q, **WATER)
    assert len(p._cache) == 2
    assert p.predict(FORCE_Q, **WATER)["vector"] == first


@pytest.mark.parametrize("kwargs, match", [
    (dict(question=""), "nonempty"),
    (dict(question="x" * 301), "longer"),
    (dict(question=FORCE_Q, atomic_numbers=None, positions=None), "exactly one"),
    (dict(question=FORCE_Q, atomic_numbers=[8, 1], positions=[[0, 0, 0]] * 3), "shape"),
    (dict(question=FORCE_Q, atomic_numbers=[17, 1], positions=[[0, 0, 0], [1, 0, 0]]), "unsupported elements"),
    (dict(question=FORCE_Q, atomic_numbers=[6] * (MAX_ATOMS + 1), positions=[[i, 0, 0] for i in range(MAX_ATOMS + 1)]),
     "atoms"),
    (dict(question=FORCE_Q, unit="nm"), "unit"),
    (dict(question=FORCE_Q, task="charge"), "task"),
    (dict(question="How do I run a PCR on this sample?"), "dipole or per-atom force"),
])
def test_bad_requests_are_rejected(pred, kwargs, match):
    kwargs = {**WATER, **kwargs}
    q = kwargs.pop("question")
    with pytest.raises(InputError, match=match):
        pred.predict(q, **kwargs)


@pytest.mark.parametrize("question, task", [
    ("Which way does the dipole point, and how strong is it?", "dipole"),
    ("Report the total dipole of the structure.", "dipole"),
    ("Which direction is the carbon atom bonded to one hydrogen pulled, and how hard?", "force"),
    ("How is the oxygen atom in a ring bonded to two carbons pushed?", "force"),
    ("Give the force vector acting on the fluorine atom.", "force"),
])
def test_task_inference_covers_v0_wordings(question, task):
    assert infer_task(question) == task


def test_smiles_input_reports_parser_atom(pred):
    pytest.importorskip("rdkit")
    r = pred.predict("What is the force on the oxygen atom bonded to one carbon and one hydrogen?", smiles="CCO")
    assert r["parsed_atom"] == 2 and len(r["positions_angstrom"]) == 9
    with pytest.raises(InputError, match="invalid SMILES"):
        pred.predict(FORCE_Q, smiles="C1CC")


def test_single_request_is_fast_on_cpu(pred):
    pred.predict(FORCE_Q, **WATER)
    t = time.perf_counter()
    pred.predict(FORCE_Q, **WATER)
    assert time.perf_counter() - t < 1.0  # model part only; the text encoder is stubbed here


def test_fastapi_app(pred):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from src.api import create_app

    client = TestClient(create_app(pred))
    ok = client.post("/predict", json={"question": FORCE_Q, **WATER})
    assert ok.status_code == 200 and ok.json()["task"] == "force"
    bad = client.post("/predict", json={"question": FORCE_Q, "atomic_numbers": [92], "positions": [[0, 0, 0]]})
    assert bad.status_code == 422
