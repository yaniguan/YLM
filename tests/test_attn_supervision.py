import math

import torch

from src.attn_supervision import attention_supervision_loss, atom_hit_rate
from src.model import YLM


def test_loss_values_and_mask():
    attn = torch.tensor([[0.5, 0.25, 0.25], [0.1, 0.9, 0.0], [0.3, 0.3, 0.4]])
    atom = torch.tensor([0, 1, 0])
    task = torch.tensor([1, 1, 0])
    loss = attention_supervision_loss(attn, atom, task)
    assert math.isclose(loss.item(), -(math.log(0.5) + math.log(0.9)) / 2, rel_tol=1e-5)
    mask = torch.tensor([[True, True, False], [True, True, True], [True, True, True]])
    loss_m = attention_supervision_loss(attn, atom, task, mask)
    assert math.isclose(loss_m.item(), -(math.log(0.5 / 0.75) + math.log(0.9)) / 2, rel_tol=1e-5)
    assert atom_hit_rate(attn, atom, task) == 1.0


def test_no_force_rows_is_zero_with_grad():
    attn = torch.rand(2, 4, requires_grad=True)
    loss = attention_supervision_loss(attn, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    assert loss.item() == 0.0
    loss.backward()
    assert math.isnan(atom_hit_rate(attn, torch.zeros(2), torch.zeros(2)))


def test_training_raises_atom_hit_on_toy_batch():
    torch.manual_seed(0)
    B, N, T, D = 16, 5, 3, 8
    model = YLM(D, hidden_irreps='4x0e + 2x0o + 3x1o + 2x1e + 2x2e',
                cross_irreps='3x0e + 2x1o + 1x1e + 1x2e', n_layers=2, option="B")
    Z = torch.randint(1, 9, (B, N))
    pos = torch.randn(B, N, 3) * 1.5
    mask = torch.ones(B, N, dtype=torch.bool)
    atom = torch.randint(0, N, (B,))
    # text encodes the asked atom's element (toy: one-hot of Z[atom]) so the target is learnable
    text = torch.zeros(B, T, D)
    text[torch.arange(B), 0, Z[torch.arange(B), atom] % D] = 1.0
    tmask = torch.ones(B, T, dtype=torch.bool)
    task = torch.ones(B, dtype=torch.long)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    _, a0 = model(Z, pos, mask, text, tmask, task)
    l0 = attention_supervision_loss(a0, atom, task, mask).item()
    for _ in range(60):
        _, a = model(Z, pos, mask, text, tmask, task)
        loss = attention_supervision_loss(a, atom, task, mask)
        opt.zero_grad(); loss.backward(); opt.step()
    _, a1 = model(Z, pos, mask, text, tmask, task)
    assert attention_supervision_loss(a1, atom, task, mask).item() < l0
    assert atom_hit_rate(a1, atom, task) >= atom_hit_rate(a0, atom, task)


def test_resume_checks_attn_sup_and_allows_legacy_checkpoint(tmp_path, monkeypatch):
    import importlib.util
    import json
    from pathlib import Path
    import pickle
    import sys
    import pytest

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    labels = [
        {
            "numbers": [8, 1, 1],
            "pos": [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.4, 1.7, 0.0]],
            "dipole": [0.1, 0.2, 0.0],
            "forces": [[0.01, -0.02, 0.0], [-0.005, 0.01, 0.0], [-0.005, 0.01, 0.0]],
        },
        {
            "numbers": [6, 8],
            "pos": [[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]],
            "dipole": [-0.3, 0.0, 0.0],
            "forces": [[0.02, 0.0, 0.0], [-0.02, 0.0, 0.0]],
        },
    ]
    with open(data_dir / "qm9_xtb.pkl", "wb") as handle:
        pickle.dump(labels, handle)

    questions = []
    for sp in ("train", "val", "test", "test_heldout"):
        questions.append({"mol": 0, "split": sp, "task": "dipole", "atom": -1, "question": "dipole?"})
        questions.append({"mol": 1, "split": sp, "task": "force", "atom": 0, "question": "force?"})
    (data_dir / "tensorqa_v0.json").write_text(json.dumps(questions), encoding="utf-8")

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "04_train.py"
    spec = importlib.util.spec_from_file_location("train04_resume", script_path)
    train04 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train04)

    monkeypatch.chdir(tmp_path)
    base_cmd = [
        "04_train.py", "--text", "taskid", "--name", "sup_run", "--tiny",
        "--epochs", "2", "--batch_size", "2", "--eval_rotations", "1", "--device", "cpu",
    ]

    # 1) Start with --attn_sup 0.1 (stop after 1 step / epoch 0), then try resuming with --attn_sup 0.0 -> error
    monkeypatch.setattr(sys, "argv", base_cmd + ["--max_steps", "1", "--attn_sup", "0.1"])
    train04.main()
    monkeypatch.setattr(sys, "argv", base_cmd + ["--resume", "--attn_sup", "0.0"])
    with pytest.raises(ValueError, match="attn_sup"):
        train04.main()

    # 2) Pre-change checkpoint without `attn_sup` in saved["args"] still resumes cleanly with --attn_sup 0
    monkeypatch.setattr(sys, "argv", [
        "04_train.py", "--text", "taskid", "--name", "legacy_run", "--tiny",
        "--epochs", "2", "--batch_size", "2", "--eval_rotations", "1", "--device", "cpu",
        "--max_steps", "1", "--attn_sup", "0.0",
    ])
    train04.main()
    latest_path = tmp_path / "runs" / "legacy_run" / "latest.pt"
    saved = torch.load(latest_path, map_location="cpu", weights_only=False)
    saved["args"].pop("attn_sup", None)
    torch.save(saved, latest_path)

    with pytest.raises(ValueError, match="attn_sup"):
        monkeypatch.setattr(sys, "argv", [
            "04_train.py", "--text", "taskid", "--name", "legacy_run", "--tiny",
            "--epochs", "2", "--batch_size", "2", "--eval_rotations", "1", "--device", "cpu",
            "--resume", "--attn_sup", "0.2",
        ])
        train04.main()

    monkeypatch.setattr(sys, "argv", [
        "04_train.py", "--text", "taskid", "--name", "legacy_run", "--tiny",
        "--epochs", "2", "--batch_size", "2", "--eval_rotations", "1", "--device", "cpu",
        "--resume", "--attn_sup", "0.0",
    ])
    train04.main()
    log_txt = (tmp_path / "runs" / "legacy_run" / "log.txt").read_text(encoding="utf-8")
    assert "resumed epoch 1" in log_txt
