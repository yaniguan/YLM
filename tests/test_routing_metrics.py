import math
import torch
from src.routing_metrics import RoutingAudit
from src.irreps_to_text import IrrepsToText


def test_hand_computed_and_deduplicated():
    audit = RoutingAudit(topk=(1, 2))
    attn = torch.tensor([[.25, .75, 90.], [.1, .9, 0.], [.8, .2, 0.]])
    mask = torch.tensor([[True, True, False]] * 3)
    pos = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [20., 0., 0.]]] * 3)
    scores = {'invariant': torch.tensor([[[1., 3., 900.]]] * 3),
              'directional': torch.tensor([[[0., 0., 900.]]] * 3)}
    for _ in range(4):
        audit.update(['force', 'dipole', 'force'], attn, [0, -1, 0], [1, 0, 1], mask, pos, scores)
    r = audit.report()
    assert r['n_questions'] == 1
    assert math.isclose(r['entropy'], -(.25 * math.log(.25) + .75 * math.log(.75)), rel_tol=1e-6)
    assert r['target_rank'] == 2 and r['top1'] == 0 and r['top2'] == 1
    assert r['neighbor_error'] == 1 and r['other_error'] == 0
    assert r['invariant_scale'] == 1 and r['directional_scale'] == 0


def test_score_capture_preserves_outputs():
    layer = IrrepsToText(4, '2x0e + 1x1o', '1x1o', option='B')
    h, x = torch.randn(1, 2, 4), torch.randn(1, 3, 5)
    mask = torch.tensor([[True, True, False]])
    original = layer(h, x, mask)
    assert layer.last_scores is None
    layer.audit_enabled = True
    audited = layer(h, x, mask)
    for a, b in zip(original, audited):
        torch.testing.assert_close(a, b)
    scores = layer.last_scores
    combined = (scores['invariant'] + scores['directional']).masked_fill(~mask[:, None], -torch.inf)
    torch.testing.assert_close(combined.softmax(-1), audited[1])
    assert not scores['directional'].requires_grad


def test_far_error_and_empty_force_report():
    a = RoutingAudit()
    mask = torch.tensor([[True, True]])
    pos = torch.tensor([[[0., 0., 0.], [5., 0., 0.]]])
    a.update([0], torch.tensor([[.2, .8]]), [-1], [0], mask, pos)
    assert a.report()['n_questions'] == 0
    a.update([0], torch.tensor([[.2, .8]]), [0], [1], mask, pos)
    assert a.report()['other_error'] == 1


def test_evaluation_counts_identity_only(monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / 'scripts/04_train.py'
    spec = importlib.util.spec_from_file_location('routing_train', path)
    train = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train)

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cross = type('Cross', (), {})()

        def forward(self, *inputs):
            self.cross.last_scores = {'invariant': torch.zeros(2, 1, 2),
                                      'directional': torch.ones(2, 1, 2)}
            return torch.ones(2, 3), torch.tensor([[.9, .1], [.2, .8]])

    inputs = (torch.ones(2, 2, dtype=torch.long), torch.zeros(2, 2, 3),
              torch.ones(2, 2, dtype=torch.bool), torch.zeros(2, 1, 4),
              torch.ones(2, 1, dtype=torch.bool), torch.tensor([0, 1]))
    monkeypatch.setattr(train, 'get_batch', lambda *a: (inputs, torch.ones(2, 3)))
    split = {'mol': torch.tensor([0, 0]), 'task': inputs[-1], 'atom': torch.tensor([-1, 1])}
    model = Fake()
    result = train.evaluate(model, None, split, torch.ones(2), 'cpu', 3, routing_audit=True)
    assert result['routing']['n_questions'] == 1
    assert result['routing']['exact'] == 1
    assert model.cross.last_scores is None and not model.cross.audit_enabled
