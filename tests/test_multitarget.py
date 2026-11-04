import numpy as np
import pytest
import torch
from src.tasks import downstream as ds
from src.tasks.common import Example, Seg
from src.tasks.tokenizer import CharTokenizer
from src.ylm_lm import YLMConfig, YLMCausalLM


def test_masked_loss_per_target_and_no_missing_gradient():
    pred = torch.tensor([[1., 8.], [3., 4.]], requires_grad=True)
    y = torch.tensor([[0., float('nan')], [1., 2.]])
    loss = ds.masked_multitarget_loss(pred, y)
    torch.testing.assert_close(loss, torch.tensor((2.5 + 4.) / 2))
    loss.backward()
    assert pred.grad[0, 1] == 0
    assert torch.isfinite(pred.grad).all()
    blank = torch.full_like(y, float('nan'))
    pred.grad.zero_()
    ds.masked_multitarget_loss(pred, blank).backward()
    assert not pred.grad.any()


def test_one_target_matches_scalar_mse():
    pred = torch.randn(5, 1, requires_grad=True)
    y = torch.randn(5, 1)
    torch.testing.assert_close(ds.masked_multitarget_loss(pred, y), torch.nn.functional.mse_loss(pred, y))


def test_training_only_statistics_and_constant_target():
    y = np.array([[1., 4.], [3., np.nan], [1000., -999.]])
    centre, scale = ds.multitarget_stats(y, [0, 1])
    np.testing.assert_allclose(centre, [2., 4.])
    np.testing.assert_allclose(scale, [1., 1e-6])
    with pytest.raises(ValueError, match='observed training'):
        ds.multitarget_stats(np.array([[1., np.nan]]), [0])


def test_tiny_joint_training_restores_best_validation(monkeypatch):
    torch.manual_seed(4)
    tok = CharTokenizer()
    model = YLMCausalLM(YLMConfig(mode='text', vocab_size=tok.vocab_size,
                                n_layer=1, n_embd=8, n_head=2, n_positions=32))
    examples = [Example([Seg(str(i))]) for i in range(6)]
    y = np.array([[0., 1.], [2., np.nan], [np.nan, 3.], [1., 2.], [2., np.nan], [np.nan, 1.]])
    observed = []
    actual_loss = ds.masked_multitarget_loss

    def traced(pred, target):
        loss = actual_loss(pred, target)
        if not pred.requires_grad:
            observed.append((float(loss), {k: v.clone() for k, v in model.state_dict().items()}))
        return loss

    monkeypatch.setattr(ds, 'masked_multitarget_loss', traced)
    # Use a large learning rate to test choosing the best validation model.
    prediction = ds.fine_tune_multitarget(model, tok, examples, y, [0, 1, 2], [3], [4, 5],
                                         epochs=3, batch_size=3, max_len=16, lr=.1)
    assert prediction.shape == (2, 2) and np.isfinite(prediction).all()
    assert len(observed) == 3
    best = min(observed, key=lambda pair: pair[0])[1]
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, best[key])
