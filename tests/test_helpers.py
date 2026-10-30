import pytest
import torch
from src.data import collate_molecules
from src.inference import predict_vector
from src.model import YLM


def test_collate_and_predict():
    torch.manual_seed(3)
    model = YLM(8, hidden_irreps='4x0e + 3x1o', cross_irreps='2x1o', n_layers=1)
    samples = [dict(Z=torch.tensor([6, 1, 8][:n]), pos=torch.randn(n, 3),
                    text=torch.randn(t, 8), task=i)
               for i, (n, t) in enumerate([(3, 2), (2, 4)])]
    batch = collate_molecules(samples)
    assert batch['atom_mask'].tolist() == [[True]*3, [True, True, False]]
    assert batch['text_mask'].tolist() == [[True, True, False, False], [True]*4]
    assert batch['Z'][1, 2] == 0
    assert batch['task'].tolist() == [0, 1]
    with torch.no_grad():
        batched = model(**batch)[0]
    for i, sample in enumerate(samples):
        seen = []
        def encode(question):
            seen.append(question)
            return sample['text']
        prediction = predict_vector(model, sample['Z'], sample['pos'], 'dipole?', encode,
                                    sample['task'])
        torch.testing.assert_close(prediction, batched[i])
        assert prediction.shape == (3,) and not prediction.requires_grad
        assert seen == ['dipole?'] and model.training
    model.eval()
    predict_vector(model, samples[0]['Z'], samples[0]['pos'], '?', lambda _: samples[0]['text'])
    assert not model.training


@pytest.mark.parametrize('samples', [[], [dict(Z=torch.empty(0, dtype=torch.long))],
    [dict(Z=torch.ones(2, dtype=torch.long), pos=torch.zeros(1, 3))],
    [dict(Z=torch.ones(2, dtype=torch.long), pos=torch.zeros(2, 3), text=torch.zeros(0, 8))]])
def test_invalid_collate(samples):
    with pytest.raises(ValueError):
        collate_molecules(samples)
