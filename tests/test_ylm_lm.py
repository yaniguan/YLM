"""Small offline tests for Oct-30 LM paths. No pretrained HF weights/network."""
import numpy as np
import pytest
import torch

from src.ylm_lm import YLMConfig, YLMCausalLM, collate_geometry
from src.tasks.common import (Example, Seg, collate, model_inputs, train_lm, seed_everything,
                              batch_tokens, fit_token_budget)
from src.tasks.tokenizer import CharTokenizer
from src.tasks.conformers import ConformerCache, single_geometry
from src.tasks.sft import build_qa, build_preferences, dpo_loss


def config(mode='text'):
    return YLMConfig(mode=mode, vocab_size=CharTokenizer().vocab_size, n_layer=1,
                     n_embd=16, n_head=2, n_positions=96, key_dim=8,
                     cross_irreps='2x0e + 2x1o + 1x1e + 1x2e')


def geometry():
    return single_geometry([8, 1, 1], [[0, 0, 0], [1.4, 0.2, 0], [-0.4, 1.3, 0.1]])


@pytest.mark.parametrize('mode', ['prefix', 'xattnA', 'xattnB', 'xattnC'])
@pytest.mark.parametrize('reflection', [False, True])
def test_loss_invariance(mode, reflection):
    seed_everything(3)
    model = YLMCausalLM(config(mode)).eval()
    for adapter in model.adapters:
        adapter.gate.data.fill_(0.7)  # Nonzero: test the geometry path, not a disabled adapter.
    tok = CharTokenizer()
    geom = geometry()
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotation[:, 0] *= torch.linalg.det(rotation)
    if reflection:
        rotation[:, 0] *= -1
    moved = dict(geom, pos=np.asarray(geom['pos']) @ rotation.numpy().T + [4, -7, 2])
    a = collate([Example([Seg('water E='), Seg('-1.2', 'num')], geom)], tok, 40)
    b = collate([Example([Seg('water E='), Seg('-1.2', 'num')], moved)], tok, 40)
    with torch.no_grad():
        out_a, out_b = model(**model_inputs(a)), model(**model_inputs(b))
    torch.testing.assert_close(out_a['loss'], out_b['loss'], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(out_a['logits'], out_b['logits'], atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize('mode', ['xattnA', 'xattnB', 'xattnC'])
def test_zero_gate_equals_text(mode):
    seed_everything(1)
    baseline = YLMCausalLM(config()).eval()
    adapted = YLMCausalLM(config(mode)).eval()
    adapted.lm.load_state_dict(baseline.lm.state_dict())
    batch = collate([Example([Seg('CC energy=-1.0')], geometry())], CharTokenizer(), 40)
    with torch.no_grad():
        a, b = baseline(**model_inputs(batch)), adapted(**model_inputs(batch))
    torch.testing.assert_close(a['logits'], b['logits'], atol=0, rtol=0)
    assert baseline.param_counts()['geometry_pathway'] == 0
    assert adapted.param_counts()['total'] > baseline.param_counts()['total']


def test_conformer_generation_and_cache(tmp_path):
    cache = ConformerCache(str(tmp_path / 'conformers.pkl'))
    first = cache.get('CO', seed=4)
    assert first is not None
    z, pos = first
    assert len(z) == 6
    assert pos.shape == (6, 3)
    assert np.isfinite(pos).all()
    cache.save()
    again = ConformerCache(str(tmp_path / 'conformers.pkl')).get('CO', seed=4)
    np.testing.assert_array_equal(pos, again[1])
    ions = cache.get('[Li+].[Cl-]')
    assert ions is not None
    assert np.linalg.norm(ions[1][0] - ions[1][1]) > 9.45


def test_calisol_parsing(tmp_path, caplog):
    from src.tasks.calisol import load_calisol, split_calisol, species_mapping
    mapping = tmp_path / 'mapping.csv'
    mapping.write_text('\ufeffLiPF6 ; salt ; [Li+].F[P-](F)(F)(F)(F)F\\r\\n'
                       'Ethylene carbonate ; C1COC(=O)O1;\\r\\n'
                       'Dimethyl carbonate ; COC(=O)OC;')
    assert species_mapping(mapping)['EC'] == 'C1COC(=O)O1'
    data = tmp_path / 'calisol.csv'
    data.write_text('doi,k,T,c,salt,c units,solvent ratio type,EC,DMC,Unknown\n'
                    'a,1,298.2,1,LiPF6,mol/L,m,0.5,0.5,0\n'
                    'b,2,300,1,LiPF6,mol/kg,w,0.2,0.8,0\n'
                    'c,3,300,1,LiPF6,mol/L,m,0,0,1\n')
    class FakeCache:
        def get(self, smiles):
            return np.array([6]), np.zeros((1, 3))
        def save(self):
            pass
    rows = load_calisol(data, mapping, cache=FakeCache())
    assert len(rows) == 2
    assert rows[0]['targets']['log_conductivity'] == pytest.approx(np.log10(1.001))
    assert 'ratio=molar' in rows[0]['example'].text
    assert 'mol/kg' in rows[1]['example'].text
    assert 'Unknown' in caplog.text
    split = split_calisol(rows, method='group')
    assert sum(bool(v) for v in split.values()) == 1  # Same formulation remains together.
    positions = rows[0]['example'].geom['pos']
    assert min(np.linalg.norm(positions[i] - positions[j]) for i in range(3) for j in range(i)) > 9.45


def test_resume_roundtrip(tmp_path):
    tok = CharTokenizer()
    examples = [Example([Seg('CC')]) for _ in range(8)]
    seed_everything(12)
    initial = YLMCausalLM(config()).state_dict()
    seed_everything(23)
    full = YLMCausalLM(config())
    full.load_state_dict(initial)
    # Reset RNG after construction so dropout sees identical states.
    seed_everything(99)
    kwargs = dict(tokens_budget=24, batch_size=2, lr=1e-3, max_len=16, seed=4, log_every=100)
    state_full = train_lm(full, examples, tok, out_dir=str(tmp_path / 'full'), **kwargs)
    resumed = YLMCausalLM(config())
    resumed.load_state_dict(initial)
    seed_everything(99)
    train_lm(resumed, examples, tok, out_dir=str(tmp_path / 'resume'), stop_after_steps=1, **kwargs)
    restored = YLMCausalLM(config())
    state = train_lm(restored, examples, tok, out_dir=str(tmp_path / 'resume'), resume=True, **kwargs)
    assert state['tokens'] == state_full['tokens'] == 24
    assert state['step'] == state_full['step']
    for name, tensor in full.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], atol=0, rtol=0)


def test_answer_mask_and_dpo():
    row = {'smiles': 'CO', 'geom': geometry(), 'targets': {'lumo': -0.123}, 'qm9_index': 1}
    qa = build_qa([row])
    batch = collate(qa, CharTokenizer(), 80, ['answer'])
    assert (batch['labels'][batch['kinds'] != 4] == -100).all()
    assert (batch['labels'] != -100).sum() > 0
    pair = build_preferences(qa)[0]
    assert pair.chosen.text != pair.rejected.text
    loss = dpo_loss(torch.tensor([0.0]), torch.tensor([-1.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    assert float(loss) < np.log(2)


def test_missing_geometry_and_prefix_budget():
    geom = collate_geometry([None, geometry()])
    assert not geom['atom_mask'][0].any()
    model = YLMCausalLM(config('prefix'))
    batch = collate([Example([Seg('CC')], geometry())] * 2, CharTokenizer(), 20)
    fitted = fit_token_budget(batch, model, 10)
    assert batch_tokens(fitted, model) <= 10
    assert fit_token_budget(batch, model, 4) is None
