import copy
import numpy as np
import pytest
from src.tasks.qm9_pretrain import pretrain_example, corpus, check_corpus_resume
from src.tasks.conformers import single_geometry


def row(i=0):
    return dict(qm9_index=i, smiles='CO', energy=-2.5, dipole=np.array([3., 4., 0.]),
                targets={'energy': -2.5, 'dipole_norm': 5.},
                geom=single_geometry([6, 8], [[0., 0., 0.], [1., 2., 3.]]))


def test_invariant_target_rotation_and_reflection():
    original = row()
    transformed = copy.deepcopy(original)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., -1.]])
    transformed['geom']['pos'] = transformed['geom']['pos'] @ rotation.T
    transformed['dipole'] = transformed['dipole'] @ rotation.T
    a = pretrain_example(original, target_mode='invariant')
    b = pretrain_example(transformed, target_mode='invariant')
    assert a.text == b.text
    assert a.meta['targets'] == {'energy': -2.5, 'dipole_norm': 5.}
    assert {s.prop for s in a.segs if s.kind == 'num'} == {'energy', 'dipole_norm'}
    assert 'mu=[' not in a.text
    assert a.geom is original['geom']


def test_norm_is_computed_from_vector_and_legacy_unchanged():
    r = row()
    legacy = pretrain_example(r)
    assert legacy.text == '<mol> CO </mol> E=-2.5 mu=[3.0,4.0,0.0] |mu|=5.0'
    assert legacy.text == pretrain_example(r, target_mode='cartesian').text
    r['targets']['dipole_norm'] = 999
    assert pretrain_example(r, target_mode='invariant').meta['targets']['dipole_norm'] == 5
    with pytest.raises(ValueError):
        pretrain_example(r, target_mode='unknown')


def test_same_membership_and_objective_for_arms():
    rows = [row(i) for i in range(30)]
    legacy = corpus(rows, seed=3)
    invariant = corpus(rows, seed=3, target_mode='invariant')
    xyz = corpus(rows, seed=3, with_xyz=True, target_mode='invariant')
    for split in legacy:
        assert [e.meta['qm9_index'] for e in legacy[split]] == [e.meta['qm9_index'] for e in invariant[split]]
        for a, b in zip(invariant[split], xyz[split]):
            assert a.meta['targets'] == b.meta['targets']
            assert [s for s in a.segs if s.prop] == [s for s in b.segs if s.prop]


def test_resume_mode_guard():
    check_corpus_resume({}, 'cartesian')
    check_corpus_resume({'corpus_target_mode': 'invariant'}, 'invariant')
    with pytest.raises(ValueError, match='corpus target mode'):
        check_corpus_resume({}, 'invariant')
    with pytest.raises(ValueError):
        check_corpus_resume({'args': {'target_mode': 'invariant'}}, 'cartesian')
