import json
import sys
import types
import numpy as np
import pytest
from src.tasks import mp_battery as mp
from src.tasks.downstream import mp_composition_features, ridge_predict
from src.tasks.conformers import single_geometry


def structure(zs):
    return [types.SimpleNamespace(is_ordered=True, specie=types.SimpleNamespace(Z=z)) for z in zs]


def test_fractions_permutation_replication_and_ion():
    zs = [3, 26, 8, 8, 8, 8]
    a = mp.elemental_fractions(structure(zs))
    np.testing.assert_array_equal(a, mp.elemental_fractions(structure(zs[::-1])))
    np.testing.assert_array_equal(a, mp.elemental_fractions(structure(zs * 4)))
    np.testing.assert_allclose(a[[2, 25, 7]], [1/6, 1/6, 4/6])
    rows = [{'composition_fractions': a, 'working_ion': ion} for ion in ('Li', 'Na', 'unknown')]
    features = mp_composition_features(rows)
    assert features.shape == (3, 237)
    assert features[0, 118 + 3] == 1 and features[1, 118 + 11] == 1
    assert features[2, 118] == 1
    np.testing.assert_array_equal(features[0, :118], features[1, :118])


def test_loader_uses_full_structure_not_cluster(tmp_path, monkeypatch):
    # Synthetic cached MP record: no pymatgen/mp-api installation or network needed.
    core = types.ModuleType('pymatgen.core')
    core.Structure = types.SimpleNamespace(from_dict=lambda saved: structure(saved['zs']))
    monkeypatch.setitem(sys.modules, 'pymatgen', types.ModuleType('pymatgen'))
    monkeypatch.setitem(sys.modules, 'pymatgen.core', core)
    monkeypatch.setenv('MP_API_KEY', 'offline-test-only')
    monkeypatch.setattr(mp, 'crystal_cluster', lambda s: single_geometry([3], [[0, 0, 0]]))
    path = tmp_path / 'mp.json'
    path.write_text(json.dumps({'complete': True, 'records': [dict(
        id='battery', material_id='charged', formula='LiFeO4', working_ion='Li',
        structure={'zs': [3, 26, 8, 8, 8, 8]}, targets={'average_voltage': 3.})]}))
    rows = mp.load_mp_battery(cache_path=path)
    assert len(rows[0]['example'].geom['Z']) == 1
    vector = mp_composition_features(rows)[0]
    assert vector[7] == 4/6 and vector[2] == 1/6
    assert rows[0]['working_ion'] == 'Li'


def test_ridge_uses_training_scaling_and_heldout_labels_are_unused():
    features = np.array([[0., 0.], [2., 0.], [4., 0.], [8., 100.]])
    y = np.array([0., 2., 4., 9.])
    pred, alpha = ridge_predict(features, y, [0, 1], [2], [3], alphas=(1.,))
    # Training mean/std=(1,1); constant second feature has zero fitted weight.
    # Weight=2/(2+1), test standardized first feature=7, target centre=1.
    np.testing.assert_allclose(pred, [1 + 7 * 2/3])
    changed = y.copy()
    changed[3] = -99999
    np.testing.assert_array_equal(pred, ridge_predict(features, changed, [0, 1], [2], [3], alphas=(1.,))[0])
    assert alpha == 1.
    # Validation labels select regularization; test labels cannot select it.
    _, weak = ridge_predict(features, y, [0, 1], [2], [3], alphas=(.01, 100.))
    changed[2] = 1.
    _, strong = ridge_predict(features, changed, [0, 1], [2], [3], alphas=(.01, 100.))
    assert weak == .01 and strong == 100.


def test_missing_provenance_and_invalid_structures_rejected():
    with pytest.raises(ValueError, match='full-structure'):
        mp_composition_features([{'example': 'truncated geometry is insufficient'}])
    with pytest.raises(ValueError, match='Empty'):
        mp.elemental_fractions([])
    disordered = structure([3])
    disordered[0].is_ordered = False
    with pytest.raises(ValueError, match='ordered'):
        mp.elemental_fractions(disordered)
