import copy
import pytest
from src.tasks.exposure import training_groups, merge_exposure, assert_no_group_leak


def config(task, rows):
    return {'group_exposure': merge_exposure({}, {task: training_groups(task, rows, [0])})}


@pytest.mark.parametrize('field', ['smiles', 'scaffold'])
def test_reddb_different_ids_same_chemistry_rejected(field):
    rows = [dict(id='train', smiles='CC', scaffold=''),
            dict(id='other', smiles='CO', scaffold='other')]
    rows[1][field] = rows[0][field]
    with pytest.raises(ValueError, match=field):
        assert_no_group_leak(config('reddb', rows), 'reddb', rows, {'val': [1], 'test': []})


def test_calisol_formulation_overlap_and_disjoint_groups():
    rows = [dict(id='0', group='salt|EC|DMC'), dict(id='1', group='salt|EC|DMC'),
            dict(id='2', group='other|PC')]
    saved = config('calisol', rows)
    with pytest.raises(ValueError, match='groups overlap'):
        assert_no_group_leak(saved, 'calisol', rows, {'val': [], 'test': [1]})
    assert_no_group_leak(saved, 'calisol', rows, {'val': [], 'test': [2]})


def test_multistage_union_does_not_mutate_and_unknown_is_sticky():
    first = config('reddb', [dict(smiles='CC', scaffold='')])
    original = copy.deepcopy(first)
    second = merge_exposure(first, {'calisol': {'group': ['salt|EC']}})
    third = merge_exposure({'group_exposure': second}, {'reddb': {'smiles': ['CO'], 'scaffold': ['ring']}})
    assert first == original
    assert third['groups']['reddb']['smiles'] == ['CC', 'CO']
    assert third['groups']['calisol']['group'] == ['salt|EC']
    old = {'sft_train_ids': {'reddb': ['old-row']}}
    unknown = merge_exposure(old, {'reddb': {'smiles': ['CC'], 'scaffold': ['']}})
    with pytest.raises(ValueError, match='historical group provenance'):
        assert_no_group_leak({'group_exposure': unknown}, 'reddb', [], {'val': [], 'test': []})


def test_versions_and_missing_fields_fail_explicitly():
    with pytest.raises(ValueError, match='version'):
        merge_exposure({'group_exposure': {'version': 99}}, {})
    with pytest.raises(ValueError, match='provenance'):
        training_groups('calisol', [{'id': '0'}], [0])
    assert_no_group_leak({}, 'calisol', [], {'val': [], 'test': []})
