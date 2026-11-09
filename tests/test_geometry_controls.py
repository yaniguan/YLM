import copy
import numpy as np
from src.tasks.common import Example, Seg
from src.tasks.conformers import single_geometry
from src.tasks.geometry_controls import geometry_control
from src.tasks.downstream import controlled_examples


def rows():
    result = []
    for i, zs in enumerate(([6, 8, 1], [8, 1, 6], [6, 1])):
        result.append(dict(id=str(i), targets={'y': i},
                           example=Example([Seg('same text')], single_geometry(zs, np.arange(len(zs)*3).reshape(-1, 3) + i), {'tag': i})))
    return result


def test_donor_element_alignment_determinism_and_metadata():
    source = rows()
    original = copy.deepcopy(source)
    a, report = geometry_control(source, 'donor', indices=[0, 2], donor_indices=[1], seed=4)
    b, second = geometry_control(source, 'donor', indices=[0, 2], donor_indices=[1], seed=4)
    assert report == second and report['changed'] == 1 and report['unmatched'] == 1
    assert report['donors'] == {'0': '1'}
    np.testing.assert_array_equal(a[0]['example'].geom['pos'], source[1]['example'].geom['pos'][[2, 0, 1]])
    for i in range(3):
        assert a[i]['example'].segs == source[i]['example'].segs
        assert a[i]['example'].meta == source[i]['example'].meta
        assert a[i]['targets'] == source[i]['targets']
        for key in ('Z', 'role', 'amount', 'species'):
            np.testing.assert_array_equal(a[i]['example'].geom[key], source[i]['example'].geom[key])
        np.testing.assert_array_equal(source[i]['example'].geom['pos'], original[i]['example'].geom['pos'])
        np.testing.assert_array_equal(a[i]['example'].geom['pos'], b[i]['example'].geom['pos'])


def test_zero_and_text_only_rendering():
    source = rows()
    rendered, report = controlled_examples(source, mode='zero', indices=[0])
    assert report['changed'] == 1
    assert not rendered[0].geom['pos'].any()
    assert rendered[0].text == source[0]['example'].text
    np.testing.assert_array_equal(rendered[1].geom['pos'], source[1]['example'].geom['pos'])
    original, _ = controlled_examples(source, mode='original')
    assert [e.text for e in original] == [e.text for e in rendered]


def test_roles_and_species_prevent_incompatible_donors():
    source = rows()[:2]
    source[1]['example'].geom['role'][0] = 2
    _, report = geometry_control(source, 'donor', indices=[0], donor_indices=[1])
    assert report['unmatched'] == 1
    source[1]['example'].geom = None
    _, report = geometry_control(source, 'zero', indices=[1])
    assert report['missing'] == 1
