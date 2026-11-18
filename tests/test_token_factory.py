"""Tests for src/token_factory.py. No data files: small molecules from SMILES and fake labels."""
import random
from collections import Counter

import numpy as np
import pytest
from rdkit import Chem

from src.parse_question import resolve
from src.token_factory import (
    EXPECTED_REJECT,
    FORCE_TEMPLATES,
    FORCE_TEMPLATES_HELDOUT,
    STAGES,
    build_teacher_prompt,
    describe_neighbors,
    join_parts,
    mutate,
    neighbor_parts,
    parse_teacher_output,
    propose_templates,
    run_factory,
)

SMILES = {0: "CCO", 1: "CC(=O)O", 2: "OC1CCCCC1", 3: "COC1CCOC1", 9: "CCN"}
TRAIN = {0, 1, 2, 3}  # 9 is a "test" molecule


@pytest.fixture(scope="module")
def data():
    mols = {m: Chem.AddHs(Chem.MolFromSmiles(s)) for m, s in SMILES.items()}
    rng = np.random.default_rng(0)
    labels = {m: dict(numbers=np.array([a.GetAtomicNum() for a in mol.GetAtoms()]),
                      forces=rng.normal(size=(mol.GetNumAtoms(), 3)) * 0.01) for m, mol in mols.items()}
    return mols, labels


def check(item, data):
    """-> name of the first failing stage, or None if the item passes all verifiers."""
    mols, labels = data
    ctx = dict(mols=mols, labels=labels, train_ids=TRAIN)
    for name, fn in STAGES:
        ok, reason = fn(item, ctx)
        if not ok:
            return name
    return None


def atom_of(mol, z, nb):
    """index of the atom with element z and neighbour symbols nb (e.g. 'CHH')"""
    for a in mol.GetAtoms():
        if a.GetAtomicNum() == z and sorted(n.GetSymbol() for n in a.GetNeighbors()) == sorted(nb):
            return a.GetIdx()
    raise ValueError


def item(mol, atom, question, arm="A"):
    return dict(mol=mol, atom=atom, question=question, arm=arm)


def test_neighbor_parts_match_02(data):
    for mol in data[0].values():
        for a in mol.GetAtoms():
            parts = neighbor_parts(Counter(n.GetAtomicNum() for n in a.GetNeighbors()))
            assert describe_neighbors(a) == ("bonded to " + join_parts(parts) if parts else "with no bonds")


def test_correct_arm_a_item_passes(data):
    mols, _ = data
    c = atom_of(mols[0], 6, "CHHH")
    q = "What is the force on the carbon atom bonded to three hydrogens and one carbon?"
    assert q in [it["question"] for it in propose_templates(0, mols[0]) if it["atom"] == c]
    assert check(item(0, c, q), data) is None


def test_ring_flag_item_passes(data):
    mols, _ = data
    ring_items = [it for it in propose_templates(3, mols[3]) if "in a ring" in it["question"]]
    assert ring_items
    assert all(check(it, data) is None for it in ring_items)


def test_every_mutation_kind_rejected_by_expected_verifier(data):
    mols, _ = data
    rng = random.Random(0)
    seen = set()
    for m in TRAIN:
        for base in propose_templates(m, mols[m]):
            for v in mutate(base, mols[m], rng):
                assert check(v, data) == EXPECTED_REJECT[v["mutation"]], v
                seen.add(v["mutation"])
    assert seen == set(EXPECTED_REJECT)


def test_article_false_accept_rejected_by_v2a(data):
    mols, _ = data
    o = atom_of(mols[0], 8, "CH")
    q = "What is the force on the oxygen atom bonded to a nitrogen?"
    assert resolve(q, mols[0]) == o  # the known hole in the rule parser
    assert check(item(0, o, q), data) == "V2a"


def test_negation_and_heldout_rejected(data):
    mols, _ = data
    o = atom_of(mols[0], 8, "CH")
    d = "the oxygen atom bonded to one hydrogen and one carbon"
    assert check(item(0, o, f"What is the force on {d}?"), data) is None
    assert check(item(0, o, "What is the force on the oxygen atom not bonded to one hydrogen and one carbon?"),
                 data) == "V2a"
    for t in FORCE_TEMPLATES_HELDOUT:
        assert check(item(0, o, t.format(d=d)), data) == "V2a"


def test_wrong_atom_rejected_by_v2c(data):
    mols, _ = data
    o = atom_of(mols[0], 8, "CH")
    q = "What is the force on the oxygen atom bonded to one hydrogen and one carbon?"
    assert check(item(0, o + 1 if o == 0 else 0, q), data) == "V2c"


def test_partial_list_with_no_exact_match_rejected(data):
    mols, _ = data
    c = atom_of(mols[0], 6, "CHHH")
    q = "What is the force on the carbon atom bonded to two hydrogens and one carbon?"
    assert resolve(q, mols[0]) == -1
    assert check(item(0, c, q), data) == "V2c"


def test_at_least_reading_rejects_carbonyl_oxygen(data):
    mols, _ = data
    o = atom_of(mols[1], 8, "C")
    q = "What is the force on the oxygen atom bonded to one carbon?"
    assert resolve(q, mols[1]) == o  # exact reading is fine, the hydroxyl O also has "one carbon"
    assert check(item(1, o, q), data) == "V2d"


def test_format_and_split(data):
    q = "What is the force on the oxygen atom bonded to one hydrogen and one carbon?"
    assert check(dict(mol=0, question=q, arm="A"), data) == "V0"
    assert check(item(0, "2", q), data) == "V0"
    assert check(item(0, 2, "short"), data) == "V0"
    mols, _ = data
    n = atom_of(mols[9], 7, "CHH")
    assert check(item(9, n, "What is the force on the nitrogen atom bonded to two hydrogens and one carbon?"),
                 data) == "V1"


def test_dedup_and_stats(data):
    mols, labels = data
    items = propose_templates(0, mols[0])
    repeat = dict(items[0])  # exact repeat
    upper = dict(items[1], question=items[1]["question"].upper())  # same after lowercasing
    spaced = dict(items[2], question=items[2]["question"].replace(" the ", "  the "))  # fails V2a, not dedup
    c_items = mutate(items[0], mols[0], random.Random(0))
    existing = {(0, items[0]["atom"])}
    kept, rejected, stats = run_factory(items + [repeat, upper, spaced] + c_items, mols, labels, TRAIN, existing)
    assert stats["dedup_removed"] == {"A": 2}
    assert stats["rejection_reasons"]["A"] == {"V2a: frame is not a train force frame": 1}
    assert len(kept) == stats["kept"]["A"] == len(items)
    assert all(it["arm"] == "A" and len(it["force_hartree_per_bohr"]) == 3 for it in kept)
    assert stats["proposals"]["C"] == len(c_items) and "C" not in stats["kept"]
    assert all(v.get("missed", 0) == 0 for v in stats["mutations"].values())
    assert stats["new_mol_atom_pairs"]["A"] == len({it["atom"] for it in items}) - 1
    assert sum(stats["coverage"].values()) == len(kept)


def test_teacher_prompt_and_parser(data):
    mols, _ = data
    prompt = build_teacher_prompt(mols[1])
    assert "0 | carbon | three hydrogens, one carbon | no" in prompt
    assert all(t.replace("{d}", "<description>") in prompt for t in FORCE_TEMPLATES)
    assert not any(t.split("{d}")[0] in prompt for t in FORCE_TEMPLATES_HELDOUT)
    q = "What is the force on the oxygen atom bonded to one hydrogen and one carbon?"
    good = parse_teacher_output('{"atom": 2, "question": "%s"}' % q, mol=0)
    assert good["atom"] == 2 and good["question"] == q and good["arm"] == "B" and good["mol"] == 0
    junky = parse_teacher_output('Sure! {bad} here: {"atom": 2, "question": "%s"} hope it helps' % q, mol=0)
    assert junky["atom"] == 2 and junky["question"] == q
    assert parse_teacher_output("no json here", mol=0) is None
    assert parse_teacher_output('{"atom": 2}', mol=0) is None
