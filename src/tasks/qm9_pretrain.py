"""QM9 xTB corpus; qm9_index is the zero-based SDF index from 01_make_labels.py."""
from __future__ import annotations
import pickle
import numpy as np
from src.tasks.common import Example, Seg, fmt_num, molecule_segs, qm9_split
from src.tasks.conformers import single_geometry


def load_qm9(pkl_path='data/qm9_xtb.pkl', sdf_path='data/gdb9.sdf', limit=None):
    from rdkit import Chem
    with open(pkl_path, 'rb') as handle:
        rows = sorted(pickle.load(handle), key=lambda r: int(r['qm9_index']))
    if limit is not None:
        rows = rows[:limit]
    wanted = {int(r['qm9_index']) for r in rows}
    last_wanted = max(wanted, default=-1)
    smiles = {}
    for i, mol in enumerate(Chem.SDMolSupplier(sdf_path, removeHs=False)):
        if i in wanted and mol is not None:
            smiles[i] = Chem.MolToSmiles(Chem.RemoveHs(mol))
        if i >= last_wanted:
            break
    result = []
    for row in rows:
        i = int(row['qm9_index'])
        if i not in smiles:
            continue
        r = dict(row, smiles=smiles[i])
        mu = np.asarray(r['dipole'], dtype=float)
        if not np.isfinite(np.r_[r['energy'], mu, np.asarray(r['pos']).ravel()]).all():
            continue
        r['targets'] = {'energy': float(r['energy']), 'dipole_norm': float(np.linalg.norm(mu))}
        r['geom'] = single_geometry(r['numbers'], r['pos'])
        result.append(r)
    return result


def pretrain_example(row, with_xyz=False):
    segs = molecule_segs(row['smiles'], row['geom'], with_xyz)
    segs += [Seg(' E='), Seg(fmt_num(row['energy'], 6), 'num', 'energy'), Seg(' mu=[')]
    for i, value in enumerate(row['dipole']):
        if i:
            segs.append(Seg(','))
        segs.append(Seg(fmt_num(float(value), 4), 'num', f'dipole_{i}'))
    segs += [Seg('] |mu|='), Seg(fmt_num(row['targets']['dipole_norm'], 4), 'num', 'dipole_norm')]
    return Example(segs, row['geom'], {'qm9_index': int(row['qm9_index']), 'targets': row['targets']})


def corpus(rows, seed=0, with_xyz=False):
    splits = qm9_split([r['qm9_index'] for r in rows], seed)
    return {name: [pretrain_example(r, with_xyz) for r in rows if r['qm9_index'] in ids]
            for name, ids in splits.items()}
