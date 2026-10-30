"""RedDB targets and group-preserving random/scaffold splits."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from src.tasks.common import Example, molecule_segs, group_split
from src.tasks.conformers import ConformerCache, single_geometry

TARGETS = ('reaction_energy', 'reactant_solubility', 'reactant_lumo', 'reactant_homo')


def load_reddb(path='data/electrolyte/reddb_reaction.tab', limit=None, seed=0, cache=None):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    frame = pd.read_csv(path, sep='\t')
    if limit is not None:
        frame = frame.sample(n=min(limit, len(frame)), random_state=seed)
    cache = cache or ConformerCache()
    rows = []
    for _, r in frame.iterrows():
        mol = Chem.MolFromSmiles(str(r['reactant_smiles']))
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        conf = cache.get(smi)
        if conf is None:
            continue
        targets = {}
        for key in TARGETS:
            try:
                value = float(r[key])
            except (ValueError, TypeError):
                continue
            if np.isfinite(value):
                targets[key] = value
        if not targets:
            continue
        geom = single_geometry(*conf)
        rows.append({'id': str(r['reaction_id']), 'smiles': smi,
                     'example': Example(molecule_segs(smi), geom), 'targets': targets,
                     'scaffold': MurckoScaffold.MurckoScaffoldSmiles(mol=mol),
                     'fingerprints': [(smi, 1.0)], 'conditions': []})
    cache.save()
    logging.getLogger(__name__).info('RedDB retained %d/%d rows', len(rows), len(frame))
    return rows


def split_reddb(rows, seed=0, method='scaffold'):
    if method not in {'random', 'scaffold'}:
        raise ValueError('RedDB split must be random or scaffold')
    # Repeated reactants cannot cross splits. Acyclic scaffolds share the empty group.
    return group_split([r['scaffold'] if method == 'scaffold' else r['smiles'] for r in rows], seed)
