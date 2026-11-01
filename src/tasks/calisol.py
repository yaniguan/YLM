"""CALiSol cleaning; retain concentration units and ratio basis without invented conversions."""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from src.tasks.common import Example, Seg, group_split, random_split
from src.tasks.conformers import ConformerCache, build_mixture

LOG = logging.getLogger(__name__)
ALIASES = {'EC': 'Ethylene carbonate', 'PC': 'Propylene carbonate',
           'DMC': 'Dimethyl carbonate', 'EMC': 'Ethyl Methyl Carbonate'}
META_COLUMNS = {'doi', 'k', 'T', 'c', 'salt', 'c units', 'solvent ratio type'}
RATIO_NAMES = {'mol': 'molar', 'm': 'molar', 'w': 'mass', 'v': 'volume'}


def species_mapping(path='data/electrolyte/calisolsmile.csv'):
    from rdkit import Chem
    with open(path, encoding='utf-8-sig') as handle:
        text = handle.read().replace('\\r\\n', '\n')
    mapping = {}
    for line in text.splitlines():
        fields = [f.strip() for f in line.split(';') if f.strip()]
        if len(fields) < 2:
            continue
        smi = fields[-1]
        if Chem.MolFromSmiles(smi) is None:
            LOG.warning('Invalid mapping row: %s', line)
            continue
        for name in fields[:-1]:
            mapping[name] = smi
    for alias, full in ALIASES.items():
        if full in mapping:
            mapping[alias] = mapping[full]
    return mapping


def load_calisol(path='data/electrolyte/calisol23.csv', mapping_path='data/electrolyte/calisolsmile.csv',
                 limit=None, seed=0, cache=None):
    frame = pd.read_csv(path)
    mapping = species_mapping(mapping_path)
    solvents = [c for c in frame.columns if c not in META_COLUMNS]
    missing = (set(solvents) | set(frame['salt'].dropna().astype(str).str.strip())) - mapping.keys()
    for name in sorted(missing):
        LOG.warning('Unmapped CALiSol species: %s (affected rows dropped)', name)
    cache = cache or ConformerCache()
    result = []
    if limit is not None:
        frame = frame.sample(n=min(limit, len(frame)), random_state=seed)
    for idx, row in frame.iterrows():
        try:
            k, temp, conc = float(row['k']), float(row['T']), float(row['c'])
            fractions = np.array([float(row[s]) for s in solvents])
        except (ValueError, TypeError):
            continue
        if not np.isfinite(np.r_[k, temp, conc, fractions]).all() or k < 0 or temp <= 0 or conc < 0:
            continue
        if (fractions < 0).any() or fractions.sum() <= 0:
            continue
        salt = str(row['salt']).strip()
        active = [(s, float(f)) for s, f in zip(solvents, fractions / fractions.sum()) if f > 0]
        if salt not in mapping or any(s not in mapping for s, _ in active):
            continue
        components = [(salt, conc, 1)] + [(s, f, 2) for s, f in active]
        species, fps = [], []
        for name, amount, role in components:
            conf = cache.get(mapping[name])
            if conf is None:
                break
            Z, pos = conf
            species.append(dict(Z=Z, pos=pos, role=role, amount=amount))
            fps.append((mapping[name], amount))
        if len(species) != len(components):
            continue
        basis = str(row['solvent ratio type']).strip()
        basis = 'mol' if basis == 'm' else basis
        units = str(row['c units']).strip()
        units = 'mol/L' if units.lower() == 'mol/l' else units
        prompt = (f'salt={salt} c={conc:g} {units} T={temp:.1f}K solvents ' +
                  ' '.join(f'{s}:{f:.4f}' for s, f in active) + f' ratio={RATIO_NAMES.get(basis, basis)}')
        result.append({'id': str(idx), 'example': Example([Seg(prompt)], build_mixture(species)),
                       'targets': {'log_conductivity': float(np.log10(k + 1e-3))}, 'conductivity': k,
                       'group': salt + '|' + '|'.join(sorted(s for s, _ in active)),
                       'doi': str(row['doi']), 'fingerprints': fps, 'conditions': [temp, conc],
                       'ratio_basis': basis, 'concentration_units': units})
    cache.save()
    LOG.info('CALiSol retained %d/%d rows', len(result), len(frame))
    return result


def split_calisol(rows, seed=0, method='group'):
    if method == 'random':
        return random_split(len(rows), seed)
    if method not in {'group', 'doi'}:
        raise ValueError('CALiSol split must be random, group, or doi')
    return group_split([r[method] for r in rows], seed)


def conductivity_from_log(values):
    return np.maximum(0.0, np.power(10.0, np.clip(values, -12, 12)) - 1e-3)
