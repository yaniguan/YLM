"""Deterministic coordinate controls. Donors carry no labels or text."""
from __future__ import annotations
import copy
import hashlib
import numpy as np


def _groups(geom):
    """Canonical species ordering by role and elemental multiset."""
    groups = []
    for sid in np.unique(geom['species']):
        ids = np.flatnonzero(np.asarray(geom['species']) == sid)
        signature = tuple(sorted((int(geom['role'][i]), int(geom['Z'][i])) for i in ids))
        groups.append((signature, ids))
    return sorted(groups, key=lambda item: item[0])


def _mapping(recipient, donor):
    a, b = _groups(recipient), _groups(donor)
    if [g[0] for g in a] != [g[0] for g in b]:
        return None
    mapping = np.empty(len(recipient['Z']), dtype=int)
    for (_, ai), (_, bi) in zip(a, b):
        key = lambda geom, i: (int(geom['role'][i]), int(geom['Z'][i]))
        ai = sorted(ai, key=lambda i: key(recipient, i))
        bi = sorted(bi, key=lambda i: key(donor, i))
        mapping[ai] = bi
    return mapping


def geometry_control(rows, mode='original', *, indices=None, donor_indices=None, seed=0):
    """Copy rows and replace selected coordinates, preserving all other fields.

    Runner selects only test rows and draws donors from training rows. Unmatched
    recipients keep their original geometry and are explicitly counted.
    Matching respects per-species elemental multisets and atom roles. Amounts
    and species IDs always remain those of the recipient.
    """
    if mode not in {'original', 'zero', 'donor'}:
        raise ValueError('Unknown geometry control')
    result = copy.deepcopy(rows)
    ids = list(range(len(rows))) if indices is None else list(indices)
    donors = list(range(len(rows))) if donor_indices is None else list(donor_indices)
    report = dict(mode=mode, requested=len(ids), changed=0, unmatched=0, missing=0, donors={})
    if mode == 'original':
        return result, report
    for i in ids:
        geom = rows[i]['example'].geom
        if geom is None:
            report['missing'] += 1
            continue
        if mode == 'zero':
            result[i]['example'].geom['pos'] = np.zeros_like(geom['pos'])
        else:
            candidates = []
            for j in donors:
                if i == j or str(rows[i]['id']) == str(rows[j]['id']):
                    continue
                donor = rows[j]['example'].geom
                mapping = _mapping(geom, donor) if donor is not None else None
                if mapping is not None:
                    candidates.append((str(rows[j]['id']), j, mapping))
            candidates.sort(key=lambda item: (item[0], item[1]))
            if not candidates:
                report['unmatched'] += 1
                continue
            digest = hashlib.sha256(f'{seed}|{rows[i]["id"]}'.encode()).digest()
            _, j, mapping = candidates[int.from_bytes(digest[:8], 'big') % len(candidates)]
            result[i]['example'].geom['pos'] = np.asarray(rows[j]['example'].geom['pos'])[mapping].copy()
            report['donors'][str(rows[i]['id'])] = str(rows[j]['id'])
        report['changed'] += 1
    return result, report
