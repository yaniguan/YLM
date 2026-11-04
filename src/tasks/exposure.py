"""Versioned chemistry-group exposure across SFT and DPO stages."""
from __future__ import annotations
import copy

VERSION = 1
FIELDS = {'calisol': ('group',), 'reddb': ('smiles', 'scaffold'), 'mp_battery': ('material_id',)}


def training_groups(task, rows, indices):
    """Use loader-provided splitting keys, including the empty acyclic scaffold."""
    if task not in FIELDS:
        raise ValueError(f'No group definition for {task}')
    groups = {key: set() for key in FIELDS[task]}
    for i in indices:
        for key in groups:
            if key not in rows[i] or rows[i][key] is None:
                raise ValueError(f'Missing {task} group provenance: {key}')
            groups[key].add(str(rows[i][key]))
    return {key: sorted(values) for key, values in groups.items()}


def merge_exposure(config, current):
    """Union known groups; never infer chemistry from historical row IDs."""
    saved = config.get('group_exposure')
    if saved is not None and saved.get('version') != VERSION:
        raise ValueError('Unsupported group-exposure manifest version')
    result = copy.deepcopy(saved) if saved else dict(version=VERSION, groups={}, unknown_sources=[])
    unknown = set(result['unknown_sources'])
    for source in set(config.get('qa_splits', {})) | set(config.get('sft_train_ids', {})):
        if source in FIELDS and source not in result['groups']:
            unknown.add(source)
    for source, fields in current.items():
        if source not in FIELDS or set(fields) != set(FIELDS[source]):
            raise ValueError('Group manifest definitions differ from current schema')
        previous = result['groups'].setdefault(source, {})
        for key, values in fields.items():
            previous[key] = sorted(set(previous.get(key, [])) | set(values))
    result['unknown_sources'] = sorted(unknown)
    return result


def assert_no_group_leak(config, task, rows, split):
    manifest = merge_exposure(config, {})
    if task in manifest['unknown_sources']:
        raise ValueError(f'Missing historical group provenance for {task}; row IDs cannot establish group isolation')
    trained = manifest['groups'].get(task)
    if trained is None:
        return
    if set(trained) != set(FIELDS[task]):
        raise ValueError('Incomplete group-exposure manifest')
    heldout = training_groups(task, rows, [i for k in ('val', 'test') for i in split[k]])
    for key, values in heldout.items():
        if set(values) & set(trained[key]):
            raise ValueError(f'Downstream held-out {task} {key} groups overlap post-training exposure')
