"""Optional public MP insertion electrodes; finite-cluster crystal approximation."""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from src.tasks.common import Example, Seg, group_split
from src.tasks.conformers import single_geometry, ANGSTROM_TO_BOHR

TARGETS = ('average_voltage', 'capacity_grav', 'energy_grav')


def crystal_cluster(structure, radius=6.0, max_atoms=64):
    """Replicate cells; nearest <=64 atoms within radius Angstrom of cell centre.

    Surfaces are unpassivated; periodic connectivity and long-range fields are absent.
    """
    from itertools import product
    lattice = np.asarray(structure.lattice.matrix)
    centre = np.array([0.5, 0.5, 0.5]) @ lattice
    bounds = np.ceil(radius * np.linalg.norm(np.linalg.inv(lattice), axis=0)).astype(int) + 1
    atoms = []
    for shift in product(*(range(-int(b), int(b) + 1) for b in bounds)):
        for site in structure:
            if not site.is_ordered:
                raise ValueError('Disordered crystal has no integer atomic numbers')
            pos = (np.asarray(site.frac_coords) + shift) @ lattice - centre
            dist = np.linalg.norm(pos)
            if dist <= radius:
                atoms.append((dist, int(site.specie.Z), pos))
    atoms.sort(key=lambda a: a[0])
    atoms = atoms[:max_atoms]
    if not atoms or any(z >= 100 for _, z, _ in atoms):
        raise ValueError('Empty cluster or unsupported element')
    return single_geometry([a[1] for a in atoms], np.array([a[2] for a in atoms]) * ANGSTROM_TO_BOHR)


def load_mp_battery(limit=None, cache_path='data/electrolyte/mp_battery.json', log=print):
    if not os.environ.get('MP_API_KEY'):
        log('MP_API_KEY is unset; skipping Materials Project insertion electrodes.')
        return []
    from pymatgen.core import Structure
    path = Path(cache_path)
    records = None
    if path.exists():
        saved = json.loads(path.read_text())
        if isinstance(saved, dict):
            cached = saved['records']
            if saved['complete'] or (limit is not None and len(cached) >= limit):
                records = cached
    if records is None:
        from mp_api.client import MPRester
        records = []
        with MPRester(os.environ['MP_API_KEY']) as mpr:
            docs = mpr.materials.insertion_electrodes.search(
                num_chunks=1 if limit is not None else None, chunk_size=max(1, limit or 1000))
            if limit is not None:
                docs = docs[:limit]
            for doc in docs:
                material_id = str(doc.id_charge)
                structure = mpr.get_structure_by_material_id(material_id)
                records.append({'id': str(doc.battery_id), 'material_id': material_id,
                                'formula': structure.composition.reduced_formula,
                                'working_ion': str(doc.working_ion),
                                'structure': structure.as_dict(),
                                'targets': {k: getattr(doc, k, None) for k in TARGETS}})
            version = mpr.get_database_version()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'records': records, 'complete': limit is None,
                                        'database_version': version,
                                        'retrieved_at': datetime.now(timezone.utc).isoformat()}))
        temporary.replace(path)
    rows = []
    for r in records[:limit] if limit is not None else records:
        try:
            geom = crystal_cluster(Structure.from_dict(r['structure']))
            targets = {k: float(v) for k, v in r['targets'].items() if v is not None and np.isfinite(float(v))}
        except (ValueError, TypeError):
            continue
        if targets:
            rows.append(dict(id=r['id'], material_id=r['material_id'], targets=targets,
                             example=Example([Seg('charged crystal=' + r['formula'] +
                                                  ' working_ion=' + r.get('working_ion', 'unknown'))], geom),
                             fingerprints=[], conditions=[]))
    return rows


def split_mp(rows, seed=0, method='group'):
    return group_split([r['material_id'] for r in rows], seed)
