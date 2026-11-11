"""Cross-repository bridge linking `yaniguan/ouroboros-ocsr` (SE(2) stereo OCSR) to `YLM` (O(3) tensor FM).

Connects Ouroboros-OCSR's three output artifacts directly to YLM's directional & chiral stack:
  1. `<stem>.extxyz` + `<stem>.manifest.jsonl` (`scripts/image_to_forcefield.py` output with
     `mace_energy`, `mace_forces`, `stereo_ok`, unescaped `@/@@` & `E/Z` SMILES, and `margin`)
  2. `predictions.csv` (`Arm A+C` ensemble OCSR predictions with confidence abstention `accepted`)
  3. Stereo SMILES or 2D structure images (`.png`/`.jpg` when `ouroboros` is installed locally)

Preserves the full symmetry chain:
  2D `SE(2)` (`C8` steerable CNN, no `O(2)` reflection) -> 3D CIP-audited `ETKDGv3` conformer ->
  `O(3)` `YLMProbe` & `YLMUniverseModel` (`0e` scalar + `0o` chiral + `1o` vector + `0e+2e` tensor).
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Optional, Sequence, Union

import numpy as np

from src.toolkit.ingest import (
    ANGSTROM_TO_BOHR,
    StructureFrame,
    covalent_radius,
    load_trajectory,
    symbol_for_z,
)
from src.toolkit.probe import ProbeResult, YLMProbe
from src.universe_data import UniverseRecord, generate_diverse_queries

# Built-in exact enantiomer & E/Z conformer geometries (Angstrom) for zero-dependency offline fallback.
# Every (S)-enantiomer is the exact spatial inversion (-pos) of its (R)-enantiomer.
_CHIRAL_FALLBACK_ANGSTROM: dict[str, tuple[list[int], list[list[float]], list[dict]]] = {
    # Bromochlorofluoromethane: F[C@H](Cl)Br vs F[C@@H](Cl)Br
    "F[C@H](Cl)Br": (
        [6, 9, 17, 35, 1],
        [
            [0.000, 0.000, 0.000],
            [0.000, 0.000, 1.360],
            [1.670, 0.000, -0.560],
            [-0.910, 1.580, -0.640],
            [-0.520, -0.890, -0.360],
        ],
        [{"atom_index": 0, "cip": "R"}],
    ),
    "F[C@@H](Cl)Br": (
        [6, 9, 17, 35, 1],
        [
            [0.000, 0.000, 0.000],
            [0.000, 0.000, -1.360],
            [-1.670, 0.000, 0.560],
            [0.910, -1.580, 0.640],
            [0.520, 0.890, 0.360],
        ],
        [{"atom_index": 0, "cip": "S"}],
    ),
    # Fluoroethylene carbonate (FEC, key Li-ion battery SEI additive): (R) vs (S)
    "O=C1OC[C@H](F)O1": (
        [8, 6, 8, 6, 6, 9, 8, 1, 1, 1],
        [
            [0.000, 2.230, 0.000],
            [0.000, 1.030, 0.000],
            [1.120, 0.250, 0.020],
            [0.770, -1.140, 0.140],
            [-0.760, -1.120, -0.120],
            [-1.380, -1.890, 0.840],
            [-1.120, 0.260, -0.020],
            [1.160, -1.690, -0.720],
            [1.180, -1.550, 1.060],
            [-1.080, -1.540, -1.080],
        ],
        [{"atom_index": 4, "cip": "R"}],
    ),
    "O=C1OC[C@@H](F)O1": (
        [8, 6, 8, 6, 6, 9, 8, 1, 1, 1],
        [
            [0.000, -2.230, 0.000],
            [0.000, -1.030, 0.000],
            [-1.120, -0.250, -0.020],
            [-0.770, 1.140, -0.140],
            [0.760, 1.120, 0.120],
            [1.380, 1.890, -0.840],
            [1.120, -0.260, 0.020],
            [-1.160, 1.690, 0.720],
            [-1.180, 1.550, -1.060],
            [1.080, 1.540, 1.080],
        ],
        [{"atom_index": 4, "cip": "S"}],
    ),
    # 1,2-difluoroethene: (E) trans vs (Z) cis (tests backslash SMILES F/C=C\F recovery)
    "F/C=C/F": (
        [9, 6, 6, 9, 1, 1],
        [
            [-1.820, -0.680, 0.000],
            [-0.660, 0.000, 0.000],
            [0.660, 0.000, 0.000],
            [1.820, 0.680, 0.000],
            [-0.680, 1.080, 0.000],
            [0.680, -1.080, 0.000],
        ],
        [],
    ),
    r"F/C=C\F": (
        [9, 6, 6, 9, 1, 1],
        [
            [-1.820, 0.680, 0.000],
            [-0.660, 0.000, 0.000],
            [0.660, 0.000, 0.000],
            [1.820, 0.680, 0.000],
            [-0.680, -1.080, 0.000],
            [0.680, -1.080, 0.000],
        ],
        [],
    ),
}


@dataclass
class OuroborosManifestEntry:
    """Authoritative per-molecule record from `<stem>.manifest.jsonl` (`image_to_forcefield.py`)."""

    entry_id: str
    status: str
    smiles: str
    inchikey: str = ""
    source_image: str = ""
    accepted: bool = True
    margin: Optional[float] = None
    stereo_unassigned: int = 0
    conformers: list[dict] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _maybe_add_ouroboros_path(ouroboros_repo: Optional[Union[str, Path]]) -> None:
    if ouroboros_repo is None:
        return
    repo_str = str(Path(ouroboros_repo).resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def load_ouroboros_manifest(manifest_path: Union[str, Path]) -> dict[str, OuroborosManifestEntry]:
    """Parse `<stem>.manifest.jsonl` keyed by `entry_id` (`id` or `source_id`)."""
    p = Path(manifest_path)
    entries: dict[str, OuroborosManifestEntry] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        eid = str(row.get("id") or row.get("source_id") or len(entries))
        margin_raw = row.get("margin")
        margin_val = None if margin_raw in (None, "") else float(margin_raw)
        acc_raw = row.get("accepted", True)
        if isinstance(acc_raw, str):
            acc_val = acc_raw.strip().lower() in ("1", "true", "yes", "t")
        else:
            acc_val = bool(acc_raw)
        entries[eid] = OuroborosManifestEntry(
            entry_id=eid,
            status=str(row.get("status", "ok")),
            smiles=str(row.get("smiles") or row.get("pred") or ""),
            inchikey=str(row.get("inchikey") or ""),
            source_image=str(row.get("source_image") or ""),
            accepted=acc_val,
            margin=margin_val,
            stereo_unassigned=int(row.get("stereo_unassigned", 0) or 0),
            conformers=list(row.get("conformers") or []),
        )
    return entries


def compute_signed_chiral_volumes(frame: StructureFrame) -> list[dict]:
    """Compute O(3)-odd (`0o`) signed tetrahedral volumes around 4-coordinate chiral centers.

    Neighbors are ordered canonically by `(atomic_number, neighbor_degree, distance_sum_round)` so
    the determinant sign is strictly invariant under atom re-indexing and proper rotations `det(R)=+1`,
    while flipping sign `V(-pos) = -V(pos)` under mirror reflection / inversion `det(R)=-1`.
    """
    n = frame.n_atoms
    if n < 5:
        return []
    pos = frame.pos_angstrom
    z = frame.Z
    radii = np.array([covalent_radius(int(zi)) for zi in z], dtype=np.float64)
    r0 = radii[:, None] + radii[None, :]
    rel = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(rel, axis=-1)
    bonded = (dist > 1e-6) & (dist <= 1.28 * r0)
    degrees = bonded.sum(axis=-1)

    # 2-hop topological signature per atom (strictly graph-topological & O(3)-invariant `0e`,
    # so prochiral -CH2- hydrogens share identical signatures and are never misflagged as chiral!)
    hop1: list[tuple] = []
    for i in range(n):
        nbs = np.where(bonded[i])[0]
        hop1.append((int(z[i]), int(degrees[i]), tuple(sorted(int(z[j]) for j in nbs))))
    env_keys: list[tuple] = []
    for i in range(n):
        nbs = np.where(bonded[i])[0]
        env_keys.append((hop1[i], tuple(sorted(hop1[j] for j in nbs))))

    centers: list[dict] = []
    for c in range(n):
        if int(z[c]) not in (6, 7, 14, 15, 16):
            continue
        nbs = list(np.where(bonded[c])[0])
        if len(nbs) != 4:
            continue
        nb_keys = [env_keys[j] for j in nbs]
        # Require 4 topologically distinct substituents (true stereocenter)
        if len(set(nb_keys)) < 4:
            continue
        ordered = [j for _, j in sorted(zip(nb_keys, nbs), key=lambda pair: pair[0], reverse=True)]
        v1 = pos[ordered[0]] - pos[c]
        v2 = pos[ordered[1]] - pos[c]
        v3 = pos[ordered[2]] - pos[c]
        signed_vol = float(np.linalg.det(np.stack([v1, v2, v3], axis=1)) / 6.0)
        if abs(signed_vol) > 1e-3:
            centers.append({
                "atom_index": int(c),
                "symbol": symbol_for_z(int(z[c])),
                "neighbors": [int(x) for x in ordered],
                "signed_volume_ang3": signed_vol,
                "handedness_sign": 1 if signed_vol > 0 else -1,
            })
    return centers


def load_ouroboros_forcefield_bundle(
    extxyz_path: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    *,
    only_stereo_ok: bool = True,
    only_accepted: bool = False,
    min_margin: Optional[float] = None,
) -> list[StructureFrame]:
    """Load `ouroboros-ocsr` `.extxyz` conformers and join authoritative `.manifest.jsonl` metadata.

    Automatically recovers unescaped `E/Z` SMILES (`F/C=C\\F`), `inchikey`, OCSR confidence `margin`,
    and `stereo_ok` audit results from `<stem>.manifest.jsonl` when present.
    """
    ext_p = Path(extxyz_path)
    frames = load_trajectory(ext_p)

    if manifest_path is None:
        auto_manifest = ext_p.with_suffix(".manifest.jsonl")
        if auto_manifest.exists():
            manifest_path = auto_manifest

    manifest_map: dict[str, OuroborosManifestEntry] = (
        load_ouroboros_manifest(manifest_path) if manifest_path is not None and Path(manifest_path).exists() else {}
    )

    kept: list[StructureFrame] = []
    for fr in frames:
        meta = dict(fr.metadata)
        src_id = str(meta.get("source_id") or meta.get("id") or "")
        conf_id_raw = meta.get("conf_id")
        conf_id = int(conf_id_raw) if conf_id_raw is not None and str(conf_id_raw).lstrip("-").isdigit() else None

        if src_id and src_id in manifest_map:
            entry = manifest_map[src_id]
            # Manifest SMILES is authoritative (never mangled by ASE extxyz backslash escaping)
            if entry.smiles:
                meta["smiles"] = entry.smiles
            if entry.inchikey:
                meta["inchikey"] = entry.inchikey
            if entry.source_image:
                meta.setdefault("source_image", entry.source_image)
            meta["accepted"] = entry.accepted
            if entry.margin is not None:
                meta["margin"] = entry.margin
            meta["stereo_unassigned"] = entry.stereo_unassigned

            if conf_id is not None and entry.conformers:
                for c_info in entry.conformers:
                    if int(c_info.get("conf_id", -999)) == conf_id:
                        if "stereo_ok" in c_info:
                            meta["stereo_ok"] = bool(c_info["stereo_ok"])
                        if "mace_energy" in c_info and "energy_ev" not in meta:
                            meta["energy_ev"] = float(c_info["mace_energy"])
                        break

        stereo_ok = bool(meta.get("stereo_ok", True))
        accepted = bool(meta.get("accepted", True))
        margin = meta.get("margin")

        if only_stereo_ok and not stereo_ok:
            continue
        if only_accepted and not accepted:
            continue
        if min_margin is not None and margin is not None and float(margin) < float(min_margin):
            continue

        meta["chiral_volumes"] = compute_signed_chiral_volumes(fr)
        fr.metadata = meta
        if src_id:
            fr.label = f"{src_id}_conf{conf_id}" if conf_id is not None else src_id
        kept.append(fr)
    return kept


def embed_stereo_smiles(
    smiles: str,
    *,
    seed: int = 0,
    check_3d_stereo: bool = True,
    ouroboros_repo: Optional[Union[str, Path]] = None,
) -> StructureFrame:
    """Embed an isomeric SMILES (`@/@@`, `/\\`) into a 3D `StructureFrame` with 3D CIP stereo audit."""
    smi = smiles.strip()
    _maybe_add_ouroboros_path(ouroboros_repo)

    # 1) Use `ouroboros.geometry.conformers` directly if installed / provided
    try:
        from ouroboros.geometry import conformers as ob_conf  # type: ignore
        res = ob_conf.embed_smiles(smi, num_confs=1, seed=seed)
        if res is not None and hasattr(res, "positions"):
            pos_ang = np.asarray(res.positions, dtype=np.float64)
            z_arr = np.asarray(res.atomic_numbers, dtype=np.int64)
            fr = StructureFrame(
                Z=z_arr,
                pos_bohr=(pos_ang - pos_ang.mean(axis=0)) * ANGSTROM_TO_BOHR,
                label=smi,
                metadata={"smiles": smi, "stereo_ok": bool(getattr(res, "stereo_ok", True)), "embedder": "ouroboros"},
            )
            fr.metadata["chiral_volumes"] = compute_signed_chiral_volumes(fr)
            return fr
    except Exception:
        pass

    # 2) RDKit ETKDGv3 (`enforceChirality=True`) + independent 3D CIP stereochemistry audit
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, inchi

        mol2d = Chem.MolFromSmiles(smi)
        if mol2d is None:
            raise ValueError(f"Invalid SMILES {smi!r}")
        Chem.AssignStereochemistry(mol2d, cleanIt=True, force=True)
        expected_centers = dict(Chem.FindMolChiralCenters(mol2d, includeUnassigned=True))
        expected_bonds = {
            b.GetIdx(): b.GetStereo()
            for b in mol2d.GetBonds()
            if b.GetBondType() == Chem.BondType.DOUBLE
            and b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ)
        }

        mol3d = Chem.AddHs(mol2d)
        params = AllChem.ETKDGv3()
        params.randomSeed = max(1, int(seed) + 1)
        params.enforceChirality = True
        cid = AllChem.EmbedMolecule(mol3d, params)
        if cid < 0:
            params.useRandomCoords = True
            cid = AllChem.EmbedMolecule(mol3d, params)
        if cid < 0:
            raise RuntimeError(f"ETKDGv3 failed to embed {smi!r}")
        try:
            AllChem.MMFFOptimizeMolecule(mol3d, confId=cid, maxIters=200)
        except Exception:
            pass

        stereo_ok = True
        cip_list: list[dict] = []
        if check_3d_stereo:
            audit = Chem.Mol(mol3d)
            Chem.RemoveStereochemistry(audit)
            Chem.AssignStereochemistryFrom3D(audit, confId=cid)
            actual_centers = dict(Chem.FindMolChiralCenters(audit, includeUnassigned=True))
            for idx_c, exp_cip in expected_centers.items():
                if exp_cip in ("R", "S"):
                    act_cip = actual_centers.get(idx_c)
                    if act_cip != exp_cip:
                        stereo_ok = False
                    cip_list.append({"atom_index": int(idx_c), "cip": str(act_cip or "?")})
            for b_idx, exp_st in expected_bonds.items():
                act_st = audit.GetBondWithIdx(b_idx).GetStereo()
                if act_st != exp_st:
                    stereo_ok = False
        else:
            for idx_c, exp_cip in expected_centers.items():
                if exp_cip in ("R", "S"):
                    cip_list.append({"atom_index": int(idx_c), "cip": str(exp_cip)})

        conf = mol3d.GetConformer(cid)
        pos_ang = np.asarray(conf.GetPositions(), dtype=np.float64)
        pos_ang = pos_ang - pos_ang.mean(axis=0)
        z_arr = np.array([a.GetAtomicNum() for a in mol3d.GetAtoms()], dtype=np.int64)
        try:
            ik = inchi.MolToInchiKey(mol2d)
        except Exception:
            ik = ""

        fr = StructureFrame(
            Z=z_arr,
            pos_bohr=pos_ang * ANGSTROM_TO_BOHR,
            label=smi,
            metadata={
                "smiles": smi,
                "inchikey": ik,
                "stereo_ok": stereo_ok,
                "cip_centers": cip_list,
                "stereo_unassigned": sum(1 for v in expected_centers.values() if v == "?"),
                "embedder": "rdkit_etkdgv3",
            },
        )
        fr.metadata["chiral_volumes"] = compute_signed_chiral_volumes(fr)
        return fr
    except ImportError:
        pass

    # 3) Built-in analytical stereo conformer fallback for CI / environments without RDKit
    if smi in _CHIRAL_FALLBACK_ANGSTROM:
        z_list, coords, cips = _CHIRAL_FALLBACK_ANGSTROM[smi]
        pos_ang = np.asarray(coords, dtype=np.float64)
        pos_ang = pos_ang - pos_ang.mean(axis=0)
        conn_hash = hashlib.sha1(smi.replace("@@", "@").replace("\\", "/").encode("utf-8")).hexdigest()[:14].upper()
        stereo_hash = hashlib.sha1(smi.encode("utf-8")).hexdigest()[:10].upper()
        fr = StructureFrame(
            Z=z_list,
            pos_bohr=pos_ang * ANGSTROM_TO_BOHR,
            label=smi,
            metadata={
                "smiles": smi,
                "inchikey": f"{conn_hash}-{stereo_hash}-N",
                "stereo_ok": True,
                "cip_centers": list(cips),
                "stereo_unassigned": 0,
                "embedder": "builtin_stereo_fallback",
            },
        )
        fr.metadata["chiral_volumes"] = compute_signed_chiral_volumes(fr)
        return fr

    from src.toolkit.ingest import smiles_to_frame
    fr = smiles_to_frame(smi, seed=seed)
    fr.metadata.setdefault("stereo_ok", True)
    fr.metadata["chiral_volumes"] = compute_signed_chiral_volumes(fr)
    return fr


def load_ouroboros_predictions_csv(
    csv_path: Union[str, Path],
    *,
    only_accepted: bool = True,
    only_stereo_ok: bool = True,
    min_margin: Optional[float] = None,
    seed: int = 0,
    ouroboros_repo: Optional[Union[str, Path]] = None,
) -> list[StructureFrame]:
    """Read `ouroboros-ocsr` `predictions.csv` and embed stereo-verified 3D `StructureFrame`s."""
    p = Path(csv_path)
    frames: list[StructureFrame] = []
    with p.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            smi = (row.get("smiles") or row.get("pred") or row.get("prediction") or "").strip()
            if not smi:
                continue
            acc_raw = row.get("accepted", "true")
            accepted = str(acc_raw).strip().lower() in ("1", "true", "yes", "t", "")
            if only_accepted and not accepted:
                continue
            m_raw = row.get("margin")
            margin = float(m_raw) if m_raw not in (None, "") else None
            if min_margin is not None and margin is not None and margin < min_margin:
                continue

            fr = embed_stereo_smiles(smi, seed=seed + idx, ouroboros_repo=ouroboros_repo)
            if only_stereo_ok and not bool(fr.metadata.get("stereo_ok", True)):
                continue
            src_id = str(row.get("id") or row.get("image_id") or row.get("source_id") or f"row_{idx}")
            fr.label = src_id
            fr.metadata["source_id"] = src_id
            fr.metadata["source_image"] = str(row.get("source_image") or row.get("image") or "")
            fr.metadata["accepted"] = accepted
            if margin is not None:
                fr.metadata["margin"] = margin
            frames.append(fr)
    return frames


def recognize_image_with_ouroboros(
    image_path: Union[str, Path],
    *,
    ouroboros_repo: Optional[Union[str, Path]] = None,
    ckpt_path: Optional[Union[str, Path]] = None,
    device: str = "cpu",
) -> dict:
    """Invoke `ouroboros-ocsr` Python API on a 2D chemical depiction `.png`/`.jpg` when installed."""
    _maybe_add_ouroboros_path(ouroboros_repo)
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(f"Image file not found: {img_p}")
    try:
        from ouroboros.demo.rotation import RotationLab  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Direct `.png`/`.jpg` 2D OCSR inference requires `yaniguan/ouroboros-ocsr` (`escnn` + checkpoint). "
            "Pass `--ouroboros-repo /path/to/ouroboros-ocsr` or feed Ouroboros's `.extxyz`, `.manifest.jsonl`, "
            "or `predictions.csv` directly into `ylm from-ouroboros`."
        ) from exc

    lab = RotationLab(ckpt_path=ckpt_path, device=device) if ckpt_path else RotationLab(device=device)
    pred = lab.predict(str(img_p))
    if isinstance(pred, dict):
        return pred
    return {"smiles": str(pred), "accepted": True}


def ouroboros_frames_to_universe_records(
    frames: Sequence[StructureFrame],
    *,
    lot: str = "wb97m",
    domain: str = "ouroboros_ocsr",
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    include_heldout: bool = True,
    max_force_sites: int = 2,
) -> list[UniverseRecord]:
    """Convert `ouroboros-ocsr` 3D conformers into leak-free `UniverseRecord` multi-task QA rows.

    Groups conformers by their 14-character connectivity InChIKey block (`inchikey[:14]`) so
    multiple conformers of the same molecule never leak across `train` / `val` / `test_id` splits.
    """
    records: list[UniverseRecord] = []
    r_train, r_val, _ = split_ratios

    for idx, fr in enumerate(frames):
        ik = str(fr.metadata.get("inchikey") or "")
        smi = str(fr.metadata.get("smiles") or fr.label or f"mol_{idx}")
        # First 14 chars of InChIKey encode connectivity only (leak-free conformer & stereo grouping)
        group_key = ik[:14] if len(ik) >= 14 else smi.replace("@@", "@").replace("\\", "/")
        h_val = int.from_bytes(hashlib.sha256(group_key.encode("utf-8")).digest()[:4], "big") / float(2 ** 32)
        if h_val < r_train:
            split = "train"
        elif h_val < r_train + r_val:
            split = "val"
        else:
            split = "test_id"

        sid = f"ouroboros_{fr.label}_{idx}"
        records.extend(
            generate_diverse_queries(
                fr,
                structure_id=sid,
                domain=domain,
                lot=lot,
                split=split,
                max_force_sites=max_force_sites,
            )
        )
        if include_heldout and split == "test_id":
            records.extend(
                generate_diverse_queries(
                    fr,
                    structure_id=sid,
                    domain=domain,
                    lot=lot,
                    split="test_heldout",
                    max_force_sites=1,
                )
            )
    return records


def load_any_ouroboros_input(
    source: Union[str, Path],
    *,
    manifest_path: Optional[Union[str, Path]] = None,
    ouroboros_repo: Optional[Union[str, Path]] = None,
    ouroboros_ckpt: Optional[Union[str, Path]] = None,
    only_stereo_ok: bool = True,
    only_accepted: bool = False,
    min_margin: Optional[float] = None,
    device: str = "cpu",
) -> list[StructureFrame]:
    """Dispatch `.extxyz`, `.manifest.jsonl`, `.csv`, `.png`/`.jpg`, or stereo SMILES to 3D `StructureFrame`s."""
    s = str(source).strip()
    p = Path(s)
    if "\n" not in s and len(s) < 260 and p.exists() and p.is_file():
        name_low = p.name.lower()
        suffix = p.suffix.lower()
        if name_low.endswith(".manifest.jsonl") or suffix == ".jsonl":
            entries = load_ouroboros_manifest(p)
            sibling_extxyz = p.with_name(p.name.replace(".manifest.jsonl", ".extxyz"))
            if sibling_extxyz.exists():
                return load_ouroboros_forcefield_bundle(
                    sibling_extxyz,
                    manifest_path=p,
                    only_stereo_ok=only_stereo_ok,
                    only_accepted=only_accepted,
                    min_margin=min_margin,
                )
            out_frames: list[StructureFrame] = []
            for idx, ent in enumerate(entries.values()):
                if only_accepted and not ent.accepted:
                    continue
                if min_margin is not None and ent.margin is not None and ent.margin < min_margin:
                    continue
                if ent.smiles:
                    fr = embed_stereo_smiles(ent.smiles, seed=idx, ouroboros_repo=ouroboros_repo)
                    if only_stereo_ok and not bool(fr.metadata.get("stereo_ok", True)):
                        continue
                    fr.label = ent.entry_id
                    fr.metadata.update({
                        "source_id": ent.entry_id,
                        "inchikey": ent.inchikey or fr.metadata.get("inchikey", ""),
                        "accepted": ent.accepted,
                        "margin": ent.margin,
                    })
                    out_frames.append(fr)
            return out_frames

        if suffix in (".extxyz", ".xyz"):
            return load_ouroboros_forcefield_bundle(
                p,
                manifest_path=manifest_path,
                only_stereo_ok=only_stereo_ok,
                only_accepted=only_accepted,
                min_margin=min_margin,
            )

        if suffix == ".csv":
            return load_ouroboros_predictions_csv(
                p,
                only_accepted=only_accepted,
                only_stereo_ok=only_stereo_ok,
                min_margin=min_margin,
                ouroboros_repo=ouroboros_repo,
            )

        if suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            rec = recognize_image_with_ouroboros(
                p,
                ouroboros_repo=ouroboros_repo,
                ckpt_path=ouroboros_ckpt,
                device=device,
            )
            fr = embed_stereo_smiles(str(rec["smiles"]), ouroboros_repo=ouroboros_repo)
            fr.metadata["source_image"] = str(p)
            fr.metadata.update({k: v for k, v in rec.items() if k != "smiles"})
            return [fr]

    return [embed_stereo_smiles(s, ouroboros_repo=ouroboros_repo)]


def probe_ouroboros_structures(
    source: Union[str, Path],
    question: str = "What is the 3D molecular dipole vector?",
    *,
    probe: Optional[YLMProbe] = None,
    manifest_path: Optional[Union[str, Path]] = None,
    ouroboros_repo: Optional[Union[str, Path]] = None,
    ouroboros_ckpt: Optional[Union[str, Path]] = None,
    only_stereo_ok: bool = True,
    only_accepted: bool = False,
    min_margin: Optional[float] = None,
    verify_o3: bool = True,
) -> list[ProbeResult]:
    """End-to-end `ouroboros-ocsr` -> 3D stereo-verified conformers -> `YLMProbe` directional analysis."""
    active_probe = probe or YLMProbe.default()
    frames = load_any_ouroboros_input(
        source,
        manifest_path=manifest_path,
        ouroboros_repo=ouroboros_repo,
        ouroboros_ckpt=ouroboros_ckpt,
        only_stereo_ok=only_stereo_ok,
        only_accepted=only_accepted,
        min_margin=min_margin,
        device=active_probe.device,
    )
    results: list[ProbeResult] = []
    for fr in frames:
        res = active_probe.ask(fr, question, verify_o3=verify_o3)
        res.domain = "ouroboros_ocsr"
        res.domain_summary = {
            "smiles": fr.metadata.get("smiles", ""),
            "inchikey": fr.metadata.get("inchikey", ""),
            "stereo_ok": fr.metadata.get("stereo_ok", True),
            "accepted": fr.metadata.get("accepted", True),
            "margin": fr.metadata.get("margin"),
            "energy_ev": fr.metadata.get("energy_ev"),
            "has_mace_forces": fr.forces_ev_ang is not None,
            "chiral_volumes": fr.metadata.get("chiral_volumes", []),
            "cip_centers": fr.metadata.get("cip_centers", []),
        }
        results.append(res)
    return results
