"""Tests for literature-aligned Electrolyte, Reactive MD (1o/1e/2e + Grotthuss), and Drug SBDD domain packs."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import torch
from e3nn import o3

from src.toolkit.domains import (
    analyze_drug,
    analyze_drug_geometry,
    analyze_electrolyte,
    analyze_electrolyte_geometry,
    detect_md_frame_events,
    scan_md_trajectory,
)
from src.toolkit.ingest import ANGSTROM_TO_BOHR, StructureFrame, load_structure, load_trajectory
from src.toolkit.probe import YLMProbe

REPO_ROOT = Path(__file__).resolve().parents[1]


def _rand_o3(seed: int, improper: bool = True) -> np.ndarray:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        r = o3.rand_matrix(dtype=torch.float64).numpy()
    return -r if improper else r


def test_electrolyte_domain_ssip_cip_agg_and_centrosymmetric_2e_channel():
    probe = YLMProbe.default()
    f_ssip = load_structure(REPO_ROOT / "examples" / "li_ec_shell.xyz")
    res = analyze_electrolyte(f_ssip, probe=probe, verify_o3=True)
    info = res.domain_summary

    assert info["motif"] == "SSIP"
    assert info["coordination_number"] == 2
    assert info["smooth_coordination_number"] > 0.5
    assert info["soft_speciation"]["SSIP"] > 0.95
    assert info["mean_ligand_radial_alignment_cos"] is not None
    assert info["mean_ligand_radial_alignment_cos"] > 0.5  # Carbonyl O and water O dipoles point radially outward
    assert info["dipole_locking_regime"] == "strong_ion_dipole_locked"
    assert res.o3_check is not None and res.o3_check.passed

    # Exact O(3) transformation (improper reflection + translation):
    # - 0e invariants stay identical
    # - 1o escape_vector transforms as R v
    # - 2e cage_quadrupole_2e transforms as R Q R^T
    R = _rand_o3(11, improper=True)
    moved = f_ssip.transformed(R, np.array([3.1, -2.4, 1.7]))
    info_rot = analyze_electrolyte_geometry(moved)

    assert info_rot["smooth_coordination_number"] == pytest.approx(info["smooth_coordination_number"], abs=1e-4)
    assert info_rot["mean_ligand_radial_alignment_cos"] == pytest.approx(info["mean_ligand_radial_alignment_cos"], abs=1e-4)
    np.testing.assert_allclose(
        np.asarray(info_rot["escape_vector"]),
        np.asarray(info["escape_vector"]) @ R.T,
        atol=1e-5,
    )
    q0 = np.asarray(info["cage_quadrupole_2e"])
    q_rot = np.asarray(info_rot["cage_quadrupole_2e"])
    np.testing.assert_allclose(np.trace(q0), 0.0, atol=1e-5)
    np.testing.assert_allclose(q_rot, R @ q0 @ R.T, atol=1e-4)

    # Centrosymmetric periodic crystal cage: 4 O atoms in xy-plane around Li -> 1o escape cancels to 0,
    # while 2e quadrupole open_channel_axis_pm identifies the unobstructed z-channel ([0, 0, +-1])!
    centro = StructureFrame(
        Z=[3, 8, 8, 8, 8],
        pos_bohr=np.array([
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -2.0, 0.0],
        ]) * ANGSTROM_TO_BOHR,
        cell_bohr=np.eye(3) * 10.0 * ANGSTROM_TO_BOHR,
        pbc=np.array([True, True, True]),
        role=[1, 2, 2, 2, 2],
    )
    c_info = analyze_electrolyte_geometry(centro)
    assert c_info["centrosymmetric_cage"] is True
    np.testing.assert_allclose(c_info["escape_vector"], [0.0, 0.0, 0.0], atol=1e-7)
    assert c_info["channel_anisotropy_2e"] > 0.5
    assert abs(c_info["open_channel_axis_pm"][2]) == pytest.approx(1.0, abs=1e-4)

    # Bridging F- between two Li+ -> AGG motif
    agg_frame = StructureFrame(
        Z=[3, 9, 3],
        pos_bohr=np.array([[0.0, 0.0, 0.0], [1.9, 0.0, 0.0], [3.8, 0.0, 0.0]]) * ANGSTROM_TO_BOHR,
        role=[1, 1, 1],
    )
    agg_info = analyze_electrolyte_geometry(agg_frame)
    assert agg_info["motif"] == "AGG"
    assert agg_info["soft_speciation"]["AGG"] > 0.3


def test_md_trajectory_grotthuss_cv_rattling_and_segnn_1o_1e_2e_decomposition():
    probe = YLMProbe.default()
    traj_path = REPO_ROOT / "examples" / "proton_transfer_traj.extxyz"
    scan = scan_md_trajectory(traj_path, probe=probe, verify_o3=True)

    assert scan.n_frames_total == 3
    bridge_events = [e for e in scan.events if e["type"] == "proton_transfer_bridge"]
    assert len(bridge_events) == 3

    # Frame 0: H closer to O#0 -> negative delta; Frame 1: shared Zundel TS (|delta| ~ 0.04 A); Frame 2: positive delta -> committed!
    assert bridge_events[0]["grotthuss_delta_angstrom"] < -0.2
    assert bridge_events[1]["bridge_state"] == "Zundel_shared_TS"
    assert abs(bridge_events[1]["grotthuss_delta_angstrom"]) <= 0.10
    assert bridge_events[2]["grotthuss_delta_angstrom"] > 0.2
    assert all(e["transfer_commitment"] == "committed_grotthuss_transfer" for e in bridge_events)
    assert scan.summary["peak_saliency_frame"] == 1  # Zundel saddle frame has highest composite saliency

    # Frustrated rattling control: H approaches midpoint in frame 1 then retreats back to original donor O#0 in frame 2
    frames = load_trajectory(traj_path)
    rattling_f2 = StructureFrame(
        Z=frames[0].Z.copy(),
        pos_bohr=frames[0].pos_bohr.copy(),
        forces_ev_ang=frames[0].forces_ev_ang.copy(),
    )
    rattling_events = [
        e for e in detect_md_frame_events([frames[0], frames[1], rattling_f2])
        if e["type"] == "proton_transfer_bridge"
    ]
    assert all(e["transfer_commitment"] == "frustrated_rattling" for e in rattling_events)

    # Verify exact O(3) transformation laws for 0e work, 1e axial torque (det(R) * R tau), and 2e virial shear (R Q R^T)
    R_imp = _rand_o3(23, improper=True)   # det(R) = -1
    f1 = frames[1]
    f1_rot = f1.transformed(R_imp, np.array([-1.5, 2.2, 0.8]))

    r_orig = probe.ask(f1, "What is the force on the transferring hydrogen atom?", domain="md")
    r_rot = probe.ask(f1_rot, "What is the force on the transferring hydrogen atom?", domain="md")
    s0 = r_orig.domain_summary
    s1 = r_rot.domain_summary

    assert s0["mechanical_work_ev"] == pytest.approx(s1["mechanical_work_ev"], abs=1e-4)
    assert s0["total_recorded_work_ev"] == pytest.approx(s1["total_recorded_work_ev"], abs=1e-4)
    # 1e pseudovector flips sign relative to 1o under improper reflection (det(R_imp) == -1)!
    tau0 = np.asarray(s0["kinematic_torque_1e"])
    tau1 = np.asarray(s1["kinematic_torque_1e"])
    np.testing.assert_allclose(tau1, np.linalg.det(R_imp) * (tau0 @ R_imp.T), atol=1e-4)
    # 2e symmetric traceless tensor is even-parity: Q' = R Q R^T even when det(R) = -1
    v0 = np.asarray(s0["kinematic_virial_2e"])
    v1 = np.asarray(s1["kinematic_virial_2e"])
    np.testing.assert_allclose(v1, R_imp @ v0 @ R_imp.T, atol=1e-4)


def test_drug_domain_2e_rings_hba_hbd_open_valence_and_linker_bridges():
    probe = YLMProbe.default()
    f_pyr = load_structure(REPO_ROOT / "examples" / "indole_pocket.xyz")
    res = analyze_drug(f_pyr, probe=probe, verify_o3=True)
    info = res.domain_summary

    assert res.o3_check is not None and res.o3_check.passed
    assert len(info["aromatic_rings"]) == 1
    ring = info["aromatic_rings"][0]
    assert ring["size"] == 6
    assert ring["pocket_pi_vector_1o"] is not None
    # Pyridine N#0 lone pair points toward +y where pocket O#10 sits at (0.05, 4.15, 0.15) -> HBA locked!
    assert len(info["hbond_acceptors"]) == 1
    assert info["hbond_acceptors"][0]["acceptor_atom"] == 0
    assert info["hbond_acceptors"][0]["pharmacophore_locked"] is True
    assert len(info["pocket_hbonds"]) == 1
    # C#1-H#6 points straight into pocket C#11 at ~1.96 A (< 2.2 A clash cutoff) so it must be filtered out!
    exit_carbons = [c["carbon_atom"] for c in info["exit_vectors"]]
    assert 1 not in exit_carbons
    assert len(exit_carbons) >= 2

    # Improper O(3) transformation covariance check on 2e ring tensor, 1o ligand dipole, and 1o exit vectors
    R = _rand_o3(31, improper=True)
    moved = f_pyr.transformed(R, np.array([2.0, -3.0, 1.0]))
    info_rot = analyze_drug_geometry(moved)

    t0 = np.asarray(ring["plane_tensor_2e"])
    t_rot = np.asarray(info_rot["aromatic_rings"][0]["plane_tensor_2e"])
    np.testing.assert_allclose(t_rot, R @ t0 @ R.T, atol=1e-4)
    np.testing.assert_allclose(
        np.asarray(info_rot["ligand_polar_dipole_debye"]),
        np.asarray(info["ligand_polar_dipole_debye"]) @ R.T,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(info_rot["exit_vectors"][0]["direction"]),
        np.asarray(info["exit_vectors"][0]["direction"]) @ R.T,
        atol=1e-5,
    )

    # Heavy-atom-only (no explicit hydrogens) benzene input still infers 6 open-valence sp2 exit vectors!
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    hex_pos = np.stack([1.39 * np.cos(angles), 1.39 * np.sin(angles), np.zeros(6)], axis=-1) * ANGSTROM_TO_BOHR
    heavy_only = StructureFrame(Z=[6] * 6, pos_bohr=hex_pos)
    heavy_info = analyze_drug_geometry(heavy_only)
    assert len(heavy_info["aromatic_rings"]) == 1
    assert len(heavy_info["exit_vectors"]) == 6

    # Two-fragment Linker / PROTAC bridging detection (anchors 5.02 A apart -> ~3-atom linker)
    frag2_pos = hex_pos.copy()
    frag2_pos[:, 0] += 7.8 * ANGSTROM_TO_BOHR
    two_frags = StructureFrame(
        Z=[6] * 12,
        pos_bohr=np.concatenate([hex_pos, frag2_pos], axis=0),
    )
    br_info = analyze_drug_geometry(two_frags)
    assert br_info["n_ligand_fragments"] == 2
    assert len(br_info["fragment_bridges"]) >= 1
    assert br_info["fragment_bridges"][0]["cos_theta_a"] > 0.8
