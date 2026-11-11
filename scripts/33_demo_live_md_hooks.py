"""End-to-end CPU demonstration of OpenMMReactionReporter and LAMMPSReactionHook on reactive MD.

Streams a 40-frame reactive trajectory featuring:
  1. Grotthuss proton transfer across a Zundel [O1...H*...O2]+ bridge (saddle climb W < 0 -> relaxation W > 0)
  2. Nucleophilic C-Cl bond cleavage & C-F bond formation (SN2 substitution)

Outputs:
  - `reactions_openmm.json` & `reactions_lammps.json` (with frame, fs/ps timestamps, work_ev, cos(F, dr))
  - `reaction_timeline.svg` (zero-dependency publication timeline plot of mechanical work & reaction events)
  - `event_frameXX_<type>.html` (interactive 3D dual-arrow displacement vs. force views)
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.toolkit.ingest import ANGSTROM_TO_BOHR, StructureFrame  # noqa: E402
from src.toolkit.md_hooks import (  # noqa: E402
    LAMMPSReactionHook,
    OpenMMReactionReporter,
    plot_reaction_timeline,
)
from src.toolkit.probe import ProbeResult, VectorPrediction  # noqa: E402
from src.toolkit.viz import render_html_report  # noqa: E402


class _DuckState:
    def __init__(self, pos_ang: np.ndarray, forces_ev_ang: np.ndarray, time_fs: float):
        self._pos = pos_ang
        self._forces = forces_ev_ang
        self._time_fs = time_fs

    def getPositions(self, asNumpy: bool = True):
        return self._pos

    def getForces(self, asNumpy: bool = True):
        return self._forces

    def getTime(self):
        return self._time_fs


class _DuckSimulation:
    def __init__(self, current_step: int):
        self.currentStep = current_step


class _DuckLAMMPS:
    def __init__(self, ntimestep: int, types: np.ndarray, pos_ang: np.ndarray, forces_ev_ang: np.ndarray):
        self._data = (ntimestep, types, pos_ang, forces_ev_ang)

    def extract_atom_numpy(self):
        return self._data


def build_reactive_demo_trajectory(n_frames: int = 40, dt_fs: float = 2.0) -> tuple[list[int], list[np.ndarray], list[np.ndarray], float]:
    """Synthesize a smooth 40-frame reactive trajectory (Angstrom, eV/A) with proton transfer + SN2 exchange."""
    # Atoms:
    # 0: O1 (donor), 1: H* (transferring proton), 2: O2 (acceptor), 3: H_d1, 4: H_a1
    # 5: C (electrophile, separated at y=+6.0 A), 6: Cl (leaving), 7: F (nucleophile), 8: H_c1, 9: H_c2
    zs = [8, 1, 8, 1, 1, 6, 17, 9, 1, 1]
    pos_frames: list[np.ndarray] = []
    force_frames: list[np.ndarray] = []

    for t in range(n_frames):
        # Stage 1 (frames 6..18): H* hops from x=-0.28 A (on O1 at -1.24 A) to x=+0.28 A (on O2 at +1.24 A)
        s1 = 1.0 / (1.0 + math.exp(-(t - 12.0) / 1.7))
        hx = -0.28 + 0.56 * s1
        fh_x = 3.2 * math.sin(math.pi * (hx / 0.28))

        # Stage 2 (frames 22..34): F approaches C (-3.15 -> -1.36 A) while Cl leaves C (+1.74 -> +3.45 A)
        s2 = 1.0 / (1.0 + math.exp(-(t - 28.0) / 1.8))
        fx_pos = -3.15 + 1.79 * s2
        clx_pos = 1.74 + 1.71 * s2
        f_f_x = -2.1 * math.cos(math.pi * s2)
        f_cl_x = 2.4 * math.sin(math.pi * s2)

        pos = np.array([
            [-1.24, 0.00, 0.00],
            [hx, 0.02 * math.sin(0.3 * t), 0.00],
            [1.24, 0.00, 0.00],
            [-1.82, 0.78, 0.15],
            [1.82, -0.78, -0.15],
            [0.00, 6.00, 0.00],
            [clx_pos, 6.00, 0.04],
            [fx_pos, 6.00, -0.04],
            [0.00, 6.98, 0.45],
            [0.00, 5.02, -0.45],
        ], dtype=np.float64)

        forces = np.zeros_like(pos)
        forces[1, 0] = fh_x
        forces[0, 0] = -0.5 * fh_x
        forces[2, 0] = -0.5 * fh_x
        forces[7, 0] = f_f_x
        forces[6, 0] = f_cl_x
        forces[5, 0] = -(f_f_x + f_cl_x)

        pos_frames.append(pos)
        force_frames.append(forces)

    return zs, pos_frames, force_frames, dt_fs


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description="Demo live OpenMM & LAMMPS reaction hooks + timeline plotting on CPU.")
    parser.add_argument("--out_dir", type=str, default="runs/live_md_demo")
    parser.add_argument("--dt_fs", type=float, default=2.0)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zs, pos_list, force_list, dt_fs = build_reactive_demo_trajectory(n_frames=40, dt_fs=args.dt_fs)

    # 1) Stream through OpenMMReactionReporter
    omm_json = out_dir / "reactions_openmm.json"
    omm_svg = out_dir / "reaction_timeline.svg"
    reporter = OpenMMReactionReporter(
        omm_json,
        reportInterval=1,
        plot_path=omm_svg,
        atomic_numbers=zs,
    )
    for step_idx, (pos_ang, f_ev) in enumerate(zip(pos_list, force_list)):
        st = _DuckState(pos_ang, f_ev, time_fs=step_idx * dt_fs)
        sim = _DuckSimulation(current_step=step_idx)
        reporter.report(sim, st)
    reporter.monitor.save_summary(omm_json, omm_svg)

    # 2) Stream through LAMMPSReactionHook ('metal' units: Angstrom, eV/A)
    uniq_z = sorted(set(zs))
    type_to_z = {idx + 1: z_val for idx, z_val in enumerate(uniq_z)}
    z_to_type = {z_val: idx + 1 for idx, z_val in enumerate(uniq_z)}
    atom_types = np.array([z_to_type[z] for z in zs], dtype=np.int64)

    lmp_json = out_dir / "reactions_lammps.json"
    hook = LAMMPSReactionHook(
        type_to_atomic_number=type_to_z,
        out_json=lmp_json,
        dt_fs=dt_fs,
    )
    for step_idx, (pos_ang, f_ev) in enumerate(zip(pos_list, force_list)):
        hook(_DuckLAMMPS(step_idx, atom_types, pos_ang, f_ev))
    hook.monitor.save_summary(lmp_json)

    # 3) Render interactive 3D dual-arrow HTML snapshots at detected key frames
    html_files: list[str] = []
    for ev in reporter.monitor.events:
        t_idx = max(1, min(len(pos_list) - 1, ev.frame_index))
        delta_ang = pos_list[t_idx] - pos_list[t_idx - 1]
        frame = StructureFrame(
            Z=zs,
            pos_bohr=pos_list[t_idx] * ANGSTROM_TO_BOHR,
            delta_pos_bohr=delta_ang * ANGSTROM_TO_BOHR,
            forces_ev_ang=force_list[t_idx],
            label=f"frame_{ev.frame_index}_{ev.event_type}",
        )
        vecs = []
        attn_pool = [0.0] * len(zs)
        for a_idx in ev.atoms[:2]:
            f_v = force_list[t_idx][a_idx]
            dr_v = delta_ang[a_idx]
            anchor = pos_list[t_idx][a_idx].tolist()
            vecs.append(VectorPrediction(
                name=f"Force F[{a_idx}]",
                vector_au=f_v.tolist(),
                vector_physical=f_v.tolist(),
                unit="eV/A",
                magnitude=float(np.linalg.norm(f_v)),
                anchor_xyz_angstrom=anchor,
                anchor_atom=int(a_idx),
            ))
            vecs.append(VectorPrediction(
                name=f"Motion dr[{a_idx}]",
                vector_au=dr_v.tolist(),
                vector_physical=dr_v.tolist(),
                unit="A/step",
                magnitude=float(np.linalg.norm(dr_v)),
                anchor_xyz_angstrom=anchor,
                anchor_atom=int(a_idx),
            ))
            attn_pool[a_idx] = 1.0 / max(1, len(ev.atoms[:2]))

        probe_res = ProbeResult(
            engine="LiveReactionMonitor",
            question=f"Reaction event at {ev.time_fs:.1f} fs (frame {ev.frame_index})",
            answer=ev.description,
            vectors=vecs,
            tokens=["[EVENT]", ev.event_type],
            atom_attention=[attn_pool, attn_pool],
            pooled_atom_attention=attn_pool,
            channel_norms={"0e": [1.0, 1.0], "1o": [1.0, 1.0]},
            domain="md_reaction",
            domain_summary=ev.to_dict(),
            frame=frame,
        )
        html_name = f"event_frame{ev.frame_index:02d}_{ev.event_type}.html"
        hp = out_dir / html_name
        hp.write_text(
            render_html_report(
                probe_res,
                title=f"{ev.event_type} @ {ev.time_fs:.1f} fs ({ev.description})",
            ),
            encoding="utf-8",
        )
        html_files.append(str(hp))

    plot_reaction_timeline(reporter.monitor.history, reporter.monitor.events, omm_svg)

    summary = {
        "n_frames": len(pos_list),
        "dt_fs": dt_fs,
        "openmm_events": [e.to_dict() for e in reporter.monitor.events],
        "lammps_events_count": len(hook.monitor.events),
        "parity_match": len(reporter.monitor.events) == len(hook.monitor.events),
        "timeline_svg": str(omm_svg),
        "html_views": html_files,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
