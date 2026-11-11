"""Live MD reaction monitor & drop-in hooks for OpenMM and LAMMPS with timeline plotting.

Streams MD frames in real time (zero LLM memory overhead on non-reactive steps), detects
bond-forming, bond-breaking, and committed Grotthuss proton-transfer events with hysteresis,
triggers optional YLMProbe directional force-vs-displacement attribution on reactive frames,
and exports publication-ready SVG / PNG timeline plots.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from src.toolkit.ingest import (
    ANGSTROM_TO_BOHR,
    BOHR_TO_ANGSTROM,
    StructureFrame,
    covalent_radius,
    minimum_image_delta_bohr,
    symbol_for_z,
)
from src.toolkit.probe import YLMProbe


@dataclass
class ReactionEvent:
    frame_index: int
    time_fs: float
    time_ps: float
    event_type: str  # "bond_formed" | "bond_broken" | "proton_transfer"
    atoms: list[int]
    labels: list[str]
    distance_angstrom: float
    work_ev: float
    force_disp_cosine: float
    description: str
    probe_vector_ev_ang: Optional[list[float]] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _smooth_bond_order(dist_ang: np.ndarray, r0_mat: np.ndarray) -> np.ndarray:
    """Smooth rational covalent bond order B_ij in [0, 1]: 1 / (1 + (d_ij / (1.25 * r0))^8)."""
    ratio = dist_ang / np.maximum(1.25 * r0_mat, 1e-6)
    with np.errstate(over="ignore"):
        bo = 1.0 / (1.0 + np.power(ratio, 8))
    np.fill_diagonal(bo, 0.0)
    return bo


class LiveReactionMonitor:
    """Streaming detector for bond formation, bond cleavage, and proton transfer during MD."""

    def __init__(
        self,
        probe: Optional[YLMProbe] = None,
        *,
        form_threshold: float = 0.68,
        break_threshold: float = 0.22,
        cooldown_steps: int = 2,
    ):
        self.probe = probe
        self.form_threshold = float(form_threshold)
        self.break_threshold = float(break_threshold)
        self.cooldown_steps = int(cooldown_steps)

        self._prev_pos_bohr: Optional[np.ndarray] = None
        self._bond_state: Optional[np.ndarray] = None
        self._last_event_step: dict[tuple[int, int], int] = {}
        self._proton_donor: dict[int, int] = {}
        self._r0_mat: Optional[np.ndarray] = None

        self.events: list[ReactionEvent] = []
        self.history: list[dict] = []

    def step(
        self,
        frame_index: int,
        time_fs: float,
        Z: Sequence[int],
        pos_angstrom: np.ndarray,
        forces_ev_ang: Optional[np.ndarray] = None,
        cell_angstrom: Optional[np.ndarray] = None,
        pbc: Optional[Sequence[bool]] = None,
    ) -> list[ReactionEvent]:
        z_arr = np.asarray(Z, dtype=np.int64)
        pos_ang = np.asarray(pos_angstrom, dtype=np.float64)
        n = len(z_arr)
        pos_bohr = pos_ang * ANGSTROM_TO_BOHR
        cell_bohr = None if cell_angstrom is None else np.asarray(cell_angstrom, dtype=np.float64) * ANGSTROM_TO_BOHR
        pbc_arr = None if pbc is None else np.asarray(pbc, dtype=bool)

        if self._prev_pos_bohr is None:
            delta_bohr = np.zeros_like(pos_bohr)
        else:
            delta_bohr = minimum_image_delta_bohr(pos_bohr, self._prev_pos_bohr, cell_bohr, pbc_arr)
        delta_ang = delta_bohr * BOHR_TO_ANGSTROM

        if self._r0_mat is None or self._r0_mat.shape != (n, n):
            radii = np.array([covalent_radius(int(z)) for z in z_arr], dtype=np.float64)
            self._r0_mat = radii[:, None] + radii[None, :]

        # Pairwise minimum-image distances in Angstrom
        if cell_bohr is None:
            rel_ang = pos_ang[:, None, :] - pos_ang[None, :, :]
        else:
            flat = minimum_image_delta_bohr(
                np.repeat(pos_bohr, n, axis=0),
                np.tile(pos_bohr, (n, 1)),
                cell_bohr,
                pbc_arr,
            )
            rel_ang = flat.reshape(n, n, 3) * BOHR_TO_ANGSTROM
        dist_ang = np.linalg.norm(rel_ang, axis=-1)
        bo = _smooth_bond_order(dist_ang, self._r0_mat)

        # Per-atom mechanical work W_i = F_i . dr_i (eV)
        f_arr = None if forces_ev_ang is None else np.asarray(forces_ev_ang, dtype=np.float64)
        atom_work = np.zeros(n, dtype=np.float64) if f_arr is None else np.sum(f_arr * delta_ang, axis=-1)
        atom_cos = np.zeros(n, dtype=np.float64)
        if f_arr is not None:
            fn = np.linalg.norm(f_arr, axis=-1)
            dn = np.linalg.norm(delta_ang, axis=-1)
            valid = (fn > 1e-8) & (dn > 1e-8)
            atom_cos[valid] = atom_work[valid] / (fn[valid] * dn[valid])

        step_events: list[ReactionEvent] = []
        time_ps = float(time_fs) / 1000.0

        # Initialize hysteresis bond state on the first frame
        if self._bond_state is None:
            self._bond_state = bo >= 0.5 * (self.form_threshold + self.break_threshold)
            self._init_proton_donors(z_arr, dist_ang)
        else:
            # 1) Detect committed Grotthuss proton transfers H between heavy donors (N, O, F, S, Cl)
            heavy_donors = [idx for idx, z in enumerate(z_arr) if z in (7, 8, 9, 16, 17)]
            transferred_pairs: set[tuple[int, int]] = set()
            if len(heavy_donors) >= 2:
                for h_idx in np.where(z_arr == 1)[0]:
                    h_i = int(h_idx)
                    d_list = sorted((float(dist_ang[h_i, hd]), hd) for hd in heavy_donors)
                    d1, nearest_hd = d_list[0]
                    prev_hd = self._proton_donor.get(h_i)
                    if prev_hd is not None and nearest_hd != prev_hd and d1 < 1.25 and dist_ang[h_i, prev_hd] > 1.30:
                        pair_key = (min(h_i, nearest_hd), max(h_i, nearest_hd))
                        if frame_index - self._last_event_step.get(pair_key, -999) >= self.cooldown_steps:
                            self._last_event_step[pair_key] = frame_index
                            self._proton_donor[h_i] = nearest_hd
                            transferred_pairs.add((min(h_i, prev_hd), max(h_i, prev_hd)))
                            transferred_pairs.add(pair_key)
                            syms = [f"{symbol_for_z(int(z_arr[prev_hd]))}#{prev_hd}", f"H#{h_i}", f"{symbol_for_z(int(z_arr[nearest_hd]))}#{nearest_hd}"]
                            p_vec = self._query_probe(z_arr, pos_bohr, cell_bohr, pbc_arr, delta_bohr, h_i)
                            ev = ReactionEvent(
                                frame_index=int(frame_index),
                                time_fs=float(time_fs),
                                time_ps=time_ps,
                                event_type="proton_transfer",
                                atoms=[int(prev_hd), h_i, int(nearest_hd)],
                                labels=syms,
                                distance_angstrom=float(d1),
                                work_ev=float(atom_work[h_i]),
                                force_disp_cosine=float(atom_cos[h_i]),
                                description=f"Proton transfer {syms[0]} -> {syms[1]} -> {syms[2]} at {time_ps:.3f} ps (d={d1:.2f} A)",
                                probe_vector_ev_ang=p_vec,
                            )
                            step_events.append(ev)

            # 2) Detect general covalent bond formation / cleavage with hysteresis
            for i in range(n):
                for j in range(i + 1, n):
                    was_bonded = bool(self._bond_state[i, j])
                    now_val = float(bo[i, j])
                    ev_type = None
                    if (not was_bonded) and now_val >= self.form_threshold:
                        self._bond_state[i, j] = self._bond_state[j, i] = True
                        ev_type = "bond_formed"
                    elif was_bonded and now_val <= self.break_threshold:
                        self._bond_state[i, j] = self._bond_state[j, i] = False
                        ev_type = "bond_broken"

                    if ev_type is not None and (i, j) not in transferred_pairs:
                        if frame_index - self._last_event_step.get((i, j), -999) >= self.cooldown_steps:
                            self._last_event_step[(i, j)] = frame_index
                            si = f"{symbol_for_z(int(z_arr[i]))}#{i}"
                            sj = f"{symbol_for_z(int(z_arr[j]))}#{j}"
                            p_vec = self._query_probe(z_arr, pos_bohr, cell_bohr, pbc_arr, delta_bohr, i)
                            verb = "formed" if ev_type == "bond_formed" else "broken"
                            ev = ReactionEvent(
                                frame_index=int(frame_index),
                                time_fs=float(time_fs),
                                time_ps=time_ps,
                                event_type=ev_type,
                                atoms=[i, j],
                                labels=[si, sj],
                                distance_angstrom=float(dist_ang[i, j]),
                                work_ev=float(atom_work[i] + atom_work[j]),
                                force_disp_cosine=float(0.5 * (atom_cos[i] + atom_cos[j])),
                                description=f"Bond {verb}: {si}-{sj} at {time_ps:.3f} ps (d={dist_ang[i, j]:.2f} A)",
                                probe_vector_ev_ang=p_vec,
                            )
                            step_events.append(ev)

        # Record Grotthuss delta(t) = min_{D!=A} (d_DH - d_AH) for plotting if H + >=2 donors exist
        grotthuss_delta = 0.0
        heavy_donors = [idx for idx, z in enumerate(z_arr) if z in (7, 8, 9, 16, 17)]
        if len(heavy_donors) >= 2 and np.any(z_arr == 1):
            best_abs = 999.0
            for h_i in np.where(z_arr == 1)[0]:
                ds = sorted(float(dist_ang[int(h_i), hd]) for hd in heavy_donors)
                delta_h = ds[0] - ds[1]  # negative when localized on donor, ~0 at Zundel TS
                if abs(delta_h) < best_abs:
                    best_abs = abs(delta_h)
                    grotthuss_delta = delta_h

        self.history.append({
            "frame_index": int(frame_index),
            "time_fs": float(time_fs),
            "time_ps": time_ps,
            "max_displacement_ang": float(np.linalg.norm(delta_ang, axis=-1).max()) if n else 0.0,
            "total_work_ev": float(atom_work.sum()),
            "bond_count": int(np.triu(self._bond_state, k=1).sum()),
            "grotthuss_delta_ang": float(grotthuss_delta),
            "n_events": len(step_events),
        })
        self.events.extend(step_events)
        self._prev_pos_bohr = pos_bohr.copy()
        return step_events

    def _init_proton_donors(self, z_arr: np.ndarray, dist_ang: np.ndarray) -> None:
        heavy_donors = [idx for idx, z in enumerate(z_arr) if z in (7, 8, 9, 16, 17)]
        if not heavy_donors:
            return
        for h_idx in np.where(z_arr == 1)[0]:
            h_i = int(h_idx)
            d1, hd = min((float(dist_ang[h_i, d]), d) for d in heavy_donors)
            if d1 < 1.35:
                self._proton_donor[h_i] = hd

    def _query_probe(
        self,
        z_arr: np.ndarray,
        pos_bohr: np.ndarray,
        cell_bohr: Optional[np.ndarray],
        pbc_arr: Optional[np.ndarray],
        delta_bohr: np.ndarray,
        atom_idx: int,
    ) -> Optional[list[float]]:
        if self.probe is None:
            return None
        frame = StructureFrame(
            Z=z_arr,
            pos_bohr=pos_bohr,
            cell_bohr=cell_bohr,
            pbc=pbc_arr,
            delta_pos_bohr=delta_bohr,
        )
        sym = symbol_for_z(int(z_arr[atom_idx])).lower()
        res = self.probe.ask(frame, f"What is the force vector on the {sym} atom?", task=1)
        return res.vectors[0].vector_physical if res.vectors else None

    def save_summary(self, json_path: Union[str, Path], plot_path: Optional[Union[str, Path]] = None) -> dict:
        payload = {
            "n_frames_monitored": len(self.history),
            "n_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
            "history": self.history,
        }
        out_p = Path(json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if plot_path is not None:
            plot_reaction_timeline(self.history, self.events, plot_path)
        return payload


class OpenMMReactionReporter:
    """Drop-in OpenMM Reporter (`simulation.reporters.append(OpenMMReactionReporter(...))`)."""

    def __init__(
        self,
        out_json: Union[str, Path],
        reportInterval: int = 20,
        *,
        plot_path: Optional[Union[str, Path]] = None,
        probe: Optional[YLMProbe] = None,
        atomic_numbers: Optional[Sequence[int]] = None,
    ):
        self.out_json = Path(out_json)
        self._reportInterval = int(reportInterval)
        self.plot_path = Path(plot_path) if plot_path is not None else None
        self.monitor = LiveReactionMonitor(probe=probe)
        self._atomic_numbers = list(atomic_numbers) if atomic_numbers is not None else None
        self._frame_counter = 0

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        # Request (steps, positions, velocities, forces, energies, enforcePeriodicBox)
        return (steps, True, False, True, False, True)

    def report(self, simulation, state):
        try:
            from openmm import unit as omm_unit
            pos_nm = state.getPositions(asNumpy=True).value_in_unit(omm_unit.nanometer)
            pos_ang = np.asarray(pos_nm, dtype=np.float64) * 10.0
            forces_kj = state.getForces(asNumpy=True).value_in_unit(omm_unit.kilojoule_per_mole / omm_unit.nanometer)
            # 1 kJ/(mol*nm) = 0.0010364269 eV/Angstrom
            forces_ev_ang = np.asarray(forces_kj, dtype=np.float64) * 0.001036426965
            time_ps = state.getTime().value_in_unit(omm_unit.picosecond)
            time_fs = float(time_ps) * 1000.0
        except ImportError:
            # Allow lightweight duck-typed simulation/state mocks in unit tests without OpenMM installed
            pos_ang = np.asarray(state.getPositions(), dtype=np.float64)
            forces_ev_ang = np.asarray(state.getForces(), dtype=np.float64)
            time_fs = float(state.getTime())

        if self._atomic_numbers is None:
            self._atomic_numbers = [int(atom.element.atomic_number) for atom in simulation.topology.atoms()]

        evs = self.monitor.step(
            frame_index=self._frame_counter,
            time_fs=time_fs,
            Z=self._atomic_numbers,
            pos_angstrom=pos_ang,
            forces_ev_ang=forces_ev_ang,
        )
        self._frame_counter += 1
        if evs:
            self.monitor.save_summary(self.out_json, self.plot_path)


class LAMMPSReactionHook:
    """Callback for LAMMPS `fix ID all python/invoke N post_force hook`."""

    def __init__(
        self,
        type_to_atomic_number: dict[int, int],
        out_json: Union[str, Path],
        *,
        dt_fs: float = 0.5,
        force_to_ev_ang: float = 1.0,  # LAMMPS `metal` units are already eV/Angstrom; `real` = 0.0433641
        plot_path: Optional[Union[str, Path]] = None,
        probe: Optional[YLMProbe] = None,
    ):
        self.type_to_z = {int(k): int(v) for k, v in type_to_atomic_number.items()}
        self.out_json = Path(out_json)
        self.dt_fs = float(dt_fs)
        self.force_to_ev_ang = float(force_to_ev_ang)
        self.plot_path = Path(plot_path) if plot_path is not None else None
        self.monitor = LiveReactionMonitor(probe=probe)
        self._frame_counter = 0

    def __call__(self, lmp_ptr, vflag: int = 0) -> None:
        if hasattr(lmp_ptr, "extract_atom_numpy"):
            # Direct mock / wrapper interface
            ntimestep, types, pos_ang, forces_raw = lmp_ptr.extract_atom_numpy()
        else:
            from lammps import lammps
            lmp = lammps(ptr=lmp_ptr)
            ntimestep = int(lmp.extract_global("ntimestep"))
            natoms = int(lmp.get_natoms())
            types = np.asarray(lmp.numpy.extract_atom("type")[:natoms], dtype=np.int64)
            pos_ang = np.asarray(lmp.numpy.extract_atom("x")[:natoms], dtype=np.float64)
            forces_raw = np.asarray(lmp.numpy.extract_atom("f")[:natoms], dtype=np.float64)

        z_arr = [self.type_to_z[int(t)] for t in types]
        evs = self.monitor.step(
            frame_index=self._frame_counter,
            time_fs=float(ntimestep) * self.dt_fs,
            Z=z_arr,
            pos_angstrom=pos_ang,
            forces_ev_ang=forces_raw * self.force_to_ev_ang,
        )
        self._frame_counter += 1
        if evs:
            self.monitor.save_summary(self.out_json, self.plot_path)


def plot_reaction_timeline(
    history: Sequence[dict],
    events: Sequence[ReactionEvent],
    out_path: Union[str, Path],
) -> Path:
    """Render a reaction timeline plot (`.svg` zero-dependency vector plot, or `.png`/`.pdf` via matplotlib)."""
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        out_p.write_text("<svg xmlns='http://www.w3.org/2000/svg' width='600' height='120'></svg>", encoding="utf-8")
        return out_p

    times = [float(h["time_fs"]) for h in history]
    works = [float(h["total_work_ev"]) for h in history]
    bonds = [int(h["bond_count"]) for h in history]
    deltas = [float(h.get("grotthuss_delta_ang", 0.0)) for h in history]

    if out_p.suffix.lower() in (".png", ".pdf"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(8.5, 6.2), sharex=True)
            axes[0].plot(times, bonds, color="#2563eb", lw=2, label="Covalent bond count")
            axes[0].set_ylabel("Bonds")
            axes[1].plot(times, deltas, color="#7c3aed", lw=2, label="Grotthuss delta (A)")
            axes[1].axhline(0.0, color="#94a3b8", ls="--", lw=1)
            axes[1].set_ylabel("delta (A)")
            axes[2].plot(times, works, color="#059669", lw=2, label="Directional work F.dr (eV)")
            axes[2].axhline(0.0, color="#94a3b8", ls="--", lw=1)
            axes[2].set_ylabel("Work (eV)")
            axes[2].set_xlabel("Simulation time (fs)")
            for ev in events:
                col = "#dc2626" if ev.event_type == "bond_broken" else ("#d97706" if ev.event_type == "proton_transfer" else "#16a34a")
                for ax in axes:
                    ax.axvline(ev.time_fs, color=col, ls=":", lw=1.4)
                axes[0].annotate(f"{ev.event_type} (f#{ev.frame_index})", xy=(ev.time_fs, bonds[min(ev.frame_index, len(bonds) - 1)]),
                                 fontsize=8, color=col, rotation=15)
            fig.tight_layout()
            fig.savefig(out_p, dpi=160)
            plt.close(fig)
            return out_p
        except ImportError:
            out_p = out_p.with_suffix(".svg")

    # Clean, dependency-free standalone SVG timeline plot
    w, h = 760, 320
    pad_l, pad_r, pad_t, pad_b = 64, 28, 36, 44
    pw, ph = w - pad_l - pad_r, h - pad_t - pad_b
    t_min, t_max = min(times), max(times)
    if abs(t_max - t_min) < 1e-9:
        t_max = t_min + 1.0

    def sx(t: float) -> float:
        return pad_l + ((t - t_min) / (t_max - t_min)) * pw

    y_vals = works if any(abs(v) > 1e-9 for v in works) else deltas
    y_label = "Directional Work F·dr (eV)" if any(abs(v) > 1e-9 for v in works) else "Proton Sharing delta (A)"
    y_min, y_max = min(min(y_vals), -0.05), max(max(y_vals), 0.05)
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1.0

    def sy(v: float) -> float:
        return pad_t + ph - ((v - y_min) / (y_max - y_min)) * ph

    pts = " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(times, y_vals))
    event_svg = []
    colors = {"bond_formed": "#16a34a", "bond_broken": "#dc2626", "proton_transfer": "#d97706"}
    for idx_e, ev in enumerate(events):
        ex = sx(ev.time_fs)
        col = colors.get(ev.event_type, "#2563eb")
        ty = pad_t + 16 + (idx_e % 3) * 16
        event_svg.append(
            f"<line x1='{ex:.1f}' y1='{pad_t}' x2='{ex:.1f}' y2='{pad_t + ph}' stroke='{col}' stroke-width='1.8' stroke-dasharray='4,3'/>"
            f"<circle cx='{ex:.1f}' cy='{ty}' r='4' fill='{col}'/>"
            f"<text x='{ex + 6:.1f}' y='{ty + 4}' font-family='sans-serif' font-size='11' fill='{col}'>"
            f"f#{ev.frame_index} ({ev.time_fs:.1f} fs): {ev.event_type} {'-'.join(ev.labels)}</text>"
        )

    y_zero = sy(0.0)
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <rect width="{w}" height="{h}" rx="10" fill="#0f172a"/>
  <text x="{pad_l}" y="22" font-family="sans-serif" font-size="13" font-weight="bold" fill="#f8fafc">
    YLM Live MD Reaction Timeline ({len(events)} reactive events across {len(history)} frames)
  </text>
  <rect x="{pad_l}" y="{pad_t}" width="{pw}" height="{ph}" fill="#1e293b" stroke="#334155"/>
  <line x1="{pad_l}" y1="{y_zero:.1f}" x2="{pad_l + pw}" y2="{y_zero:.1f}" stroke="#475569" stroke-dasharray="3,3"/>
  <polyline fill="none" stroke="#38bdf8" stroke-width="2.2" points="{pts}"/>
  {''.join(event_svg)}
  <text x="{pad_l + pw / 2:.1f}" y="{h - 12}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#94a3b8">
    Simulation Time ({t_min:.1f} fs — {t_max:.1f} fs)
  </text>
  <text x="18" y="{pad_t + ph / 2:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#94a3b8" transform="rotate(-90 18 {pad_t + ph / 2:.1f})">
    {y_label}
  </text>
</svg>"""
    out_p.write_text(svg, encoding="utf-8")
    return out_p
