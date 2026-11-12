"""Molecular dynamics runner and OpenMM / MACE / NequIP force helpers.

Includes:
  1. `TorchMLIPForceProvider`: computes `(energy_ev, forces_ev_ang)` from MACE, NequIP, ASE, or custom functions.
  2. `AnalyticalReactiveForceProvider`: fast built-in Morse bond potential (`F = -dE/dx`) for CPU testing.
  3. `MLIPTrajectoryRunner`: runs Velocity-Verlet / Langevin MD steps while watching for bond changes and H+ jumps.
  4. `build_openmm_mlip_simulation`: connects MACE or NequIP TorchScript models to OpenMM with reaction logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import numpy as np
import torch

from src.mace_backbone import MACEEncoder
from src.nequip_backbone import NequIPEncoder
from src.toolkit.domains import scan_md_trajectory
from src.toolkit.ingest import (
    ANGSTROM_TO_BOHR,
    StructureFrame,
    covalent_radius,
    load_structure,
    symbol_for_z,
)
from src.toolkit.md_hooks import LiveReactionMonitor, OpenMMReactionReporter, ReactionEvent
from src.toolkit.probe import YLMProbe

# Exact physical unit conversion: 1 eV / amu = 0.009648533212 Å^2 / fs^2
EV_OVER_AMU_TO_ANG2_FS2 = 0.009648533212
KB_EV_PER_K = 8.617333262e-5

ATOMIC_MASS_AMU: dict[int, float] = {
    1: 1.008, 2: 4.0026, 3: 6.94, 4: 9.0122, 5: 10.81, 6: 12.011, 7: 14.007, 8: 15.999,
    9: 18.998, 10: 20.180, 11: 22.990, 12: 24.305, 13: 26.982, 14: 28.085, 15: 30.974,
    16: 32.06, 17: 35.45, 18: 39.948, 19: 39.098, 20: 40.078, 26: 55.845, 27: 58.933,
    28: 58.693, 29: 63.546, 30: 65.38, 35: 79.904, 47: 107.87, 53: 126.90, 78: 195.08, 79: 196.97,
}


def atomic_masses_amu(Z: Sequence[int]) -> np.ndarray:
    return np.array([ATOMIC_MASS_AMU.get(int(z), max(2.0 * float(z), 1.0)) for z in Z], dtype=np.float64)


class AnalyticalReactiveForceProvider:
    """Smooth O(3)-invariant soft-Morse pairwise potential with exact conservative `-dE/dr` forces.

    Useful for zero-dependency CPU validation of reactive MD workflows before switching to
    pretrained `MACE-MP-0`, `MACE-OFF23`, `NequIP`, or `Allegro` checkpoints on GPU.
    """

    def __init__(self, well_depth_ev: float = 1.6, stiffness: float = 1.85, cutoff_ang: float = 6.0):
        self.well_depth_ev = float(well_depth_ev)
        self.stiffness = float(stiffness)
        self.cutoff_ang = float(cutoff_ang)

    def __call__(
        self,
        Z: np.ndarray,
        pos_angstrom: np.ndarray,
        cell_angstrom: Optional[np.ndarray] = None,
        pbc: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray]:
        pos_t = torch.as_tensor(pos_angstrom, dtype=torch.float64).detach().requires_grad_(True)
        n = pos_t.shape[0]
        if n <= 1:
            return 0.0, np.zeros_like(pos_angstrom, dtype=np.float64)

        radii = torch.tensor([covalent_radius(int(z)) for z in Z], dtype=torch.float64)
        r0 = radii[:, None] + radii[None, :]  # (N, N)
        diff = pos_t[:, None, :] - pos_t[None, :, :]
        if cell_angstrom is not None and (pbc is None or np.any(pbc)):
            cell_t = torch.as_tensor(cell_angstrom, dtype=torch.float64)
            pinv = torch.linalg.pinv(cell_t)
            frac = diff @ pinv
            diff = diff - torch.round(frac) @ cell_t

        eye = torch.eye(n, dtype=torch.bool)
        dist = (diff.pow(2).sum(dim=-1) + 1e-12).sqrt()
        # Smooth Morse pair energy D * ((1 - exp(-a*(r - r0)))^2 - 1) * cutoff_envelope
        exp_term = torch.exp((-self.stiffness * (dist - r0)).clamp(min=-8.0, max=6.0))
        morse = self.well_depth_ev * ((1.0 - exp_term).pow(2) - 1.0)
        env = torch.cos(0.5 * torch.pi * (dist / self.cutoff_ang).clamp(max=1.0)).pow(2)
        pair_e = morse * env * (~eye).to(torch.float64)
        total_e = 0.5 * pair_e.sum()
        forces = -torch.autograd.grad(total_e, pos_t)[0]
        return float(total_e.detach().item()), forces.detach().cpu().numpy()


class TorchMLIPForceProvider:
    """Uniform `(energy_ev, forces_ev_ang)` adapter over MACEEncoder, NequIPEncoder, ASE, or callables."""

    def __init__(
        self,
        backend: Union[MACEEncoder, NequIPEncoder, Callable, object],
        *,
        device: str = "cpu",
    ):
        self.backend = backend
        self.device = device

    @classmethod
    def from_spec(
        cls,
        *,
        mace: Optional[str] = None,
        nequip: Optional[str] = None,
        device: str = "cpu",
    ) -> "TorchMLIPForceProvider":
        if mace:
            enc = MACEEncoder(backbone=mace, include_mace_forces=True).to(device)
            return cls(enc, device=device)
        if nequip:
            enc = NequIPEncoder(backbone=nequip, include_nequip_forces=True, device=device).to(device)
            return cls(enc, device=device)
        return cls(AnalyticalReactiveForceProvider(), device=device)

    def __call__(
        self,
        Z: np.ndarray,
        pos_angstrom: np.ndarray,
        cell_angstrom: Optional[np.ndarray] = None,
        pbc: Optional[np.ndarray] = None,
    ) -> tuple[float, np.ndarray]:
        z_arr = np.asarray(Z, dtype=np.int64)
        pos_ang = np.asarray(pos_angstrom, dtype=np.float64)

        if isinstance(self.backend, NequIPEncoder):
            z_t = torch.as_tensor(z_arr, dtype=torch.long, device=self.device)
            pos_t = torch.as_tensor(pos_ang, dtype=torch.float32, device=self.device)
            cell_t = None if cell_angstrom is None else torch.as_tensor(cell_angstrom, dtype=torch.float32, device=self.device)
            pbc_t = None if pbc is None else torch.as_tensor(pbc, dtype=torch.bool, device=self.device)
            e_ev, f_ev_ang = self.backend.predict_energy_and_forces(z_t, pos_t, cell_t, pbc_t)
            return float(e_ev.detach().cpu().item()), f_ev_ang.detach().cpu().numpy().astype(np.float64)

        if isinstance(self.backend, MACEEncoder):
            enc = self.backend
            dtype = enc._mace_dtype(torch.float32)
            z_1 = torch.as_tensor(z_arr[None, :], dtype=torch.long, device=self.device)
            pos_1 = torch.as_tensor(pos_ang[None, :, :], dtype=dtype, device=self.device)
            mask_1 = torch.ones_like(z_1, dtype=torch.bool, device=self.device)
            cell_1 = None if cell_angstrom is None else torch.as_tensor(cell_angstrom[None, :, :], dtype=dtype, device=self.device)
            pbc_1 = None if pbc is None else torch.as_tensor(pbc[None, :], dtype=torch.bool, device=self.device)
            data, _, _ = enc._build_mace_batch(z_1, pos_1, mask_1, cell_1, pbc_1, dtype)
            with torch.enable_grad():
                data["positions"] = data["positions"].detach().requires_grad_(True)
                mace_out = enc.mace(data, training=False, compute_force=True)
            e_key = "energy" if "energy" in mace_out else ("total_energy" if "total_energy" in mace_out else None)
            e_val = float(mace_out[e_key].sum().detach().cpu().item()) if e_key is not None else 0.0
            f_arr = mace_out["forces"].detach().cpu().numpy().astype(np.float64)
            return e_val, f_arr

        # ASE Calculator duck-typing (`atoms.get_forces()`, `atoms.get_potential_energy()`)
        if hasattr(self.backend, "calculate") and hasattr(self.backend, "get_forces"):
            from ase import Atoms  # type: ignore
            atoms = Atoms(
                numbers=z_arr.tolist(),
                positions=pos_ang,
                cell=cell_angstrom if cell_angstrom is not None else None,
                pbc=pbc if pbc is not None else False,
            )
            atoms.calc = self.backend
            f_arr = np.asarray(atoms.get_forces(), dtype=np.float64)
            e_val = float(atoms.get_potential_energy())
            return e_val, f_arr

        if callable(self.backend):
            return self.backend(z_arr, pos_ang, cell_angstrom, pbc)

        raise TypeError(f"Unsupported MLIP force provider {type(self.backend)!r}")


@dataclass
class MLIPMDRunResult:
    frames: list[StructureFrame]
    events: list[ReactionEvent]
    history: list[dict]
    energies_ev: list[float]
    extxyz_path: Optional[str] = None
    events_json_path: Optional[str] = None
    plot_path: Optional[str] = None
    html_path: Optional[str] = None

    def to_summary_dict(self) -> dict:
        return {
            "n_frames": len(self.frames),
            "n_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
            "initial_energy_ev": self.energies_ev[0] if self.energies_ev else 0.0,
            "final_energy_ev": self.energies_ev[-1] if self.energies_ev else 0.0,
            "extxyz_path": self.extxyz_path,
            "events_json_path": self.events_json_path,
            "plot_path": self.plot_path,
            "html_path": self.html_path,
        }


def write_extxyz_trajectory(
    frames: Sequence[StructureFrame],
    energies_ev: Sequence[float],
    path: Union[str, Path],
) -> Path:
    """Write a multi-frame `.extxyz` trajectory with `energy` and `forces` columns."""
    out_p = Path(path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    for idx, frame in enumerate(frames):
        e_val = float(energies_ev[idx]) if idx < len(energies_ev) else float(frame.metadata.get("energy_ev", 0.0))
        header = [f'Properties=species:S:1:pos:R:3:forces:R:3', f"energy={e_val:.8f}", f"frame={idx}"]
        if frame.cell_angstrom is not None:
            lat = " ".join(f"{v:.6f}" for v in frame.cell_angstrom.reshape(-1))
            header.append(f'Lattice="{lat}"')
            pbc_flags = frame.pbc if frame.pbc is not None else np.ones(3, dtype=bool)
            pbc_str = " ".join("T" if bool(b) else "F" for b in pbc_flags)
            header.append(f'pbc="{pbc_str}"')
        pos_ang = frame.pos_angstrom
        f_ang = frame.forces_ev_ang if frame.forces_ev_ang is not None else np.zeros_like(pos_ang)
        lines = [str(frame.n_atoms), " ".join(header)]
        for z, (x, y, zc), (fx, fy, fz) in zip(frame.Z, pos_ang, f_ang):
            lines.append(
                f"{symbol_for_z(int(z)):2s} {x:13.6f} {y:13.6f} {zc:13.6f} {fx:13.6f} {fy:13.6f} {fz:13.6f}"
            )
        chunks.append("\n".join(lines))
    out_p.write_text("\n".join(chunks) + "\n", encoding="utf-8")
    return out_p


class MLIPTrajectoryRunner:
    """Velocity-Verlet + Langevin MD runner coupled to `LiveReactionMonitor` and `YLMProbe`."""

    def __init__(
        self,
        force_provider: Union[TorchMLIPForceProvider, MACEEncoder, NequIPEncoder, Callable],
        *,
        probe: Optional[YLMProbe] = None,
        dt_fs: float = 0.5,
        temperature_k: float = 300.0,
        friction_per_fs: float = 0.01,
        seed: int = 42,
        device: str = "cpu",
    ):
        if isinstance(force_provider, TorchMLIPForceProvider):
            self.force_provider = force_provider
        else:
            self.force_provider = TorchMLIPForceProvider(force_provider, device=device)
        self.probe = probe
        self.dt_fs = float(dt_fs)
        self.temperature_k = float(temperature_k)
        self.friction_per_fs = float(friction_per_fs)
        self.seed = int(seed)

    def run(
        self,
        structure: Union[str, Path, dict, StructureFrame],
        *,
        steps: int = 25,
        report_every: int = 1,
        initial_velocities_ang_fs: Optional[np.ndarray] = None,
        out_extxyz: Optional[Union[str, Path]] = None,
        out_events_json: Optional[Union[str, Path]] = None,
        out_plot: Optional[Union[str, Path]] = None,
        out_html: Optional[Union[str, Path]] = None,
        query: str = "What is the force vector on the reacting atom?",
    ) -> MLIPMDRunResult:
        init_frame = load_structure(structure)
        z_arr = init_frame.Z.copy()
        pos_ang = init_frame.pos_angstrom.copy()
        cell_ang = None if init_frame.cell_angstrom is None else init_frame.cell_angstrom.copy()
        pbc_arr = None if init_frame.pbc is None else init_frame.pbc.copy()
        masses = atomic_masses_amu(z_arr)  # (N,)
        inv_m = (EV_OVER_AMU_TO_ANG2_FS2 / masses)[:, None]  # (N, 1) converts eV/Å -> Å/fs^2

        rng = np.random.default_rng(self.seed)
        if initial_velocities_ang_fs is not None:
            vel = np.asarray(initial_velocities_ang_fs, dtype=np.float64).reshape(pos_ang.shape)
        elif self.temperature_k > 0.0:
            sigma_v = np.sqrt(KB_EV_PER_K * self.temperature_k * EV_OVER_AMU_TO_ANG2_FS2 / masses)[:, None]
            vel = rng.normal(0.0, 1.0, size=pos_ang.shape) * sigma_v
            # Remove net center-of-mass linear momentum
            vel -= (masses[:, None] * vel).sum(axis=0, keepdims=True) / masses.sum()
        else:
            vel = np.zeros_like(pos_ang)

        monitor = LiveReactionMonitor(probe=self.probe)
        frames: list[StructureFrame] = []
        energies_ev: list[float] = []

        e_ev, forces = self.force_provider(z_arr, pos_ang, cell_ang, pbc_arr)
        dt = self.dt_fs
        gamma = self.friction_per_fs
        c1 = float(np.exp(-gamma * dt)) if gamma > 0.0 else 1.0
        noise_scale = (
            np.sqrt(max(0.0, 1.0 - c1 * c1) * KB_EV_PER_K * self.temperature_k * EV_OVER_AMU_TO_ANG2_FS2 / masses)[:, None]
            if (gamma > 0.0 and self.temperature_k > 0.0)
            else np.zeros((len(z_arr), 1), dtype=np.float64)
        )

        prev_pos_bohr = pos_ang * ANGSTROM_TO_BOHR
        for step_idx in range(steps + 1):
            if step_idx % max(1, report_every) == 0:
                time_fs = step_idx * dt
                cur_pos_bohr = pos_ang * ANGSTROM_TO_BOHR
                delta_bohr = np.zeros_like(cur_pos_bohr) if len(frames) == 0 else (cur_pos_bohr - prev_pos_bohr)
                prev_pos_bohr = cur_pos_bohr.copy()
                fr = StructureFrame(
                    Z=z_arr.copy(),
                    pos_bohr=cur_pos_bohr,
                    cell_bohr=None if cell_ang is None else cell_ang * ANGSTROM_TO_BOHR,
                    pbc=None if pbc_arr is None else pbc_arr.copy(),
                    delta_pos_bohr=delta_bohr,
                    forces_ev_ang=forces.copy(),
                    label=f"{init_frame.label}_step{step_idx}",
                    metadata={"energy_ev": e_ev, "time_fs": time_fs, "step": step_idx},
                )
                frames.append(fr)
                energies_ev.append(e_ev)
                monitor.step(
                    frame_index=len(frames) - 1,
                    time_fs=time_fs,
                    Z=z_arr,
                    pos_angstrom=pos_ang,
                    forces_ev_ang=forces,
                    cell_angstrom=cell_ang,
                    pbc=pbc_arr,
                )

            if step_idx == steps:
                break

            # Velocity-Verlet half-kick + drift
            vel_half = vel + 0.5 * dt * forces * inv_m
            pos_ang = pos_ang + dt * vel_half
            if cell_ang is not None and (pbc_arr is None or np.any(pbc_arr)):
                pinv = np.linalg.pinv(cell_ang)
                frac = pos_ang @ pinv
                mask_axes = np.ones(3, dtype=bool) if pbc_arr is None else pbc_arr
                frac[:, mask_axes] = frac[:, mask_axes] % 1.0
                pos_ang = frac @ cell_ang

            e_ev, forces = self.force_provider(z_arr, pos_ang, cell_ang, pbc_arr)
            vel = vel_half + 0.5 * dt * forces * inv_m

            # Optional Ornstein-Uhlenbeck Langevin thermostat step
            if gamma > 0.0 and self.temperature_k > 0.0:
                vel = c1 * vel + noise_scale * rng.normal(0.0, 1.0, size=vel.shape)

        extxyz_str = str(write_extxyz_trajectory(frames, energies_ev, out_extxyz)) if out_extxyz else None
        events_json_str = None
        plot_str = str(out_plot) if out_plot else None
        if out_events_json or out_plot:
            target_json = out_events_json or Path(str(out_plot)).with_suffix(".json")
            monitor.save_summary(target_json, plot_path=out_plot)
            events_json_str = str(target_json)

        html_str = None
        if out_html:
            probe = self.probe or YLMProbe.default()
            scan = scan_md_trajectory(frames, probe=probe, query=query, every=1)
            scan.to_html(out_html)
            html_str = str(out_html)

        return MLIPMDRunResult(
            frames=frames,
            events=monitor.events,
            history=monitor.history,
            energies_ev=energies_ev,
            extxyz_path=extxyz_str,
            events_json_path=events_json_str,
            plot_path=plot_str,
            html_path=html_str,
        )


def build_openmm_mlip_simulation(
    topology,
    positions_nm,
    *,
    potential_name_or_torchscript: str = "mace-off23-small",
    out_events_json: Union[str, Path] = "openmm_reactions.json",
    out_plot: Optional[Union[str, Path]] = "openmm_reactions.svg",
    report_interval: int = 10,
    temperature_k: float = 300.0,
    dt_ps: float = 0.0005,
    probe: Optional[YLMProbe] = None,
):
    """Connect OpenMM (`openmm-ml` or `openmm-torch`) with `OpenMMReactionReporter` in one helper.

    Accepts either an `openmm-ml` potential name (`'mace-off23-small'`, `'ani2x'`) or a path to a
    compiled TorchScript `.pt`/`.pth` potential (`NequIP` / `MACE` via `openmmtorch.TorchForce`).
    """
    try:
        import openmm
        from openmm import app, unit
    except ImportError as exc:
        raise ImportError(
            "OpenMM is not installed in this Python environment. "
            "Install `conda install -c conda-forge openmm openmm-torch openmm-ml` on your GPU workstation, "
            "or use `MLIPTrajectoryRunner` (`ylm run-md`) for pure-PyTorch reactive MD."
        ) from exc

    spec_path = Path(potential_name_or_torchscript)
    if spec_path.exists():
        try:
            from openmmtorch import TorchForce  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"Loading TorchScript MLIP {spec_path} into OpenMM requires `openmm-torch`."
            ) from exc
        system = openmm.System()
        for atom in topology.atoms():
            system.addParticle(atom.element.mass)
        system.addForce(TorchForce(str(spec_path)))
    else:
        try:
            from openmmml import MLPotential  # type: ignore
        except ImportError as exc:
            raise ImportError(
                f"Loading ML potential {potential_name_or_torchscript!r} into OpenMM requires `openmm-ml`."
            ) from exc
        potential = MLPotential(potential_name_or_torchscript)
        system = potential.createSystem(topology)

    integrator = openmm.LangevinMiddleIntegrator(
        temperature_k * unit.kelvin,
        1.0 / unit.picosecond,
        dt_ps * unit.picoseconds,
    )
    sim = app.Simulation(topology, system, integrator)
    sim.context.setPositions(positions_nm)
    reporter = OpenMMReactionReporter(
        out_events_json,
        reportInterval=report_interval,
        plot_path=out_plot,
        probe=probe,
    )
    sim.reporters.append(reporter)
    return sim, reporter
