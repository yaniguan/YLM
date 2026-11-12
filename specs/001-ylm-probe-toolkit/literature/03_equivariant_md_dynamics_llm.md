# Molecular Dynamics, Reaction Paths, Proton Hops, and MD Language Models (40 Papers, 2020–2026)

This note summarizes 40 papers behind YLM's trajectory scanner (`ylm scan-traj` in `src/toolkit/domains.py`) and live MD hooks (`src/toolkit/md_hooks.py`).

## Quick Guide to Terms
- **Plain number (`0e`)**: A single value that stays the same when you rotate or mirror a molecule (like potential energy $E$, bond length, or mechanical work $W_i = \mathbf{F}_i \cdot \Delta\mathbf{r}_i$).
- **3D arrow (`1o`)**: A vector $(x, y, z)$ that rotates with the molecule and flips sign in a mirror (like force $\mathbf{F}_i$, velocity $\mathbf{v}_i$, or step-to-step atom movement $\Delta\mathbf{r}_i$).
- **Spin / torque axis (`1e`)**: A rotation axis $\mathbf{F}_i \times \Delta\mathbf{r}_i$ (cross product) that rotates with the molecule and does **not** flip sign in a mirror.
- **Proton-sharing number ($\delta$)**: For a hydrogen $\text{H}^*$ sitting between donor oxygen $\text{O}_1$ and acceptor oxygen $\text{O}_2$, $\delta = d(\text{O}_1\text{-H}^*) - d(\text{O}_2\text{-H}^*)$. When $\delta < 0$, $\text{H}^*$ belongs to $\text{O}_1$; when $\delta \approx 0$, both oxygens share it equally; when $\delta > 0$, it has hopped to $\text{O}_2$.

---

## 1. 40-Paper Summary Table

### A. 3D Force-Field Networks (#1–#10)

| # | Paper | What it uses | What it does & what is missing |
| :-: | :--- | :--- | :--- |
| 1 | **PaiNN** (Schütt et al., *ICML* 2021) | Numbers + 3D arrows (`1o`) | Fast neural force field that also predicts dipoles; looks at one frame at a time with no text output. |
| 2 | **SpookyNet** (Unke et al., *Nat. Commun.* 2021) | 3D coords + total charge & spin | Handles charged ions and radicals well, but only scores static snapshots with no MD event text. |
| 3 | **NequIP** (Batzner et al., *Nat. Commun.* 2022) | 3D spherical features ($l \le 3$) | Accurate, data-efficient 3D force field; collapses all rich 3D features into one energy number at the end. |
| 4 | **TorchMD-NET** (Thölke & De Fabritiis, *ICLR* 2022) | Numbers + 3D arrows (`1o`) | Attention-based 3D potential for fast MD; sees single frames only and cannot answer English questions. |
| 5 | **MACE** (Batatia et al., *NeurIPS* 2022) | 4-body 3D spherical features | Captures multi-atom angles in just 2 message-passing steps; outputs energy and forces silently without text. |
| 6 | **SO3krates** (Frank et al., *NeurIPS* 2022) | Distances + spherical angles | Separates distance and angle branches for long-range forces; single-frame only, no language readout. |
| 7 | **Allegro** (Musaelian et al., *Nat. Commun.* 2023) | Local atom-pair 3D features | Drops multi-hop message passing so LAMMPS can scale to 100M atoms; cannot explain reactions in words. |
| 8 | **Equiformer** (Liao & Smidt, *ICLR* 2023) | 3D spherical graph attention | Strong 3D transformer for energies and forces on static structures; no frame-to-frame motion or text. |
| 9 | **eSCN** (Passaro & Zitnick, *ICML* 2023) | Bond-aligned 3D rotations | Speeds up high-angle 3D math by aligning along each bond; still a silent single-frame force calculator. |
| 10 | **ViSNet** (Wang et al., *Nat. Commun.* 2024) | Angles, torsions + 3D arrows | Tracks bond angles and twists explicitly for biomolecule MD; needs outside scripts to explain events. |

### B. Motion-Aware Networks and Big-Step MD Generators (#11–#20)

| # | Paper | What it uses | What it does & what is missing |
| :-: | :--- | :--- | :--- |
| 11 | **EGNN** (Satorras et al., *ICML* 2021) | Positions + velocity arrows (`1o`) | Updates atom positions and velocities cleanly, but lacks higher-angle features and text output. |
| 12 | **SEGNN** (Brandstetter et al., *ICLR* 2022) | Positions + velocity & force (`1o`) | Shows how to feed velocity and force arrows into spherical layers; outputs numbers/coords only, no words. |
| 13 | **GMN** (Huang et al., *ICLR* 2022) | Positions + velocities + rigid rods | Keeps rigid bonds fixed during motion, which means bonds can never break or form in a reaction. |
| 14 | **ProtMD** (Wu et al., *Adv. Sci.* 2022) | Protein MD frames + step shifts | Pretrains on protein MD frames to predict drug binding strength; does not describe MD events in text. |
| 15 | **DiffMD** (Wu & Li, *AAAI* 2023) | Past frames + velocity arrows | Generates the next MD frame by denoising; skips exact force-times-displacement work and text output. |
| 16 | **Timewarp** (Klein et al., *NeurIPS* 2023) | 3D coords -> big time jumps | Jumps forward by $10^5\text{--}10^6\text{ fs}$ in one step, skipping right over fast bond-breaking moments. |
| 17 | **ITO** (Schreiner et al., *NeurIPS* 2023) | 3D coords + lag time $N\Delta t$ | Samples future shapes at chosen time lags; smooths out fast barrier crossings and has no text Q&A. |
| 18 | **GeoTDM** (Han et al., *NeurIPS* 2024) | Multi-frame 3D path | Diffuses whole 3D trajectories at once; does not break down per-atom force work or explain reactions. |
| 19 | **MDGen** (Jing et al., *arXiv* 2024) | Protein torsion-angle tokens | Treats a protein trajectory like a video of bond twists; fixed bond graphs cannot break bonds or hop protons. |
| 20 | **EquiJump** (Costa et al., *arXiv* 2024) | 3D coords + local side-chain arrows | Takes big time steps for protein folding; skips sub-picosecond chemistry and has no language readout. |

### C. Chemical Reactions, Transition States, and Proton Hops (#21–#30)

| # | Paper | What it uses | What it does & what is missing |
| :-: | :--- | :--- | :--- |
| 21 | **Water–TiO2 Proton Transfer** (Calegari Andrade et al., *Chem. Sci.* 2020) | Proton-sharing $\delta$ + O-H count | Uses $\delta = d_1 - d_2$ to map proton hops on oxide surfaces; relies on hand-written analysis scripts. |
| 22 | **Deep-TICA** (Bonati et al., *PNAS* 2021) | Atom-pair distances (`0e`) | Learns slow reaction numbers to speed up rare-event sampling; black-box numbers lose 3D arrow directions. |
| 23 | **Transition1x** (Schreiner et al., *Sci. Data* 2022) | 9.6M reaction-path 3D frames | Huge DFT energy/force dataset across 10k organic reactions; static reaction paths with no thermal velocities or text. |
| 24 | **Machine-Guided Path Sampling** (Jung et al., *Nat. Comput. Sci.* 2023) | Internal distances (`0e`) | Predicts the chance ($p_B \approx 0.5$) that a frame crosses a barrier; outputs one probability, not plain English. |
| 25 | **mlcolvar** (Bonati et al., *J. Chem. Phys.* 2023) | Distances + smooth coordination | Open library for slow reaction variables in PLUMED; uses scalar features only, without velocity arrows or text. |
| 26 | **OA-ReactDiff** (Duan et al., *Nat. Comput. Sci.* 2023) | Reactant & product 3D coords | Generates 3D transition-state structures in seconds; static endpoints-to-saddle tool, not a live MD watcher. |
| 27 | **ZundEig** (Di Pino et al., *J. Phys. Chem. B* 2023) | Local shapes + proton $\delta$ | Maps how excess protons in acid water shift between Eigen ($|\delta|>0.3\text{ \AA}$) and Zundel ($\delta\approx 0$) states. |
| 28 | **EquiReact / 3DReact** (van Gerwen et al., *JCIM* 2024) | Reactant & product 3D tensors | Predicts reaction barrier height $\Delta E^\ddagger$ from start and end structures; does not watch live MD frames. |
| 29 | **Doubly Gated Water Proton Hop** (Gomez et al., *Nat. Chem.* 2024) | Proton $\delta$ + H-bond switches | Proves real proton hops in water need H-bond switches at both ends, separating real hops from back-and-forth rattling. |
| 30 | **React-OT** (Duan et al., *Nat. Mach. Intell.* 2025) | Reactant-to-product 3D flow | Fast deterministic transition-state finder; solves static gas-phase saddles rather than live heated MD trajectories. |

### D. Language Models and Agents for Chemistry & MD (#31–#40)

| # | Paper | What it uses | What it does & why 3D dynamics are missed |
| :-: | :--- | :--- | :--- |
| 31 | **MolCA** (Liu et al., *EMNLP* 2023) | 2D bond graph + text | Answers questions about 2D molecule graphs; has zero 3D coordinates, forces, or MD frames. |
| 32 | **Coscientist** (Boiko et al., *Nature* 2023) | Text docs + SMILES + Python | Plans lab syntheses by running Python scripts and searching docs; never sees 3D atom coordinates or forces. |
| 33 | **ChemCrow** (Bran et al., *Nat. Mach. Intell.* 2024) | SMILES + 18 RDKit/chem tools | Great tool-calling chemistry assistant for 1D strings; has no 3D structure or MD trajectory input. |
| 34 | **3D-MoLM** (Li et al., *ICLR* 2024) | Single 3D frame distances -> LLM | Reads one static 3D molecule through rotation-blind distances; cannot read forces, motion $\Delta\mathbf{r}$, or MD hops. |
| 35 | **ChatMol Copilot** (Sun et al., *ACL* 2024) | Text + PyMOL / docking CLI | Runs PyMOL and docking commands from chat; the LLM only reads text logs, never 3D force or motion vectors. |
| 36 | **AtomAgents** (Ghafarollahi & Buehler, *PNAS* 2025) | LAMMPS scripts + 2D plot images | Multi-agent system that writes LAMMPS files and reads 2D summary plots; blind to per-atom 3D forces and motion. |
| 37 | **MDCrow** (Campbell et al., *arXiv* 2025) | OpenMM + MDTraj scripts | Runs standard MD scripts (RMSD, radius of gyration); misses any reaction event not already hard-coded in a script. |
| 38 | **EquiLLM** (Li et al., *ICML* 2025) | 3D coords -> LLM -> 3D coords | Uses an LLM inside a 3D coordinate predictor; outputs future `(x, y, z)` numbers, **not** English explanations. |
| 39 | **El Agente** (Zou et al., *Matter* 2025) | ORCA quantum log text files | Automates quantum chemistry jobs by reading text log files; does not watch live 3D MD trajectories. |
| 40 | **MDAgent2** (Shi et al., *arXiv* 2026) | Simulation specs + Q&A text | Writes LAMMPS/GROMACS input files and answers textbook questions; cannot inspect live 3D atom forces or hops. |

---

## 2. Plain-English Takeaways for YLM (`ylm scan-traj` & MD Hooks)

1. **Why force fields and LLM script agents both miss part of the picture**:
   - Force fields like `NequIP` and `MACE` (#1–#10) know every 3D angle inside out, yet collapse everything to one silent energy number.
   - LLM agents like `MDCrow` and `AtomAgents` (#31–#40) speak English, yet only read summary text logs written by outside scripts.
   - **YLM** feeds both 3D atom positions and step-to-step motion arrows ($\Delta\mathbf{r}_i$) straight into a 3D-to-text model (`use_motion_1o=True`).

2. **How YLM separates real chemistry from random heat jiggling**:
   - **Work along the step ($W_i = \mathbf{F}_i \cdot \Delta\mathbf{r}_i$)**: Dot-multiplying force and movement gives a rotation-proof number showing whether forces are **pushing an atom forward** ($W_i > 0$, downhill release) or **pulling it back** ($W_i < 0$, climbing an energy barrier).
   - **Real proton hops vs. back-and-forth rattling**: Using the proton-sharing number $\delta = d(\text{D-H}^*) - d(\text{A-H}^*)$, YLM checks whether $\delta$ actually crosses zero and stays on the new atom (`committed_grotthuss_transfer`) or bounces right back (`frustrated_rattling`).
   - **Live OpenMM / LAMMPS hooks (`src/toolkit/md_hooks.py`)**: Plugs directly into running MD simulations to log bond-forming, bond-breaking, and proton-transfer events automatically on the fly.
