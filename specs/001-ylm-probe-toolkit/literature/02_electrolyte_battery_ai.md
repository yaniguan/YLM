# Battery Electrolytes, Ion Shells, Crystals, and Materials LLMs (40 Papers, 2020–2026)

This note summarizes 40 papers behind YLM's electrolyte and crystal tool (`ylm solvation` in `src/toolkit/domains.py`).

## Quick Guide to Terms
- **Solvation shell**: The tight ring of solvent molecules and salt anions surrounding a `Li+` or `Na+` ion.
- **SSIP / CIP / AGG**:
  - **SSIP**: `Li+` touches only solvent molecules (no salt anions nearby).
  - **CIP**: `Li+` touches one salt anion that is not shared with another `Li+`.
  - **AGG**: A salt anion bridges two or more `Li+` ions at once.
- **Escape arrow (`1o`)**: The 3D direction where the solvent cage has the widest opening so `Li+` can slip out most easily.
- **Crystal repeat box (PBC)**: A 3D unit cell that repeats endlessly in X, Y, and Z (like bricks in a wall).

---

## 1. 40-Paper Summary Table

### A. Liquid Electrolytes and Ion Solvation Shells (#1–#11)

| # | Paper | System | What it does & what is missing |
| :-: | :--- | :--- | :--- |
| 1 | **LIBE** (Spotte-Smith et al., *Sci. Data* 2021) | 17k ion/solvent clusters | Quantum energies of isolated battery fragments, without bulk liquid cage competition. |
| 2 | **Carbonate Solvation** (Hou, Persson et al., *Chem. Sci.* 2021) | Liquid carbonate MD | Links SSIP/CIP/AGG ratios to ion diffusion, but requires long MD runs first. |
| 3 | **Ultra-Low-Temp Li** (Holoubek, Ping Liu et al., *Nat. Energy* 2021) | Low-temp Li shells | Shows strong solvent dipoles lock `Li+` tightly in SSIP cages, while CIP/AGG cages let `Li+` escape faster. |
| 4 | **RedDB** (Sorkun, Er et al., *Sci. Data* 2022) | 31k redox molecules | Single-molecule redox potentials and solubility numbers; no 3D ion shells. |
| 5 | **Li Metal Electrolyte ML** (Kim, Cui et al., *PNAS* 2023) | Liquid formulations | Finds lower oxygen content boosts battery efficiency, using element ratios without 3D shapes. |
| 6 | **ESP Solvent Map** (Wu, He et al., *Adv. Energy Mater.* 2023) | Single solvents | Uses min/max surface charge numbers `(ESP_min, ESP_max)`, dropping full 3D cage angles. |
| 7 | **SolvationAnalysis** (Cohen, Persson et al., *JOSS* 2023) | MD trajectories | Standard Python definitions for coordination number and SSIP/CIP/AGG, after running full MD. |
| 8 | **CALiSol-23** (de Blasio et al., *Sci. Data* 2024) | 13.8k salt+solvent mixes | Experimental conductivity numbers across temperatures; no 3D atom coordinates. |
| 9 | **BAMBOO** (Gong et al., *arXiv* 2024) | Liquid electrolyte MD | Fast 3D force field for liquid mixtures, still needing nanosecond MD runs to extract properties. |
| 10 | **OMol25** (Levine et al., *arXiv* 2025) | ~100M DFT clusters | Huge DFT dataset including solvated metal ions; predicts energies/forces without text Q&A. |
| 11 | **OEDB** (Nakamura et al., *npj Comput. Mater.* 2026) | Automated MD database | Diffusion numbers and Raman spectra from MD, without 3D cage escape arrows. |

### B. Solid Crystals and Ion Hopping Networks (#12–#21)

| # | Paper | System | What it does & what is missing |
| :-: | :--- | :--- | :--- |
| 12 | **Charge-Density Insertion** (Shen, Persson et al., *npj Comput. Mater.* 2020) | Cathode crystals | Finds empty spots for `Li+` inside DFT charge grids; outputs static sites and voltages. |
| 13 | **CAVD / BVSE** (He, Shi et al., *Sci. Data* 2020) | 29k solid electrolytes | Fast geometric tunnel scan on rigid crystals, ignoring atom relaxation. |
| 14 | **M3GNet** (Chen & Ong, *Nat. Comput. Sci.* 2022) | Periodic crystals | Universal crystal force field built from distances and 3-body bond angles (plain numbers only). |
| 15 | **CHGNet** (Deng, Ceder et al., *Nat. Mach. Intell.* 2023) | Battery crystals | Tracks metal charge/spin states (`Fe2+/Fe3+`) on crystals; still needs NEB/MD for hopping paths. |
| 16 | **MACE-MP-0** (Batatia, Csányi et al., *arXiv* 2024 / *JCP* 2025) | 89-element crystals & liquids | Pretrained 3D equivariant force field across the periodic table; plugged into YLM via `MACEEncoder`. |
| 17 | **EquiformerV2** (Liao, Smidt et al., *ICLR* 2024) | Surfaces & crystals | Fast high-order 3D spherical attention for energies and forces; no language interface. |
| 18 | **SevenNet** (Park, Han et al., *JCTC* 2024) | Multi-GPU crystal MD | Scales NequIP-style 3D force fields across many GPUs in LAMMPS. |
| 19 | **MatterSim** (Yang et al., *arXiv* 2024) | Crystals & melts | Trained across wide temperatures and pressures for energies, forces, and stress. |
| 20 | **Orb** (Neumann et al., *arXiv* 2024) | Crystal screening | Very fast force field trained with rotation augmentation rather than exact rotation symmetry. |
| 21 | **LiTraj** (Dembitskiy et al., *npj Comput. Mater.* 2025) | 122k `Li+` hop barriers | Benchmarks `Li+` migration barrier heights ($E_a$), predicting one scalar number rather than a 3D hop vector. |

### C. Materials and Battery Language Models (#22–#31)

| # | Paper | System | What it does & why 3D arrows are lost |
| :-: | :--- | :--- | :--- |
| 22 | **MatSciBERT** (Gupta et al., *npj Comput. Mater.* 2022) | 150k materials papers | Text encoder for materials words (used as one of YLM's frozen text encoders); reads text only. |
| 23 | **BatteryBERT** (Huang & Cole, *JCIM* 2022) | 400k battery papers | Extracts battery facts from papers; reads text only, no 3D structures. |
| 24 | **CrysMMNet** (Das et al., *UAI* 2023) | Crystal graph + text | Combines distance/angle graphs with text to predict scalar band gaps; no 3D vectors. |
| 25 | **DARWIN** (Xie et al., *arXiv* 2023) | Science Q&A text | Fine-tuned LLMs over chemical formulas and SMILES strings; no 3D coordinates. |
| 26 | **LLM-Prop** (Rubungo et al., *arXiv* 2023 / *npj* 2025) | Text descriptions of crystals | Converts bond lengths/angles into English paragraphs to predict scalar band gaps. |
| 27 | **Crystal-Text-LLM** (Gruver et al., *ICLR* 2024) | Coordinates as text | Fine-tunes Llama-2 to print crystal coordinates as decimals, losing exact 3D rotation rules. |
| 28 | **CrystaLLM** (Antunes et al., *Nat. Commun.* 2024) | CIF files as text | Trains a GPT model on raw `.cif` text files; has no internal 3D vector math. |
| 29 | **HoneyComb** (Zhang et al., *EMNLP Findings* 2024) | Materials tool agent | Calls external Python scripts and knowledge bases; the LLM itself cannot see 3D geometry. |
| 30 | **MatExpert** (Ding et al., *arXiv* 2024) | Multi-step CIF agent | Generates and checks `.cif` text strings step by step without 3D equivariant layers. |
| 31 | **ChatBattery** (Liu, Bengio, Guo et al., *arXiv* 2025) | Cathode formulas | Suggests new cathode chemical formulas tested in wet labs; works on element counts, not 3D cages. |

### D. Multi-Component Liquid Mixtures (#32–#40)

| # | Paper | System | What it does & what is missing |
| :-: | :--- | :--- | :--- |
| 32 | **Clio Robot Electrolyte** (Dave, Viswanathan et al., *Nat. Commun.* 2022) | Liquid mixes | Robot lab + Bayesian optimization over fixed solvent ratios; no 3D molecular geometry. |
| 33 | **Polymer Electrolyte MPNN** (Bradford, Shao-Horn et al., *ACS Cent. Sci.* 2023) | Polymer + salt | 2D molecular graphs + Arrhenius temperature law for conductivity; no 3D cage geometry. |
| 34 | **SolvBERT** (Yu et al., *Digit. Discov.* 2023) | Solute + solvent SMILES | Joins SMILES strings with separators, so swapping list order changes the input. |
| 35 | **DiffMix** (Zhu, Viswanathan et al., *Nat. Commun.* 2024) | Liquid mixes | Order-invariant 2D graph pooling + physics equations for conductivity; scalar outputs only. |
| 36 | **Thermo-Consistent GNN** (Rittig & Mitsos, *Chem. Sci.* 2024) | Liquid mixtures | Enforces Gibbs–Duhem rules on scalar activity numbers using 2D graphs. |
| 37 | **MolSets** (Zhang, Chen et al., *PRX Energy* 2024) | Salt + solvent sets | Order-invariant set pooling with smooth zero-concentration limits on 2D graphs. |
| 38 | **Uni-ELF** (Zeng, E, Zhang et al., *arXiv* 2024) | Liquid electrolytes | Pretrains single-molecule `Uni-Mol` distances against 1D radial curves $g(r)$, averaging away 3D cage directions. |
| 39 | **Chem-FM Electrolyte** (Zohair, IBM et al., *npj Comput. Mater.* 2025) | SMILES mixes | Uses 1D SMILES embeddings weighted by recipe ratios to find high-conductivity mixes. |
| 40 | **Unified Electrolyte Gen** (Yang et al., *Nat. Mach. Intell.* 2025/2026) | Recipe prediction & generation | Predicts properties and generates solvent recipes in scalar space without 3D cage vectors. |

---

## 2. Plain-English Takeaways for YLM (`ylm solvation`)

1. **Why 3D directions matter in batteries**:
   - When `Li+` is surrounded evenly on all sides by strong solvent dipoles (symmetric SSIP), every pull cancels out and `Li+` is trapped tightly.
   - When an anion or weak solvent enters one side (asymmetric CIP/AGG), it opens a **3D escape gap** on the weakest side. Averaging everything into a 1D curve $g(r)$ erases that opening!

2. **What `src/toolkit/domains.py` (`analyze_electrolyte_geometry`) computes**:
   - **Smooth neighbor count & SSIP / CIP / AGG tag**: Counts coordinating atoms smoothly around `Li+` or `Na+` (including across 3D crystal box walls) and classifies the shell as SSIP, CIP, or AGG.
   - **Dipole lock angle (`cos theta`)**: Checks how strongly each solvent molecule points its negative end straight at the ion.
   - **3D escape arrow (`1o`)**: Points toward the least crowded opening in the solvent cage so you can see where the ion breaks free.
   - **Symmetric crystal tunnel axis (`2e`)**: In symmetric crystals where opposite pulls cancel any single arrow to `0`, a 3x3 stretch matrix still finds the open two-way tunnel axis.
