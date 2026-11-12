# Drug Design, 3D Growth Directions, and Molecule Language Models (40 Papers, 2020–2026)

This note summarizes 40 papers behind YLM's drug-pocket tool (`ylm drug` in `src/toolkit/domains.py`).

## Quick Guide to Terms
- **Plain number (`0e`)**: A single value that does not change when you rotate a molecule (like binding energy or distance).
- **3D arrow (`1o`)**: A direction in space $(x, y, z)$ that turns when the molecule turns (like a dipole, force, or safe direction to attach a new chemical group).
- **Flat ring plane (`2e`)**: A 3x3 symmetric matrix that describes how a flat ring (like benzene) sits in 3D without caring which side of the ring is "up" or "down".

---

## 1. 40-Paper Summary Table

### A. 3D Pocket Docking and Molecule Generation (#1–#11)

| # | Paper | What it uses | What it does & what is missing |
| :-: | :--- | :--- | :--- |
| 1 | **Pocket2Mol** (Peng et al., *ICML* 2022) | Numbers + 3D arrows | Grows a molecule atom by atom inside a protein pocket; cannot take plain-English questions. |
| 2 | **EquiBind** (Stärk et al., *ICML* 2022) | Numbers + 3D coords | Predicts where a drug sits in a protein in one step, but often pushes atoms too close together. |
| 3 | **TANKBind** (Lu et al., *NeurIPS* 2022) | Atom-to-atom distances | Predicts protein-drug distances; distance tables cannot tell left-handed from right-handed shapes. |
| 4 | **TargetDiff** (Guan et al., *ICLR* 2023) | Atom types + 3D coords | Denoises random 3D points into a drug inside a pocket; sometimes bends flat rings. |
| 5 | **DiffDock** (Corso et al., *ICLR* 2023) | 3D spherical features | Predicts docking poses by shifting, spinning, and twisting bonds; does not answer text questions. |
| 6 | **DecompDiff** (Guan et al., *ICML* 2023) | 3D coords + sub-pockets | Splits a drug into a central core and side arms reaching into pocket corners; no text interface. |
| 7 | **DiffSBDD** (Schneuing et al., *Nat. Comput. Sci.* 2024) | Atom types + 3D coords | Generates or edits parts of a 3D ligand inside a pocket; needs force-field cleanup afterward. |
| 8 | **MolCRAFT** (Qu et al., *ICML* 2024) | Smooth probs + 3D coords | Reduces atom clashes during 3D generation, but cannot follow text instructions. |
| 9 | **VoxBind** (Pinheiro et al., *ICML* 2024) | 3D pixel grid (voxels) | Uses a 3D grid instead of exact atom coordinates, so rotating the input slightly changes the answer. |
| 10 | **SurfDock** (Cao et al., *Nat. Methods* 2025) | Pocket surface + 3D coords | Uses protein surface shape and charge for cleaner docking; does not rank C-H growth arrows. |
| 11 | **PocketXMol** (Peng et al., *Cell* 2026) | Unified 3D atom features | Handles docking, linker design, and peptides in one model, controlled by masks rather than English text. |

### B. Growing Fragments and Connecting Two Pieces (#12–#21)

| # | Paper | What it uses | What it does & what is missing |
| :-: | :--- | :--- | :--- |
| 12 | **DeLinker** (Imrie et al., *JCIM* 2020) | 2D graph + distance/angle | Connects two fragments using distance and angle numbers, ignoring nearby protein atoms that may collide. |
| 13 | **DEVELOP** (Imrie et al., *Chem. Sci.* 2021) | 2D graph + 3D voxel grid | Adds 3D grid hints for hydrogen bonds and rings, but outputs a 2D graph rather than exact 3D arrows. |
| 14 | **DeepFrag** (Green et al., *Chem. Sci.* 2021) | 3D voxel grid -> list pick | Picks a side group from a fixed library instead of giving a continuous 3D growth arrow. |
| 15 | **3DLinker** (Huang et al., *ICML* 2022) | Numbers + 3D arrows | Connects two 3D fragments without hand-picked anchor atoms, but ignores the surrounding protein pocket. |
| 16 | **FragDiff** (Peng et al., *OpenReview* 2023) | Numbers + 3D arrows | Grows 3D fragments step by step inside a pocket; early mistakes carry forward and there is no text Q&A. |
| 17 | **D3FG** (Lin et al., *NeurIPS* 2023) | Rigid functional groups | Keeps rings flat by treating common groups as rigid blocks; limited to a fixed group list. |
| 18 | **LinkerNet** (Guan et al., *NeurIPS* 2023) | 3D poses + atom coords | Adjusts fragment angles while building a linker (useful for PROTACs); no text reasoning. |
| 19 | **DiffLinker** (Igashov et al., *Nat. Mach. Intell.* 2024) | Atom types + 3D coords | Links two or more fragments inside a pocket; acts as a black box without scoring each C-H arrow. |
| 20 | **SyntheMol** (Swanson et al., *Nat. Mach. Intell.* 2024) | 2D reaction building blocks | Designs easy-to-make antibiotics in 2D, without 3D pocket shape or growth directions. |
| 21 | **TamGen** (Wu et al., *Nat. Commun.* 2024) | Pocket distances -> SMILES | Reads a 3D pocket and writes a 1D SMILES string, losing the 3D pose and arrow directions. |

### C. Hydrogen Bonds, Charge Shapes, and All-Atom Folding (#22–#30)

| # | Paper | What it uses | What it does & what is missing |
| :-: | :--- | :--- | :--- |
| 22 | **ESP-Sim** (Bolcato et al., *JCIM* 2022) | 3D shape + charge grid | Scores how well two pre-aligned molecules match in 3D shape and electric potential. |
| 23 | **E3Bind** (Zhang et al., *ICLR* 2023) | Pair features + 3D coords | Refines flexible ligand coordinates in a pocket without explicit lone-pair or ring directions. |
| 24 | **PGMG** (Zhu et al., *Nat. Commun.* 2023) | Distance graph -> SMILES | Turns feature-point distances into SMILES strings without keeping 3D cone or ring angles. |
| 25 | **IPDiff** (Huang et al., *ICLR* 2024) | 3D interaction network | Guides 3D diffusion with a learned binding prior; does not output interpretable arrows. |
| 26 | **PharmacoNet** (Seo & Kim, *Chem. Sci.* 2024) | 3D voxel hotspots | Finds H-bond, ring, and hydrophobic spots in a pocket for fast screening using distance tolerances. |
| 27 | **RoseTTAFold All-Atom** (Krishna et al., *Science* 2024) | Protein frames + 3D atoms | Folds proteins together with small molecules and metal ions; heavy to run and has no text Q&A. |
| 28 | **Chai-1** (Boitreaud et al., *bioRxiv* 2024) | Sequence + 3D diffusion | Predicts protein-ligand 3D structures using rotation data augmentation rather than exact rotation symmetry. |
| 29 | **Boltz-1 / Boltz-2** (Wohlwend / Passaro et al., 2024–2025) | Open all-atom 3D + energy | Open-source protein-ligand structure and binding-strength predictor; outputs one number, not site arrows. |
| 30 | **ShEPhERD** (Adams et al., *ICLR* 2025) | `e3nn` 3D shapes + arrows | Diffuses 3D shape, electric charge, and directional pharmacophore arrows, but has no language interface. |

### D. Molecule + Text Language Models (#31–#40)

| # | Paper | What it uses | What it does & why 3D arrows are lost |
| :-: | :--- | :--- | :--- |
| 31 | **MoleculeSTM** (Liu et al., *Nat. Mach. Intell.* 2023) | 2D graph + text | Matches 2D molecules to text descriptions; has no 3D coordinates at all. |
| 32 | **DrugChat** (Liang et al., *arXiv* 2023) | 2D graph -> 1 vector -> LLM | Compresses a whole 2D molecule into one summary vector for chat; no 3D atoms or directions. |
| 33 | **MolCA** (Liu et al., *EMNLP* 2023) | 2D graph + SMILES -> LLM | Uses a Q-Former bridge on 2D graphs; outputs text only, no 3D geometry. |
| 34 | **Mol-Instructions** (Fang et al., *ICLR* 2024) | 1D text strings | Large text dataset of SELFIES and protein sequences; no 3D coordinates. |
| 35 | **3D-MoLM** (Li et al., *ICLR* 2024) | Uni-Mol distances -> LLM | Feeds atom-distance features into Llama-2. Because distances do not change when a molecule spins, 3D arrow directions are completely erased. |
| 36 | **nach0** (Livne et al., *Chem. Sci.* 2024) | SMILES + patent text | Trains on papers and reaction strings; does not take 3D protein-ligand coordinates. |
| 37 | **BioT5+** (Pei et al., *ACL Findings* 2024) | SELFIES + chemical names | Improves text and property Q&A from 1D strings without 3D coordinates. |
| 38 | **Lingo3DMol** (Feng et al., *Nat. Mach. Intell.* 2024) | Rounded grid coordinates | Writes 3D coordinates on a discrete grid as tokens, so smooth 3D rotations break token consistency. |
| 39 | **BindGPT** (Zholus et al., *AAAI* 2025) | XYZ numbers as text tokens | Prints `(x, y, z)` at `0.1 Å` resolution as text; rotating the pocket changes every token. |
| 40 | **3D-MolT5** (Pei et al., *ICLR* 2025) | Local 3D hash IDs + text | Turns local atom neighborhoods into rotation-invariant IDs, throwing away global 3D directions. |

---

## 2. Plain-English Takeaways for YLM (`ylm drug`)

1. **Why earlier models miss the middle ground**:
   - Models #1–#30 keep 3D geometry right, but you cannot ask them a question in English.
   - Models #31–#40 speak English, but either compress the molecule into rotation-blind distance numbers (`3D-MoLM`) or print rounded `(x, y, z)` text strings that break when rotated (`BindGPT`).
   - **YLM** keeps 3D arrows alive while reading English words, and outputs exact `<vec>` 3D vectors that rotate with the molecule.

2. **What `src/toolkit/domains.py` (`analyze_drug_geometry`) checks**:
   - **Flat aromatic rings**: Measured with a 3x3 symmetric plane matrix so flipping "top" and "bottom" of a benzene ring never flips a random sign.
   - **Hydrogen-bond donors and acceptors**: Finds both `N-H` / `O-H` donors and lone-pair acceptors (`=O`, `:N`, `-O-`) and marks H-bonds already locked to the protein pocket so we do not break them.
   - **Safe 3D growth arrows (exit vectors)**: Steps along each candidate `C-H` (or open heavy-atom) ray from `1.5 Å` to `4.5 Å` to filter out protein wall crashes and rank directions that point cleanly into open sub-pockets.
   - **Two-piece linker geometry**: When two separate fragments sit in a pocket, measures the gap distance and angles needed to bridge them.
