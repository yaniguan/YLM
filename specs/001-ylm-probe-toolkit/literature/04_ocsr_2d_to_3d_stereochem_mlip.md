# Reading 2D Molecule Drawings into 3D Handedness and Force Fields (38 Papers, 2016–2025)

This note summarizes 38 papers behind the `Ouroboros-OCSR` + `YLM` bridge (`src/toolkit/ouroboros_bridge.py`, `src/mace_backbone.py`, and `src/nequip_backbone.py`).

## Quick Guide to Terms
- **2D spin without mirror flip (`SE(2)` / `C8`)**: Rotating a flat chemical drawing on paper by $45^\circ$ or $90^\circ$ keeps the same molecule. Flipping the paper in a mirror is **not** allowed because solid wedges (pointing at you) and dashed wedges (pointing away) trade places, turning a right-handed $(R)$ drug into its left-handed $(S)$ twin!
- **Plain number (`0e`)**: Stays the same under both 3D rotation and mirror flip (like atom-pair distance $d_{ij}$ or total energy $E$).
- **Handedness number (`0o`)**: Stays the same when you rotate a 3D molecule in your hands, **but flips sign (`+` to `-`) in a mirror**—just like a left glove vs. a right glove.
- **3D arrow (`1o`)**: A vector $(x, y, z)$ that rotates with the molecule (like electric dipole $\boldsymbol{\mu}$ or atom force $\mathbf{F}_i$).

---

## 1. 38-Paper Summary Table

### A. Reading 2D Chemical Drawings into Bond Graphs (#1–#10)

| # | Paper | What it uses | What it does & what is missing |
| :-: | :--- | :--- | :--- |
| 1 | **OCSR Tools Review** (Rajan et al., *J. Cheminform.* 2020) | Rule-based parsers (`OSRA`, `MolVec`) | Benchmarks classic rule-based drawing readers and shows where noisy scans break wedge bonds. |
| 2 | **ChemGrapher** (Oldenhof et al., *JCIM* 2020) | Pixel segmentation CNN | Labels atoms, bonds, and charges pixel by pixel to rebuild a 2D graph; not rotation-proof on paper. |
| 3 | **Img2Mol** (Clevert et al., *Chem. Sci.* 2021) | CNN -> latent vector -> SMILES | Handles noisy drawing styles well, but loses exact 2D atom positions and rotation symmetry. |
| 4 | **SwinOCSR** (Xu et al., *J. Cheminform.* 2022) | Swin Transformer -> SELFIES | Reads upright drawings well, but needs heavy $0^\circ\text{--}360^\circ$ spin training because pixel windows are axis-locked. |
| 5 | **MolScribe** (Qian et al., *JCIM* 2023) | Swin + 2D atom coords + bond types | Predicts 2D atom spots and solid/dashed wedges so RDKit can recover $(R)/(S)$ handedness accurately. |
| 6 | **RxnScribe** (Qian et al., *JCIM* 2023) | Box detector for full reaction schemes | Finds reactants, arrows, conditions, and products in multi-step paper figures; 2D only. |
| 7 | **DECIMER.ai** (Rajan et al., *Nat. Commun.* 2023) | ViT -> SELFIES trained on 400M images | Brute-forces rotation invariance with hundreds of millions of rotated training images. |
| 8 | **MolGrapher** (Morin et al., *ICCV* 2023) | Atom keypoint finder + GNN | Reads patent drawings and shorthand labels (`Boc`, `Ph`, `Me`); not built with rotation-locked filters. |
| 9 | **MolNexTR** (Chen et al., *J. Cheminform.* 2024) | ConvNeXt strokes + ViT context | Combines local pen-stroke filters with full-image attention and expands shorthand chemical groups. |
| 10 | **MarkushGrapher** (Morin et al., *CVPR* 2025) | Image + text table parser | Reads patent templates with variable `R1`/`R2` placeholders (which must be expanded before 3D physics). |

### B. Rotation-Safe 2D Image Networks (#11–#19)

| # | Paper | What it uses | Why it matters for 2D wedge/hash drawings |
| :-: | :--- | :--- | :--- |
| 11 | **G-CNNs** (Cohen & Welling, *ICML* 2016) | $90^\circ$ rotation filters (`p4` vs `p4m`) | Shows the key split between pure rotations (`p4`, safe for wedges) and mirror flips (`p4m`, flips `R`/`S`). |
| 12 | **Steerable CNNs** (Cohen & Welling, *ICLR* 2017) | Steerable feature channels | General math framework behind both 2D `escnn` (`Ouroboros-OCSR`) and 3D `e3nn` (`YLM`). |
| 13 | **Harmonic Networks** (Worrall et al., *CVPR* 2017) | Circular wave filters $e^{im\theta}$ | Smooth $360^\circ$ 2D rotation symmetry—the 2D cousin of 3D spherical harmonics. |
| 14 | **RotEqNet** (Marcos et al., *ICCV* 2017) | Rotated filters + 2D angle arrows | Tracks thin line directions (like bond strokes and wedge tapers) with very few parameters. |
| 15 | **SE(2) G-CNNs** (Bekkers et al., *MICCAI* 2018) | 2D lift -> group conv -> angle pool | Direct blueprint for `Ouroboros-OCSR`'s 8-angle (`C8`) rotation-safe CNN blocks. |
| 16 | **E2CNN** (Weiler & Cesa, *NeurIPS* 2019) | Exact 2D rotation/mirror filter solver | Solves all 2D rotation groups and shows why `C8` without mirror flips preserves handedness. |
| 17 | **LieConv & EMLP** (Finzi et al., *ICML* 2020/2021) | Continuous group convolutions | Explains tiny pixel-grid rounding differences at $45^\circ$ vs exact $90^\circ$ image turns. |
| 18 | **Functional Non-Linearities** (Poulenard & Guibas, *CVPR* 2021) | Angle-grid activation functions | Connects 2D angle activations with 3D spherical gated layers. |
| 19 | **escnn** (Cesa, Lang & Weiler, *ICLR* 2022) | PyTorch $E(n)$ steerable library | Exact library powering `Ouroboros-OCSR` (`rot2dOnR2(N=8)`), giving $<10^{-6}$ error under $90^\circ$ spins. |

### C. 3D Handedness (`R`/`S`) and 3D Conformer Builders (#20–#28)

| # | Paper | What it uses | What it does & why distance-only models fail |
| :-: | :--- | :--- | :--- |
| 20 | **Tensor Field Networks** (Thomas et al., *arXiv* 2018) | 3D spherical harmonics + tensor product | Direct ancestor of `e3nn` and `YLM`, enabling handedness (`0o`), arrows (`1o`), and shapes (`2e`). |
| 21 | **Structural Incompleteness** (Pozdnyakov et al., *PRL* 2020) | Counterexample 3D atom clouds | Proves mathematically that atom-pair distances alone give the **exact same features** to different/mirror 3D shapes. |
| 22 | **TetraDMPNN** (Pattanaik et al., *arXiv* 2020) | Signed 4-neighbor tetrahedral order | Breaks mirror symmetry at 4-bond carbon centers so 2D message passing can tell $(R)$ from $(S)$. |
| 23 | **ETKDGv3** (Wang et al., *JCIM* 2020) | Signed chiral volume + ring torsion rules | RDKit's 2D-to-3D builder used in `Ouroboros-OCSR` (`enforceChirality=True`) to build valid 3D isomers. |
| 24 | **GeoMol** (Ganea et al., *NeurIPS* 2021) | Local bond angles + signed chiral tag | Shows 3D generators must feed in explicit $(R)/(S)$ signs or mirror twins get mixed up. |
| 25 | **ChIRo** (Adams, Pattanaik & Coley, *ICLR* 2022) | Signed torsion $(\cos\phi, \sin\phi)$ | Uses odd-parity $\sin\phi$ twists so mirror molecules get opposite signs during drug binding scoring. |
| 26 | **Torsional Diffusion** (Jing et al., *NeurIPS* 2022) | Twists bonds on fixed local geometry | Keeps chiral centers locked while diffusing bond rotations so $(R)$ never flips into $(S)$. |
| 27 | **Uni-Mol** (Zhou et al., *ICLR* 2023) | Atom-to-atom distances $d_{ij}$ only | Pretrained on 209M 3D shapes, **but distances never change in a mirror**, so Uni-Mol is 100% blind to $(R)$ vs $(S)$ and 3D arrows! |
| 28 | **MCF** (Wang et al., *ICML* 2024) | Unconstrained 3D coordinate diffusion | Diffuses raw `(x, y, z)` without symmetry locks, showing why a final 3D `stereo_ok` handedness check is essential. |

### D. Universal 3D Force Fields and OpenMM MD Tools (#29–#38)

| # | Paper | What it uses | How YLM connects to it |
| :-: | :--- | :--- | :--- |
| 29 | **NequIP** (Batzner et al., *Nat. Commun.* 2022) | Full `e3nn` even + odd 3D tensors | Loaded directly via `NequIPEncoder` (`src/nequip_backbone.py`) to expose internal 3D features and forces. |
| 30 | **MACE** (Batatia et al., *NeurIPS* 2022) | 4-body `e3nn` spherical contractions | Core architecture wrapped by `MACEEncoder` (`src/mace_backbone.py`); 4-body math naturally sees chiral centers. |
| 31 | **Allegro** (Musaelian et al., *Nat. Commun.* 2023) | Local pair `O(3)` tensors | Supported through `nequip-deploy` TorchScript models (`allegro:*`) for fast MD force extraction. |
| 32 | **MACE-MP-0** (Batatia et al., *J. Chem. Phys.* 2025) | 89-element crystal & materials MACE | Mounted in `YLMUniverseModel` for periodic crystals and battery electrolytes (`[LOT=pbe_u]`). |
| 33 | **OpenMM 8** (Eastman et al., *J. Phys. Chem. B* 2024) | `OpenMM-Torch` & `OpenMM-ML` | Connects PyTorch ML force fields directly to OpenMM MD, bridged by `OpenMMReactionReporter` in `YLM`. |
| 34 | **EquiformerV2** (Liao et al., *ICLR* 2024) | High-order spherical transformer ($l \le 6$) | Shows higher spherical channels (`2e` and above) scale cleanly for universal 3D foundation models. |
| 35 | **SevenNet** (Park et al., *JCTC* 2024) | Multi-GPU parallel NequIP for LAMMPS | Pairs naturally with `LAMMPSReactionHook` (`src/toolkit/md_hooks.py`) for large reactive simulations. |
| 36 | **MatterSim** (Yang et al., *arXiv* 2024) | Multi-temperature/pressure 3D model | Motivates `YLMUniverseModel`'s `[LOT=...]` quantum-method tags (`wb97m`, `r2scan`, `pbe_u`, `ccsd`, `xtb`). |
| 37 | **MACE-OFF23** (Kovács et al., *JACS* 2025) | Organic MACE trained at $\omega\text{B97M}$ DFT | Shared bridge model: labels `Ouroboros-OCSR` 3D conformers with forces and powers `YLM`'s organic backbone. |
| 38 | **eSEN** (Fu et al., *ICML* 2025) | Smooth energy surface + exact gradients | Confirms smooth energies and exact $-\nabla E$ forces are vital for reliable MD physics checks. |

---

## 2. Four Plain-English Takeaways (`Ouroboros-OCSR` + `YLM`)

1. **Why 2D image readers must rotate—but NEVER mirror-flip**:
   - Spinning a textbook drawing on the page keeps the exact same molecule, so `Ouroboros-OCSR` uses 8-angle rotation filters (`C8`).
   - Mirror-flipping a 2D drawing swaps left and right around a wedge bond, accidentally turning $(R)$ into $(S)$—so mirror flips are deliberately left out in 2D!

2. **Why distance-only 3D models (`Uni-Mol`, `SchNet`, `GraphMVP`) are blind to mirror twins**:
   - Look at your left hand and right hand in a mirror: every finger-to-finger distance is **100% identical**, yet a left glove does not fit your right hand.
   - Because distance-only models only look at atom-to-atom distances $d_{ij}$, they give the **exact same numbers** to $(R)$ and $(S)$ mirror molecules, and they output zero 3D direction arrows.

3. **Why total energy alone cannot catch a flipped chiral center**:
   - In empty space, a left-handed molecule $(R)$ and its mirror twin $(S)$ have the **exact same total energy** $E$.
   - So a standard force field that only prints energy $E$ will never warn you if an $(R)$ drug got flipped into $(S)$!

4. **How `Ouroboros-OCSR` + `YLM` keeps every symmetry honest from start to finish**:
   - **Step 1 (`Ouroboros-OCSR`)**: Reads a 2D drawing with rotation-safe `C8` filters -> builds 3D coordinates with `ETKDGv3` -> double-checks 3D $(R)/(S)$ labels (`stereo_ok`).
   - **Step 2 (`YLMUniverseModel` + `YLMProbe`)**: Reads the 3D atoms with full 3D spherical features (`0e` numbers + `0o` handedness sign + `1o` 3D arrows + `2e` 3x3 shapes), so $(R)$ vs $(S)$ flips sign cleanly while forces and dipoles rotate accurately with the molecule.
