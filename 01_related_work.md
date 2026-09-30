# Related Work

Format per paper: **Name** (year, venue) — link
- Method: one line.
- For us: one line.

`(verify)` = I am not fully sure about the ID or venue. Check before citing.

---

## A. Equivariant networks

Focus: how each one does "attention" (if it has attention).

**Tensor Field Networks (TFN)** (2018, arXiv) — https://arxiv.org/abs/1802.08219
- Method: features are irreps; layers are CG tensor products with `Y_l(bond direction)` times a learned radial function. No attention.
- For us: the basic building block. Every later model is a variation of this.

**Cormorant** (2019, NeurIPS) — https://arxiv.org/abs/1906.04015
- Method: CG products between atom features, no attention.
- For us: early proof that irreps work for molecules.

**SE(3)-Transformer** (2020, NeurIPS) — https://arxiv.org/abs/2006.10503
- Method: attention inside the 3D graph. Query/key are irreps; attention weight = invariant dot product of q and k (sum over each l of `q_l . k_l`); value is a TFN message (equivariant).
- For us: **the key pattern: invariant weights x equivariant values.** Our Option A is this pattern with text queries.

**EGNN** (2021, ICML) — https://arxiv.org/abs/2102.09844
- Method: only scalars + coordinates (l=0 and l=1 via relative vectors). Very simple.
- For us: cheap baseline encoder. Used inside EquiLLM.

**PaiNN** (2021, ICML) — https://arxiv.org/abs/2102.03150
- Method: scalar + vector (l=0, l=1) features, gating. Predicts dipoles and polarizability tensors.
- For us: cheap l<=1 encoder, and a known readout for dipole/polarizability.

**NequIP** (2022, Nature Comms) — https://arxiv.org/abs/2101.03164
- Method: TFN-style message passing, E(3) with parity, very data efficient for forces.
- For us: possible 3D encoder.

**Allegro** (2023, Nature Comms) — https://arxiv.org/abs/2204.05249
- Method: strictly local, edge-based equivariant features, scales to huge systems.
- For us: shows high-l can be made fast if local.

**MACE** (2022, NeurIPS) — https://arxiv.org/abs/2206.07697
- Method: higher body-order messages built by repeated CG products (ACE basis). No attention.
- For us: strongest off-the-shelf encoder. Pretrained foundation checkpoints exist (MACE-MP-0). Good first choice.

**SEGNN** (2022, ICLR) — https://arxiv.org/abs/2110.02905
- Method: "steerable MLPs": node features and messages are irreps, nonlinearity via gated CG products conditioned on geometry.
- For us: shows how to mix in extra conditioning (they condition on geometric attributes). Similar to how we could condition on text.

**TorchMD-NET / Equivariant Transformer** (2022, ICLR) — https://arxiv.org/abs/2202.02541
- Method: attention with scalar weights, updates both scalar and vector features.
- For us: another example of "invariant weights, equivariant values", l<=1 only.

**So3krates** (2022, NeurIPS) — https://arxiv.org/abs/2205.14276
- Method: attention whose weights depend on invariants of spherical harmonic coordinates.
- For us: shows attention can "see" geometry through invariant summaries of high-l features.

**Equiformer** (2023, ICLR) — https://arxiv.org/abs/2206.11990
- Method: graph attention with irreps. Attention weights from invariant (l=0) part of a CG product; values are irreps; adds non-linear message passing.
- For us: main reference for "equivariant attention" design.

**eSCN** (2023, ICML) — https://arxiv.org/abs/2302.03655
- Method: rotate each edge to align with z-axis, then CG product becomes cheap (SO(2) convolution). Cost drops from ~L^6 to ~L^3.
- For us: the trick that makes high l (like l=4 for elastic tensor) affordable.

**EquiformerV2** (2024, ICLR) — https://arxiv.org/abs/2306.12059
- Method: Equiformer + eSCN trick, scales to l=6/8, attention re-normalization, separable S2 activation.
- For us: best reference for high-l attention that actually trains at scale.

**e3nn** (2022, arXiv software paper) — https://arxiv.org/abs/2207.09453
- Method: PyTorch library for irreps, spherical harmonics, CG products, Cartesian<->irreps conversion.
- For us: **the library we will use.** Code: https://github.com/e3nn/e3nn

Newer (2025-2026) found in search, not read in detail:
- EquiformerV3 — https://arxiv.org/abs/2604.09130
- E2Former-V2 — https://arxiv.org/abs/2601.16622

### How equivariant attention is built (summary)

All models above do the same thing:
1. Compute attention **weights** from **invariants only** (l=0 features, or dot products `q_l . k_l`).
2. Apply those weights to **equivariant values** (irreps).
3. Weighted sum of equivariant values = still equivariant.

Softmax of non-invariant numbers would break equivariance. That is why weights must be invariant.

---

## B. Molecule / material + text models

**Text2Mol** (2021, EMNLP) — ACL Anthology, no arXiv I know of
- Method: retrieve molecules from text; GNN on 2D graph + text encoder, contrastive.
- For us: introduced ChEBI-20 text-molecule data.

**MolT5** (2022, EMNLP) — https://arxiv.org/abs/2204.11817
- Method: T5 trained on SMILES + text; translates molecule <-> caption.
- For us: 1D only (SMILES string). No 3D at all.

**MoMu** (2022, arXiv) — https://arxiv.org/abs/2209.05481
- Method: CLIP-style contrastive between 2D molecular graph (GIN) and text.
- For us: 2D only.

**MoleculeSTM** (2023, Nature Machine Intelligence) — https://arxiv.org/abs/2212.10789
- Method: contrastive text <-> molecule (SMILES or 2D graph), used for text-guided editing.
- For us: 2D only. Output is one invariant vector per molecule.

**GIT-Mol** (2024, Computers in Biology and Medicine) — https://arxiv.org/abs/2308.06911 (verify)
- Method: graph + image + text into one LLM via cross-attention adapter.
- For us: 2D only.

**MolCA** (2023, EMNLP) — https://arxiv.org/abs/2310.12798
- Method: 2D graph encoder -> Q-Former (BLIP-2 style) -> LLM soft prompts.
- For us: shows the standard "encoder -> Q-Former -> LLM" pipeline. 2D only.

**3D-MoLM** (2024, ICLR) — https://arxiv.org/abs/2401.13923
- Method: Uni-Mol 3D encoder -> Q-Former -> LLaMA. Instruction tuning on 3D molecule-text data (PubChem-derived). Answers property questions as text.
- For us: **closest "3D molecule + LLM" system.** But Uni-Mol is invariant (uses only distances), so direction is gone before the LLM sees it.

**3D-MolT5** (2024, arXiv) — https://arxiv.org/abs/2406.05797 (verify)
- Method: turns 3D structure into discrete tokens (3D fingerprint) and trains T5.
- For us: another invariant 3D->text route.

**Uni-Mol** (2023, ICLR) — ChemRxiv / OpenReview (no arXiv I know of)
- Method: SE(3)-invariant Transformer on atoms, pair representation from distances; has a coordinate head that outputs positions (equivariant via distance-weighted updates).
- For us: popular 3D encoder, but its features are invariant (pairwise distances).

**Uni-Mol2** (2024, NeurIPS) — https://arxiv.org/abs/2406.14969
- Method: scaled Uni-Mol to ~1B params.
- For us: same invariance issue.

**TEDMol** (text-guided 3D molecule diffusion, 2024) — https://arxiv.org/abs/2410.03803
- Method: text embedding conditions an equivariant diffusion model.
- For us: text *conditions* an equivariant model, but only as an invariant (scalar) signal. That is the "Option A" direction of info flow. Not a Q&A system.

**Chemical Language Model Linker** (2024) — https://arxiv.org/abs/2410.20182
- Method: modular adapters linking text and molecule encoders.
- For us: adapter-style design reference.

### Materials side

**MatSciBERT** (2022, npj Comput Mater) — https://arxiv.org/abs/2109.15290
- Method: BERT further pretrained on materials papers.
- For us: candidate text encoder.

**BatteryBERT** (2022, J. Chem. Inf. Model.) — search "BatteryBERT Huang Cole" (verify ID)
- Method: BERT pretrained on battery papers.
- For us: **candidate text encoder for battery text.**

**CLaSP: Bridging Text and Crystal Structures** (2025, Mach. Learn.: Sci. Technol.; NeurIPS 2024 workshop) — https://arxiv.org/abs/2501.12919
- Method: CLIP-style contrastive between crystal structure (graph encoder) and paper titles/abstracts, ~400k structures.
- For us: **closest text-crystal alignment work.** Global invariant embedding only.

**Contrastive Learning of English Language and Crystal Graphs** (2025) — https://arxiv.org/abs/2502.16451
- Method: text-crystal graph contrastive.
- For us: same category as CLaSP.

**Text-guided crystal generation (Chemeleon)** (2025, Nature Comms) — https://www.nature.com/articles/s41467-025-59636-y
- Method: text embeddings guide crystal diffusion.
- For us: text -> structure, invariant conditioning.

**LLM-Prop** (2023) — https://arxiv.org/abs/2310.14029 (verify)
- Method: T5 encoder on text descriptions of crystals predicts band gap etc.
- For us: baseline (ii) style: "describe structure in text, predict a number".

**LLMs for elastic constant tensor** (2024) — https://arxiv.org/abs/2411.12280
- Method: text description of composition + structure fed to Llama2-7b to predict elastic tensor.
- For us: **direct baseline (ii)** for tensor prediction from text. Not rotation equivariant.

**CrystaLLM** (2024, Nature Comms) — https://arxiv.org/abs/2307.04340
- Method: GPT on CIF text files.
- For us: "coordinates as text" baseline.

**Gruver et al., fine-tuned LLMs generate stable materials** (2024, ICLR) — https://arxiv.org/abs/2402.04379
- Method: Llama-2 on crystal text strings with rotation/translation augmentation.
- For us: shows that augmentation is the "cheap" way LLMs handle symmetry. Our argument is against this.

**CatalyticMLLM** (2026) — https://arxiv.org/abs/2605.17254
- Method: EquiformerV2 encoder -> linear projection -> Qwen2.5-VL. Property prediction and CIF generation.
- For us: equivariant encoder, but no sign l>=1 reaches the LLM; no vector outputs, no rotation tests.

**MatterChat** (2026, Nature Machine Intelligence) — https://arxiv.org/abs/2502.13107
- Method: frozen MACE-MP-0 or CHGNet atom embeddings -> BLIP-2 style bridge with 32 queries -> frozen LLM.
- For us: **closest materials chat model.** Direct baseline for crystals (week 9+). Uses the same MACE-MP-0 encoder we plan to use, so the comparison is clean.

**Zatom-1** (2026) — https://arxiv.org/abs/2602.22251
- Method: plain (non-equivariant) Transformer, flow matching over molecules + materials.
- For us: counter-example — some groups drop equivariance entirely. Reviewers may cite this.

### Tensor property prediction (no text, but same outputs we want)

- PaiNN (above): dipole, polarizability.
- **Equivariant GNNs for tensor material properties** — https://arxiv.org/abs/2406.03563
- **Dielectric tensor prediction** (npj Comput Mater 2024) — https://www.nature.com/articles/s41524-024-01450-z
- **StrainTensorNet** (elastic) — https://arxiv.org/abs/2306.12818
- **GoeCTP: canonicalization for crystal tensors** — https://arxiv.org/abs/2410.02372
- **CEITNet: high-order crystal tensors (ICML 2026)** — https://arxiv.org/abs/2602.04323
- **Scalable dielectric tensor prediction** — https://arxiv.org/abs/2601.04755
- MatTen (elastic tensor with e3nn, Wen et al. 2024, Digital Discovery) (verify ID)

For us: these are the "upper bound without text" baselines. Our model should not be worse than them on the same tensor task.

---

## C. 3D vision / point cloud + language

Useful for mechanisms, not for symmetry (almost none are equivariant).

**ScanRefer** (2020, ECCV) — https://arxiv.org/abs/1912.08830
- Method: find the object in a 3D scan that a sentence describes.
- For us: "text points at a location" = like text pointing at an atom.

**ULIP** (2023, CVPR) — https://arxiv.org/abs/2212.05171
- Method: align point cloud, image, text (CLIP-style).
- For us: contrastive pretraining recipe.

**PointLLM** (2024, ECCV) — https://arxiv.org/abs/2308.16911
- Method: point cloud encoder -> projector -> LLM tokens.
- For us: simplest "3D tokens into LLM" pipeline. Not equivariant.

**3D-LLM** (2023, NeurIPS) — https://arxiv.org/abs/2307.12981
- Method: multi-view 2D features lifted to 3D, fed to LLM.
- For us: not equivariant. Uses augmentation.

**3D-VisTA** (2023, ICCV) — https://arxiv.org/abs/2308.04352
- Method: simple Transformer for 3D scene + text, pretraining.
- For us: shows simple fusion works when data is big.

**LEO** (2024, ICML) — https://arxiv.org/abs/2311.12871
- Method: embodied 3D generalist agent, 3D object tokens into LLM.
- For us: outputs actions (directions) — similar to our "say a vector" problem.

**SpatialVLM** (2024, CVPR) — https://arxiv.org/abs/2401.12168
- Method: generate huge spatial Q&A data from images, train VLM to answer distances/directions in text.
- For us: **useful idea for data: auto-generate Q&A from structures with templates.**

**Vector Neurons** (2021, ICCV) — https://arxiv.org/abs/2104.12229
- Method: make each neuron a 3D vector; linear layers mix channels, not xyz. Cheap SO(3) equivariance for point clouds.
- For us: cheapest possible l=1 equivariant layer. Good for a first prototype.

**Frame Averaging** (2022, ICLR) — https://arxiv.org/abs/2110.03336
- Method: make any network equivariant by averaging over a few PCA-based frames.
- For us: a baseline that makes a plain LLM "equivariant" without irreps. Reviewers will ask about this.

**EquiVLA** (2026) — https://arxiv.org/abs/2606.19784
- Method: frozen VLM sees only invariant visual tokens + language; equivariant tokens (C8 planar group) bypass it and are fused after with an invariant gate.
- For us: same "language sees invariants only" pattern as EquiLLM, in 2D robotics.

---

## D. LLM / Transformer tokens + equivariant (non-scalar) features

This is the novelty-critical group.

**EquiLLM: Large Language-Geometry Model** (2025, ICML) — https://arxiv.org/abs/2502.11149
- Method: equivariant encoder (EGNN) -> **only invariant features** go into the LLM -> LLM output (invariant) + equivariant vectors from the encoder -> equivariant adapter (EGNN layer) outputs vectors.
- For us: **closest work.** Differences:
  - LLM never sees l>=1 features. Direction info bypasses the LLM.
  - Only l<=1 (EGNN vectors). No l=2 tensors.
  - Tasks: MD, human motion, antibody design. No text Q&A, no literature text, no tensor properties.
  - So the language part cannot *select* which directional feature to use via attention on l>=1 data. It only produces scalars that scale vectors afterward.

**TEDMol** (above, B) — text scalar conditions an equivariant model.

**SEGNN / conditioning via steerable attributes** (above, A) — shows how to inject extra info into CG products.

**EquiFiLM** (2026) — https://arxiv.org/abs/2607.05559 (found in search, not read)
- Method: conditions equivariant force fields on per-input scalars (FiLM-style).
- For us: "scalar condition x equivariant features" — same math as our Option A, but condition is a number, not text.

**EquiVLA** (above, C) — needs a close read.

Nothing found (in my searches) where text tokens do cross-attention *over l>=1 irreps* and return an l>=1 answer with guaranteed equivariance, evaluated on tensor Q&A.
