# Novelty Check

## Question

Has anyone built a model where **language tokens directly cross-attend to l>=1 equivariant
features**, and the whole thing **stays equivariant**?

## Update 2026-09-29 (week 1)

Read EquiVLA, EquiLLM, CatalyticMLLM, and MatterChat. All four keep language on invariant features.
The pattern is always: equivariant encoder -> invariant bridge -> LLM, with vectors either dropped or fused after the LLM.
The claim still holds.

Note: this repo (github.com/yaniguan/YLM) is public and already shows up in web search for these keywords.
ICML is double-blind: don't link the repo from the submission, and consider making it private until the deadline.

## Short answer

Not that I found (searches done 2026-09-29). The closest is **EquiLLM** (ICML 2025), and it
deliberately does **not** let the LLM touch l>=1 features.

This is based on ~9 web searches plus what I already knew. It is not a full literature review.
Redo the searches below before submitting.

## Closest works, compared

| Work | Text/LLM sees l>=1? | Output is vector/tensor? | Exact equivariance? | Text Q&A? | Difference from YLM |
|---|---|---|---|---|---|
| EquiLLM (2502.11149) | No. Only invariants go into the LLM. | Yes (vectors, l=1) | Yes | No (MD, motion, antibody) | Direction info goes *around* the LLM. LLM output only scales vectors afterwards in an EGNN adapter. No l=2. No literature text. |
| 3D-MoLM (2401.13923) | No. Uni-Mol features are invariant. | No (text only) | Invariant only | Yes | Can't say a direction at all. |
| TEDMol (2410.03803) | Text is a scalar condition on an equivariant diffusion model | Generates coordinates | Yes | No | Text -> geometry generation, not Q&A. Text never attends to l>=1 features. |
| EquiFiLM (2607.05559) | Condition is a scalar, not text | Forces | Yes | No | Same math as our Option A with a number instead of text. |
| CLaSP (2501.12919) | No. One global invariant embedding. | No | Invariant | Retrieval only | Contrastive only. |
| LLM elastic tensor (2411.12280) | Structure is written as text | Yes, 21 numbers | No | Sort of | No symmetry guarantee at all. |
| SE(3)-Transformer, Equiformer(V2) | No text | Yes | Yes | No | They have the math we reuse (invariant weights x equivariant values), but queries come from atoms, not words. |
| EquiVLA (2606.19784) | No. Language tokens only see invariant visual tokens (frozen VLM). Equivariant tokens bypass the VLM and are fused after it with an invariant gate. | Robot actions | Yes, but only the discrete planar group C8 (8 rotations in 2D) | Instructions | Same pattern as EquiLLM, for robots. 2D discrete group, no SO(3), no l>=1 irreps. |
| CatalyticMLLM (2605.17254) | Not shown. EquiformerV2 encoder -> linear projection -> Qwen2.5-VL. No sign that l>=1 features reach the LLM. | Energy, CIF text | Not tested | Yes | Uses an equivariant encoder but throws away equivariance at the projection. No vectors/tensors out. No rotation tests. |
| MatterChat (2502.13107, Nat. Mach. Intell. 2026) | Most likely no: MACE-MP-0/CHGNet atom embeddings -> BLIP-2 style bridge (32 queries) -> frozen LLM. (verify which irreps are used) | Scalars, text | Not tested | Yes | Closest *materials* chat model. Collapse point: the bridge queries read invariant atom embeddings. |

## Where existing models collapse 3D to invariants

This is the evidence for the motivation. For each model: the exact step where direction is lost.

1. **MolT5, MoMu, MoleculeSTM, MolCA, GIT-Mol**: never had 3D. Input is SMILES or a 2D graph.
   Direction never existed.
2. **3D-MoLM**: the 3D encoder is Uni-Mol. Uni-Mol builds its attention bias from **pairwise distances**
   (Gaussian basis of |r_i - r_j|). Distances are invariant. So every atom feature that goes into the
   Q-Former is invariant. The Q-Former then pools into a fixed number of tokens, which are also invariant.
   **Collapse point: the first layer of the encoder (distance featurization).**
3. **Uni-Mol / Uni-Mol2**: same as above. The coordinate head can output positions, but those are not
   passed to any language model.
4. **3D-MolT5**: turns 3D into discrete fingerprint tokens (invariant). **Collapse: tokenization.**
5. **CLaSP, crystal-text contrastive models**: graph encoder -> **global mean pool -> one invariant vector**.
   **Collapse: pooling to a single embedding.**
6. **PointLLM, 3D-LLM, LEO (vision side)**: not equivariant at all. They rely on data augmentation or a
   fixed scene frame. **Collapse: no symmetry to begin with.**
7. **MatterChat, CatalyticMLLM**: equivariant encoder (MACE-MP-0 / EquiformerV2), but atom embeddings are projected / Q-Former-pooled into LLM tokens, which cannot carry l>=1 safely. **Collapse: the encoder -> LLM bridge.**
8. **EquiVLA** (robots): same as EquiLLM; language sees invariants only.
9. **EquiLLM**: encoder is equivariant, but **only invariant features are passed to the LLM** by design.
   **Collapse: the encoder -> LLM interface.**
10. **LLM-Prop, CrystaLLM, LLM-for-elastic-tensor, Gruver et al.**: structure is text. Rotation is handled
   (if at all) by augmentation. **No guarantee.**
11. **TEDMol, Chemeleon**: text is squashed to one invariant vector and used as a condition.
   **Collapse: the text side is invariant (which is fine), but there is no path for text to read
   direction features.**

## One-sentence claim you can defend

"Existing molecule/material-language models either never see 3D direction, or reduce it to invariants
before the language model, so language cannot select, combine, or report directional quantities;
YLM is the first to let text tokens attend over l>=1 irreps with exact SE(3) equivariance."

Make this claim only after the re-check below.

## Re-check list (do this before writing)

Search Google Scholar, arXiv, OpenReview (ICLR/NeurIPS/ICML 2025-2026) for:
- "equivariant" + "cross-attention" + "language"
- "irreps" + "LLM" / "text"
- "steerable" + "language model"
- "tensor property" + "language model"
- "equivariant multimodal" + "molecule"
- papers citing EquiLLM (2502.11149)
- papers citing 3D-MoLM (2401.13923)
- ~~read EquiVLA and CatalyticMLLM~~ done 2026-09-29 (see table)
- read MatterChat full text (check which MACE irreps go into the bridge)

## Sources used in this check

- https://arxiv.org/abs/2502.11149 (EquiLLM)
- https://arxiv.org/abs/2401.13923 (3D-MoLM)
- https://arxiv.org/abs/2410.03803 (TEDMol)
- https://arxiv.org/abs/2501.12919 (CLaSP)
- https://arxiv.org/abs/2411.12280 (LLM elastic tensor)
- https://arxiv.org/abs/2606.19784 (EquiVLA)
- https://arxiv.org/abs/2607.05559 (EquiFiLM)
- https://arxiv.org/abs/2605.17254 (CatalyticMLLM)
- https://arxiv.org/abs/2602.22251 (Zatom-1)
- https://arxiv.org/abs/2502.13107 (MatterChat)
