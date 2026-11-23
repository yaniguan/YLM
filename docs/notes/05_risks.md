# Top 5 Risks

## 1. Not enough paired data (text + structure + vector/tensor label)

- Problem: nobody writes "the dipole vector is (0.1, -0.5, 1.3)" in papers. Natural text + tensor labels basically don't exist together.
- Fix:
  - Make the benchmark from templates + LLM paraphrases (like SpatialVLM did).
  - Use literature text (COD abstracts, battery database) only for pretraining the text-structure alignment.
  - Compute our own labels with xTB/DFT where needed. We control quality.
  - Say this openly in the paper. Reviewers accept templated data if the held-out-template test is honest.

## 2. Reviewer: "Scalars are enough" / "EquiLLM already does this"

- Problem: an invariant LLM + equivariant adapter (EquiLLM) or frame averaging might match us.
- Fix:
  - Design tasks where **text must pick a direction** (T3-T5 in the benchmark). An adapter after the LLM can only scale vectors the encoder already made; it can't let a word choose "the force on this specific oxygen" by attending to l>=1 features.
  - Run baseline i-b (EquiLLM-style) and vi (frame averaging) ourselves, same encoder, same data.
  - Report sample efficiency (10%, 30%, 100% data). Equivariant models usually win most at low data.
  - Report equivariance error under random frames. Baselines that cheat with dataset orientation will fail here.
  - If we lose on some task, report it. Don't hide it.

## 3. High-l irreps are slow and memory-hungry

- Problem: elastic tensor needs l = 4. CG products get expensive fast.
- Fix:
  - Month 1-2: only l <= 2 (dipole, polarizability, dielectric, stress). This already covers most of the story.
  - Cross-attention cost is `T * N`, not per edge. Keep T small (question length ~32 tokens).
  - Use the eSCN / EquiformerV2 trick for l = 4 later.
  - Use a frozen pretrained encoder (MACE-MP-0) so we don't train the 3D part.

## 4. The text becomes just a task label

- Problem: the model may ignore the words and just recognize "this is the dipole question".
- Fix:
  - Baseline v (one-hot task ID). If it matches us, the language contributes nothing. We must see a gap on T3-T5.
  - Hold out question templates and paraphrases at test time.
  - Add compositional questions (combine atom selection + projection).
  - Show attention maps: for "force on the oxygen", attention should sit on the oxygen.

## 5. Connecting to a real pretrained LLM is hard

- Problem: the LLM stream must stay invariant (LayerNorm etc.). Large LLMs are slow to fine-tune. Training can be unstable.
- Fix:
  - Start with a small text encoder (BERT-size: MatSciBERT or BatteryBERT, or a ~0.5B LLM). No generation needed at first.
  - Use the two-stream design (`h_t` invariant + `e_t` irreps side stream). Only invariants of `e_t` go into the LLM.
  - Freeze the LLM, train only the YLM blocks + LoRA.
  - Only move to a bigger generative LLM after the small version works.

## Smaller risks worth knowing

- **Label conventions**: dielectric tensors (electronic vs ionic vs total), units, DFT functional differences between datasets. Fix: one source per task, write units in the data card.
- **Charged molecules**: dipole depends on origin. Fix: neutral only, or fix origin.
- **Crystal periodicity**: need a periodic encoder (MACE, EquiformerV2 handle this). Don't use a molecule-only encoder on crystals.
- **Formula -> structure matching** in the battery database is ambiguous (polymorphs). Fix: keep only formulas with one clear MP ground-state structure, or drop.
