# Data and Benchmark

Sizes and licenses are from memory. Check each one on the official page before using.

## 1. Data with vector / tensor labels

| Dataset | What labels | Size | License | How to get |
|---|---|---|---|---|
| QM9 | Standard release has only **|dipole|** (scalar) and **isotropic polarizability** (scalar). No vectors. | ~134k small organic molecules | CC0-ish / open (verify) | PyTorch Geometric `QM9`, or figshare |
| QM9 + recomputed labels (we make it) | dipole vector (1o), full polarizability tensor (0e+2e) | as many as we compute | ours | Run GFN2-xTB (fast, seconds/molecule) or PySCF DFT on QM9 geometries |
| QM7b polarizability (Wilkins et al. 2019, PNAS) | full polarizability tensors (CCSD and DFT) | ~7k molecules | open (verify) | Materials Cloud archive |
| QM7-X | dipole vectors, polarizability, forces (verify which are tensors) | ~4.2M conformers | CC BY 4.0 (verify) | Zenodo |
| MD17 / rMD17 | forces (1o per atom) | ~100k frames per molecule | open | sgdml.org / figshare |
| OC20 | forces on adsorbate+slab | ~130M+ frames (S2EF), smaller splits exist (200k) | CC BY 4.0 | fair-chem repo download scripts |
| MPtrj (from Materials Project) | forces, stress (0e+2e) | ~1.5M frames, ~146k materials | CC BY 4.0 | figshare (CHGNet paper) |
| Materials Project dielectric | dielectric tensor (0e+2e) | several thousand (verify count; original 2017 set ~1k) | CC BY 4.0 | `mp-api` with free API key |
| Materials Project elastic | elastic tensor (2x0e+2x2e+4e) | ~10k+ (verify) | CC BY 4.0 | `mp-api` |
| Materials Project piezo | piezo tensor (2x1o+2o+3o) | ~1k-3k (verify) | CC BY 4.0 | `mp-api` |
| JARVIS-DFT | dielectric, elastic, piezo tensors, many scalars | ~70k+ materials total, fewer with tensors | NIST, public | `jarvis-tools` |

**Pick for month 1**: QM9 + our own xTB dipole vectors and polarizability tensors.
Reason: small molecules, cheap, we control every label, no license worries.

**Pick for the battery story (month 2-3)**: Materials Project dielectric + elastic tensors on
Li/Na-containing compounds, plus MPtrj forces.

## 2. Data with text paired to structures

| Source | What | Size | Notes |
|---|---|---|---|
| Huang & Cole battery database (Sci. Data 2020) | text-mined (compound, property) pairs: capacity, voltage, conductivity, Coulombic efficiency, energy | ~292k records, ~17k compounds, from ~229k papers | Compound name only, **no structure**. Must match formula -> MP structure (polymorph ambiguity). Also has source DOIs. |
| BatteryDataExtractor / BatteryBERT | text-mining tools and a battery-domain BERT | - | Use as text encoder. |
| COD (Crystallography Open Database) + paper abstracts | structure + DOI of the paper it came from -> get abstract via OpenAlex/Crossref | ~500k structures | What CLaSP did. COD is open. Abstracts: check publisher terms; OpenAlex has many. |
| Materials Project + ICSD references | structure + literature references | ~150k materials | Some references only. |
| ChEBI-20 / PubChem descriptions | molecule + text description | ~33k (ChEBI-20); ~300k (PubChem, 3D-MoLM data) | Few QM9 molecules have descriptions. |
| Templated Q&A (we make it) | question templates filled from labels, paraphrased by an LLM | unlimited | The main source for the benchmark. SpatialVLM did this for images. |

Honest point: **there is almost no natural text that asks for a dipole vector.** The benchmark text will be
templated + paraphrased. The literature text is for pretraining the alignment, not for the tensor labels.

## 3. Minimal benchmark: "TensorQA-v0"

Input: (structure, question text). Output: a vector or tensor (sometimes a scalar).

Question types (start with 5):

| Type | Example question | Answer | Needs text to select? |
|---|---|---|---|
| T1 global vector | "What is the dipole moment of this molecule?" | 1o | No (task label) |
| T2 global tensor | "Give the polarizability tensor." | 0e+2e | No |
| T3 atom-selected vector | "What is the force on the oxygen atom bonded to two carbons?" | 1o | **Yes** |
| T4 projection | "How polarizable is the molecule along the C=O bond?" | 0e (= b^T alpha b) | **Yes**, and mixes direction + tensor |
| T5 comparison | "Is the dipole closer to the N-H bond or the C-O bond direction?" | text / class | **Yes** |

T3-T5 are the important ones. T1-T2 alone can be solved with a task ID (no language needed).

Splits: split by molecule (scaffold split), and hold out some question templates to test paraphrase generalization.

Size for v0: ~20k molecules x ~5 questions = ~100k Q&A pairs. Enough to start.

## 4. Rotation consistency protocol

For each test example:
1. Pick K = 32 random rotations (uniform: `e3nn.o3.rand_matrix`). Plus the mirror image for chirality tests.
2. Run the model on each rotated input.
3. Report:
   - **Task error**, averaged over the K rotations:
     - vectors: MAE per component, and cosine similarity of the direction, and error of the length.
     - tensors: Frobenius norm error, and error of eigenvalues (invariant), and angle of the main eigenvector.
     - scalars / text: MAE or accuracy.
   - **Equivariance error**: `||f(Rx) - D(R) f(x)|| / ||f(x)||`, average and max over rotations.
   - **Answer consistency** (for text/class answers): fraction of rotations that give the same answer.
4. Also test **random input frames** (not the dataset's saved orientation). Many datasets save molecules
   in a standard orientation; non-equivariant baselines cheat by memorizing that frame.

Expected: our model has equivariance error ~1e-6 (float32 noise). Baselines (ii) and (iii) will not.

## 5. Baselines

| # | Baseline | What it tests |
|---|---|---|
| i-a | **Scalar collapse + canonical frame**: invariant 3D encoder -> LLM -> numbers in a PCA frame -> rotate back | "Just use invariants" |
| i-b | **EquiLLM-style**: invariants into LLM, equivariant adapter after the LLM | Strongest existing design. The one we must beat on T3-T5. |
| ii | **Coordinates as text** into a small LLM (e.g. Qwen2.5-0.5B/1.5B fine-tuned), with and without rotation augmentation | "LLMs can just learn it" |
| iii | **3D-MoLM** / **MolCA** fine-tuned to write numbers | Existing molecular MLLMs |
| iv | **No text, task-specific equivariant model** (PaiNN / MACE + tensor head), one model per question type | Upper bound for T1-T2 |
| v | **Our model with text replaced by a one-hot task ID** | Is the language doing any work? |
| vi | **Frame averaging** around a plain Transformer | Cheap equivariance without irreps |

Must-have ablations for our model: Option A only / A+B / A+B+C; max l = 1 vs 2; with vs without parity/pseudoscalars.
