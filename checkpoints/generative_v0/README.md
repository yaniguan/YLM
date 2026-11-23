# Generative `<vec>` Checkpoint (`checkpoints/generative_v0`)

Compact offline-loadable `YLMCausalLM` (`mode="xattnA"`, `vec_readout=True`, `CharTokenizer`, `2.76M` total parameters) trained by [`scripts/65_train_generative_ylm.py`](../../scripts/65_train_generative_ylm.py) on the official `TensorQA-v0.1.0` QM9 GFN2-xTB splits (`4,800` train examples, frozen 60-epoch QM9 equivariant `Encoder`).

## Measured Test Metrics (`results.json`)

| Split | `<vec>` emission rate | Dipole cosine | Dipole MAE (Debye) | Force cosine | O(3) rot / mirror rel. err |
|---|---:|---:|---:|---:|---:|
| `test` (`n=200`) | **0.995** | **0.9830** | **0.2691** | 0.2468 | `2.64e-6` / `2.64e-6` |
| `test_heldout` (`n=200`) | **0.950** | **0.9161** | **0.4847** | 0.1096 | `2.50e-6` / `2.50e-6` |

### Scope & Honest Takeaway
- **What works well**: Autoregressive text decoding with `<vec>` emission (`99.5%` on `test`, `95.0%` on unseen `test_heldout` templates), molecular dipole vector readout (`0.983` / `0.916` cosine), and exact O(3) rotation + reflection equivariance (`~2.6e-6` float32 round-off) with zero external tokenizer or Hugging Face downloads.
- **Current bottleneck**: Because the text decoder here is a `2.47M`-parameter character-level GPT-2 trained from scratch (`CharTokenizer`), single-atom force disambiguation over 80-character connectivity clauses reaches `0.247` cosine (vs `0.960` when paired with a pretrained `MatSciBERT` encoder in `checkpoints/v0/ylm_matscibert_s0.pt` or `0.872` with `GroupProgModel`). Multi-GPU fine-tuning of a pretrained BPE causal LM (`hf:Qwen/Qwen2.5-0.5B`) with `vec_readout=True` is tracked in `Issues #56, #58`.

## Usage

```python
from src.toolkit.ingest import StructureFrame
from src.toolkit.probe import YLMProbe

probe = YLMProbe.from_checkpoint("checkpoints/generative_v0", device="cpu")
water = StructureFrame(Z=[8, 1, 1], pos_bohr=[[0.0, 0.0, 0.0], [1.81, 0.0, 0.0], [-0.45, 1.75, 0.0]], label="water")
out = probe.ask(water, "What is the dipole moment of the molecule?", verify_o3=True)
print(out.answer, out.vectors[0].vector_physical, out.o3_check.passed)
```
