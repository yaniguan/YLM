# Technical Design: Irreps-to-Text Cross-Attention

## The setup

- `N` atoms. Atom `i` has position `r_i` and a feature `V_i` from the 3D encoder.
  `V_i` is a list of irreps blocks: `V_i^(0)` (C numbers), `V_i^(1)` (C x 3), `V_i^(2)` (C x 5), ...
- `T` text tokens. Token `t` has a hidden vector `h_t` (d numbers) from the text model.
- Rotation `R`. After rotating the molecule: `V_i^(l) -> D_l(R) V_i^(l)`. Text `h_t` does not change.

## The one rule

**Softmax only sees invariant numbers. Anything with a direction is only ever multiplied by
invariant numbers, or combined through CG products.**

If you follow this rule, the output is equivariant. That is all there is to it.

Why: if the weights `a_ti` are the same before and after rotation, then
`sum_i a_ti * D V_i = D * sum_i a_ti V_i`. The `D` comes out.

Translation: the encoder only uses relative vectors `r_i - r_j`, so translation does nothing.

---

## Option A: invariant weights x equivariant values (simplest, do this first)

```
q_t     = W_q h_t                                  # from text, invariant
k_i     = W_k inv(V_i)                             # from atom, invariant
a_ti    = softmax_i( q_t . k_i / sqrt(d) )         # invariant weight
for each l:
    val_i^(l) = W_v^(l) V_i^(l)                    # mix channels only, never touch the 2l+1 axis
    g_t^(l)   = sigmoid(W_g^(l) h_t)               # per-channel gate from text, invariant
    out_t^(l) = g_t^(l) * sum_i a_ti val_i^(l)     # equivariant
```

`inv(V_i)` = the l=0 part, plus the norm of each l>=1 channel (`||V_i^(l,c)||`).
(Optionally also dot products between channels of the same l.)

- **Text controls**: which atoms (`a_ti`), which channels (`g_t`).
- **Text cannot**: make new directions. The answer is always a mix of directions the atoms already have.
- **Cost**: `O(T * N * C * sum_l(2l+1))`. Same as normal cross-attention, times a small factor.
- **Equivariance**: exact, by construction.
- **Known from**: SE(3)-Transformer, Equiformer (inside the graph), EquiLLM adapter, EquiFiLM (scalar condition).
  New part here: queries come from words.

## Option B: direction-aware keys (text asks "which atoms point along X?")

In Option A, keys only use norms, so attention cannot "see" how atoms are oriented relative to each other.
Fix: give the query its own irreps, built from the molecule, weighted by text.

```
G^(l)     = pool_i V_i^(l)                       # global molecule irreps (e.g. mean), equivariant
Q_t^(l)   = diag(W_qg^(l) h_t) G^(l)             # text picks which global channels to use
K_i^(l)   = W_k^(l) V_i^(l)                      # equivariant key
score_ti  = q_t . k_i  +  sum_l sum_c < Q_t^(l,c), K_i^(l,c) >   # dot product of same-l blocks: invariant
a_ti      = softmax_i(score_ti)
values as in Option A
```

- **Text controls**: plus "attend to atoms whose feature aligns with the molecule's main axis / dipole / etc."
- **Cost**: a bit more than A (one extra dot product per l).
- **Equivariance**: exact. `<D x, D y> = <x, y>` because D is orthogonal.
- **Expressive gain**: can pick atoms by relative orientation. Useful for "force component along the Li-O bond".
- **Note**: text still can't point anywhere by itself. It has no direction. It can only point via the molecule.

## Option C: text-controlled CG tensor product (needed for tensors)

To answer with an l=2 tensor starting from l=1 vectors you need `1 x 1 -> 2`.
Options A and B cannot do that (they only take weighted sums). So:

```
o_t   = Option A or B output (irreps)
w_t   = MLP(h_t)                               # invariant weights, one per CG path
y_t   = TensorProduct(o_t, o_t, weights = w_t) # e3nn FullyConnectedTensorProduct / "weighted TP"
```

Also possible: `TensorProduct(o_t, G, w_t)` to combine the token's feature with the molecule's global feature.

Note: "treat text as l=0 and CG it with atom features" is the special case `0 x l -> l`,
which is only scaling. That equals the gate in Option A. It adds nothing new.
The real power is text choosing the weights of the `l1 x l2 -> l3` paths (a hypernetwork).

- **Text controls**: which products to form (dot, cross, outer products...).
- **Cost**: CG products scale badly with max l. But here it is done once per text token (T times),
  not per edge. With T ~ 32 and L <= 2 this is cheap. For L = 4 (elastic tensor) use eSCN-style tricks or keep T small.
- **Equivariance**: exact (e3nn guarantees it).

## Recommendation

Build `A` first. Add `B` scores. Add `C` only at the output.
Final block = A+B cross-attention, then one text-weighted tensor product before the readout.

## Where does the equivariant info live inside the language model?

The LLM's hidden stream must stay invariant (it goes through LayerNorm, softmax, MLPs that would break equivariance).
So each text token carries **two streams**:
- `h_t`: normal invariant hidden vector (the LLM).
- `e_t`: an irreps vector (the "geometry side stream").

Per YLM block:
1. `e_t += IrrepsToText(h_t, V)` (Options A/B/C above)
2. `h_t += MLP(inv(e_t))` — the LLM gets invariants of the side stream (norms, dot products between `e_t` and `e_s`). This lets later words "know" what was found.
3. Optional: equivariant self-attention between text tokens' `e_t` (same rule: invariant weights, equivariant values).

Difference from EquiLLM: there, the LLM never gets to query l>=1 features and the vectors are only fused after the LLM. Here, every YLM layer lets words query l>=1 features, and the result feeds back into the words.

---

## Output: how does the language model "say" a vector?

| Way | How | Equivariant? | Good for |
|---|---|---|---|
| 1. Equivariant readout head | Special token `<VEC>` or `<TENSOR>`. Its `e_t` goes through a linear layer to the target irreps (e.g. `1o`, or `0e+2e`), then `e3nn.io.CartesianTensor` turns it into a 3x3 matrix. | Exact | Main method |
| 2. Numbers as text tokens | LLM writes `"[0.12, -0.53, 1.30]"`. | No. Depends on how the coordinates were given. Only works with augmentation. | Baseline (ii) |
| 3. Invariant description in words | "points from O toward the H atoms", "largest along the c axis" | Invariant (correct) | Human-readable answers; use `h_t` and the normal LM head |

**Use 1 + 3.** The LLM writes text; when it emits `<VEC>` the head fills in the numbers. Loss = normal LM
loss on text + MSE on the irreps output. When showing to a human, print the numbers in the input frame.

Watch out:
- **Dipole of charged molecules depends on where the origin is.** Use neutral molecules, or fix origin = center of mass.
- **Symmetric tensors**: predict `0e + 2e` (6 numbers), not 9 numbers. The symmetry is then automatic.
- **Crystal tensors must respect the crystal's point group.** An equivariant model gets this for free if
  the input is symmetric (a nice selling point).

---

## Parity and chirality: SE(3) or E(3)?

Facts:
- Mirror image of an (R) molecule is the (S) molecule.
- Text saying "(R)-..." does not flip when you mirror the structure. So text is **not** mirror-invariant in meaning.
- A fully E(3)-invariant score (only `0e` features) gives the **same** number for R and S. It **cannot** tell them apart.
- A pseudoscalar `0o` (e.g. triple product `r1 . (r2 x r3)`) flips sign under mirror. It **can** tell them apart.

Choice:
- **Use SE(3) in practice** (rotations + translations only), but **keep e3nn parity labels** on all irreps.
- Allow `0o` features into the attention keys and the text-structure matching score.
- Then: rotations never change anything (good). Mirrors flip `0o` (good — R and S look different).
- Parity labels still help: dipole is `1o`, magnetic moment is `1e`; the readout head produces the right type.

Test: (1) rotate -> answer rotates. (2) mirror a chiral molecule -> invariant answer "R" should change to "S".
(3) mirror an achiral molecule -> vector answer should equal mirrored vector.

Crystals: most battery materials are not chiral, but polar properties (piezo, pyroelectric) depend on the
lack of inversion symmetry. Parity labels keep these correct.

---

## Minimal equivariance test (put in the repo on day 1)

```python
import torch
from e3nn import o3

def test_equivariance(model, pos, atom_types, text_ids, irreps_out):
    R = o3.rand_matrix()
    D = irreps_out.D_from_matrix(R)
    y1 = model(pos @ R.T, atom_types, text_ids)   # rotate input
    y2 = model(pos, atom_types, text_ids) @ D.T   # rotate output
    err = (y1 - y2).abs().max().item()
    assert err < 1e-4, err                         # float32. Use float64 to get ~1e-10.
```

Run it for every model, every commit.
