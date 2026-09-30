# Glossary

Plain meanings. No proofs.

## Symmetry words

**Rotation R.** A 3x3 matrix that turns things in 3D. Does not stretch. det(R) = +1.

**Reflection / parity / inversion.** Mirror flip. Inversion is x -> -x. det = -1.

**SO(3).** All rotations.

**O(3).** All rotations plus mirror flips.

**SE(3).** Rotations + translations (moving things around).

**E(3).** Rotations + translations + mirror flips.

**Invariant.** Output does not change when input rotates.
Example: energy, band gap, the length of a vector.

**Equivariant.** Output rotates the same way the input rotates.
Example: force on an atom. Rotate the molecule by R, the force becomes R * force.

Written as a rule: `f(R x) = R f(x)`.

## Irreps (irreducible representations)

The simplest "building blocks" of how things rotate. Each block has a number `l`.

| l | size (2l+1) | what it is | example |
|---|---|---|---|
| 0 | 1 | a plain number (scalar) | energy, band gap |
| 1 | 3 | a vector | dipole, force |
| 2 | 5 | a traceless symmetric 3x3 matrix | the "shape" part of polarizability |
| 3 | 7 | higher order | part of piezo tensor |
| 4 | 9 | higher order | part of elastic tensor |

Under rotation R, an `l` block is multiplied by a (2l+1)x(2l+1) matrix called the
**Wigner-D matrix** `D_l(R)`. For l=0, D is just 1 (nothing changes). For l=1, D is basically R.

**Parity label (e / o).** In e3nn you see `1o`, `1e`, `2e` etc.
- `o` = odd: flips sign under inversion. A normal vector (dipole, force) is `1o`.
- `e` = even: does not flip. Angular momentum / magnetic field are `1e` ("pseudovector").
- A scalar that flips sign under mirror is `0o` ("pseudoscalar"). This is the thing that
  can tell left-handed from right-handed molecules.

## Spherical harmonics Y_lm

Functions on the sphere. Feed in a direction (unit vector), get out an `l` block.
`Y_1(direction)` is basically the direction itself.
Equivariant networks use `Y_l(r_ij)` of the bond direction between atoms i and j.

## Clebsch-Gordan (CG) tensor product

The only safe way to "multiply" two irreps and still get irreps.
- `l1 x l2` can produce any `l` from `|l1 - l2|` to `l1 + l2`.
- `1 x 1 -> 0` is the dot product.
- `1 x 1 -> 1` is the cross product.
- `1 x 1 -> 2` is the traceless outer product.
- `0 x l -> l` is just "scale the vector by a number". This is the cheap one.

Cost grows fast with max l (roughly L^6 naive, L^3 with tricks like eSCN).

## Tensors we care about

- **Dipole moment**: vector, `1o`. 3 numbers.
- **Polarizability / dielectric tensor**: symmetric 3x3 matrix = `0e + 2e` (1 + 5 = 6 numbers).
- **Piezoelectric tensor**: 3x3x3 with symmetry, 18 numbers = `2x1o + 2o + 3o` (3+3+5+7).
- **Elastic tensor**: 3x3x3x3 with symmetry, 21 numbers = `2x0e + 2x2e + 4e`.

You can convert a Cartesian tensor to irreps and back with `e3nn.io.CartesianTensor`.

## ML words

**Cross-attention.** Tokens of type A (queries) look at tokens of type B (keys, values).
Output = weighted average of B's values, weights = softmax(q . k).

**Readout head.** The small layer at the end that turns hidden features into the final answer.

**Frame averaging / canonicalization.** Instead of building symmetry into the network,
rotate the input to a standard pose (or average over many poses). Cheaper, but only
approximately or conditionally equivariant.
