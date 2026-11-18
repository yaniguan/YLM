# LabLean

LabLean is a small Lean 4 language for bio lab protocols, with a checker.
A protocol is a list of steps (pick tip, transfer, mix, incubate, centrifuge, seal, thaw, ...).
The checker must accept a protocol before it goes to the robot:

    text -> protocol (LabLean steps) -> checker -> robot

Valid protocols are proved valid in Lean. Broken ones are rejected with an error message.
No Mathlib; core Lean only.

## Build

    source ~/.elan/env
    lake build          # builds the library, all examples and proofs
    lake exe lablean    # prints the checker result for every example

Toolchain: `leanprover/lean4:v4.36.0`.

## Units

* Volume: `Nat` in **0.1 uL** steps. `(2.5 : Vol)` is 2.5 uL, stored as 25.
* Temperature: whole degrees C. Time: seconds.

## Well index convention (96-well plate)

Row-major, 1-based numbers: 1 = A1, 12 = A12, 13 = B1, ..., 65 = F5, ..., 96 = H12.
`wellOfIndex : String -> Fin 96 -> Well` takes a 0-based `i` (well number `i + 1`):
row = i / 12 (A..H), column = i % 12 + 1. `label 65 = "F5"` is checked with `decide`.

## Rules the checker enforces

* volume inside the pipette range (P2 0.2-2, P20 2-20, P200 20-200, P1000 100-1000 uL)
* no overflow (`dst + v <= capacity`), enough liquid in the source
* must hold a tip to transfer or mix
* a tip remembers every reagent it has touched; it may enter a source only if the
  source already contains all of them (one source -> many destinations is fine)
* frozen reagents must be thawed first
* no pipetting into or out of a sealed plate
* incubation temperature 4..98 C
* centrifuge: even slot count, slot i opposite slot i + n/2, opposite loads within 5 uL,
  plates must be sealed, speed <= 15000 rpm

## What is proven (no `sorry`, no extra axioms)

* `check_sound`: `check init steps = true -> ValidProtocol init steps`
  (and `check_complete`, `check_iff`: they agree).
* `transfer_conserves`, `step_conserves`, `run_conserves`: total liquid volume over all
  containers never changes in a successful step or run (waste is a container too).
* `indexOfWell_wellOfIndex` (and the reverse): the well index round trip.
* `pcr_valid`, `dilution_valid`: the two example protocols are valid.
* Six broken protocols are proved invalid (`overflow_rejected`, `unbalanced_rejected`,
  `reusedTip_rejected`, ...), and their exact errors are checked.

All example proofs use `decide` (kernel check). `native_decide` is not used.

## What is NOT proven

This is a model, not the physical world. It says nothing about pipette error,
evaporation, liquid class (viscous, foamy liquids), dead volume, real rotor masses
(we compare volumes), temperature dynamics, timing, or chemistry (does the PCR work?).
Mixed contents are tracked only as an estimate; the total volume is exact.

Other limits:

* The spec (`ValidProtocol`) and the checker use the same rule list. `check_sound` proves the
  checker applies the rules correctly. It does not prove the rules are the right lab rules;
  people still have to review them.
* The tip rule works on reagent names, not concentrations. A tip that goes back upstream in a
  dilution (for example from A5 to A2) is not caught, because both wells hold the same reagents.
* Only the start of PCR is modeled (one 94 C step). Thermal cycling is not modeled.

## Path to training

An LLM (YLM) turns protocol text into LabLean steps. The checker gates them.
Pass/fail is a verifiable reward for GRPO: cheap and exact. On its own it is easy to game
(an empty protocol passes), so the reward must also check that the protocol does the task,
for example the final volumes and contents of the target wells.

## Sources

* Addgene, PCR protocol, protocols.io: https://dx.doi.org/10.17504/protocols.io.betyjepw
* iGEM, plate reader fluorescein calibration, protocols.io: https://dx.doi.org/10.17504/protocols.io.6zrhf56

## Files

* `LabLean/Basic.lean` - units, reagents, containers, wells, pipettes, steps, errors
* `LabLean/Semantics.lean` - rules (Props), `step`, `run`, `check`, `ValidProtocol`
* `LabLean/Theorems.lean` - soundness and volume conservation
* `LabLean/Examples/PCR.lean`, `SerialDilution.lean`, `Broken.lean`
* `Main.lean` - `lake exe lablean`
