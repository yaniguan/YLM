# LabLean

LabLean is a small Lean 4 language for bio lab protocols, with a checker.
A protocol is a list of steps (pick tip, transfer, mix, incubate, centrifuge, seal, thaw, ...).
The checker must accept a protocol before it goes to the robot:

    text -> protocol (LabLean steps) -> checker -> robot

No Mathlib; core Lean only. Toolchain: `leanprover/lean4:v4.36.0`.

## Build

    source ~/.elan/env
    lake build          # library, examples and all proofs
    lake exe lablean    # prints the checker result for every example

## Design

* `Spec.lean` is the spec. `StepOK` is an inductive with one constructor per kind of step, and
  each lab rule is a named field (`range`, `fits`, `enough`, `clean`, `thawed`, `balanced`, ...).
  `ValidProtocol s l` = `WellFormed s` (start state) and every step is `StepOK` in the state reached.
* `Checker.lean` is separate `Bool` code with error messages.
* `Theorems.lean` proves they agree: `check s l = true ↔ ValidProtocol s l`.
  `decide` on `ValidProtocol` runs the checker; this is correct because of that theorem.

## Units

* Volume: 0.1 uL steps (`(2.5 : Vol)` is stored as 25). Reagent amounts inside a container:
  1e-5 uL steps. Mass: mg. Density: mg/mL. Temperature: whole degrees C. Time: seconds.

## Well ids (96-well plate)

Row-major, 1-based: 1 = A1, 12 = A12, 13 = B1, ..., 65 = F5, ..., 96 = H12.
`wellOfIndex` / `indexOfWell` convert between a 0-based `Fin 96` and (row, column); both round
trips are proved. A well's container id is `plate:label`, e.g. `wellId "CAL" 65 = "CAL:F5"`.

## The rules

Start state (`WellFormed`): container ids unique; each container within capacity; no container id
equals a plate id; every well's plate is declared.

Per step:

* **Pipette range**: `minV ≤ v ≤ maxV` (P2 0.2-2, P20 2-20, P200 20-200, P1000 100-1000 uL).
* **No overflow**: `dst.vol + v ≤ dst.cap`. **Enough**: `v ≤ src.vol` (also for `mix`).
* **Tip on** to transfer or mix; no second tip on top of a tip.
* **Thawed**: no reagent in the source is frozen.
* **Not sealed**: no pipetting into or out of a well of a sealed plate.
* **Incubate**: 4..98 C.
* **Tip discipline (contamination)**. A tip remembers every liquid it touched since it was
  picked up: each source, and each destination *after* the dispense. A reagent's
  concentration in a liquid is the volume fraction of that reagent's stock. The tip may
  enter a container `c` (aspirate or mix) only if, for every liquid `L` it touched:
  1. every reagent in `L` is also in `c` (no foreign reagent), and
  2. for every reagent in `L` that is not a diluent (water, PBS), `c` is not more concentrated
     than `L` (1% tolerance, for rounding).

  So: the same tip from one source to many empty wells is OK. Down a dilution row is OK.
  Back up the row, back into the stock, into a different reagent, or back to the source
  after touching a well that held other liquid: rejected. Dispensing is not checked by
  itself, because the dispensed liquid comes from a source that was already checked.
* **Centrifuge**: even slot count `n ≥ 2`, and slot `i` is opposite slot `i + n/2`. Each item is in
  at most one slot, and every item is known. Plates must be sealed. Speed ≤ 15000 rpm.
  **Balance by mass**: the mass of an item is its plastic (`tare` for tubes; plate mass for plates)
  plus the liquid mass (amount × reagent density, default 1000 mg/mL). For a plate, that is the
  sum over its wells. Opposite slots must be within **100 mg (0.1 g)**. An empty slot weighs 0.

## What is proven (no `sorry`, no `native_decide`)

* `check_sound`, `check_complete`, `check_iff`: the checker accepts ↔ `ValidProtocol`.
  `checks_iff` (per step) and `initChecks_iff` (start state) are the building blocks.
* `step_conserves`, `steps_conserve`, `run_conserves`: total volume over all containers is
  unchanged by every valid step (liquid only moves; waste is a container; mix moves nothing).
* `step_fits`, `steps_fit`, `valid_fits`: in a valid protocol no container ever exceeds capacity.
* `indexOfWell_wellOfIndex`, `wellOfIndex_indexOfWell`: the well index round trip;
  `label 65 = "F5"` by `decide`.
* `pcr_valid`, `dilution_valid`: the two example protocols are valid.
* Nine broken protocols are proved invalid, and each one's exact error is checked
  (overflow, unbalanced by volume, unbalanced by density, reused tip, not thawed, sealed plate,
  no tip, tip back to stock, duplicate ids).

## What is NOT proven

* That the rules are the right lab rules. The spec is written separately from the checker and is
  readable, but people still have to review it.
* The physical world: pipette error, dead volume, evaporation, liquid class (viscous, foamy),
  temperature dynamics, timing, chemistry (does the PCR work?).
* Contents after mixing are integer estimates; total volume is exact. Mass has small rounding.
* A tip that touched only a diluent may enter any liquid that holds that diluent.
* Not modeled: number of mix cycles, unsealing, seal required for hot incubation, keeping
  reagents on ice (`keepCold`), thermal cycling (the PCR example ends with one 94 C step).

## Examples

* `Examples/PCR.lean` - Addgene PCR, 4 reactions, master mix for 5 (25% overage, 43 uL left).
* `Examples/SerialDilution.lean` - iGEM fluorescein 2-fold dilution, row A only.
* `Examples/Broken.lean` - protocols that must be rejected.

Departures from the sources are listed at the top of each example file.

## Path to training

A model turns protocol text into LabLean steps; the checker gates them. Pass/fail is a cheap,
exact reward, but on its own it is easy to game (an empty protocol passes), so the reward must also
check that the protocol does the task (e.g. final volumes and contents of the target wells).

## Sources

* Addgene, PCR protocol, protocols.io: https://dx.doi.org/10.17504/protocols.io.betyjepw
* iGEM, plate reader fluorescein calibration, protocols.io: https://dx.doi.org/10.17504/protocols.io.6zrhf56

## Files

* `LabLean/Basic.lean` - units, reagents, containers, wells, pipettes, steps, errors
* `LabLean/Semantics.lean` - lookup, concentration, mass, effects of steps (`apply`)
* `LabLean/Spec.lean` - the rules as Props: `StepOK`, `StepsOK`, `WellFormed`, `ValidProtocol`
* `LabLean/Checker.lean` - Bool checks with errors: `step`, `run`, `check`, `report`
* `LabLean/Theorems.lean` - checker ↔ spec, volume conservation, capacity invariant
* `LabLean/Examples/` - PCR, serial dilution, broken protocols
* `Main.lean` - `lake exe lablean`
