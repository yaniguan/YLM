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

* `Spec.lean` is the spec. `StepOK` is an inductive with one constructor per kind of step; each
  lab rule is a named Prop field (`range`, `fits`, `enough`, `clean`, `thawed`, `balanced`, ...).
  `ValidProtocol s l` = `WellFormed s` (start state) and every step is `StepOK` in the state reached.
  The spec uses no checker code. It shares the lab model (`Semantics.lean`: lookup, mass, the effect
  of each step) with the checker.
* `Checker.lean` is separate `Bool` code with error messages.
* `Theorems.lean` proves they agree: `check s l = true ↔ ValidProtocol s l`.
  `Decidable (ValidProtocol s l)` is built from that theorem, so `decide` runs the checker.

## Units

Volume: 0.1 uL steps (`(2.5 : Vol)` is stored as 25; a literal like `0.25` is cut to 0.2 uL).
Reagent amounts inside a container: 1e-5 uL steps. Mass: mg. Density: mg/mL.
Temperature: whole degrees C. Time: seconds.

## Well ids (96-well plate)

Row-major, 1-based: 1 = A1, 12 = A12, 13 = B1, ..., 65 = F5, ..., 96 = H12.
`wellOfIndex` / `indexOfWell` convert between a 0-based `Fin 96` and (row, column); both round
trips are proved. A well's container id is `plate:label`, e.g. `wellId "CAL" 65 = "CAL:F5"`.

## The rules

**Start state (`WellFormed`)**: container ids unique; each container within capacity; no container
id equals a plate id; every well's plate is declared; reagent names unique; every reagent named in a
container is declared; in each container every reagent is listed once and the amounts add up to the
volume.

**Per step**:

* **Transfer**: source and destination exist and differ; tip on; pipette range
  `minV ≤ v ≤ maxV` (P2 0.2-2, P20 2-20, P200 20-200, P1000 100-1000 uL); neither is a well of a
  sealed plate; tip discipline for the source; no reagent in the source is frozen;
  `v ≤ src.vol`; `dst.vol + v ≤ dst.cap`.
* **Mix**: the container exists; tip on; range; not sealed; tip discipline; thawed; `v ≤ vol`.
* **Pick tip** only with no tip on; **drop tip** only with a tip on.
* **Incubate**: the target is a tube or a plate; 4..98 C.
* **Seal**: a known plate, not already sealed. **Thaw**: a known reagent.
* **Tip discipline (contamination)**. A tip remembers every liquid it touched since it was picked
  up: each source, and each destination *after* the dispense. A reagent's concentration in a liquid
  is the volume fraction of that reagent's stock. The tip may enter a container `c` (aspirate or mix)
  only if, for every liquid `L` it touched:
  1. every reagent in `L` is also in `c` (no foreign reagent), and
  2. for every reagent in `L` not marked `diluent`, `c` is not more concentrated than `L`
     (0.1% tolerance, only for integer rounding).

  So: one tip from one source to many empty wells is OK. Down a dilution row is OK. Back up the row
  (`Broken.backUpRow`), back into the stock, or into a different reagent's source: rejected.
  Dispensing is not checked by itself: by the rule, any residue on the tip only holds reagents of the
  source, and the source liquid goes into the destination anyway.
* **Centrifuge**: `n` slots, `n` even and `n ≥ 2`; slot `i` is opposite slot `i + n/2`. Each item is
  a tube or a whole plate (not a single well), is in at most one slot, and plates are sealed.
  Speed ≤ 15000 rpm. **Balance by mass**: item mass = plastic (`tare` for a tube, plate mass for a
  plate) + liquid mass (sum of amount × reagent density, default 1000 mg/mL, rounded down to mg);
  a plate adds all its wells. An empty slot weighs 0 mg. Opposite slots must differ by at most
  **100 mg**. This is a chosen, conservative value, not a manufacturer figure.

## What is proven (no `sorry`, no `native_decide`)

* `check_sound`, `check_complete`, `check_iff`: the checker accepts ↔ `ValidProtocol`.
  Built from `checks_iff` (per step) and `initChecks_iff` (start state).
* `step_conserves`, `steps_conserve`, `run_conserves`: total volume over all containers is unchanged
  by every valid step (waste is a container; mix moves nothing).
* `step_fits`, `steps_fit`, `valid_fits`: in a valid protocol no container ever exceeds capacity.
* `indexOfWell_wellOfIndex`, `wellOfIndex_indexOfWell`: the well index round trip.
* `pcr_valid`, `dilution_valid`: the two example protocols are valid.
* 13 broken protocols are proved invalid, each with its exact error.

## What is NOT proven or not modeled

* That these are the right lab rules. The spec is readable, but people still have to review it.
* The `diluent` flag is trusted: a reagent wrongly marked diluent skips clause 2 of the tip rule.
* Tip rule assumes contact dispensing (the tip touches the destination liquid). Free-jet dispensing
  would allow more reuse; the rule is conservative there.
* Physical world: pipette error, dead volume, evaporation, viscous liquids, timing, chemistry.
* Contents after mixing are integer estimates; total volume is exact.
* Rotor: one speed limit for tubes and plates (real plate rotors are much slower); symmetric
  odd-count loads (3 tubes at 120 degrees) are rejected; tube lids are not modeled.
* Not modeled: number of mix cycles, unsealing, seal required for hot incubation, keeping reagents
  on ice (`keepCold`), thermal cycling (the PCR example ends with one 94 C step).

## Examples

* `Examples/PCR.lean` - PCR, 4 reactions, master mix for 5 (25% overage, 43 uL left).
* `Examples/SerialDilution.lean` - fluorescein 2-fold dilution, row A only.
* `Examples/Broken.lean` - protocols that must be rejected.

Departures from the sources are listed at the top of each example file. Volumes are our
transcription of the sources; check them against the sources before use.

## Path to training

A model turns protocol text into LabLean steps; the checker gates them. Pass/fail is a cheap, exact
reward, but alone it is easy to game (an empty protocol passes), so the reward must also check that
the protocol does the task (e.g. final volumes and contents of the target wells).

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
