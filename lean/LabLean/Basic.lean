/-!
# LabLean basic types

Units (all are plain `Nat`/`Int` inside, so every check is decidable):
* `Vol`  : volume in **0.1 uL** steps. `(2.5 : Vol)` is stored as `25`.
* `Temp` : temperature in whole degrees C.
* `Time` : time in whole seconds.
* Reagent amounts inside a container: `Nat` in units of `1 / amtScale` of a `Vol` step
  (1e-5 uL), so mixing estimates stay fine-grained.
* Mass: whole mg. Density: mg/mL.
-/
namespace LabLean

/-- Volume, counted in 0.1 uL steps (`t` = tenths of a microliter). -/
structure Vol where
  t : Nat
deriving DecidableEq

/-- `(43 : Vol)` means 43 uL. -/
instance : OfNat Vol n := ⟨⟨n * 10⟩⟩
/-- `(2.5 : Vol)` means 2.5 uL. Digits after the first decimal are cut off. -/
instance : OfScientific Vol where
  ofScientific m isNeg e := if isNeg then ⟨m * 10 / 10 ^ e⟩ else ⟨m * 10 * 10 ^ e⟩

instance : Repr Vol := ⟨fun v _ => s!"{v.t / 10}.{v.t % 10}uL"⟩

/-- Reagent amounts are counted in `1 / amtScale` of a `Vol` step (1e-5 uL). -/
def amtScale : Nat := 10000

/-- Temperature in whole degrees C. -/
structure Temp where
  c : Int
deriving DecidableEq, Repr

instance : OfNat Temp n := ⟨⟨n⟩⟩

/-- Time in seconds. -/
structure Time where
  s : Nat
deriving DecidableEq, Repr

def sec (n : Nat) : Time := ⟨n⟩
def mins (n : Nat) : Time := ⟨n * 60⟩

/-! ## Reagents -/

inductive Storage where
  | frozen | thawed | onIce | roomTemp
deriving DecidableEq, Repr

/-- A named solution (a stock). Its "concentration" in a container is the volume fraction
of that stock in the container. -/
structure Reagent where
  name : String
  state : Storage
  /-- After thawing, keep it on ice (enzymes, buffers for PCR). -/
  keepCold : Bool := false
  /-- Water, PBS: used only to dilute. A tip carrying it does not raise any concentration. -/
  diluent : Bool := false
  /-- mg/mL. -/
  density : Nat := 1000
deriving DecidableEq, Repr

/-! ## Containers -/

/-- A tube, reservoir or plate well. Ids must be unique (see `WellFormed`).
`contents` lists reagent names with amounts (units of `1 / amtScale` of 0.1 uL). After mixing,
the amounts are an estimate (integer division); `vol` is the exact total. -/
structure Container where
  id : String
  /-- The plate this well belongs to (`none` for tubes). -/
  plate : Option String := none
  cap : Vol
  vol : Vol
  contents : List (String × Nat) := []
  /-- Mass of the empty tube in mg (0 for wells: the plate carries the plastic mass). -/
  tare : Nat := 0
deriving DecidableEq, Repr

/-- A 1.5 mL tube (plastic mass assumed 1 g) holding `v` of one reagent. -/
def tube (id : String) (reagent : String) (v : Vol) (cap : Vol := 1500) : Container :=
  { id, cap, vol := v, contents := [(reagent, v.t * amtScale)], tare := 1000 }

/-- A snapshot of a liquid (amounts and total volume), as remembered by a used tip. -/
structure Liquid where
  amts : List (String × Nat)
  vol : Nat
deriving DecidableEq, Repr

def Container.liquid (c : Container) : Liquid := ⟨c.contents, c.vol.t⟩

/-! ## 96-well plate addresses

Rows 0..7 are A..H, columns 0..11 are labels 1..12.

**Index convention (row-major).** Wells are counted row by row, 1-based:
number 1 = A1, 12 = A12, 13 = B1, ..., 65 = F5, ..., 96 = H12.
A value `i : Fin 96` is 0-based, so `i` stands for well number `i + 1`.
-/

structure Well where
  plate : String
  row : Fin 8
  col : Fin 12
deriving DecidableEq, Repr

def Well.label (w : Well) : String :=
  String.singleton (Char.ofNat (65 + w.row.val)) ++ toString (w.col.val + 1)

/-- 0-based row-major index to well: row = i / 12, col = i % 12. -/
def wellOfIndex (plate : String) (i : Fin 96) : Well :=
  ⟨plate, ⟨i.val / 12, by have := i.isLt; omega⟩, ⟨i.val % 12, by have := i.isLt; omega⟩⟩

def indexOfWell (w : Well) : Fin 96 :=
  ⟨w.row.val * 12 + w.col.val, by have := w.row.isLt; have := w.col.isLt; omega⟩

/-- Label of well number `n` (1-based, row-major). Out of range gives "?". -/
def label (n : Nat) : String :=
  if h : 1 ≤ n ∧ n ≤ 96 then (wellOfIndex "" ⟨n - 1, by omega⟩).label else "?"

theorem indexOfWell_wellOfIndex (p : String) (i : Fin 96) :
    indexOfWell (wellOfIndex p i) = i := by
  apply Fin.ext; simp only [indexOfWell, wellOfIndex]; omega

theorem wellOfIndex_indexOfWell (w : Well) : wellOfIndex w.plate (indexOfWell w) = w := by
  cases w with
  | mk p r c =>
    simp only [wellOfIndex, indexOfWell, Well.mk.injEq, true_and]
    constructor <;> apply Fin.ext <;> simp <;> omega

example : label 1 = "A1" := by decide
example : label 12 = "A12" := by decide
example : label 13 = "B1" := by decide
example : label 65 = "F5" := by decide
example : label 96 = "H12" := by decide

/-- Container id of well number `n` on plate `p`, e.g. `wellId "CAL" 65 = "CAL:F5"`.
Ids are unique across plates. -/
def wellId (p : String) (n : Nat) : String := p ++ ":" ++ label n

/-- Well number `n` of plate `p`, holding `v` of one reagent (empty by default). -/
def well (p : String) (n : Nat) (cap : Vol) (reagent : String := "") (v : Vol := 0) :
    Container :=
  { id := wellId p n, plate := some p, cap, vol := v,
    contents := if v.t = 0 then [] else [(reagent, v.t * amtScale)] }

/-! ## Pipettes and tips -/

structure Pipette where
  name : String
  minV : Vol
  maxV : Vol
deriving DecidableEq, Repr

def P2    : Pipette := ⟨"P2", 0.2, 2⟩
def P20   : Pipette := ⟨"P20", 2, 20⟩
def P200  : Pipette := ⟨"P200", 20, 200⟩
def P1000 : Pipette := ⟨"P1000", 100, 1000⟩

/-- `held l`: a tip is on the pipette and has touched the liquids `l` (`[]` = fresh tip). -/
inductive TipState where
  | none | held (touched : List Liquid)
deriving DecidableEq, Repr

/-! ## Steps -/

inductive Step where
  | pickTip
  | dropTip
  | transfer (p : Pipette) (src dst : String) (v : Vol)
  /-- Pipette `v` up and down `times` times in a container. -/
  | mix (p : Pipette) (c : String) (v : Vol) (times : Nat)
  /-- Incubate a container or a whole plate (by plate id). -/
  | incubate (target : String) (t : Temp) (time : Time)
  /-- Rotor slots in order; `none` = empty slot. Items are tube ids or plate ids. -/
  | centrifuge (rotor : List (Option String)) (rpm : Nat) (time : Time)
  | seal (plate : String)
  | thaw (reagent : String)
deriving Repr

inductive Error where
  | unknown (id : String)
  | notTubeOrPlate (id : String)
  | noTip
  | tipAlreadyOn
  | outOfRange (pipette : String) (v : Vol)
  | sameContainer (id : String)
  | sealed (id : String)
  | contaminated (id : String)
  | notEnough (id : String) (has want : Vol)
  | overflow (id : String) (has add cap : Vol)
  | notThawed (reagent : String)
  | badTemp (t : Temp)
  | badRotor (slots : Nat)
  | tooFast (rpm : Nat)
  | twiceInRotor
  | notSealed (plate : String)
  | unbalanced (slot1 slot2 : Nat) (mg1 mg2 : Nat)
  | alreadySealed (plate : String)
  -- start state (`WellFormed`)
  | duplicateIds
  | overfull (id : String)
  | idIsPlate (id : String)
  | unknownPlate (plate : String)
  | duplicateReagents
  | undeclaredReagent (container : String)
  | badContents (container : String)
deriving Repr, DecidableEq

structure LabState where
  cs : List Container
  reagents : List Reagent := []
  /-- Plate ids with the plastic mass of the plate (mg). -/
  plates : List (String × Nat) := []
  tip : TipState := .none
  sealed : List String := []
deriving Repr

end LabLean
