/-!
# LabLean basic types

Units (all are plain `Nat`/`Int` inside, so every check is decidable):
* `Vol`  : volume in **0.1 uL** steps. `(2.5 : Vol)` is stored as `25`.
* `Temp` : temperature in whole degrees C.
* `Time` : time in whole seconds.
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

structure Reagent where
  name : String
  state : Storage
  /-- After thawing, keep it on ice (enzymes, buffers for PCR). -/
  keepCold : Bool := false
deriving DecidableEq, Repr

/-- The "needs thawed" rule: a reagent can be pipetted only if it is not frozen. -/
abbrev Reagent.Usable (r : Reagent) : Prop := r.state ≠ .frozen

/-! ## Containers -/

/-- A tube, reservoir or plate well.
`contents` lists reagent names with amounts (0.1 uL). After mixing, the amounts are
an estimate (integer division); `vol` is the exact total that the checker uses. -/
structure Container where
  id : String
  /-- The plate this well belongs to (`none` for tubes). -/
  plate : Option String := none
  cap : Vol
  vol : Vol
  contents : List (String × Nat) := []
deriving DecidableEq, Repr

/-- A tube holding `v` of one reagent. -/
def tube (id : String) (cap : Vol) (reagent : String) (v : Vol) : Container :=
  { id, cap, vol := v, contents := [(reagent, v.t)] }

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

/-- `used l`: the tip has touched liquid holding the reagents in `l`. -/
inductive TipState where
  | none | fresh | used (touched : List String)
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
  | notSealed (plate : String)
  | unbalanced (slot1 slot2 : Nat) (vol1 vol2 : Vol)
  | alreadySealed (plate : String)
deriving Repr, DecidableEq

structure LabState where
  cs : List Container
  reagents : List Reagent := []
  tip : TipState := .none
  sealed : List String := []
deriving Repr

end LabLean
