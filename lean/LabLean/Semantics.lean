import LabLean.Basic
/-!
# Semantics: the lab model and what each step does

Shared by the spec (`Spec.lean`) and the checker (`Checker.lean`): container lookup,
concentration, mass, and `apply` (the effect of a step). Nothing here checks a rule.
-/
namespace LabLean

/-- Opposite rotor positions may differ by at most this many mg (0.1 g). -/
def rotorTol : Nat := 100
def maxRpm : Nat := 15000
/-- Tip rule tolerance: a liquid counts as "not more concentrated" up to 1%. -/
def concTol : Nat := 1

/-! ## Containers -/

def findC (id : String) : List Container → Option Container
  | [] => none
  | c :: l => if c.id = id then some c else findC id l

/-- Change the first container with this id. -/
def modifyC (id : String) (f : Container → Container) : List Container → List Container
  | [] => []
  | c :: l => if c.id = id then f c :: l else c :: modifyC id f l

/-- Sum of all volumes (0.1 uL). -/
def total : List Container → Nat
  | [] => 0
  | c :: l => c.vol.t + total l

def names (c : Container) : List String := c.contents.map Prod.fst
def Liquid.names (L : Liquid) : List String := L.amts.map Prod.fst

/-- Amount of reagent `r` in a contents list (0 if absent). -/
def amt (l : List (String × Nat)) (r : String) : Nat := (l.lookup r).getD 0

def touched : TipState → List Liquid
  | .held l => l
  | .none => []

def plateIds (s : LabState) : List String := s.plates.map Prod.fst

/-! ## Mass (for rotor balance) -/

def densityOf (s : LabState) (r : String) : Nat :=
  match s.reagents.find? (·.name == r) with
  | some x => x.density
  | none => 1000

/-- mg = amount (1e-5 uL) * density (mg/mL) / 1e8. -/
def massOf (s : LabState) (c : Container) : Nat :=
  c.tare + (c.contents.map fun (r, x) => x * densityOf s r / 100000000).sum

/-- Mass of a tube, or of a plate (plastic + all its wells). `none` if unknown. -/
def itemMass (s : LabState) (id : String) : Option Nat :=
  match findC id s.cs with
  | some c => some (massOf s c)
  | none =>
    match s.plates.lookup id with
    | some tare => some (tare + ((s.cs.filter (·.plate == some id)).map (massOf s)).sum)
    | none => none

/-- Mass in rotor slot `i` (empty slot = 0). -/
def slotMass (s : LabState) (rotor : List (Option String)) (i : Nat) : Nat :=
  match rotor[i]? with
  | some (some id) => (itemMass s id).getD 0
  | _ => 0

/-! ## Effects -/

/-- Take `v` out. Amounts are rounded down, so the concentration never goes up. -/
def takeOut (v : Nat) (c : Container) : Container :=
  { c with vol := ⟨c.vol.t - v⟩,
           contents := c.contents.map fun (r, x) => (r, x * (c.vol.t - v) / c.vol.t) }

def addAmt (r : String) (x : Nat) : List (String × Nat) → List (String × Nat)
  | [] => [(r, x)]
  | (q, y) :: l => if q = r then (q, y + x) :: l else (q, y) :: addAmt r x l

/-- Put `v` of the liquid from `a` into `c`. -/
def putIn (a : Container) (v : Nat) (c : Container) : Container :=
  { c with vol := ⟨c.vol.t + v⟩,
           contents := a.contents.foldl (fun acc (r, x) => addAmt r (x * v / a.vol.t) acc)
                         c.contents }

/-- The tip touches more liquids (duplicates are kept once). -/
def touch (s : LabState) (ls : List Liquid) : TipState :=
  .held (ls.foldl (fun acc L => if acc.contains L then acc else L :: acc) (touched s.tip))

def apply (s : LabState) : Step → LabState
  | .pickTip => { s with tip := .held [] }
  | .dropTip => { s with tip := .none }
  | .transfer _ src dst v =>
    match findC src s.cs, findC dst s.cs with
    | some a, some b =>
      -- the tip touches the source, then the destination after the dispense
      { s with cs := modifyC dst (putIn a v.t) (modifyC src (takeOut v.t) s.cs),
               tip := touch s [a.liquid, (putIn a v.t b).liquid] }
    | _, _ => s
  | .mix _ c _ _ =>
    match findC c s.cs with
    | some a => { s with tip := touch s [a.liquid] }
    | none => s
  | .incubate .. => s
  | .centrifuge .. => s
  | .seal p => { s with sealed := p :: s.sealed }
  | .thaw r =>
    { s with reagents := s.reagents.map fun x =>
        if x.name = r ∧ x.state = .frozen
        then { x with state := if x.keepCold then .onIce else .thawed } else x }

end LabLean
