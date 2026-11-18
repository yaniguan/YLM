import LabLean.Basic
/-!
# Semantics: rules, step function, checker

Each step has a list of `Check`s. A check is a decidable `Prop` plus the error to report
when it fails. `StepOK s st` (the spec) says all checks hold. `step` (the checker) returns
the first failing check's error, or the new state.

Rules (stated simply):
* **Pipette range**: `minV ≤ v ≤ maxV`.
* **No overflow**: `dst.vol + v ≤ dst.cap`. **Enough liquid**: `v ≤ src.vol`.
* **Tip discipline**: you must hold a tip to transfer or mix. A tip remembers every reagent
  in every container it has entered (source and destination). A tip may enter a source
  (aspirate or mix) only if that source already contains every reagent on the tip.
  So one source to many destinations with one tip is fine, but a tip that touched
  reagent X cannot go into a tube of reagent Y.
* **Thawed**: every reagent in the source must not be frozen.
* **Sealed**: no pipetting into or out of a well of a sealed plate.
* **Incubate**: temperature must be in `4..98` C (heat block / thermocycler range).
* **Centrifuge**: the rotor has an even number of slots `n ≥ 2`; slot `i` is opposite
  slot `i + n/2`. Opposite slots must hold volumes within `rotorTol` (empty slot = 0).
  A plate weighs the sum of its wells. Plates must be sealed. Speed `≤ maxRpm`.
-/
namespace LabLean

def rotorTol : Nat := 50      -- 5.0 uL
def maxRpm : Nat := 15000

/-! ## Helpers on the container list -/

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

def tipNames : TipState → List String
  | .used l => l
  | _ => []

def union (a b : List String) : List String := a ++ b.filter (fun x => !a.contains x)

def isSealed (s : LabState) (c : Container) : Bool :=
  match c.plate with
  | some p => s.sealed.contains p
  | none => false

def isPlate (s : LabState) (p : String) : Bool := s.cs.any (·.plate == some p)

def plateVol (s : LabState) (p : String) : Nat := total (s.cs.filter (·.plate == some p))

/-- Volume of a tube, or of a whole plate. `none` if unknown. -/
def weight (s : LabState) (id : String) : Option Nat :=
  match findC id s.cs with
  | some c => some c.vol.t
  | none => if isPlate s id then some (plateVol s id) else none

def slotVol (s : LabState) (rotor : List (Option String)) (i : Nat) : Nat :=
  match rotor[i]? with
  | some (some id) => (weight s id).getD 0
  | _ => 0

/-! ## Preconditions as Props (all decidable) -/

abbrev InRange (p : Pipette) (v : Vol) : Prop := p.minV.t ≤ v.t ∧ v.t ≤ p.maxV.t
abbrev NoOverflow (c : Container) (v : Vol) : Prop := c.vol.t + v.t ≤ c.cap.t
abbrev Enough (c : Container) (v : Vol) : Prop := v.t ≤ c.vol.t
abbrev HasTip (s : LabState) : Prop := s.tip ≠ .none
abbrev TipClean (s : LabState) (c : Container) : Prop := ∀ r ∈ tipNames s.tip, r ∈ names c
abbrev SafeTemp (t : Temp) : Prop := 4 ≤ t.c ∧ t.c ≤ 98
abbrev Balanced (a b : Nat) : Prop := a ≤ b + rotorTol ∧ b ≤ a + rotorTol

structure Check where
  prop : Prop
  [dec : Decidable prop]
  err : Error

instance (c : Check) : Decidable c.prop := c.dec

def need (p : Prop) [Decidable p] (e : Error) : Check := ⟨p, e⟩

/-- All reagents in container `c` must be thawed. -/
def thawChecks (s : LabState) (c : Container) : List Check :=
  s.reagents.map fun r => need (r.name ∈ names c → r.Usable) (.notThawed r.name)

/-- Checks for entering a source with a tip (aspirate or mix). -/
def sourceChecks (s : LabState) (p : Pipette) (id : String) (a : Container) (v : Vol) :
    List Check :=
  [ need (HasTip s) .noTip,
    need (InRange p v) (.outOfRange p.name v),
    need (isSealed s a = false) (.sealed id),
    need (TipClean s a) (.contaminated id),
    need (Enough a v) (.notEnough id a.vol v) ] ++ thawChecks s a

def checks (s : LabState) : Step → List Check
  | .pickTip => [need (s.tip = .none) .tipAlreadyOn]
  | .dropTip => [need (HasTip s) .noTip]
  | .transfer p src dst v =>
    match findC src s.cs, findC dst s.cs with
    | some a, some b =>
      sourceChecks s p src a v ++
      [ need (src ≠ dst) (.sameContainer src),
        need (isSealed s b = false) (.sealed dst),
        need (NoOverflow b v) (.overflow dst b.vol v b.cap) ]
    | none, _ => [need False (.unknown src)]
    | _, none => [need False (.unknown dst)]
  | .mix p c v _ =>
    match findC c s.cs with
    | some a => sourceChecks s p c a v
    | none => [need False (.unknown c)]
  | .incubate x t _ =>
    [ need ((weight s x).isSome) (.unknown x),
      need (SafeTemp t) (.badTemp t) ]
  | .centrifuge rotor rpm _ =>
    let n := rotor.length
    [ need (n % 2 = 0 ∧ 0 < n) (.badRotor n),
      need (rpm ≤ maxRpm) (.tooFast rpm) ] ++
    (rotor.filterMap id).flatMap (fun x =>
      [ need ((weight s x).isSome) (.unknown x),
        need ((findC x s.cs).isNone → x ∈ s.sealed) (.notSealed x) ]) ++
    (List.range (n / 2)).map (fun i =>
      let a := slotVol s rotor i
      let b := slotVol s rotor (i + n / 2)
      need (Balanced a b) (.unbalanced i (i + n / 2) ⟨a⟩ ⟨b⟩))
  | .seal p =>
    [ need (isPlate s p) (.unknown p),
      need (p ∉ s.sealed) (.alreadySealed p) ]
  | .thaw r => [need (s.reagents.any (·.name == r)) (.unknown r)]

/-! ## Effects (what a step does, assuming its checks pass) -/

def takeOut (v : Nat) (c : Container) : Container :=
  { c with vol := ⟨c.vol.t - v⟩,
           contents := c.contents.map fun (r, x) => (r, x - x * v / c.vol.t) }

def addAmt (r : String) (x : Nat) : List (String × Nat) → List (String × Nat)
  | [] => [(r, x)]
  | (q, y) :: l => if q = r then (q, y + x) :: l else (q, y) :: addAmt r x l

/-- Put `v` of the liquid from `a` into `c`. -/
def putIn (a : Container) (v : Nat) (c : Container) : Container :=
  { c with vol := ⟨c.vol.t + v⟩,
           contents := a.contents.foldl (fun acc (r, x) => addAmt r (x * v / a.vol.t) acc)
                         c.contents }

def touch (s : LabState) (l : List String) : TipState := .used (union (tipNames s.tip) l)

def apply (s : LabState) : Step → LabState
  | .pickTip => { s with tip := .fresh }
  | .dropTip => { s with tip := .none }
  | .transfer _ src dst v =>
    match findC src s.cs, findC dst s.cs with
    | some a, some b =>
      { s with cs := modifyC dst (putIn a v.t) (modifyC src (takeOut v.t) s.cs),
               tip := touch s (union (names a) (names b)) }
    | _, _ => s
  | .mix _ c _ _ =>
    match findC c s.cs with
    | some a => { s with tip := touch s (names a) }
    | none => s
  | .incubate .. => s
  | .centrifuge .. => s
  | .seal p => { s with sealed := p :: s.sealed }
  | .thaw r =>
    { s with reagents := s.reagents.map fun x =>
        if x.name = r ∧ x.state = .frozen
        then { x with state := if x.keepCold then .onIce else .thawed } else x }

/-! ## Spec -/

/-- All preconditions of one step hold. -/
def StepOK (s : LabState) (st : Step) : Prop := ∀ c ∈ checks s st, c.prop

instance (s : LabState) (st : Step) : Decidable (StepOK s st) :=
  inferInstanceAs (Decidable (∀ c ∈ checks s st, c.prop))

/-- Every step's preconditions hold in the state reached so far. -/
def ValidProtocol : LabState → List Step → Prop
  | _, [] => True
  | s, st :: rest => StepOK s st ∧ ValidProtocol (apply s st) rest

instance decValid : (s : LabState) → (l : List Step) → Decidable (ValidProtocol s l)
  | _, [] => isTrue trivial
  | s, st :: rest =>
    have := decValid (apply s st) rest
    inferInstanceAs (Decidable (StepOK s st ∧ ValidProtocol (apply s st) rest))

/-! ## Checker (with error messages) -/

def firstFail : List Check → Option Error
  | [] => none
  | c :: l => if c.prop then firstFail l else some c.err

def step (s : LabState) (st : Step) : Except Error LabState :=
  match firstFail (checks s st) with
  | some e => .error e
  | none => .ok (apply s st)

def run (s : LabState) : List Step → Except Error LabState
  | [] => .ok s
  | st :: rest =>
    match step s st with
    | .error e => .error e
    | .ok s' => run s' rest

/-- The gate: `true` means the protocol may go to the robot. -/
def check (s : LabState) (l : List Step) : Bool :=
  match run s l with
  | .ok _ => true
  | .error _ => false

/-- The error of a rejected protocol, `none` if it passes. -/
def errorOf (s : LabState) (l : List Step) : Option Error :=
  match run s l with
  | .ok _ => none
  | .error e => some e

/-- Short result text, used by `#eval` and the executable. -/
def report (s : LabState) (l : List Step) : String :=
  match run s l with
  | .ok _ => "OK"
  | .error e => s!"REJECTED: {repr e}"

end LabLean
