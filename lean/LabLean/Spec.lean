import LabLean.Semantics
/-!
# Spec: what a valid protocol is, rule by rule

`StepOK s st` has one constructor per kind of step; each lab rule is a named field.
`ValidProtocol s l` = the start state is well formed, and every step is OK in the state
reached so far. This file has no checker code; `Checker.lean` computes the same thing with
`Bool`s and error messages, and `Theorems.lean` proves the two agree.
-/
namespace LabLean

/-! ## Rules as Props -/

def HasTip (s : LabState) : Prop := s.tip ≠ .none
/-- Pipette range: `minV ≤ v ≤ maxV`. -/
def InRange (p : Pipette) (v : Vol) : Prop := p.minV.t ≤ v.t ∧ v.t ≤ p.maxV.t
/-- No overflow: the destination stays within its capacity. -/
def NoOverflow (c : Container) (v : Vol) : Prop := c.vol.t + v.t ≤ c.cap.t
/-- Enough source volume. -/
def Enough (c : Container) (v : Vol) : Prop := v.t ≤ c.vol.t
/-- The container is a well of a sealed plate. -/
def IsSealed (s : LabState) (c : Container) : Prop := ∃ p, c.plate = some p ∧ p ∈ s.sealed
/-- Every reagent in the container is thawed (not frozen). -/
def Thawed (s : LabState) (c : Container) : Prop :=
  ∀ x ∈ s.reagents, x.name ∈ names c → x.state ≠ .frozen
def IsDiluent (s : LabState) (r : String) : Prop :=
  ∃ x ∈ s.reagents, x.name = r ∧ x.diluent = true
/-- `c` is not more concentrated in `r` than `L` (up to `concTol` %):
`amt c r / c.vol ≤ (1 + concTol/100) * amt L r / L.vol`, cross-multiplied. -/
def NotMoreConc (r : String) (L c : Liquid) : Prop :=
  100 * amt c.amts r * L.vol ≤ (100 + concTol) * amt L.amts r * c.vol
/-- A tip carrying liquid `L` may enter liquid `c`: every reagent of `L` is already in `c`,
and `c` is downstream of `L` for every non-diluent reagent of `L`. -/
def MayEnter (s : LabState) (L c : Liquid) : Prop :=
  ∀ r ∈ L.names, r ∈ c.names ∧ (IsDiluent s r ∨ NotMoreConc r L c)
/-- Tip discipline: every liquid the tip has touched may enter `c`. -/
def TipClean (s : LabState) (c : Container) : Prop :=
  ∀ L ∈ touched s.tip, MayEnter s L c.liquid
/-- Heat block / thermocycler range. -/
def SafeTemp (t : Temp) : Prop := 4 ≤ t.c ∧ t.c ≤ 98
/-- Rotor balance: masses (mg) of opposite positions within `rotorTol`. -/
def Balanced (a b : Nat) : Prop := a ≤ b + rotorTol ∧ b ≤ a + rotorTol
def IsPlate (s : LabState) (p : String) : Prop := p ∈ plateIds s
def Known (s : LabState) (x : String) : Prop := (∃ c ∈ s.cs, c.id = x) ∨ IsPlate s x

/-! ## One step -/

inductive StepOK : LabState → Step → Prop
  | pickTip (noTip : s.tip = .none) : StepOK s .pickTip
  | dropTip (tip : HasTip s) : StepOK s .dropTip
  | transfer (hsrc : findC src s.cs = some a) (hdst : findC dst s.cs = some b)
      (distinct : src ≠ dst) (tip : HasTip s) (range : InRange p v)
      (srcOpen : ¬ IsSealed s a) (dstOpen : ¬ IsSealed s b)
      (clean : TipClean s a) (thawed : Thawed s a)
      (enough : Enough a v) (fits : NoOverflow b v) :
      StepOK s (.transfer p src dst v)
  | mix (hc : findC c s.cs = some a) (tip : HasTip s) (range : InRange p v)
      (isOpen : ¬ IsSealed s a) (clean : TipClean s a) (thawed : Thawed s a)
      (enough : Enough a v) :
      StepOK s (.mix p c v n)
  | incubate (known : Known s x) (safe : SafeTemp t) : StepOK s (.incubate x t time)
  | centrifuge (even : rotor.length % 2 = 0) (nonempty : 0 < rotor.length)
      (speed : rpm ≤ maxRpm) (once : (rotor.filterMap id).Nodup)
      (known : ∀ x ∈ rotor.filterMap id, Known s x)
      (platesSealed : ∀ x ∈ rotor.filterMap id, IsPlate s x → x ∈ s.sealed)
      (balanced : ∀ i, i < rotor.length / 2 →
        Balanced (slotMass s rotor i) (slotMass s rotor (i + rotor.length / 2))) :
      StepOK s (.centrifuge rotor rpm time)
  | seal (plate : IsPlate s p) (notYet : p ∉ s.sealed) : StepOK s (.seal p)
  | thaw (known : ∃ x ∈ s.reagents, x.name = r) : StepOK s (.thaw r)

/-! ## Whole protocol -/

/-- Every step is OK in the state reached so far. -/
inductive StepsOK : LabState → List Step → Prop
  | nil : StepsOK s []
  | cons : StepOK s st → StepsOK (apply s st) rest → StepsOK s (st :: rest)

/-- The start state is physically possible and ids are unambiguous. -/
structure WellFormed (s : LabState) : Prop where
  uniqueIds : (s.cs.map (·.id)).Nodup
  withinCap : ∀ c ∈ s.cs, c.vol.t ≤ c.cap.t
  notPlateId : ∀ c ∈ s.cs, c.id ∉ plateIds s
  plateKnown : ∀ c ∈ s.cs, ∀ p, c.plate = some p → p ∈ plateIds s

def ValidProtocol (s : LabState) (l : List Step) : Prop := WellFormed s ∧ StepsOK s l

end LabLean
