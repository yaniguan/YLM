import LabLean.Semantics
/-!
# Checker: Bool tests with error messages

For each step, `checks` lists `(test, error)` pairs; `step` reports the first failing one.
The tests are plain `Bool` code. `Theorems.lean` proves they match the spec in `Spec.lean`.
-/
namespace LabLean

def sealedB (s : LabState) (c : Container) : Bool :=
  match c.plate with
  | some p => s.sealed.contains p
  | none => false

def diluentB (s : LabState) (r : String) : Bool :=
  s.reagents.any fun x => x.name == r && x.diluent

def mayEnterB (s : LabState) (L c : Liquid) : Bool :=
  L.names.all fun r =>
    c.names.contains r &&
    (diluentB s r || 100 * amt c.amts r * L.vol ≤ (100 + concTol) * amt L.amts r * c.vol)

def tipCleanB (s : LabState) (c : Container) : Bool :=
  (touched s.tip).all fun L => mayEnterB s L c.liquid

def knownB (s : LabState) (x : String) : Bool :=
  (findC x s.cs).isSome || (plateIds s).contains x

def nodupB : List String → Bool
  | [] => true
  | x :: l => !l.contains x && nodupB l

def hasTipB (s : LabState) : Bool :=
  match s.tip with
  | .none => false
  | .held _ => true

def thawChecks (s : LabState) (c : Container) : List (Bool × Error) :=
  s.reagents.map fun x => (!(names c).contains x.name || x.state != .frozen, .notThawed x.name)

/-- Checks for entering a container with a tip (aspirate or mix). -/
def enterChecks (s : LabState) (p : Pipette) (id : String) (a : Container) (v : Vol) :
    List (Bool × Error) :=
  [ (hasTipB s, .noTip),
    (p.minV.t ≤ v.t && v.t ≤ p.maxV.t, .outOfRange p.name v),
    (!sealedB s a, .sealed id),
    (tipCleanB s a, .contaminated id),
    (v.t ≤ a.vol.t, .notEnough id a.vol v) ] ++ thawChecks s a

def checks (s : LabState) : Step → List (Bool × Error)
  | .pickTip => [(!hasTipB s, .tipAlreadyOn)]
  | .dropTip => [(hasTipB s, .noTip)]
  | .transfer p src dst v =>
    match findC src s.cs, findC dst s.cs with
    | some a, some b =>
      (src != dst, .sameContainer src) :: enterChecks s p src a v ++
      [ (!sealedB s b, .sealed dst),
        (b.vol.t + v.t ≤ b.cap.t, .overflow dst b.vol v b.cap) ]
    | none, _ => [(false, .unknown src)]
    | _, none => [(false, .unknown dst)]
  | .mix p c v _ =>
    match findC c s.cs with
    | some a => enterChecks s p c a v
    | none => [(false, .unknown c)]
  | .incubate x t _ => [(knownB s x, .unknown x), (4 ≤ t.c && t.c ≤ 98, .badTemp t)]
  | .centrifuge rotor rpm _ =>
    let n := rotor.length
    let items := rotor.filterMap id
    [ (n % 2 == 0 && 0 < n, .badRotor n),
      (rpm ≤ maxRpm, .tooFast rpm),
      (nodupB items, .twiceInRotor) ] ++
    items.map (fun x => (knownB s x, .unknown x)) ++
    items.map (fun x => (!(plateIds s).contains x || s.sealed.contains x, .notSealed x)) ++
    (List.range (n / 2)).map (fun i =>
      let a := slotMass s rotor i
      let b := slotMass s rotor (i + n / 2)
      (a ≤ b + rotorTol && b ≤ a + rotorTol, .unbalanced i (i + n / 2) a b))
  | .seal p => [((plateIds s).contains p, .unknown p), (!s.sealed.contains p, .alreadySealed p)]
  | .thaw r => [(s.reagents.any (·.name == r), .unknown r)]

def plateKnownB (s : LabState) (c : Container) : Bool :=
  match c.plate with
  | some p => (plateIds s).contains p
  | none => true

/-- Checks on the start state (`WellFormed`). -/
def initChecks (s : LabState) : List (Bool × Error) :=
  (nodupB (s.cs.map (·.id)), .duplicateIds) ::
  s.cs.map (fun c => (c.vol.t ≤ c.cap.t, .overfull c.id)) ++
  s.cs.map (fun c => (!(plateIds s).contains c.id, .idIsPlate c.id)) ++
  s.cs.map (fun c => (plateKnownB s c, .unknownPlate (c.plate.getD "")))

def firstErr : List (Bool × Error) → Option Error
  | [] => none
  | (b, e) :: l => if b then firstErr l else some e

def step (s : LabState) (st : Step) : Except Error LabState :=
  match firstErr (checks s st) with
  | some e => .error e
  | none => .ok (apply s st)

def runSteps (s : LabState) : List Step → Except Error LabState
  | [] => .ok s
  | st :: rest =>
    match step s st with
    | .error e => .error e
    | .ok s' => runSteps s' rest

/-- Check the start state, then run every step. -/
def run (s : LabState) (l : List Step) : Except Error LabState :=
  match firstErr (initChecks s) with
  | some e => .error e
  | none => runSteps s l

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
