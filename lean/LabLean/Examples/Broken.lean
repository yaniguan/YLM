import LabLean.Theorems
/-!
# Broken protocols: the checker must reject all of them

Each one has a proof that it is NOT valid, the exact error, and an `#eval` of the report.
-/
namespace LabLean.Broken

def init : LabState :=
  { reagents := [ { name := "r1", state := .roomTemp }, { name := "r2", state := .roomTemp },
                  { name := "enzyme", state := .frozen } ],
    cs := [ tube "R1" 1500 "r1" 500,
            tube "R2" 1500 "r2" 500,
            tube "ENZ" 1500 "enzyme" 50,
            tube "T1" 1500 "water" 100,
            tube "T2" 1500 "water" 500,
            { id := label 1, plate := some "P", cap := 200, vol := 190,
              contents := [("water", 1900)] },
            { id := label 2, plate := some "P", cap := 200, vol := 0 } ] }

/-- 1. Overflow: A1 holds 190 uL of 200 uL, we add 20 uL. -/
def overflow : List Step := [.pickTip, .transfer P20 "R1" "A1" 20, .dropTip]

/-- 2. Unbalanced centrifuge: 4 slots, slot 0 is opposite slot 2: 100 uL vs 500 uL. -/
def unbalanced : List Step := [.centrifuge [some "T1", none, some "T2", none] 5000 (sec 60)]

/-- 3. Reused tip: the tip touched r1, then goes into the r2 tube. -/
def reusedTip : List Step :=
  [.pickTip, .transfer P200 "R1" "A2" 50, .transfer P200 "R2" "A2" 50, .dropTip]

/-- 4. Frozen reagent used without thawing. -/
def notThawed : List Step := [.pickTip, .transfer P20 "ENZ" "A2" 5, .dropTip]

/-- 5. Pipetting into a sealed plate. -/
def intoSealed : List Step := [.seal "P", .pickTip, .transfer P20 "R1" "A2" 10, .dropTip]

/-- 6. Transfer without a tip. -/
def noTip : List Step := [.transfer P20 "R1" "A2" 10]

theorem overflow_rejected : ¬ ValidProtocol init overflow := by decide
theorem unbalanced_rejected : ¬ ValidProtocol init unbalanced := by decide
theorem reusedTip_rejected : ¬ ValidProtocol init reusedTip := by decide
theorem notThawed_rejected : ¬ ValidProtocol init notThawed := by decide
theorem intoSealed_rejected : ¬ ValidProtocol init intoSealed := by decide
theorem noTip_rejected : ¬ ValidProtocol init noTip := by decide

-- The exact error each one gets.
example : errorOf init overflow = some (.overflow "A1" 190 20 200) := by decide
example : errorOf init unbalanced = some (.unbalanced 0 2 100 500) := by decide
example : errorOf init reusedTip = some (.contaminated "R2") := by decide
example : errorOf init notThawed = some (.notThawed "enzyme") := by decide
example : errorOf init intoSealed = some (.sealed "A2") := by decide
example : errorOf init noTip = some .noTip := by decide

#eval run init overflow |>.map (fun _ => ())   -- Except.error (Error.overflow "A1" 190.0uL 20.0uL 200.0uL)
#eval report init overflow
#eval report init unbalanced
#eval report init reusedTip
#eval report init notThawed
#eval report init intoSealed
#eval report init noTip

-- Claiming a broken protocol is valid does not compile (decide finds it is false):
-- theorem overflow_ok : ValidProtocol init overflow := by decide

end LabLean.Broken
