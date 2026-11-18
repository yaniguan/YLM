import LabLean.Examples.SerialDilution
/-!
# Broken protocols: the checker must reject all of them

Each one has a proof that it is NOT valid, the exact error, and an `#eval` of the report.
-/
namespace LabLean.Broken

def init : LabState :=
  { reagents := [ { name := "r1", state := .roomTemp }, { name := "r2", state := .roomTemp },
                  { name := "enzyme", state := .frozen },
                  { name := "water", state := .roomTemp, diluent := true },
                  { name := "glycerol50", state := .roomTemp, density := 1130 } ],
    plates := [("P", 14000)],
    cs := [ tube "R1" "r1" 500,
            tube "R2" "r2" 500,
            tube "ENZ" "enzyme" 50,
            tube "T1" "water" 100,
            tube "T2" "water" 500,
            tube "W1" "water" 1000,
            tube "G1" "glycerol50" 1000,
            well "P" 1 200 "water" 190,
            well "P" 2 200 ] }

def A1 : String := wellId "P" 1
def A2 : String := wellId "P" 2

/-- 1. Overflow: A1 holds 190 uL of 200 uL, we add 20 uL. -/
def overflow : List Step := [.pickTip, .transfer P20 "R1" A1 20, .dropTip]

/-- 2. Unbalanced: 4 slots, slot 0 is opposite slot 2: 1.1 g vs 1.5 g. -/
def unbalanced : List Step := [.centrifuge [some "T1", none, some "T2", none] 5000 (sec 60)]

/-- 3. Same volume, different mass: 1000 uL water vs 1000 uL 50% glycerol (0.13 g apart). -/
def denseUnbalanced : List Step := [.centrifuge [some "W1", some "G1"] 5000 (sec 60)]

/-- 4. Reused tip: the tip touched r1, then goes into the r2 tube. -/
def reusedTip : List Step :=
  [.pickTip, .transfer P200 "R1" A2 50, .transfer P200 "R2" A2 50, .dropTip]

/-- 5. Frozen reagent used without thawing. -/
def notThawed : List Step := [.pickTip, .transfer P20 "ENZ" A2 5, .dropTip]

/-- 6. Pipetting into a sealed plate. -/
def intoSealed : List Step := [.seal "P", .pickTip, .transfer P20 "R1" A2 10, .dropTip]

/-- 7. Transfer without a tip. -/
def noTip : List Step := [.transfer P20 "R1" A2 10]

/-- 8. Serial dilution, but the used tip goes back into the stock: it carries the 1/2 dilution
from column 2 into the full-strength stock. Caught by tip clause 1 (the stock tube is modeled as
"fluorescein" only, column 2 also holds PBS). -/
def backToStock : List Step :=
  open SerialDilution in
  fillPBS ++ fillStock ++
  [.pickTip, .transfer P200 (col 1) (col 2) 100, .mix P200 (col 2) 100 3,
   .transfer P200 "FLUO" (col 3) 20, .dropTip]

/-- 9. Serial dilution, but the tip goes back up the row: after column 3 (1/4) it mixes
column 2 (1/2). Both wells hold PBS and fluorescein, so only tip clause 2 (concentration) catches it. -/
def backUpRow : List Step :=
  open SerialDilution in
  fillPBS ++ fillStock ++
  [.pickTip, .transfer P200 (col 1) (col 2) 100, .mix P200 (col 2) 100 3,
   .transfer P200 (col 2) (col 3) 100, .mix P200 (col 3) 100 3,
   .mix P200 (col 2) 100 3, .dropTip]

/-- 10. One tube against an empty slot: 1100 mg vs 0 mg. -/
def singleTube : List Step := [.centrifuge [some "T1", none] 5000 (sec 60)]

/-- 11. A single well spun as if it were a tube (only tubes and whole plates go in a rotor). -/
def wellInRotor : List Step := [.centrifuge [some A1, some "T1"] 5000 (sec 60)]

/-- 12. Two containers with the same id. -/
def dupInit : LabState := { init with cs := init.cs ++ [tube "R1" "r1" 10] }

/-- 13. A tube whose listed contents do not add up to its volume (500 uL of "nothing" would
weigh 0 mg in the rotor). -/
def badInit : LabState := { init with cs := init.cs ++ [{ id := "X", cap := 1500, vol := 500 }] }

theorem overflow_rejected : ¬ ValidProtocol init overflow := by decide
theorem unbalanced_rejected : ¬ ValidProtocol init unbalanced := by decide
theorem denseUnbalanced_rejected : ¬ ValidProtocol init denseUnbalanced := by decide
theorem reusedTip_rejected : ¬ ValidProtocol init reusedTip := by decide
theorem notThawed_rejected : ¬ ValidProtocol init notThawed := by decide
theorem intoSealed_rejected : ¬ ValidProtocol init intoSealed := by decide
theorem noTip_rejected : ¬ ValidProtocol init noTip := by decide
theorem backToStock_rejected : ¬ ValidProtocol SerialDilution.init backToStock := by decide
theorem backUpRow_rejected : ¬ ValidProtocol SerialDilution.init backUpRow := by decide
theorem singleTube_rejected : ¬ ValidProtocol init singleTube := by decide
theorem wellInRotor_rejected : ¬ ValidProtocol init wellInRotor := by decide
theorem dupInit_rejected : ¬ ValidProtocol dupInit [] := by decide
theorem badInit_rejected : ¬ ValidProtocol badInit [] := by decide

-- The exact error each one gets.
example : errorOf init overflow = some (.overflow "P:A1" 190 20 200) := by decide
example : errorOf init unbalanced = some (.unbalanced 0 2 1100 1500) := by decide
example : errorOf init denseUnbalanced = some (.unbalanced 0 1 2000 2130) := by decide
example : errorOf init reusedTip = some (.contaminated "R2") := by decide
example : errorOf init notThawed = some (.notThawed "enzyme") := by decide
example : errorOf init intoSealed = some (.sealed "P:A2") := by decide
example : errorOf init noTip = some .noTip := by decide
example : errorOf SerialDilution.init backToStock = some (.contaminated "FLUO") := by decide
example : errorOf SerialDilution.init backUpRow = some (.contaminated "CAL:A2") := by decide
example : errorOf init singleTube = some (.unbalanced 0 1 1100 0) := by decide
example : errorOf init wellInRotor = some (.notTubeOrPlate "P:A1") := by decide
example : errorOf dupInit [] = some .duplicateIds := by decide
example : errorOf badInit [] = some (.badContents "X") := by decide

#eval report init overflow
#eval report init unbalanced
#eval report init denseUnbalanced
#eval report init reusedTip
#eval report init notThawed
#eval report init intoSealed
#eval report init noTip
#eval report SerialDilution.init backToStock
#eval report SerialDilution.init backUpRow
#eval report init singleTube
#eval report init wellInRotor
#eval report dupInit []
#eval report badInit []

-- Claiming a broken protocol is valid does not compile (decide finds it is false):
-- theorem overflow_ok : ValidProtocol init overflow := by decide

end LabLean.Broken
