import LabLean.Theorems
/-!
# Example: 2-fold serial dilution of fluorescein across one plate row

Based on the iGEM plate reader fluorescein calibration protocol on protocols.io
(https://dx.doi.org/10.17504/protocols.io.6zrhf56).
* 100 uL PBS into columns 2..12,
* 200 uL fluorescein stock into column 1,
* move 100 uL from column 1 to 2, mix 3x, 2 to 3, ... through column 11,
* the last 100 uL from column 11 goes to waste; column 12 stays PBS only (blank).
The real protocol does this in 4 rows (replicates); here we do row A only.
The same tip is used along the dilution: allowed, because each next well already
holds every reagent the tip has touched (PBS and fluorescein).
-/
namespace LabLean.SerialDilution

def reagents : List Reagent :=
  [ { name := "PBS", state := .roomTemp }, { name := "fluorescein", state := .roomTemp } ]

/-- Column `c` (1..12) of row A is well number `c`. -/
def col (c : Nat) : String := label c

def init : LabState :=
  { reagents,
    cs := [ tube "PBS" 5000 "PBS" 2000,
            tube "FLUO" 1500 "fluorescein" 500,
            { id := "WASTE", cap := 5000, vol := 0 } ] ++
          (List.range 12).map (fun i => { id := col (i + 1), plate := some "CAL", cap := 300, vol := 0 }) }

def fillPBS : List Step :=
  [.pickTip] ++ (List.range 11).map (fun i => .transfer P200 "PBS" (col (i + 2)) 100) ++ [.dropTip]

def fillStock : List Step := [.pickTip, .transfer P200 "FLUO" (col 1) 200, .dropTip]

def dilute : List Step :=
  [.pickTip] ++
  (List.range 10).flatMap (fun i =>
    [.transfer P200 (col (i + 1)) (col (i + 2)) 100, .mix P200 (col (i + 2)) 100 3]) ++
  [.transfer P200 (col 11) "WASTE" 100, .dropTip]

def dilutionSteps : List Step := fillPBS ++ fillStock ++ dilute

#eval report init dilutionSteps   -- "OK"

-- Volumes per container after the run: every well 100 uL, waste 100 uL.
#eval (run init dilutionSteps).toOption.map (·.cs.map fun c => (c.id, c.vol))

theorem dilution_valid : ValidProtocol init dilutionSteps := by decide

end LabLean.SerialDilution
