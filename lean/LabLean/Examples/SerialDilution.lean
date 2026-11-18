import LabLean.Theorems
/-!
# Example: 2-fold serial dilution of fluorescein across one plate row

Based on the iGEM plate reader fluorescein calibration protocol on protocols.io
(https://dx.doi.org/10.17504/protocols.io.6zrhf56).
* 100 uL PBS into columns 2..12,
* 200 uL 1x fluorescein stock (in PBS) into column 1,
* move 100 uL from column 1 to 2, mix 3x, 2 to 3, ... through column 11,
* the last 100 uL from column 11 goes to waste; column 12 stays PBS only (blank).
The source repeats this in several rows (replicates); here we do row A only.

The model treats the 1x stock as one named solution "fluorescein"; its concentration in a well
is the volume fraction of stock (1, 1/2, 1/4, ...). PBS is a diluent.

One tip is used for the whole dilution, moving downstream only. The tip rule allows this:
every liquid on the tip is at least as concentrated as the next well. Going back up the row,
or back into the stock, is rejected (see `Broken.backToStock`).
-/
namespace LabLean.SerialDilution

def reagents : List Reagent :=
  [ { name := "PBS", state := .roomTemp, diluent := true },
    { name := "fluorescein", state := .roomTemp } ]

/-- Column `c` (1..12) of row A is well number `c`. -/
def col (c : Nat) : String := wellId "CAL" c

def init : LabState :=
  { reagents,
    plates := [("CAL", 14000)],
    cs := [ tube "PBS" "PBS" 2000 (cap := 5000),
            tube "FLUO" "fluorescein" 500,
            { id := "WASTE", cap := 5000, vol := 0, tare := 1000 } ] ++
          (List.range 12).map (fun i => well "CAL" (i + 1) 300) }

/-- One tip for all PBS: the PBS-only wells never add a new reagent to the reservoir. -/
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
