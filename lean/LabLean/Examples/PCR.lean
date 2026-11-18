import LabLean.Theorems
/-!
# Example: PCR master mix setup

Based on the Addgene PCR protocol on protocols.io
(https://dx.doi.org/10.17504/protocols.io.betyjepw).
Per 50 uL reaction: 2 uL template, 5 uL 10X buffer, 1 uL 10 mM dNTP,
2.5 uL of each 10 uM primer, 0.2 uL Taq, 36.8 uL water. Keep reagents on ice.

We run 4 reactions. The master mix (no primers, no template) is made for
**5 reaction volumes** (4 + 1 extra reaction as overage, which is more than 10%):
water 184 uL, buffer 25 uL, dNTP 5 uL, Taq 1.0 uL = 215 uL = 5 x 43 uL.

Taq: 0.2 uL per reaction is below the P20 range (2-20 uL) and sits at the very
edge of a P2. So we never pipette 0.2 uL. In the master mix Taq is 5 x 0.2 = 1.0 uL,
which is inside the P2 range (0.2-2 uL).
-/
namespace LabLean.PCR

def reagents : List Reagent :=
  [ { name := "water", state := .roomTemp },
    { name := "buffer", state := .frozen, keepCold := true },
    { name := "dNTP", state := .frozen, keepCold := true },
    { name := "taq", state := .onIce, keepCold := true },   -- in glycerol, does not freeze
    { name := "primerF", state := .frozen, keepCold := true },
    { name := "primerR", state := .frozen, keepCold := true },
    { name := "template", state := .frozen, keepCold := true } ]

/-- The 4 reaction wells: numbers 1..4 = A1..A4. -/
def wells : List String := [label 1, label 2, label 3, label 4]

def pcrWell (id : String) : Container := { id, plate := some "PCR", cap := 200, vol := 0 }

def init : LabState :=
  { reagents,
    cs := [ tube "WATER" 1500 "water" 1000,
            tube "BUF" 1500 "buffer" 100,
            tube "DNTP" 1500 "dNTP" 50,
            tube "TAQ" 50 "taq" 10,
            tube "FWD" 1500 "primerF" 50,
            tube "REV" 1500 "primerR" 50,
            tube "TMPL" 1500 "template" 20,
            { id := "MM", cap := 1500, vol := 0 } ] ++
          wells.map pcrWell ++
          -- balance plate for the centrifuge: one well with 200 uL water
          [ { id := "BAL-A1", plate := some "BAL", cap := 300, vol := 200,
              contents := [("water", 2000)] } ] }

/-- New tip, one transfer, drop the tip. -/
def withFreshTip (p : Pipette) (src dst : String) (v : Vol) : List Step :=
  [.pickTip, .transfer p src dst v, .dropTip]

def masterMix : List Step :=
  [.thaw "buffer", .thaw "dNTP", .thaw "primerF", .thaw "primerR", .thaw "template"] ++
  withFreshTip P200 "WATER" "MM" 184 ++
  withFreshTip P200 "BUF" "MM" 25 ++
  withFreshTip P20 "DNTP" "MM" 5 ++
  withFreshTip P2 "TAQ" "MM" 1.0

/-- Mix the master mix, then one tip for all wells (one source, many destinations). -/
def dispense : List Step :=
  [.pickTip, .mix P200 "MM" 100 3] ++ wells.map (fun w => .transfer P200 "MM" w 43) ++ [.dropTip]

/-- Primers and template: a fresh tip for every addition. -/
def addPerWell (w : String) : List Step :=
  withFreshTip P20 "FWD" w 2.5 ++
  withFreshTip P20 "REV" w 2.5 ++
  [.pickTip, .transfer P20 "TMPL" w 2, .mix P20 w 10 3, .dropTip]

def finish : List Step :=
  [ .seal "PCR", .seal "BAL",
    .centrifuge [some "PCR", some "BAL"] 1000 (sec 30),   -- short spin, plate vs balance
    .incubate "PCR" 94 (mins 2) ]                          -- initial denaturation (source: 94 C, 2 min)

def pcrSteps : List Step := masterMix ++ dispense ++ wells.flatMap addPerWell ++ finish

#eval report init pcrSteps   -- "OK"

theorem pcr_valid : ValidProtocol init pcrSteps := by decide

end LabLean.PCR
