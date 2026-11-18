import LabLean.Theorems
/-!
# Example: PCR master mix setup

Based on the Addgene PCR protocol on protocols.io
(https://dx.doi.org/10.17504/protocols.io.betyjepw).
Per 50 uL reaction: 2 uL template, 5 uL 10X buffer, 1 uL 10 mM dNTP,
2.5 uL of each 10 uM primer, 0.2 uL Taq, 36.8 uL water. Keep reagents on ice.

We run 4 reactions. The master mix (water, buffer, dNTP, Taq; no primers, no template) is
43 uL per reaction and is made for **5 reactions** (4 + 1 extra, i.e. 25% overage):
water 184 uL, buffer 25 uL, dNTP 5 uL, Taq 1.0 uL = 215 uL. 4 x 43 = 172 uL is used;
43 uL is left in the tube.

Departures from the source, all deliberate:
* Taq: 0.2 uL per reaction is at the very edge of a P2, so we never pipette it alone.
  In the master mix Taq is 5 x 0.2 = 1.0 uL, inside the P2 range (0.2-2 uL).
  It is in glycerol (viscous); pipette slowly. The checker does not model this.
* Template 2 uL is the bottom of the P20 range (least accurate); the source volume is kept.
* After setup: seal, short spin against a balance plate, then one 94 C / 2 min step as the
  start of the run. Thermal cycling itself is not modeled.
-/
namespace LabLean.PCR

def reagents : List Reagent :=
  [ { name := "water", state := .roomTemp, diluent := true },
    { name := "buffer", state := .frozen, keepCold := true },
    { name := "dNTP", state := .frozen, keepCold := true },
    { name := "taq", state := .onIce, keepCold := true },   -- in glycerol, does not freeze
    { name := "primerF", state := .frozen, keepCold := true },
    { name := "primerR", state := .frozen, keepCold := true },
    { name := "template", state := .frozen, keepCold := true } ]

/-- The 4 reaction wells: numbers 1..4 = A1..A4 of plate "PCR". -/
def wells : List String := [1, 2, 3, 4].map (wellId "PCR")

def init : LabState :=
  { reagents,
    -- same plate type for the reactions and the balance plate (plastic mass assumed 14 g)
    plates := [("PCR", 14000), ("BAL", 14000)],
    cs := [ tube "WATER" "water" 1000,
            tube "BUF" "buffer" 100,
            tube "DNTP" "dNTP" 50,
            tube "TAQ" "taq" 10,
            tube "FWD" "primerF" 50,
            tube "REV" "primerR" 50,
            tube "TMPL" "template" 20,
            { id := "MM", cap := 1500, vol := 0, tare := 1000 } ] ++
          [1, 2, 3, 4].map (fun n => well "PCR" n 200) ++
          -- balance plate: one well with 200 uL water (= 4 x 50 uL of reactions)
          [ well "BAL" 1 300 "water" 200 ] }

/-- New tip, one transfer, drop the tip. -/
def withFreshTip (p : Pipette) (src dst : String) (v : Vol) : List Step :=
  [.pickTip, .transfer p src dst v, .dropTip]

def masterMix : List Step :=
  [.thaw "buffer", .thaw "dNTP", .thaw "primerF", .thaw "primerR", .thaw "template"] ++
  withFreshTip P200 "WATER" "MM" 184 ++
  withFreshTip P200 "BUF" "MM" 25 ++
  withFreshTip P20 "DNTP" "MM" 5 ++
  withFreshTip P2 "TAQ" "MM" 1.0

/-- Mix the master mix, then one tip for all wells: one source, many empty destinations.
Going back into "MM" is allowed because each well holds only master mix. -/
def dispense : List Step :=
  [.pickTip, .mix P200 "MM" 100 3] ++ wells.map (fun w => .transfer P200 "MM" w 43) ++ [.dropTip]

/-- Primers and template: a fresh tip for every addition; mix the well after the template. -/
def addPerWell (w : String) : List Step :=
  withFreshTip P20 "FWD" w 2.5 ++
  withFreshTip P20 "REV" w 2.5 ++
  [.pickTip, .transfer P20 "TMPL" w 2, .mix P20 w 10 3, .dropTip]

def finish : List Step :=
  [ .seal "PCR", .seal "BAL",
    .centrifuge [some "PCR", some "BAL"] 1000 (sec 30),   -- short spin, plate vs balance plate
    .incubate "PCR" 94 (mins 2) ]                          -- initial denaturation

def pcrSteps : List Step := masterMix ++ dispense ++ wells.flatMap addPerWell ++ finish

#eval report init pcrSteps   -- "OK"

theorem pcr_valid : ValidProtocol init pcrSteps := by decide

end LabLean.PCR
