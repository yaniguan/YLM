import LabLean
open LabLean

/-- Print the checker result for every example protocol. -/
def main : IO Unit := do
  let cases : List (String × LabState × List Step) :=
    [ ("PCR master mix", PCR.init, PCR.pcrSteps),
      ("Serial dilution", SerialDilution.init, SerialDilution.dilutionSteps),
      ("Broken: overflow", Broken.init, Broken.overflow),
      ("Broken: unbalanced", Broken.init, Broken.unbalanced),
      ("Broken: same volume, different mass", Broken.init, Broken.denseUnbalanced),
      ("Broken: reused tip", Broken.init, Broken.reusedTip),
      ("Broken: not thawed", Broken.init, Broken.notThawed),
      ("Broken: into sealed", Broken.init, Broken.intoSealed),
      ("Broken: no tip", Broken.init, Broken.noTip),
      ("Broken: tip back to stock", SerialDilution.init, Broken.backToStock),
      ("Broken: tip back up the row", SerialDilution.init, Broken.backUpRow),
      ("Broken: single tube", Broken.init, Broken.singleTube),
      ("Broken: well in rotor", Broken.init, Broken.wellInRotor),
      ("Broken: duplicate ids", Broken.dupInit, []),
      ("Broken: contents do not match volume", Broken.badInit, []) ]
  for (name, s, l) in cases do
    IO.println s!"{name}: {report s l}"
