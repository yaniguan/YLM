# Data in this folder

- `scenarios_own.json` and `items_own.jsonl` — ChemProto-Twins, written for
  this study. Synthetic seeded-error protocol items.
- `calisol_{train,dev,test}.jsonl` — derived pairwise items from CALiSol-23
  (CC-BY 4.0). Each record keeps the source DOI. Rebuild with
  `calisol_pairs.py` from a local copy of `calisol23.csv` and
  `calisol_smiles.csv`.
- `fm_pred.json` — MoLFormer-XL ridge-head predictions (logS, logD) for the
  molecules used in this run, so layer F can be rendered without downloading
  the model. The heads themselves are not stored here.

No LAB-Bench or BioProBench item text is stored in this repository.
