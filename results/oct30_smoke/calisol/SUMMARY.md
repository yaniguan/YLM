# Oct-30 downstream results

| Task | Target | Arm | Method | Seeds | Mean MAE | Mean RMSE | Mean R2 | Mean Spearman |
|---|---|---|---|---|---|---|---|---|
| calisol | log_conductivity | fingerprint | RF | 2 | 0.30747 | 0.3531 | 0.49765 | 0.65354 |
| calisol | log_conductivity | fingerprint | ridge | 2 | 0.31681 | 0.3968 | 0.54253 | 0.68142 |
| calisol | log_conductivity | prefixA | probe | 2 | 0.51724 | 0.59851 | -0.16733 | 0.30556 |
| calisol | log_conductivity | smiles | probe | 2 | 0.54312 | 0.69743 | -0.98157 | 0.36993 |
| calisol | log_conductivity | smiles_xyz | probe | 2 | 0.47423 | 0.6312 | -0.38698 | 0.0094758 |
| calisol | log_conductivity | xattnA | probe | 2 | 0.64439 | 0.87669 | -1.5517 | 0.029061 |
| calisol | log_conductivity | xattnB | probe | 2 | 0.55304 | 0.70554 | -1.0076 | 0.35421 |
| calisol | log_conductivity | xattnC | probe | 2 | 0.54476 | 0.69476 | -0.96781 | 0.3697 |

Per-seed test bootstrap intervals, predictions, split IDs and paired comparisons are in results.json.
Conductivity is evaluated on log10(k+0.001) and on the original mS/cm scale in results.json.
MP has no Morgan fingerprint baseline because molecular fingerprints do not represent crystals.

Skipped calisol log_conductivity: Empty train/validation/test split after label filtering
