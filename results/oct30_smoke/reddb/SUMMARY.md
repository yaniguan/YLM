# Oct-30 downstream results

| Task | Target | Arm | Method | Seeds | Mean MAE | Mean RMSE | Mean R2 | Mean Spearman |
|---|---|---|---|---|---|---|---|---|
| reddb | reactant_homo | fingerprint | RF | 3 | 0.017778 | 0.029168 | 0.65327 | 0.8716 |
| reddb | reactant_homo | fingerprint | ridge | 3 | 0.021553 | 0.034709 | 0.48306 | 0.77006 |
| reddb | reactant_homo | prefixA | probe | 3 | 0.019602 | 0.034892 | 0.45493 | 0.84385 |
| reddb | reactant_homo | smiles | probe | 3 | 0.018937 | 0.033044 | 0.53678 | 0.83241 |
| reddb | reactant_homo | smiles_xyz | probe | 3 | 0.022291 | 0.03664 | 0.26326 | 0.82287 |
| reddb | reactant_homo | xattnA | probe | 3 | 0.018666 | 0.03266 | 0.54225 | 0.85298 |
| reddb | reactant_homo | xattnB | probe | 3 | 0.018554 | 0.032994 | 0.53597 | 0.83253 |
| reddb | reactant_homo | xattnC | probe | 3 | 0.019067 | 0.033498 | 0.51882 | 0.83533 |
| reddb | reactant_lumo | fingerprint | RF | 3 | 0.016515 | 0.026345 | 0.40552 | 0.71024 |
| reddb | reactant_lumo | fingerprint | ridge | 3 | 0.016007 | 0.026203 | 0.39374 | 0.80186 |
| reddb | reactant_lumo | prefixA | probe | 3 | 0.015316 | 0.025203 | 0.44121 | 0.85071 |
| reddb | reactant_lumo | smiles | probe | 3 | 0.017991 | 0.027463 | 0.34921 | 0.76671 |
| reddb | reactant_lumo | smiles_xyz | probe | 3 | 0.016275 | 0.02613 | 0.40634 | 0.80861 |
| reddb | reactant_lumo | xattnA | probe | 3 | 0.018179 | 0.027262 | 0.352 | 0.78013 |
| reddb | reactant_lumo | xattnB | probe | 3 | 0.018231 | 0.027596 | 0.3406 | 0.76834 |
| reddb | reactant_lumo | xattnC | probe | 3 | 0.01811 | 0.027461 | 0.34609 | 0.76875 |
| reddb | reactant_solubility | fingerprint | RF | 3 | 0.67539 | 0.83859 | 0.50189 | 0.72672 |
| reddb | reactant_solubility | fingerprint | ridge | 3 | 0.68714 | 0.8272 | 0.56431 | 0.79372 |
| reddb | reactant_solubility | prefixA | probe | 3 | 0.52687 | 0.63974 | 0.73431 | 0.86264 |
| reddb | reactant_solubility | smiles | probe | 3 | 0.55496 | 0.67247 | 0.69441 | 0.83682 |
| reddb | reactant_solubility | smiles_xyz | probe | 3 | 0.51845 | 0.62301 | 0.74606 | 0.82484 |
| reddb | reactant_solubility | xattnA | probe | 3 | 0.52965 | 0.65307 | 0.70426 | 0.83589 |
| reddb | reactant_solubility | xattnB | probe | 3 | 0.55471 | 0.67046 | 0.69612 | 0.83705 |
| reddb | reactant_solubility | xattnC | probe | 3 | 0.51394 | 0.6264 | 0.728 | 0.85591 |
| reddb | reaction_energy | fingerprint | RF | 3 | 0.024461 | 0.028965 | -0.11614 | 0.65221 |
| reddb | reaction_energy | fingerprint | ridge | 3 | 0.023221 | 0.026644 | 0.0079197 | 0.7046 |
| reddb | reaction_energy | prefixA | probe | 3 | 0.029828 | 0.035642 | -0.94584 | 0.52013 |
| reddb | reaction_energy | smiles | probe | 3 | 0.029664 | 0.034044 | -0.73717 | 0.491 |
| reddb | reaction_energy | smiles_xyz | probe | 3 | 0.023825 | 0.028866 | -0.24992 | 0.66109 |
| reddb | reaction_energy | xattnA | probe | 3 | 0.029794 | 0.034217 | -0.71749 | 0.49469 |
| reddb | reaction_energy | xattnB | probe | 3 | 0.029935 | 0.034324 | -0.75519 | 0.48104 |
| reddb | reaction_energy | xattnC | probe | 3 | 0.0301 | 0.034369 | -0.75477 | 0.47846 |

Per-seed test bootstrap intervals, predictions, split IDs and paired comparisons are in results.json.
Conductivity is evaluated on log10(k+0.001) and on the original mS/cm scale in results.json.
MP has no Morgan fingerprint baseline because molecular fingerprints do not represent crystals.
