# YLM Colab Results and Process Notes

Inventory date: 2026-11-09 (America/Los_Angeles)

## Repository and Pull Request State

- Repository: `yaniguan/YLM`
- `main` commit: `ae8b0a2453140eb748b2f09edd675c565982e752`
- Open pull request: PR #20
  - Title: "Add an optional 0e+2e tensor readout"
  - Branch: `cursor/l2-tensor-readout-1bcd`
  - Head commit: `d6a72f3ab64fee4f37b35895781c4b8822f060ff`
  - Status: waiting for Yani's review. Do not merge or rewrite PR #20.

## Google Drive Layout

- Drive root: `YLM_runs`
- Official Option A/B/C runs: `YLM_runs/options_bc_test/runs/full_A`, `YLM_runs/options_bc_test/runs/full_B`, and `YLM_runs/options_bc_test/runs/full_C`
- Earlier dated `full_A` run folder: `20261030_160412` under `YLM_runs` (reported separately from the 2026-10-31 `options_bc_test/runs/full_A` run below)

## Official Option Runs Summary

A shorter Options A/B/C table already exists in `docs/results_options_abc.md`. This inventory keeps the two `full_A` runs separate and records the official `results.json` values listed below.

Note: The two `full_A` runs (2026-10-30 dated folder `20261030_160412` and 2026-10-31 `options_bc_test/runs/full_A`) are kept separate below. Where a specific metric is not listed in the verified summary, it is marked "not yet confirmed from the official file".

| Run | Drive Path / File | Modified Date | Test Force MAE | Test Force Cosine | Test `atom_acc` | Dipole MAE | Dipole Cosine | Heldout Force MAE | Heldout `atom_acc` | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full_C` | `YLM_runs/options_bc_test/runs/full_C/results.json` | 2026-11-01 | `0.0048709` | `0.97623` | `0.92040` | `0.07349` | `0.98348` | `0.01245` | `0.86239` | Option C, 60 epochs, batch 128, minilm, resume true; equivariance errors about `1e-6` |
| `full_B` | `YLM_runs/options_bc_test/runs/full_B/results.json` | 2026-10-31 | `0.006803` | `0.95878` | `0.69605` | `0.08944` | `0.97662` | not yet confirmed from the official file | `0.66643` | Known `0.696` vs target `0.754` on test `atom_acc` |
| `full_A` (`options_bc_test`) | `YLM_runs/options_bc_test/runs/full_A/results.json` | 2026-10-31 | `0.006676` | `0.95588` | `0.90890` | `0.08872` | `0.97811` | not yet confirmed from the official file | `0.86901` | Resume run in `options_bc_test`; worse than the 2026-10-30 `full_A` run |
| Earlier `full_A` (`20261030_160412`) | `YLM_runs/.../20261030_160412/results.json` | 2026-10-30 | `0.006118` | not yet confirmed from the official file | `0.91935` | `0.08530` | `0.97910` | not yet confirmed from the official file | `0.88155` | Better than the 2026-10-31 `options_bc_test` resume across all confirmed metrics |

## Detailed Run Records

### 1. `full_C` (Option C)

- **Drive path**: `YLM_runs/options_bc_test/runs/full_C/results.json`
- **Modified date**: 2026-11-01
- **Configuration**: Option C, 60 epochs, batch size 128, `minilm`, `resume: true` (`YLM_runs/options_bc_test/runs/full_C/results.json`)
- **Metrics** (`YLM_runs/options_bc_test/runs/full_C/results.json`):
  - Test force MAE: `0.0048709`
  - Test force cosine similarity: `0.97623`
  - Test `atom_acc`: `0.92040`
  - Dipole MAE: `0.07349`
  - Dipole cosine similarity: `0.98348`
  - Heldout force MAE: `0.01245`
  - Heldout `atom_acc`: `0.86239`
  - Equivariance errors: about `1e-6`

### 2. `full_B` (Option B)

- **Drive path**: `YLM_runs/options_bc_test/runs/full_B/results.json`
- **Modified date**: 2026-10-31
- **Metrics** (`YLM_runs/options_bc_test/runs/full_B/results.json`):
  - Test force MAE: `0.006803`
  - Test force cosine similarity: `0.95878`
  - Test `atom_acc`: `0.69605` (known `0.696` vs target `0.754`)
  - Dipole MAE: `0.08944`
  - Dipole cosine similarity: `0.97662`
  - Heldout force MAE: not yet confirmed from the official file
  - Heldout `atom_acc`: `0.66643`

### 3. `full_A` in `options_bc_test` (Oct 31 Resume)

- **Drive path**: `YLM_runs/options_bc_test/runs/full_A/results.json`
- **Modified date**: 2026-10-31
- **Metrics** (`YLM_runs/options_bc_test/runs/full_A/results.json`):
  - Test force MAE: `0.006676`
  - Test force cosine similarity: `0.95588`
  - Test `atom_acc`: `0.90890`
  - Dipole MAE: `0.08872`
  - Dipole cosine similarity: `0.97811`
  - Heldout force MAE: not yet confirmed from the official file
  - Heldout `atom_acc`: `0.86901`

### 4. Earlier `full_A` (`20261030_160412`, Oct 30 Run)

- **Drive path**: dated run folder `20261030_160412`, `results.json` under `YLM_runs`
- **Modified date**: 2026-10-30
- **Comparison note**: This run performed better than the 2026-10-31 resume in `options_bc_test/runs/full_A`.
- **Metrics** (`20261030_160412/results.json`):
  - Test force MAE: `0.006118`
  - Test force cosine similarity: not yet confirmed from the official file
  - Test `atom_acc`: `0.91935`
  - Dipole MAE: `0.08530`
  - Dipole cosine similarity: `0.97910`
  - Heldout force MAE: not yet confirmed from the official file
  - Heldout `atom_acc`: `0.88155`

## Separate Polarizability Track Note

- Oct 20 `results.json` files under `pol_s2` (in `YLM_runs`) belong to the separate YLM polarizability track (`tensorqa_pol` / `matscibert`, seed 2), and are not OCSR results. Detailed metrics for `pol_s2` are not yet confirmed from the official file in this folder.

## Artifacts Not to Push to GitHub

- Do not push `.pt` checkpoints or large binary artifacts to GitHub. Only commit lightweight markdown result summaries and configuration notes.
