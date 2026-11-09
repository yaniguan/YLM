# YLM Colab Results and Process Notes

Inventory date: 2026-11-09 (America/Los_Angeles)

## Repository and Pull Request State

- Repository: `yaniguan/YLM`
- `main` commit: `ae8b0a2453140eb748b2f09edd675c565982e752`
- Open pull requests:
  - PR #20
    - Title: "Add an optional 0e+2e tensor readout"
    - Branch: `cursor/l2-tensor-readout-1bcd`
    - Head commit: `d6a72f3ab64fee4f37b35895781c4b8822f060ff`
    - Status: waiting for Yani's review. Do not merge or rewrite PR #20.
  - PR #21
    - Title: "docs: record verified Colab Option A/B/C results under YLM_runs"
    - Branch: `cursor/colab-results-nov9-986c`
    - This file is the change in PR #21. Leave PR #21 open for Yani's review.

## Google Drive Layout

- Drive root: `YLM_runs`
- Official Option A/B/C runs: `YLM_runs/options_bc_test/runs/full_A`, `YLM_runs/options_bc_test/runs/full_B`, and `YLM_runs/options_bc_test/runs/full_C`
- Earlier dated `full_A` run folder: `20261030_160412` under `YLM_runs` (reported separately from the 2026-10-31 `options_bc_test/runs/full_A` run below). This branch does not record a full parent path for that folder.

## Official Option Runs Summary

A shorter Options A/B/C table already exists in `docs/results_options_abc.md`. This inventory keeps the two `full_A` runs in separate rows.

`docs/results_options_abc.md` already lists held-out force MAE for the official `options_bc_test` runs: A `0.0107`, B `0.00952`, C `0.0125`. Those A and B values are used below. The label "not yet confirmed from the official file" is kept only for numbers this repository does not list. That includes some fields of the earlier dated `full_A` run (`20261030_160412`), including test force cosine and held-out force MAE.

The two `full_A` rows are not ranked. This repository does not list commit, data hash, e3nn version, and eval protocol side by side for both runs. `docs/results_options_abc.md` does record the `options_bc_test` runs as branch `feat/options-bc-oct30`, with eval of 4 rotations plus reflections. It does not record a commit hash, a data hash, or an e3nn version for those runs. None of those four items is recorded here for `20261030_160412`.

Test force MAE and mean force equivariance error are different metrics. `docs/results_options_abc.md` says: mean force equivariance error is ~2e-6 for every option on rotations, O(3) (rotations + reflections), and reflection-only evaluation (A 2.09e-6, B 2.03e-6, C 2.03e-6 reflection-only mean).

| Run | Drive Path / File | Modified Date | Test Force MAE | Test Force Cosine | Test `atom_acc` | Dipole MAE | Dipole Cosine | Heldout Force MAE | Heldout `atom_acc` | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `full_C` | `YLM_runs/options_bc_test/runs/full_C/results.json` | 2026-11-01 | `0.0048709` | `0.97623` | `0.92040` | `0.07349` | `0.98348` | `0.01245` | `0.86239` | Option C, 60 epochs, batch 128, minilm, resume true. Mean force equivariance error is ~2e-6 (C 2.03e-6 reflection-only mean). This is not the force MAE. `docs/results_options_abc.md` lists held-out force MAE as `0.0125`. |
| `full_B` | `YLM_runs/options_bc_test/runs/full_B/results.json` | 2026-10-31 | `0.006803` | `0.95878` | `0.69605` | `0.08944` | `0.97662` | `0.00952` | `0.66643` | Known `0.696` vs target `0.754` on test `atom_acc`. Held-out force MAE `0.00952` is from `docs/results_options_abc.md`. |
| `full_A` (`options_bc_test`) | `YLM_runs/options_bc_test/runs/full_A/results.json` | 2026-10-31 | `0.006676` | `0.95588` | `0.90890` | `0.08872` | `0.97811` | `0.0107` | `0.86901` | 2026-10-31 `options_bc_test` run. Held-out force MAE `0.0107` is from `docs/results_options_abc.md`. Not ranked against the 2026-10-30 row. |
| Earlier `full_A` (`20261030_160412`) | folder `20261030_160412` under `YLM_runs` (full parent path is not in this branch) | 2026-10-30 | `0.006118` | not yet confirmed from the official file | `0.91935` | `0.08530` | `0.97910` | not yet confirmed from the official file | `0.88155` | Separate row from the 2026-10-31 `options_bc_test` run. Not ranked. Commit, data hash, e3nn version, and eval protocol are not listed side by side. |

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
  - Heldout force MAE: `0.01245` (`docs/results_options_abc.md` lists `0.0125`)
  - Heldout `atom_acc`: `0.86239`
- **Mean force equivariance error** (`docs/results_options_abc.md`): ~2e-6 for every option. The reflection-only mean for C is 2.03e-6 (A 2.09e-6, B 2.03e-6). This is not the test force MAE.

### 2. `full_B` (Option B)

- **Drive path**: `YLM_runs/options_bc_test/runs/full_B/results.json`
- **Modified date**: 2026-10-31
- **Metrics** (`YLM_runs/options_bc_test/runs/full_B/results.json`):
  - Test force MAE: `0.006803`
  - Test force cosine similarity: `0.95878`
  - Test `atom_acc`: `0.69605` (known `0.696` vs target `0.754`)
  - Dipole MAE: `0.08944`
  - Dipole cosine similarity: `0.97662`
  - Heldout force MAE: `0.00952` (from `docs/results_options_abc.md`)
  - Heldout `atom_acc`: `0.66643`

### 3. `full_A` in `options_bc_test` (2026-10-31)

- **Drive path**: `YLM_runs/options_bc_test/runs/full_A/results.json`
- **Modified date**: 2026-10-31
- **Metrics** (`YLM_runs/options_bc_test/runs/full_A/results.json`):
  - Test force MAE: `0.006676`
  - Test force cosine similarity: `0.95588`
  - Test `atom_acc`: `0.90890`
  - Dipole MAE: `0.08872`
  - Dipole cosine similarity: `0.97811`
  - Heldout force MAE: `0.0107` (from `docs/results_options_abc.md`)
  - Heldout `atom_acc`: `0.86901`
- **Ranking**: This row is not ranked against the 2026-10-30 `full_A` row. Commit, data hash, e3nn version, and eval protocol are not listed side by side for both runs.

### 4. Earlier `full_A` (`20261030_160412`, 2026-10-30)

- **Drive path**: dated run folder `20261030_160412`, `results.json` under `YLM_runs`. This branch does not record a full parent path, so none is given here.
- **Modified date**: 2026-10-30
- **Ranking**: This row stays separate from the 2026-10-31 `options_bc_test` `full_A` row. No ranking is given, because commit, data hash, e3nn version, and eval protocol are not listed side by side.
- **Metrics** (`20261030_160412/results.json`):
  - Test force MAE: `0.006118`
  - Test force cosine similarity: not yet confirmed from the official file
  - Test `atom_acc`: `0.91935`
  - Dipole MAE: `0.08530`
  - Dipole cosine similarity: `0.97910`
  - Heldout force MAE: not yet confirmed from the official file
  - Heldout `atom_acc`: `0.88155`

## Separate Polarizability Track Note

- Oct 20 `results.json` files under `pol_s2` (in `YLM_runs`) belong to the separate YLM polarizability track (`tensorqa_pol` / `matscibert`, seed 2). Detailed metrics for `pol_s2` are not yet confirmed from the official file in this folder.

## Artifacts Not to Push to GitHub

- Do not push `.pt` checkpoints or large binary artifacts to GitHub. Only commit lightweight markdown result summaries and configuration notes.
