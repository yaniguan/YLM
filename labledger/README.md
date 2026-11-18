# labledger: a lab data ledger

A small, dependency-free way to record every experiment in full, keep the record honest, and
turn it into multimodal training data. Schema version: **`labledger/v1`**.

## Levels

```
project
└── campaign            hypothesis
    └── experiment      one run of one protocol, status planned/running/completed/failed/aborted
        ├── step        protocol step: action, planned params, instrument, status, failure tags
        │   └── observation   what was measured or seen (one modality each)
        │       └── artifact  raw file: sha256, size, media type, path in blobs/
        └── observation       continuous stream (room temperature, long audio) with a time window
```

Each level may only sit under the level above it. The ledger refuses anything else.

## Events, not edits

The ledger is one file, `events.jsonl`. Every line is one event:

| field | meaning |
|---|---|
| `schema` | `labledger/v1` |
| `seq` | 0, 1, 2, ... (order comes from `seq`, not from clocks) |
| `op` | `add`, `supersede` (full new body + `supersedes` = old seq + `reason`), `retract` (+ `reason`) |
| `level`, `record_id`, `parent_id` | place in the tree; ids look like `exp:RUN-7` |
| `recorded_utc` | when it was written, UTC with `Z` |
| `origin` | `real`, `simulated`, `model_generated`, `synthetic_demo` |
| `license` | SPDX id or `own` |
| `provenance` | `agent` {type: human/robot/instrument/software/model, id}, `software`, `derived_from` |
| `body` | level fields (below) |
| `prev_hash`, `event_hash` | sha256 chain over canonical JSON (sorted keys, no spaces, no NaN) |

Nothing is ever changed or deleted. A typo fix is a `supersede`; a duplicate is a `retract`. Old
versions stay in the file, so you can always see what was believed when.

Raw files go into `blobs/<sha256>` once and are never changed. Artifacts point to them.

## Level fields

- **project**: `title`, `goal`
- **campaign**: `title`, `hypothesis`
- **experiment**: `title`, `protocol_id`, `status`, `started_utc`, `ended_utc`
- **step**: `index`, `action`, `status` (planned/ok/deviated/failed/skipped), `instrument_id`,
  `params` {name: quantity}, `started_utc`, `ended_utc`, `deviation` (needed if deviated),
  `failure_tags` (needed if failed)
- **observation**: `modality`, `instrument_id`, `t_utc`, `t_end_utc`, `quantities`, `text`,
  `failure_tags`, `detail` (per modality):
  - `text`: free notes in `text`
  - `reading`: one or more quantities
  - `image`: optional `smiles` and `ocsr_flag` (from the structure reader)
  - `audio`: the `lab-ears/audio-event/v1` record (detector, score, threshold, mic, ...);
    a score above threshold must carry the `acoustic_anomaly` tag; raw audio with the speech
    band is refused
  - `spectrum`, `series`: `x_unit`, `y_unit`, `n_points`; the data itself is an artifact
  - `sim`: `engine`, `system`, `values`
  - `geom`: `Z`, `pos` (N x 3), `unit` (bohr or angstrom)
- **artifact**: `sha256`, `size_bytes`, `media_type`, `uri`, `role` (raw/derived/thumbnail)

**Quantity**: `{value, unit, uncertainty?}`. Units come from a fixed list (`uL`, `Cel`, `x g`,
`nm`, `{absorbance}`, ...). Uncertainty is `std`, `spec`, `interval` or `none`, and always says
how it was obtained (`method`). NaN and Inf are refused.

**Failure tags**: instrument_fault, acoustic_anomaly, out_of_range, contamination,
pipetting_error, volume_mismatch, sample_lost, timeout, operator_error, sim_disagreement,
checker_fail, calibration_expired, other. Failures are records and are never dropped.

The JSON Schema is in [`schema/labledger.v1.schema.json`](schema/labledger.v1.schema.json)
(`python -m labledger.jsonschema` writes it). The Python checks in `schema.py` are the reference.

## Use

```python
from labledger import Ledger
L = Ledger("runs/plate-7")
L.add("project", "prj:P", {"title": "...", "goal": "..."}, origin="real", license="own",
      agent={"type": "human", "id": "person:me"})
art = L.put_blob(open("A1.csv", "rb").read(), "text/csv", "A1.csv")
L.add("artifact", "art:A1", art, parent_id="obs:A1", origin="real", license="own",
      agent={"type": "instrument", "id": "rdr:1"})
```

```bash
python -m labledger verify runs/plate-7          # schema, chain, tree, raw file checksums
python -m labledger export runs/plate-7 train.jsonl [--sealed sealed_ids.txt]
```

## Training export

`export` writes `coscientist.v1` records, the format of the post-training recipe
(`recipes/coscientist_posttrain`, PR #86). Each finished experiment gives:

1. a failure question: the whole record as text, then "which steps failed and with which tags";
   the answer comes from the records, and the verifier is `exact`;
2. a step-order question (3 or more steps), shuffled with a fixed seed.

Placeholders point into arrays, with indices 0..n-1:

| placeholder | array | from |
|---|---|---|
| `<geom:i>` | `geoms` | geom observations, converted to bohr |
| `<img:i>` | `images` | image observations that have `smiles` (others become plain text) |
| `<sim:i>` | `sims` | WignerFlow argon results (other sims become plain text) |
| `<audio:i>` | `audios` | lab-ears audio events; `export.audio_text` gives a text form |

Gates: the ledger must verify clean; only live versions are used (superseded values and
retracted records never appear); `synthetic_demo` is blocked unless `allow_synthetic=True`;
sealed ids are never exported; `commercial_ok` follows the license. Every example keeps
`ledger.record_ids` and the head hash, so it can be traced back to its raw files.

## Synthetic example

`examples/demo_campaign/` is made by `python -m labledger.demo`. **All of it is synthetic:**
ids, lots, serials, readings, scores and simulation numbers are made up to show the format.
It has three experiments: a completed dilution and plate read (image, geometry, reading with
uncertainty, spectrum file, temperature stream, one corrected value), a run that failed on a
noisy centrifuge (audio anomaly, operator note, one retracted duplicate, a skipped step), and an
argon simulation reference run.

## Measured on a laptop-class CPU (synthetic demo)

Intel Xeon (8 cores), Python 3.11+, single process:

| what | result |
|---|---|
| demo build (29 events, 24 live records, 3 blobs) | 2.2 to 2.6 ms (5 runs) |
| demo verify | 0.8 ms |
| training examples from the demo | 5 |
| append 10,000 reading events | 0.67 s |
| reload and verify 10,000 events | 0.38 s |

## Limits

- One writer per ledger directory; no locking.
- The hash chain shows tampering inside the file, not a rewrite of the whole file. Copy the
  head hash somewhere else (a git commit, a signed note) to anchor it.
- `children()` scans all records, so export is slow on very large ledgers. Fine for one lab.
- No real instrument parsers yet. The unit list is short on purpose; add units as needed.
- Deleting personal data conflicts with append-only storage. Keep personal data out of events.

## Related formats

The design borrows from ISA (investigation/study/assay), W3C PROV-O (agent, derived_from),
the Bluesky event model (append-only documents, raw data by reference), HELAO (actions with
inputs and outputs), Allotrope and AnIML (units and instrument metadata), and RO-Crate
(checksummed files).
