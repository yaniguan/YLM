#!/usr/bin/env bash
# Usage: bash recipes/coscientist_posttrain/launch.sh <stage> [--config FILE] [--i-have-a-gpu]
#   smoke                  CPU only: tests + profile of the tiny workloads
#   sft | dpo | grpo | opsd  print the planned labbench-posttrain / vLLM commands and the
#                          smoke command that works today; nothing is run or faked
#   profile                profile.py with the config (needs --i-have-a-gpu unless cpu_smoke.json)
# Env: PYTHON (default python), PROFILE (config profile name), DATA, OUT.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$DIR/../.." && pwd)"
PY="${PYTHON:-python}"
STAGE="${1:-}"
shift || true
CONFIG="$DIR/configs/h200.json"
GPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --i-have-a-gpu) GPU=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
cd "$ROOT"

need_gpu() {
  if [ "$GPU" != 1 ]; then
    echo "Not run. Re-run with --i-have-a-gpu on a GPU machine." >&2; exit 2
  fi
  command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: no GPU here." >&2; exit 2; }
}

case "$STAGE" in
  smoke)
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q tests/test_coscientist_recipe.py
    HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES= "$PY" -m recipes.coscientist_posttrain.profile \
      --config "$DIR/configs/cpu_smoke.json" --device cpu --top 8
    ;;
  sft|dpo|grpo|opsd)
    "$PY" - "$STAGE" "$CONFIG" "$DIR" <<'EOF'
import json, os, sys
stage, hw_path, d = sys.argv[1:]
st = json.load(open(os.path.join(d, "configs", "stages.json")))["stages"][stage]
hw = json.load(open(hw_path))
if "profiles" not in hw:
    sys.exit(f"{os.path.basename(hw_path)} has no stage profiles; pass --config h200.json or b200.json")
name = os.environ.get("PROFILE") or hw["profile_default"]
vals = {**hw["rollout"], **hw["train"], **hw["profiles"][name], **st, "base_model": hw["base_model"],
        "data": os.environ.get("DATA", "data/coscientist"), "out": os.environ.get("OUT", "runs/coscientist")}
print(f"Stage {stage} with {os.path.basename(hw_path)} (profile {name}).")
print("Planned commands (interface not built yet, see YLM issue #66; printed, not run):")
for key in ("planned_rollout_command", "planned_command"):
    if key in st:
        print("  " + st[key].format(**vals))
print("Works today in a labbench-posttrain checkout (CPU smoke):")
print("  " + st["smoke_command"].format(**vals))
EOF
    ;;
  profile)
    case "$CONFIG" in *cpu_smoke.json) ;; *) need_gpu ;; esac
    "$PY" -m recipes.coscientist_posttrain.profile --config "$CONFIG"
    ;;
  *)
    sed -n '2,8p' "$DIR/launch.sh"; exit 2 ;;
esac
