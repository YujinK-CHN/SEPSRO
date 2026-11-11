#!/usr/bin/env bash
#
# Local launcher: the seeds of one round (one method, one configuration, one
# env -> results/<exp-name>/, see main.py). Another method or configuration is
# another round: run it again with another --exp-name.
#
# Four knobs, because only these change between runs of a suite:
#
#   --method    random | psro | ppo | gems   (one per round)
#   --env       a key of ENV_PRESETS in config.py, or backend:id
#   --budget    training budget in ENV STEPS -- the common currency that makes
#               methods comparable: at the same --budget every method has
#               consumed the same amount of experience.
#   --seeds     comma-separated seeds
#
# Everything else lives in config.py. Edit it there. Any unrecognised flag is
# passed through to main.py unchanged, so a one-off override still works:
#   bash run.sh --method random --env kuhn_poker --budget 50000 --seeds 0 --eval-episodes 500
#
# Usage:
#   bash run.sh --method random --env kuhn_poker --budget 100000 --seeds 0,1,2
#   bash run.sh --method psro --env leduc_poker --budget 1000000 --seeds 0,1,2 \
#        --run-mode parallel --exp-name leduc_psro

set -euo pipefail

# ───────────────────────── the knobs ────────────────────────────────
METHODS=()
ENV="kuhn_poker"
BUDGET=100000
SEEDS="0"

# ───────────────────────── run control ──────────────────────────────
RUN_MODE="sequential"      # sequential | parallel
EXP_NAME="check"
PYTHON="${PYTHON:-python}"
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)    METHODS+=("$2"); shift 2 ;;
        --env)       ENV="$2"; shift 2 ;;
        --budget)    BUDGET="$2"; shift 2 ;;
        --seeds)     SEEDS="$2"; shift 2 ;;
        --run-mode)  RUN_MODE="$2"; shift 2 ;;
        --exp-name)  EXP_NAME="$2"; shift 2 ;;
        *)           PASSTHROUGH+=("$1"); shift ;;
    esac
done
[[ ${#METHODS[@]} -eq 0 ]] && METHODS=(random)
if [[ ${#METHODS[@]} -ne 1 ]]; then
    echo "One --method per round (got ${#METHODS[@]}); run another method with another --exp-name." >&2
    exit 1
fi

cd "$(dirname "$0")"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

pids=()
for method in "${METHODS[@]}"; do
    for seed in "${SEED_LIST[@]}"; do
        cmd=("$PYTHON" main.py --method "$method" --env "$ENV" --budget "$BUDGET"
             --seed "$seed" --exp-name "$EXP_NAME" "${PASSTHROUGH[@]+"${PASSTHROUGH[@]}"}")
        echo ">> ${cmd[*]}"
        if [[ "$RUN_MODE" == "parallel" ]]; then
            log_dir="results/$EXP_NAME/logs"; mkdir -p "$log_dir"
            "${cmd[@]}" > "$log_dir/seed${seed}.log" 2>&1 &
            pids+=($!)
        else
            "${cmd[@]}"
        fi
    done
done

status=0
for pid in "${pids[@]+"${pids[@]}"}"; do
    wait "$pid" || status=1
done
exit $status
