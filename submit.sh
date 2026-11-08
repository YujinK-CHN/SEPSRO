#!/usr/bin/env bash
#
# Submit SEPSRO runs to Slurm (Digital Research Alliance clusters).
#
# Same knobs as run.sh -- every (method, seed) pair becomes one task of a job
# array, so seeds run in parallel on separate nodes/cores:
#
#   bash submit.sh --account def-yourpi --exp-name si_main \
#       --method gems --method psro --env space_invaders \
#       --budget 5000000 --seeds 0,1,2 --time 48:00:00
#
# Any other flag goes to main.py unchanged, for every task:
#
#   bash submit.sh --account def-yourpi --exp-name kuhn_br --method psro \
#       --env kuhn_poker --budget 100000000 --seeds 0,1,2,3,4 \
#       --psro-oracle br --eval-metrics nash_conv
#
# Before submitting, on the login node, it builds the environment once
# (catches missing packages or Atari ROMs now, not after hours in the queue)
# and refuses run directories that already hold logs unless --overwrite is
# among the main.py flags -- the same rule main.py enforces, checked early.
#
# Resources default to CPU-only: SEPSRO's learners run on CPU (rollouts are one
# step at a time), so a GPU would sit idle. --gpus 1 requests one per task.
#
# Slurm settings come from flags or environment:
#   --account   (or $SEPSRO_ACCOUNT)   required by Alliance clusters
#   --time      (or $SEPSRO_TIME)      default 24:00:00
#   --mem       (or $SEPSRO_MEM)       default 16G
#   --cpus      (or $SEPSRO_CPUS)      default 4  (also torch / OpenMP threads)
#   --gpus      (or $SEPSRO_GPUS)      default 0
#   --gpu-type  (or $SEPSRO_GPU_TYPE)  e.g. h100; with --gpus
#   --venv      (or $SEPSRO_VENV)      default ~/venvs/sepsro
#   --out-root  (or $SEPSRO_OUT)       default <repo>/results; ~/scratch/... for big sweeps
#   --python-module                    default python/3.11 (must match setup_cluster.sh)
#   --dry-run                          write and print the job script, submit nothing
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
ACCOUNT="${SEPSRO_ACCOUNT:-}"
TIME="${SEPSRO_TIME:-24:00:00}"
MEM="${SEPSRO_MEM:-16G}"
CPUS="${SEPSRO_CPUS:-4}"
GPUS="${SEPSRO_GPUS:-0}"
GPU_TYPE="${SEPSRO_GPU_TYPE:-}"
VENV="${SEPSRO_VENV:-$HOME/venvs/sepsro}"
OUT_ROOT="${SEPSRO_OUT:-$REPO/results}"
PYTHON_MODULE="${SEPSRO_PYTHON_MODULE:-python/3.11}"
DRY_RUN=0

METHODS=()
ENV_NAME="kuhn_poker"
BUDGET=100000
SEEDS="0"
EXP_NAME="check"
PASS=()          # everything else goes to main.py untouched

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)  sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0;;
    --account)  ACCOUNT="$2";  shift 2;;
    --time)     TIME="$2";     shift 2;;
    --mem)      MEM="$2";      shift 2;;
    --cpus)     CPUS="$2";     shift 2;;
    --gpus)     GPUS="$2";     shift 2;;
    --gpu-type) GPU_TYPE="$2"; shift 2;;
    --venv)     VENV="$2";     shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --python-module) PYTHON_MODULE="$2"; shift 2;;
    --dry-run)  DRY_RUN=1;     shift;;
    --method)   METHODS+=("$2"); shift 2;;
    --env)      ENV_NAME="$2"; shift 2;;
    --budget)   BUDGET="$2";   shift 2;;
    --seeds)    SEEDS="$2";    shift 2;;
    --exp-name) EXP_NAME="$2"; shift 2;;
    --results-dir) echo "Use --out-root instead of --results-dir." >&2; exit 1;;
    *)          PASS+=("$1");  shift;;
  esac
done
[[ ${#METHODS[@]} -eq 0 ]] && METHODS=(random)

if [[ -z "$ACCOUNT" && $DRY_RUN -eq 0 ]]; then
  echo "No --account given and \$SEPSRO_ACCOUNT is unset." >&2
  echo "Alliance clusters reject jobs without one, e.g. --account def-yourpi" >&2
  exit 1
fi

IFS=',' read -ra SEED_LIST <<< "$SEEDS"
TASKS=()                                   # "method seed" per array index
for m in "${METHODS[@]}"; do
  for s in "${SEED_LIST[@]}"; do TASKS+=("$m $s"); done
done
N_TASKS=${#TASKS[@]}

# ---- checks on the login node, where failures are cheap -------------------- #
if command -v module >/dev/null 2>&1; then module load "$PYTHON_MODULE"; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate" 2>/dev/null || {
  echo "Could not activate $VENV. Run setup_cluster.sh, or pass --venv." >&2; exit 1; }

overwrite=0
for a in "${PASS[@]+"${PASS[@]}"}"; do [[ "$a" == "--overwrite" ]] && overwrite=1; done

echo "[submit] checking env '$ENV_NAME' and run directories on the login node"
( cd "$REPO" && python - "$ENV_NAME" "$OUT_ROOT" "$EXP_NAME" "$overwrite" "${TASKS[@]}" <<'PYEOF'
import os, sys
import warnings
warnings.filterwarnings('ignore')
from config import parse_env
from sepsro.envs import make_env

env_name, out_root, exp, overwrite, tasks = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] == '1', sys.argv[5:]
preset = parse_env(env_name)
try:
    env = make_env(preset)
    env.reset(seed=0)
    spec = env.spec
    env.close()
except Exception as exc:
    sys.exit(f'[submit] cannot build env {env_name!r}: {type(exc).__name__}: {exc}\n'
             '         (missing package? for space_invaders run AutoROM on the login node)')
print(f'[submit] env ok: {spec.backend}, agents {list(spec.agents)}, obs {dict(spec.obs_shape)}')

clash = []
for task in tasks:
    method, seed = task.split()
    run_dir = os.path.join(out_root, exp, preset['name'], method, f'seed{seed}')
    if any(os.path.exists(os.path.join(run_dir, f)) for f in ('eval.jsonl', 'train.jsonl')):
        clash.append(run_dir)
if clash and not overwrite:
    sys.exit('[submit] these runs already have logs (use another --exp-name, or --overwrite):\n  '
             + '\n  '.join(clash))
PYEOF
)

# ---- build the job script ------------------------------------------------- #
# Slurm reads #SBATCH lines literally (no shell quoting), so the job name is
# sanitised; the real --exp-name, parentheses and all, still reaches main.py.
SAFE_NAME="${EXP_NAME//[^A-Za-z0-9._-]/_}"
mkdir -p "$OUT_ROOT/jobs" "$OUT_ROOT/logs"
JOB="$OUT_ROOT/jobs/${SAFE_NAME}.sbatch"

q() { printf '%q' "$1"; }
TASK_LINES=""
for t in "${TASKS[@]}"; do TASK_LINES+="  $(q "$t")"$'\n'; done
MAIN_ARGS=""
for a in "${PASS[@]+"${PASS[@]}"}"; do MAIN_ARGS+=" $(q "$a")"; done

GRES_LINE=""
if [[ "$GPUS" != "0" ]]; then
  GRES_LINE="#SBATCH --gres=gpu:${GPU_TYPE:+${GPU_TYPE}:}${GPUS}"
fi

cat > "$JOB" <<EOF
#!/bin/bash
#SBATCH --account=${ACCOUNT:-UNSET}
#SBATCH --job-name=${SAFE_NAME}
#SBATCH --array=0-$((N_TASKS - 1))
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${OUT_ROOT}/logs/%x-%A_%a.out
${GRES_LINE}

set -euo pipefail
TASKS=(
${TASK_LINES})
read -r METHOD SEED <<< "\${TASKS[\$SLURM_ARRAY_TASK_ID]}"

module load ${PYTHON_MODULE}
source $(q "$VENV")/bin/activate
cd $(q "$REPO")
# Torch and NumPy otherwise size their thread pools to the whole node.
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}

echo "task    : \$SLURM_ARRAY_TASK_ID  method=\$METHOD seed=\$SEED  on \$(hostname)"
echo "commit  : \$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "started : \$(date)"

python main.py --method "\$METHOD" --seed "\$SEED" --env $(q "$ENV_NAME") --budget $(q "$BUDGET") \\
    --exp-name $(q "$EXP_NAME") --results-dir $(q "$OUT_ROOT")${MAIN_ARGS}

echo "finished: \$(date)"
EOF

echo "[submit] job script -> $JOB  ($N_TASKS task(s): ${TASKS[*]})"
if [[ $DRY_RUN -eq 1 ]]; then
  echo "---------------------------------------------------------------"
  cat "$JOB"
  echo "---------------------------------------------------------------"
  echo "[submit] --dry-run: nothing submitted."
  exit 0
fi

sbatch "$JOB"
echo "[submit] watch with: squeue -u \$USER     logs in $OUT_ROOT/logs"
