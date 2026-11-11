#!/usr/bin/env bash
#
# Submit SEPSRO runs to Slurm (Digital Research Alliance clusters).
#
# One submission = one task of one round. A round (--exp-name) is one method
# with one configuration in one env, every seed of it, all in one folder:
#
#   <out-root>/<exp-name>/
#     round.json          the round's settings; a seed with other settings is refused
#     stats/seed<N>/      statistics: config, eval / train logs, exploit and cross-play results
#     checkpoints/seed<N>/  policies (copy back only if needed)
#     slurm/              job scripts and Slurm logs
#
# Name the round after what defines it. Every seed becomes one task of a job
# array, so seeds run in parallel:
#
#   bash submit.sh --account def-yourpi --exp-name si_gems --method gems \
#       --env space_invaders --budget 30000000 --seeds 0,1,2,3,4 --time 36:00:00
#
# Any other flag goes to main.py unchanged, for every seed:
#
#   bash submit.sh --account def-yourpi --exp-name kuhn_psro_br --method psro \
#       --env kuhn_poker --budget 100000000 --seeds 0,1,2,3,4 --psro-oracle br
#
# Another method or configuration is another round: another --exp-name, another
# submission (only one --method per submission).
#
# --task picks what the jobs run (default train):
#   train      main.py, one array task per seed
#   exploit    posthoc.py exploit on the round's finished seeds: approximate
#              NashConv, one array task per (seed, checkpoint); --checkpoints
#              picks them (default final; e.g. 0,0.25,0.5,0.75,1 for a 5-point
#              curve: fractions of the budget; 'all' = every checkpoint of a seed
#              in one task). Other flags go to posthoc.py (e.g. --exploit-steps).
#   crossplay  posthoc.py crossplay on the round: one task, among its seeds and
#              against the rounds given by --against (other flags, e.g.
#              --episodes, go to posthoc.py)
#
#   bash submit.sh --account def-yourpi --exp-name si_gems --seeds 0,1,2,3,4 \
#       --task exploit --checkpoints 0,0.25,0.5,0.75,1 --time 12:00:00
#   bash submit.sh --account def-yourpi --exp-name si_gems --task crossplay \
#       --against si_psro_ppo --time 6:00:00
#
# Before submitting, on the login node, it builds the environment once
# (catches missing packages or Atari ROMs now, not after hours in the queue)
# and checks what main.py / posthoc.py would refuse -- seeds that already hold
# logs (unless --overwrite), settings that differ from the round's, missing
# checkpoints, results already there -- so nothing fails in the queue.
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
#   --task      train | exploit | crossplay   (see above)
#   --checkpoints  with --task exploit: final (default), all, step_<N> or
#               fractions of the budget, comma-separated; one task each
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
TASK=train
CHECKPOINTS=final

METHODS=()
ENV_NAME="kuhn_poker"
BUDGET=100000
SEEDS="0"
EXP_NAME=""
PASS=()          # everything else goes to main.py / posthoc.py untouched

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
    --task)     TASK="$2";     shift 2;;
    --checkpoints) CHECKPOINTS="$2"; shift 2;;
    --method)   METHODS+=("$2"); shift 2;;
    --env)      ENV_NAME="$2"; shift 2;;
    --budget)   BUDGET="$2";   shift 2;;
    --seeds)    SEEDS="$2";    shift 2;;
    --exp-name) EXP_NAME="$2"; shift 2;;
    --results-dir) echo "Use --out-root instead of --results-dir." >&2; exit 1;;
    *)          PASS+=("$1");  shift;;
  esac
done
case "$TASK" in
  train|exploit|crossplay) ;;
  *) echo "--task must be train, exploit or crossplay, got '$TASK'." >&2; exit 1;;
esac
if [[ -z "$EXP_NAME" ]]; then
  echo "--exp-name is required: it names the round (e.g. si_gems)." >&2; exit 1
fi
if [[ "$TASK" == train && ${#METHODS[@]} -ne 1 ]]; then
  echo "--task train needs exactly one --method (got ${#METHODS[@]}): a round is one method;" >&2
  echo "submit another method as another round (another --exp-name)." >&2
  exit 1
fi
if [[ "$TASK" != train && ${#METHODS[@]} -gt 0 ]]; then
  echo "--method is not used by --task $TASK (the round's config.json has it)." >&2; exit 1
fi
if [[ -z "$ACCOUNT" && $DRY_RUN -eq 0 ]]; then
  echo "No --account given and \$SEPSRO_ACCOUNT is unset." >&2
  echo "Alliance clusters reject jobs without one, e.g. --account def-yourpi" >&2
  exit 1
fi

ROUND_DIR="$OUT_ROOT/$EXP_NAME"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"
CKPT_LIST=(final)
[[ "$TASK" == exploit ]] && IFS=',' read -ra CKPT_LIST <<< "$CHECKPOINTS"
TASKS=()                                   # "seed checkpoint" per array index
for s in "${SEED_LIST[@]}"; do
  for c in "${CKPT_LIST[@]}"; do TASKS+=("$s $c"); done
done
[[ "$TASK" == crossplay ]] && TASKS=("all final")   # one job over the round
N_TASKS=${#TASKS[@]}

# ---- checks on the login node, where failures are cheap -------------------- #
if command -v module >/dev/null 2>&1; then module load "$PYTHON_MODULE"; fi
# shellcheck disable=SC1091
source "$VENV/bin/activate" 2>/dev/null || {
  echo "Could not activate $VENV. Run setup_cluster.sh, or pass --venv." >&2; exit 1; }

echo "[submit] checking round '$EXP_NAME' on the login node"
( cd "$REPO" && python - "$TASK" "$ROUND_DIR" "${METHODS[0]:-}" "$ENV_NAME" "$BUDGET" \
    "$OUT_ROOT" "$EXP_NAME" "${#TASKS[@]}" "${TASKS[@]}" "${PASS[@]+"${PASS[@]}"}" <<'PYEOF'
import json, os, sys
import warnings
warnings.filterwarnings('ignore')

task, round_dir, method, env_name, budget, out_root, exp = sys.argv[1:8]
n = int(sys.argv[8])
tasks, passed = sys.argv[9:9 + n], sys.argv[9 + n:]

from config import parse_args, parse_env
from sepsro.envs import make_env

if task != 'train':   # the round says which env
    try:
        with open(os.path.join(round_dir, 'round.json'), encoding='utf-8') as handle:
            env_name = json.load(handle)['env']
    except OSError:
        sys.exit(f'[submit] {round_dir} has no round.json: train the round first.')
try:
    env = make_env(parse_env(env_name))
    env.reset(seed=0)
    spec = env.spec
    env.close()
except Exception as exc:
    sys.exit(f'[submit] cannot build env {env_name!r}: {type(exc).__name__}: {exc}\n'
             '         (missing package? for space_invaders run AutoROM on the login node)')
print(f'[submit] env ok: {env_name}: {spec.backend}, agents {list(spec.agents)}, '
      f'obs {dict(spec.obs_shape)}')

if task == 'train':
    import main
    problems = []
    for item in tasks:
        seed = item.split()[0]
        args = parse_args(['--method', method, '--env', env_name, '--budget', budget,
                           '--seed', seed, '--exp-name', exp, '--results-dir', out_root, *passed])
        problem = main.run_conflict(args)
        if problem:
            problems.append(problem)
    if problems:
        sys.exit('[submit] refused:\n  ' + '\n  '.join(problems))
    sys.exit()

import posthoc
overwrite = '--overwrite' in passed
if task == 'crossplay':
    against = []
    if '--against' in passed:
        for a in passed[passed.index('--against') + 1:]:
            if a.startswith('--'):
                break
            against.append(a)
    mine = posthoc.finished_seeds(round_dir)
    if len(mine) < 2 and not against:
        sys.exit(f'[submit] {round_dir} has {len(mine)} finished seed(s): within-round '
                 'cross-play needs two; add --against <round>.')
    for other in against:
        if not posthoc.finished_seeds(posthoc.resolve_round(other, near=round_dir)):
            sys.exit(f'[submit] round {other} has no finished seed.')
    out = os.path.join(round_dir, 'stats', 'crossplay')
    names = (['within'] if len(mine) > 1 else []) + [f'vs_{os.path.basename(a)}' for a in against]
    done = [x for x in names if os.path.exists(os.path.join(out, f'{x}.json'))]
    if done and not overwrite:
        sys.exit(f'[submit] {out} already has {", ".join(done)} (use --overwrite).')
    print(f'[submit] cross-play: {len(mine)} seed(s), against {against or "no other round"}')
    sys.exit()

problems = []
for item in tasks:
    seed, which = item.split()
    run_dir = os.path.join(round_dir, 'stats', f'seed{seed}')
    try:
        cfg = posthoc._read_config(run_dir)
    except OSError:
        sys.exit(f'[submit] {run_dir} has no config.json: train it first.')
    for step, name in posthoc.select_checkpoints(cfg, which):   # exits on a missing one
        if os.path.exists(posthoc.exploit_path(run_dir, name)):
            problems.append(posthoc.exploit_path(run_dir, name))
if problems and not overwrite:
    sys.exit('[submit] already scored (use --overwrite):\n  ' + '\n  '.join(problems))
PYEOF
)

# ---- build the job script ------------------------------------------------- #
# Slurm reads #SBATCH lines literally (no shell quoting), so the job name is
# sanitised; the real --exp-name, parentheses and all, still reaches main.py.
SAFE_NAME="${EXP_NAME//[^A-Za-z0-9._-]/_}"
[[ "$TASK" != train ]] && SAFE_NAME="${SAFE_NAME}_${TASK}"
SLURM_DIR="$ROUND_DIR/slurm"
mkdir -p "$SLURM_DIR"
JOB="$SLURM_DIR/${SAFE_NAME}.sbatch"

q() { printf '%q' "$1"; }
TASK_LINES=""
for t in "${TASKS[@]}"; do TASK_LINES+="  $(q "$t")"$'\n'; done
MAIN_ARGS=""
for a in "${PASS[@]+"${PASS[@]}"}"; do MAIN_ARGS+=" $(q "$a")"; done

GRES_LINE=""
case "$TASK" in
  train)     COMMAND="python main.py --method $(q "${METHODS[0]}") --seed \"\$SEED\" --env $(q "$ENV_NAME") --budget $(q "$BUDGET") \\
    --exp-name $(q "$EXP_NAME") --results-dir $(q "$OUT_ROOT")${MAIN_ARGS}";;
  exploit)   COMMAND="python posthoc.py exploit $(q "$ROUND_DIR")/stats/seed\$SEED --checkpoint \"\$CKPT\"${MAIN_ARGS}";;
  crossplay) COMMAND="python posthoc.py crossplay $(q "$ROUND_DIR")${MAIN_ARGS}";;
esac
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
#SBATCH --output=${SLURM_DIR}/%x-%A_%a.out
${GRES_LINE}

set -euo pipefail
TASKS=(
${TASK_LINES})
read -r SEED CKPT <<< "\${TASKS[\$SLURM_ARRAY_TASK_ID]}"

module load ${PYTHON_MODULE}
source $(q "$VENV")/bin/activate
cd $(q "$REPO")
# Torch and NumPy otherwise size their thread pools to the whole node.
export OMP_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}

echo "task    : \$SLURM_ARRAY_TASK_ID  round=$(q "$EXP_NAME") ${TASK} seed=\$SEED checkpoint=\$CKPT  on \$(hostname)"
echo "commit  : \$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "started : \$(date)"

${COMMAND}

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
echo "[submit] watch with: squeue -u \$USER     logs in $SLURM_DIR"
