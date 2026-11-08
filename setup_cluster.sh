#!/usr/bin/env bash
#
# One-time environment setup on a Digital Research Alliance cluster.
# Run on a LOGIN node (it needs internet), from the repository root:
#
#   bash setup_cluster.sh                     # venv at ~/venvs/sepsro
#   bash setup_cluster.sh --venv ~/venvs/x    # elsewhere
#   bash setup_cluster.sh --no-atari          # skip the Atari / Space Invaders packages
#
# What it does:
#   1. loads the Python module (python/3.11, the version SESiL-II uses there);
#   2. creates a virtualenv with the Alliance-recommended --no-download;
#   3. installs every package from the Alliance wheelhouse where it exists
#      (--no-index: wheels built for the cluster), falling back to PyPI for the
#      rest (open_spiel, pettingzoo, ...), which login nodes can reach --
#      including PyPI's manylinux wheels, which Alliance's pip otherwise skips;
#   4. checks that everything imports.
#
# The Atari ROMs are NOT downloaded here: AutoROM asks you to accept the ROM
# licence, so run it yourself afterwards (see the message at the end).
#
# Compute nodes have no internet, so everything a job needs must be installed
# by this script on the login node.
set -euo pipefail

VENV="${SEPSRO_VENV:-$HOME/venvs/sepsro}"
PYTHON_MODULE="${SEPSRO_PYTHON_MODULE:-python/3.11}"
ATARI=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)          VENV="$2"; shift 2;;
    --python-module) PYTHON_MODULE="$2"; shift 2;;
    --no-atari)      ATARI=0; shift;;
    *) echo "Unknown option $1" >&2; exit 1;;
  esac
done

cd "$(dirname "$0")"

if command -v module >/dev/null 2>&1; then
  module load "$PYTHON_MODULE"
else
  echo "[setup] no 'module' command: not on an Alliance cluster, using $(command -v python3)"
fi

if [[ -d "$VENV" ]]; then
  echo "[setup] reusing existing venv $VENV"
else
  echo "[setup] creating venv $VENV"
  if command -v virtualenv >/dev/null 2>&1; then
    virtualenv --no-download "$VENV"
  else
    python3 -m venv "$VENV"
  fi
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install --no-index --upgrade pip 2>/dev/null || pip install --upgrade pip

PYVER="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
SITE="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# Alliance's Python refuses generic manylinux wheels from PyPI (it wants
# packages built for the cluster), so pip falls back to the source tarball --
# for open_spiel that is a C++20 build needing clang, which fails. The PyPI
# wheels themselves run fine there (open_spiel needs glibc >= 2.27 and
# GLIBCXX_3.4.22; the 2023 stack has newer), so ask for the manylinux tag
# explicitly and install the wheel into the venv's site-packages. Its
# dependencies then go through install() like everything else.
MANYLINUX=(--platform manylinux2014_x86_64)   # pip doesn't expand an explicit
for minor in $(seq 17 28); do                  # --platform, so list every tag
  MANYLINUX+=(--platform "manylinux_2_${minor}_x86_64")
done

install_manylinux() {
  local pkg="$1"
  pip install --no-deps --only-binary=:all: --upgrade --target "$SITE" \
      "${MANYLINUX[@]}" --implementation cp \
      --python-version "$PYVER" --abi "cp${PYVER/./}" --abi abi3 --abi none \
      "$pkg" >/dev/null 2>&1 || return 1
  local deps
  deps="$(python - "$pkg" <<'PYEOF'
import sys
from importlib.metadata import requires
from pip._vendor.packaging.requirements import Requirement
req = Requirement(sys.argv[1])
for line in requires(req.name) or []:
    dep = Requirement(line)
    if dep.marker is None or any(dep.marker.evaluate({'extra': e}) for e in req.extras | {''}):
        print(str(dep).split(';')[0].strip())
PYEOF
)"
  if [[ -n "$deps" ]]; then
    local dep_list
    mapfile -t dep_list <<< "$deps"
    install "${dep_list[@]}"
  fi
}

# Wheelhouse first; then PyPI wheels; then PyPI's manylinux wheel; source last.
install() {
  local pkg
  for pkg in "$@"; do
    if pip install --no-index "$pkg" >/dev/null 2>&1; then
      echo "[setup] $pkg  (Alliance wheelhouse)"
    elif pip install --only-binary=:all: "$pkg" >/dev/null 2>&1; then
      echo "[setup] $pkg  (PyPI)"
    elif install_manylinux "$pkg"; then
      echo "[setup] $pkg  (PyPI manylinux wheel)"
    else
      echo "[setup] $pkg  (PyPI, building from source)"
      pip install "$pkg"
    fi
  done
}

# torch: the wheelhouse build matches the cluster's CUDA; nothing in SEPSRO
# needs a GPU yet, but the same wheel works on CPU nodes.
install torch numpy scipy matplotlib pytest
install gymnasium pygame "pettingzoo>=1.24" open_spiel
if [[ $ATARI -eq 1 ]]; then
  install "pettingzoo[atari]" multi_agent_ale_py pygame-ce autorom
fi

echo "[setup] checking imports"
python - <<'PYEOF'
import importlib
for name in ('torch', 'numpy', 'scipy', 'matplotlib', 'gymnasium', 'pettingzoo', 'pyspiel'):
    module = importlib.import_module(name)
    print(f'  {name:12s} {getattr(module, "__version__", "ok")}')
PYEOF
python -c "import config, main" && echo "[setup] SEPSRO imports fine"

cat <<EOF

[setup] done. venv: $VENV
Next steps:
  * Atari ROMs (needed for --env space_invaders), on this login node:
        source $VENV/bin/activate && AutoROM
    (it shows the ROM licence and asks you to accept it)
  * Optional quick check of the codebase (~1 min, light enough for a login node):
        source $VENV/bin/activate && pytest -q
  * Submit jobs with submit.sh (bash submit.sh --help).
EOF
