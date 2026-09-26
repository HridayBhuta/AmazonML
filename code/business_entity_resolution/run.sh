#!/bin/bash
# Amazon ML Challenge 2026 - one-command runner for macOS (Apple Silicon or Intel) and Linux.
#   bash run.sh            -> setup + smoke test + baseline + experiment sweep + pick best + final zip
#   bash run.sh setup      -> only create the Python environment
#   bash run.sh <command>  -> smoke | run | sweep | compare | finalize | prune | locate  (extra args passed on)
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
mkdir -p logs
CMD="${1:-auto}"; [ $# -gt 0 ] && shift || true
LOG="$ROOT/logs/${CMD}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=================================================================="
echo " AMAZON ML CHALLENGE - STARTED ($CMD)   $(date)"
echo " Keep this Terminal window OPEN and the Mac plugged in (lid open)."
echo " Text will keep appearing below - that is the progress."
echo " Log file: $LOG"
echo "=================================================================="

# ---- one job at a time (read-only commands skip the lock) --------------------------------------
if [ "$CMD" != "compare" ] && [ "$CMD" != "locate" ]; then
  LOCK="$ROOT/.run.lock"
  if ! mkdir "$LOCK" 2>/dev/null; then
    OLD="$(cat "$LOCK/pid" 2>/dev/null || true)"
    if [ -n "$OLD" ] && kill -0 "$OLD" 2>/dev/null; then
      echo "Another run is already active (pid $OLD). Wait for it to finish, or stop it with: kill $OLD"
      exit 1
    fi
    rm -rf "$LOCK"; mkdir "$LOCK"
  fi
  echo $$ > "$LOCK/pid"
  trap 'rm -rf "$LOCK"' EXIT
fi

case "$ROOT" in
  "$HOME/Desktop"*|"$HOME/Documents"*|"$HOME/Library/Mobile Documents"*)
    echo "NOTE: this folder is on Desktop/Documents/iCloud Drive. If iCloud syncs those folders, the multi-GB"
    echo "      caches will be uploaded. Moving the folder to your home folder (Finder > Go > Home) is better." ;;
esac

pick_python() {
  for c in python3.12 python3.11 python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
           /usr/local/bin/python3.12 /usr/local/bin/python3.11; do
    if command -v "$c" >/dev/null 2>&1; then
      v=$("$c" -c 'import sys; print("%d%02d" % sys.version_info[:2])' 2>/dev/null || echo 0)
      if [ "$v" -ge 311 ] && [ "$v" -le 313 ]; then command -v "$c"; return 0; fi
    fi
  done
  return 1
}

installed_ok() {
  [ -f .venv/.complete ] && [ -x .venv/bin/python ] && .venv/bin/python - <<'PY' 2>/dev/null
import importlib.metadata as m, pathlib
want = dict(l.strip().split("==") for l in pathlib.Path("requirements.txt").read_text().splitlines() if "==" in l)
bad = [p for p, v in want.items() if m.version(p) != v]
raise SystemExit(1 if bad else 0)
PY
}

setup() {
  echo ""
  echo ">>> STEP 1: preparing Python (first time: 2-5 minutes, needs internet; later: seconds)"
  if installed_ok; then
    echo "python env: ok (.venv)"
  else
    PY="$(pick_python || true)"
    if [ -z "$PY" ]; then
      echo "No Python 3.11-3.13 found; downloading a private copy with uv (no admin rights needed)..."
      mkdir -p .tools
      if [ ! -x .tools/uv ]; then
        case "$(uname -s)-$(uname -m)" in
          Darwin-arm64) A=uv-aarch64-apple-darwin ;;
          Darwin-x86_64) A=uv-x86_64-apple-darwin ;;
          Linux-x86_64) A=uv-x86_64-unknown-linux-gnu ;;
          Linux-aarch64) A=uv-aarch64-unknown-linux-gnu ;;
          *) echo "Unsupported platform $(uname -sm)"; exit 1 ;;
        esac
        curl -fsSL "https://github.com/astral-sh/uv/releases/latest/download/$A.tar.gz" -o .tools/uv.tgz
        tar -xzf .tools/uv.tgz -C .tools
        mv ".tools/$A/uv" .tools/uv && rm -rf ".tools/$A" .tools/uv.tgz
      fi
      export UV_PYTHON_INSTALL_DIR="$ROOT/.tools/python"
      .tools/uv python install 3.12
      PY="$(.tools/uv python find 3.12)"
    fi
    echo "using $PY"
    rm -rf .venv
    "$PY" -m venv .venv
    echo "installing packages (numpy, scipy, polars, rapidfuzz, lightgbm)..."
    .venv/bin/python -m pip install --quiet --upgrade pip
    .venv/bin/python -m pip install --progress-bar off -r requirements.txt
    touch .venv/.complete
  fi
  # LightGBM needs the OpenMP runtime (libomp) on macOS.
  rm -f .venv/omp_env.sh
  if ! .venv/bin/python -c "import lightgbm" 2>/dev/null; then
    if [ "$(uname -s)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
      echo "installing libomp with Homebrew"; brew install libomp || true
    fi
  fi
  if ! .venv/bin/python -c "import lightgbm" 2>/dev/null; then
    echo "using the OpenMP runtime bundled with scikit-learn"
    .venv/bin/python -m pip install --quiet "scikit-learn==1.7.2"
    OMPDIR="$(.venv/bin/python -c 'import sklearn, os; print(os.path.join(os.path.dirname(sklearn.__file__), ".dylibs"))')"
    {
      echo "export DYLD_LIBRARY_PATH=\"$OMPDIR:\${DYLD_LIBRARY_PATH:-}\""
      echo "export DYLD_FALLBACK_LIBRARY_PATH=\"$OMPDIR:\${DYLD_FALLBACK_LIBRARY_PATH:-}\""
      echo "export ER_OMP_LIB=\"$OMPDIR/libomp.dylib\""
    } > .venv/omp_env.sh
  fi
  if [ -f .venv/omp_env.sh ]; then source .venv/omp_env.sh; fi
  # self-test with exactly the launcher the pipeline uses (python started directly by this shell)
  .venv/bin/python - <<'PY'
import os, ctypes
lib = os.environ.get("ER_OMP_LIB")
if lib: ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
import numpy as np, lightgbm as lgb, polars, rapidfuzz
X = np.random.RandomState(0).rand(200, 3); y = (X[:, 0] > .5).astype(int)
lgb.train({"objective": "binary", "verbose": -1, "num_threads": 2}, lgb.Dataset(X, y), 5)
print("environment OK: lightgbm", lgb.__version__, "polars", polars.__version__, "rapidfuzz", rapidfuzz.__version__)
PY
  echo ">>> checking the program (unit tests)..."
  .venv/bin/python -m unittest discover -s tests
}

setup
[ "$CMD" = "setup" ] && exit 0
if [ -f .venv/omp_env.sh ]; then source .venv/omp_env.sh; fi
echo ""
echo ">>> STEP 2 onwards: the program itself. Lines with [time +seconds] are progress."
echo "    Quick test (3-5 min) -> full baseline (about 1.5-2 h) -> first files in final/ -> improvements."

export PYTHONUNBUFFERED=1
if command -v caffeinate >/dev/null 2>&1; then
  # keep the Mac awake while this script runs (python itself lowers its own priority);
  # caffeinate runs beside python, not in front of it, so DYLD_* variables survive.
  caffeinate -dimsu -w $$ &
fi
set +e
.venv/bin/python src/run_pipeline.py "$CMD" "$@"
STATUS=$?
set -e
if [ $STATUS -eq 0 ]; then
  echo "== finished OK $(date). Submission files (if finalized): $ROOT/final/"
else
  echo "== FAILED (exit $STATUS) $(date). See $LOG and runs/*/pipeline.log"
fi
exit $STATUS
