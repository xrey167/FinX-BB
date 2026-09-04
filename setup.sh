#!/usr/bin/env bash
# One command from a fresh Ubuntu server to a working SO research environment.
#
#   ./setup.sh              CPU-only PyTorch (small download, what every recorded result used)
#   ./setup.sh --system     no virtualenv, install into the current Python
#
# Tested on Ubuntu 22.04 and 24.04. Nothing here needs a GPU.
set -euo pipefail

VENV=1
[ "${1:-}" = "--system" ] && VENV=0
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "== system packages"
NEED=""
for c in python3 pip3 make git; do command -v "$c" >/dev/null 2>&1 || NEED="$NEED $c"; done
python3 -c "import venv" >/dev/null 2>&1 || NEED="$NEED python3-venv"
if [ "${NO_APT:-0}" = "1" ]; then
    echo "   NO_APT=1, skipping"
elif [ -z "$NEED" ]; then
    echo "   already present: python3, pip3, make, git, venv"
elif command -v apt-get >/dev/null 2>&1; then
    echo "   missing:$NEED - installing"
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    # A slow mirror or an offline machine must not hang the whole run: apt gets a budget and its
    # failure is reported rather than fatal. The import check further down is what actually decides.
    timeout 300 $SUDO apt-get update -qq || echo "   apt-get update unfinished after 300 s, continuing"
    timeout 600 $SUDO apt-get install -y -qq python3 python3-pip python3-venv make git curl >/dev/null \
        || echo "   apt-get install failed, continuing with what is installed"
else
    echo "   apt-get not found; install python3, python3-venv, make and git yourself"
fi
python3 --version

PY=python3
if [ "$VENV" = "1" ]; then
    echo "== virtualenv in .venv"
    [ -d .venv ] || python3 -m venv .venv
    PY=".venv/bin/python"
    "$PY" -m pip install --quiet --upgrade pip
fi

if "$PY" -c "import torch, numpy, transformers" >/dev/null 2>&1; then
    echo "== python packages already importable, skipping the install"
else
    echo "== PyTorch, CPU build (about 200 MB)"
    "$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu \
        || "$PY" -m pip install --quiet torch
    echo "== everything else"
    "$PY" -m pip install --quiet -r so/requirements.txt
fi

echo "== check"
"$PY" - <<'PYCHK'
import numpy, torch, transformers, sys
print(f"   python {sys.version.split()[0]}  torch {torch.__version__}  "
      f"numpy {numpy.__version__}  transformers {transformers.__version__}")
print(f"   threads torch sees: {torch.get_num_threads()}")
PYCHK
"$PY" -m pytest so/tests -q 2>&1 | tail -2

cat <<'MSG'

Ready. Next steps, cheapest first:

  make test        unit tests only, about 10 seconds
  make smoke       a reduced version of the whole synthetic chain, about 15 minutes
  make synthetic   the recorded synthetic chain, about 3 hours on 4 cores
  make gpt2        the frozen-GPT-2 chain, about 20 hours on 4 cores (downloads GPT-2 once, ~550 MB)
  make report      rebuild docs/so-results-2026-09-02.md from whatever is in so/results/

If you created a virtualenv, either activate it (source .venv/bin/activate) or
pass it to make:  make smoke PY=.venv/bin/python
MSG
