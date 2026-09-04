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
if command -v apt-get >/dev/null 2>&1; then
    SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq python3 python3-pip python3-venv git curl >/dev/null
else
    echo "   apt-get not found, skipping (install python3, python3-venv and git yourself)"
fi
python3 --version

PY=python3
if [ "$VENV" = "1" ]; then
    echo "== virtualenv in .venv"
    [ -d .venv ] || python3 -m venv .venv
    PY=".venv/bin/python"
    "$PY" -m pip install --quiet --upgrade pip
fi

echo "== PyTorch, CPU build"
"$PY" -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
echo "== everything else"
"$PY" -m pip install --quiet -r so/requirements.txt

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
