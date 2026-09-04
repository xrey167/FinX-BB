#!/usr/bin/env bash
# The single entry point. From a freshly cloned repository on a bare Ubuntu server:
#
#   ./run.sh                 set up, run the unit tests, then the 15-minute smoke chain
#   ./run.sh test            set up and run the unit tests only
#   ./run.sh synthetic       set up and reproduce the synthetic chain      (~3 h on 4 cores)
#   ./run.sh gpt2            set up and reproduce the frozen-GPT-2 chain   (~20 h, downloads GPT-2)
#   ./run.sh all             synthetic, then gpt2                          (~23 h)
#
# Everything after the stage name is passed to make, so ./run.sh gpt2 SEEDS="0" works.
# Setup is skipped automatically once .venv exists and imports cleanly.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

STAGE="${1:-smoke}"
[ $# -gt 0 ] && shift || true
case "$STAGE" in
    test|smoke|synthetic|gpt2|report|all) ;;
    -h|--help|help) sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown stage '$STAGE' — try: test smoke synthetic gpt2 report all"; exit 2 ;;
esac

if [ -x .venv/bin/python ] && .venv/bin/python -c "import torch, numpy, transformers" 2>/dev/null; then
    echo "== environment already present, skipping setup (delete .venv to force it)"
else
    ./setup.sh
fi
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

if command -v make >/dev/null 2>&1; then
    if [ "$STAGE" = "all" ]; then
        make synthetic PY="$PY" "$@" && make gpt2 PY="$PY" "$@"
    else
        make "$STAGE" PY="$PY" "$@"
    fi
else
    # make is normally installed by setup.sh; this keeps the script usable if it is not
    echo "== make not available, calling the modules directly"
    case "$STAGE" in
        test)      "$PY" -m pytest so/tests -q ;;
        smoke)     "$PY" -m so.experiments.run_all --quick ;;
        synthetic) "$PY" -m so.experiments.run_all && "$PY" -m so.report ;;
        report)    "$PY" -m so.report ;;
        *)         echo "stage '$STAGE' needs make; install it with: sudo apt-get install make"; exit 3 ;;
    esac
fi
