#!/usr/bin/env bash
# Build the submission notebook from the canonical attack.py, push it as a Kaggle
# kernel, and poll until it finishes. Final leaderboard submission is one click in the
# notebook UI ("Submit to Competition") — CLI push alone does not attach to the LB.
#
# Usage: bash push_submission.sh [path/to/attack.py]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SUB_DIR="$SCRIPT_DIR/submission"
# Default: the canonical operation-code attack.py (kept OUTSIDE the repo).
ATTACK="${1:-/Users/xingyuanxue1122/Documents/coding/multi-step-tool-attacks/attack/attack.py}"
SLUG="cindyxue1122/msta-baseline"
PY="${PY:-/Users/xingyuanxue1122/Documents/coding/multi-step-tool-attacks/.venv/bin/python}"

echo "=== Building notebook from $ATTACK ==="
"$PY" "$SCRIPT_DIR/build_notebook.py" --attack "$ATTACK" --out "$SUB_DIR/msta_submission.ipynb"

echo "=== Pushing kernel $SLUG ==="
kaggle kernels push -p "$SUB_DIR"

echo "=== Polling status (Ctrl-C to stop watching; kernel keeps running) ==="
while true; do
    STATUS=$(kaggle kernels status "$SLUG" 2>&1 || true)
    echo "  $(date +%H:%M:%S) $STATUS"
    if echo "$STATUS" | grep -qiE "complete|error|failed|cancel"; then break; fi
    sleep 30
done

echo ""
echo "Kernel finished (Save&Run All self-test). Submit to the leaderboard with the"
echo "NEWER kaggle CLI (>=1.6; the venv has 2.2.3). Code comps need kernel + output + version:"
echo "  VER=\$(kaggle kernels status $SLUG >/dev/null; echo 2)   # use the version you just pushed"
echo "  $PY -m kaggle competitions submit -c ai-agent-security-multi-step-tool-attacks \\"
echo "      -k $SLUG -f submission.csv -v <VERSION> -m 'msg'"
echo "Then poll the score:"
echo "  kaggle competitions submissions -c ai-agent-security-multi-step-tool-attacks | head"
echo "Pull the Save&Run output (full error traces) with:"
echo "  kaggle kernels output $SLUG -p ./output"
