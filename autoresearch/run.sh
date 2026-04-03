#!/bin/bash
# Autoresearch loop — launches Junie to iteratively optimize RL/classifer.py
# Usage: ./autoresearch/run.sh [max_experiments]
# Prerequisites: Ollama must be running (docker compose up ollama -d)

set -euo pipefail

MAX=${1:-50}
RESULTS="autoresearch/results.tsv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Check Ollama health
echo "Checking Ollama..."
if ! curl -s -f "http://localhost:11434/api/tags" > /dev/null 2>&1; then
    echo "ERROR: Ollama is not running. Start it first:"
    echo "  docker compose up ollama -d"
    exit 1
fi
echo "Ollama OK."

# Ensure we're on the autoresearch/experiments branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$CURRENT_BRANCH" != "autoresearch/experiments" ]; then
    echo "Switching to autoresearch/experiments branch..."
    git checkout -b autoresearch/experiments 2>/dev/null || git checkout autoresearch/experiments
fi

# Initialize results file if needed
if [ ! -f "$RESULTS" ]; then
    echo -e "git_hash\teval_score\teval_breadth\teval_best_ms\tdescription\tstatus" > "$RESULTS"
    echo "Created $RESULTS"
fi

# Baseline run
echo ""
echo "Running baseline eval..."
python autoresearch/eval.py > run.log 2>&1 || true
BASELINE=$(grep "^eval_score:" run.log | awk '{print $2}' || echo "0")
BASELINE_BREADTH=$(grep "^eval_breadth:" run.log | awk '{print $2}' || echo "0")
BASELINE_MS=$(grep "^eval_best_ms:" run.log | awk '{print $2}' || echo "9999")
BASELINE_HASH=$(git rev-parse --short HEAD)
STATUS=$(grep "^eval_status:" run.log | awk '{print $2}' || echo "fail")

echo "Baseline eval_score: $BASELINE (breadth=$BASELINE_BREADTH, best_ms=$BASELINE_MS)"

# Log baseline to results.tsv if not already present
if ! grep -q "$BASELINE_HASH" "$RESULTS" 2>/dev/null; then
    echo -e "${BASELINE_HASH}\t${BASELINE}\t${BASELINE_BREADTH}\t${BASELINE_MS}\tbaseline\t${STATUS}" >> "$RESULTS"
fi

# Main loop — Junie handles the inner experiment cycle
echo ""
echo "Starting Junie autoresearch loop (max $MAX experiments)..."
echo "Working branch: $(git rev-parse --abbrev-ref HEAD)"
echo ""

junie "$(cat autoresearch/program.md)

Current baseline eval_score: $BASELINE
Run up to $MAX experiments. Start now."

echo ""
echo "Junie loop complete. Results:"
cat "$RESULTS"
