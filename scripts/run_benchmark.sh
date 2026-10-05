#!/usr/bin/env bash
# Run the full benchmark with real models.
#
# For each model: a baseline run of every task, then a run with the injection
# defenses switched on. Then a leaderboard (HTML + Markdown) and a before/after
# comparison per model, ready to paste into docs/BENCHMARK.md.
#
# Usage:
#   scripts/run_benchmark.sh                              # default models
#   scripts/run_benchmark.sh openai:gpt-4o-mini ollama:llama3.1
#
# Settings (environment variables):
#   REPEATS=3        trials per task (pass^k needs at least 2)
#   OUT=results/benchmark
#   DEFENSE=all      defenses for the second run (or "none" to skip it)
#   DRY_RUN=1        print the commands instead of running them
#
# API keys are read from the environment or from a .env file in the repo root
# (see .env.example). This script never prints them.
set -euo pipefail
cd "$(dirname "$0")/.."

REPEATS="${REPEATS:-3}"
OUT="${OUT:-results/benchmark}"
DEFENSE="${DEFENSE:-all}"
DRY_RUN="${DRY_RUN:-0}"
PY="${PYTHON:-python}"

if [ "$#" -gt 0 ]; then
  MODELS=("$@")
else
  MODELS=("anthropic:claude-haiku-4-5" "anthropic:claude-sonnet-4-5")
fi

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '+'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

# The label a defended run is saved under, e.g. "openai:gpt-4o +hardened_prompt+tag_untrusted".
defended_label() {
  "$PY" -c 'import sys
from agent_eval.defenses import resolve_defenses
from agent_eval.runner import run_label
print(run_label(sys.argv[1], [d.name for d in resolve_defenses(sys.argv[2])]))' "$1" "$DEFENSE"
}

TASKS=$("$PY" -m agent_eval list | wc -l | tr -d ' ')
PER_MODEL=$((TASKS * REPEATS))
[ "$DEFENSE" != "none" ] && PER_MODEL=$((PER_MODEL * 2))
echo "benchmark: ${#MODELS[@]} model(s), $TASKS tasks x $REPEATS trial(s), defense: $DEFENSE"
echo "           $PER_MODEL agent runs per model; results in $OUT"

check=()
for m in "${MODELS[@]}"; do check+=(--model "$m"); done
run "$PY" -m agent_eval check "${check[@]}"

for m in "${MODELS[@]}"; do
  echo
  echo "== $m: baseline"
  run "$PY" -m agent_eval run --model "$m" --repeats "$REPEATS" --out "$OUT"
  if [ "$DEFENSE" != "none" ]; then
    echo "== $m: defense $DEFENSE"
    run "$PY" -m agent_eval run --model "$m" --repeats "$REPEATS" --defense "$DEFENSE" --out "$OUT"
  fi
done

echo
run "$PY" -m agent_eval report "$OUT" --title "Agent Eval Benchmark" --markdown "$OUT/leaderboard.md"

if [ "$DEFENSE" != "none" ]; then
  for m in "${MODELS[@]}"; do
    slug=$(printf '%s' "$m" | tr ':/' '__')
    echo
    run "$PY" -m agent_eval compare "$m" "$(defended_label "$m")" --results "$OUT" \
      --markdown "$OUT/compare_${slug}.md"
  done
fi

echo
echo "done. Open $OUT/report.html, and copy $OUT/*.md into docs/BENCHMARK.md."
