#!/usr/bin/env bash
# Run the real-model benchmark in one command:
#
#   the 40 built-in tasks + HotpotQA-200 + HumanEval (164)
#   x every strategy (direct, react, reflection, reflexion)
#   x claude-haiku-4-5, claude-sonnet-4-5, gpt-4o-mini, llama3.1 (Ollama)
#   x 5 repeats at temperature 0.7 (for pass^k)
#
# It downloads and verifies the data, checks every model is ready, prints a cost
# estimate and asks before spending anything. Results go to $OUT/<benchmark>/,
# one fixed directory per (model, strategy), so re-running the script after a
# crash or Ctrl-C resumes where it stopped; the response cache makes finished
# calls free. At the end it writes $OUT/<benchmark>/report.html and
# leaderboard.md for each benchmark.
#
# Usage:
#   scripts/run_real_models.sh                 # everything, asks before spending
#   YES=1 scripts/run_real_models.sh           # no confirmation prompt
#   DRY_RUN=1 scripts/run_real_models.sh       # print the plan and commands only
#   STRATEGIES="react reflexion" BENCHMARKS="humaneval" scripts/run_real_models.sh
#
# Settings (environment variables, all optional):
#   MODELS        space-separated model specs (default: the four above)
#   BENCHMARKS    builtin hotpotqa humaneval
#   STRATEGIES    direct react reflection reflexion
#   REPEATS=5  TEMPERATURE=0.7  PROTOCOL=json  CONCURRENCY=4  OLLAMA_CONCURRENCY=1
#   BUDGET        USD cap per (model, benchmark, strategy) run, e.g. BUDGET=20
#   OUT=results/real
#   SKIP_UNREADY=1  drop models that fail the setup check instead of stopping
#
# API keys come from the environment or a .env file in the repo root:
#   ANTHROPIC_API_KEY, OPENAI_API_KEY (and OLLAMA_HOST if Ollama is not local).
# This script never prints them.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

DEFAULT_MODELS="anthropic:claude-haiku-4-5-20251001 anthropic:claude-sonnet-4-5-20250929 openai:gpt-4o-mini ollama:llama3.1"
read -r -a MODELS <<< "${MODELS:-$DEFAULT_MODELS}"
read -r -a BENCHMARKS <<< "${BENCHMARKS:-builtin hotpotqa humaneval}"
read -r -a STRATEGIES <<< "${STRATEGIES:-direct react reflection reflexion}"
REPEATS="${REPEATS:-5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
PROTOCOL="${PROTOCOL:-json}"
CONCURRENCY="${CONCURRENCY:-4}"
OLLAMA_CONCURRENCY="${OLLAMA_CONCURRENCY:-1}"
BUDGET="${BUDGET:-}"
OUT="${OUT:-results/real}"
YES="${YES:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_UNREADY="${SKIP_UNREADY:-0}"
PY="${PYTHON:-python}"
CLI=("$PY" -m agent_eval)

run() {
  if [ "$DRY_RUN" = "1" ]; then
    printf '+'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

join() { local IFS="$1"; shift; echo "$*"; }

echo "== data"
needs_data=()
for b in "${BENCHMARKS[@]}"; do [ "$b" != "builtin" ] && needs_data+=("$b"); done
if [ "${#needs_data[@]}" -gt 0 ]; then
  run "${CLI[@]}" download "${needs_data[@]}"
  run "${CLI[@]}" selfcheck "${needs_data[@]}"
fi

echo
echo "== model setup"
ready=()
for m in "${MODELS[@]}"; do
  if [ "$DRY_RUN" = "1" ] || "${CLI[@]}" check --model "$m"; then
    ready+=("$m")
  elif [ "$SKIP_UNREADY" = "1" ]; then
    echo "skipping $m (not ready)"
  else
    echo "stopping: $m is not ready (fix it above, or set SKIP_UNREADY=1 to skip it)" >&2
    exit 1
  fi
done
if [ "${#ready[@]}" -eq 0 ]; then
  echo "no model is ready" >&2
  exit 1
fi

echo
echo "== cost estimate (before any model call)"
plan=()
for m in "${ready[@]}"; do plan+=(--model "$m"); done
run "${CLI[@]}" plan "${plan[@]}" --benchmarks "$(join , "${BENCHMARKS[@]}")" \
  --strategies "$(join , "${STRATEGIES[@]}")" --repeats "$REPEATS" --protocol "$PROTOCOL"

if [ "$DRY_RUN" != "1" ] && [ "$YES" != "1" ]; then
  read -r -p "Run all of this? [y/N] " answer
  case "$answer" in y|Y|yes|YES) ;; *) echo "stopped before spending anything"; exit 0 ;; esac
fi

for b in "${BENCHMARKS[@]}"; do
  for m in "${ready[@]}"; do
    conc="$CONCURRENCY"
    case "$m" in ollama:*) conc="$OLLAMA_CONCURRENCY" ;; esac
    for s in "${STRATEGIES[@]}"; do
      slug=$(printf '%s_%s_%s' "$m" "$s" "$PROTOCOL" | tr ':/ ' '___')
      echo
      echo "== $b / $m / $s"
      args=(run --model "$m" --benchmark "$b" --strategy "$s" --protocol "$PROTOCOL"
            --repeats "$REPEATS" --temperature "$TEMPERATURE" --concurrency "$conc"
            --run-dir "$OUT/$b/$slug")
      [ -n "$BUDGET" ] && args+=(--budget "$BUDGET")
      run "${CLI[@]}" "${args[@]}"
    done
  done
  echo
  run "${CLI[@]}" report "$OUT/$b" --title "Agent Eval: $b" --markdown "$OUT/$b/leaderboard.md"
done

echo
echo "done. Reports: $(for b in "${BENCHMARKS[@]}"; do printf '%s ' "$OUT/$b/report.html"; done)"
echo "Copy the leaderboard.md tables into docs/BENCHMARK.md with the date, commit and hardware."
