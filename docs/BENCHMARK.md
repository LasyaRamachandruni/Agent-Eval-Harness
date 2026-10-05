# Benchmark: how reliable and how hijackable are real agents?

> **Status: results pending.** This page is a template. The tables below are
> filled in from `scripts/run_benchmark.sh`, which needs an API key and has not
> been run yet. Nothing here should be read as a measured result until the
> placeholders are replaced.

## Question

Given the same tools and the same 40 tasks, how often does each model finish
the job, does it finish it *every* time, how easily can text inside a file
take control of it, and do two cheap prompt-level defenses help without
breaking ordinary work?

## Setup

| | |
|---|---|
| Harness version | `<git commit>` |
| Date run | `<YYYY-MM-DD>` |
| Models | `<model 1>`, `<model 2>`, ... |
| Tasks | 40 (files 10, reasoning 8, multi-step 5, robustness 7, injection 10) |
| Trials per task | `<REPEATS>` |
| Temperature | 0 |
| Defenses (second run) | `hardened_prompt` + `tag_untrusted` |
| Total estimated cost | `<$ from the leaderboard>` |

Reproduce with:

```bash
pip install -e ".[anthropic]"            # and/or .[openai]
cp .env.example .env                     # add your API key(s)
scripts/run_benchmark.sh                 # or: scripts/run_benchmark.sh MODEL [MODEL ...]
```

The script runs every model without and then with defenses, writes
`results/benchmark/report.html`, and writes the Markdown tables used below
(`leaderboard.md` and one `compare_<model>.md` per model).

## Results

### Leaderboard

<!-- paste results/benchmark/leaderboard.md here -->

_Pending._

### Effect of the defenses

<!-- paste each results/benchmark/compare_<model>.md here, one subsection per model -->

_Pending._

## Findings

<!-- 3-5 short points, each backed by a number from the tables above. Prompts:
- Which model is most dependable (pass^k), not just most successful?
- Where is the gap between pass@k and pass^k widest, and on which tasks?
- Which injection tasks hijacked the most models? (see the per-task grid in report.html)
- Did the defenses raise resistance, and did they break any ordinary tasks?
- What is the most common failure label per model, and what would fix it?
-->

_Pending._

## Example failures

<!-- One or two short traces from runs.jsonl that show a failure label in
action, e.g. a followed_injection run on inj-fake-tool-result. Use
`agent-eval failures results/benchmark/<run-dir> --label followed_injection`. -->

_Pending._

## Limitations

- 40 small, synthetic tasks: results show relative behaviour on this suite,
  not general capability.
- One prompt format (a JSON action per turn) for every model; models with
  native tool calling might do better with it.
- Costs are estimates from list prices in `agent_eval/pricing.py`.
- The defenses only change prompts; they are a baseline, not a full
  security design.
