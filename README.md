# Agent Eval Harness

A small, readable harness for measuring how reliably LLM agents complete tool-using tasks — and how easily they can be pushed off course.

Most agent demos show one lucky run. This project asks the questions that matter before you ship an agent:

- **Does it finish the task?** Success rate across a suite of tasks with automatic checks.
- **Does it finish it every time?** Repeated runs to measure consistency, not just best case.
- **What does it cost?** Steps, tokens and time per task.
- **Can it be tricked?** Prompt-injection tests hidden inside the files the agent reads.
- **Why does it fail?** Every step is traced, so failures can be sorted into clear categories.

## How it works

```
task (JSON) ──► agent loop ──► tools (sandboxed workspace)
                   │  ▲
                   ▼  │
                 model (Anthropic / OpenAI / Ollama)
                   │
                   ▼
            checks ──► results (JSONL trace + summary)
```

- **Tasks** are JSON files: a prompt, some starting files, and one or more checks (for example, "the answer equals 62.5" or "todo.txt was not modified").
- **The agent** replies with one JSON action per turn: either a tool call or a final answer. This simple protocol works with any chat model, so different providers can be compared fairly.
- **Tools** run against an in-memory workspace, so runs are repeatable and nothing touches your real files.
- **The runner** executes each task, grades it, and saves the full step-by-step trace.

## Quick start

```bash
pip install -e ".[dev]"
python -m pytest                       # offline tests, no API key needed

agent-eval list                        # see the tasks
pip install -e ".[anthropic]"          # or .[openai]
export ANTHROPIC_API_KEY=...           # see .env.example
agent-eval run --model anthropic:claude-sonnet-4-5 -v
agent-eval run --model anthropic:claude-sonnet-4-5 --repeats 5   # measure consistency

# free local option
ollama pull llama3.1
agent-eval run --model ollama:llama3.1 -v
```

Results land in `results/<timestamp>_<model>/` as `runs.jsonl` (full traces, one line per run) and `summary.json`.

## Task suite

30 tasks in four categories, each with automatic checks and a reference solution:

| Category | Tasks | What it tests |
|---|---|---|
| `files` | 10 | reading, searching, editing and writing workspace files |
| `reasoning` | 8 | arithmetic, unit conversion, dates and small logic puzzles |
| `multi-step` | 5 | chaining several reads, a calculation and a write |
| `robustness` | 7 | messy input, missing or misnamed files, empty files, tool errors |

## Reliability metrics

One lucky run says little, so `--repeats N` runs every task N times and the summary reports:

- **success rate**: share of all runs that passed.
- **pass@k**: chance that at least one of k attempts succeeds (good when you can retry).
- **pass^k**: chance that *all* k attempts succeed (good when the agent must be dependable every time).
- **consistent / flaky tasks**: tasks that passed every trial vs. tasks that passed only sometimes, plus per-task variance `p(1-p)`.
- **efficiency**: average steps, tokens and seconds per run.
- **estimated cost**: tokens multiplied by a per-model price table (`agent_eval/pricing.py`); pass `--prices prices.json` to override, e.g. `{"openai:gpt-4o": [2.5, 10]}` in USD per million input/output tokens.

pass@k and pass^k use the unbiased estimators over n trials with c successes: `1 - C(n-c,k)/C(n,k)` and `C(c,k)/C(n,k)`. The wider the gap between them, the less consistent the agent.

Example end-of-run output (illustrative numbers; real benchmark results are pending):

```
27/30 runs passed (90%) across 10 tasks x 3 trial(s)
pass@3 100%  pass^3 70%  consistent 7/10  flaky 3
avg 2.9 steps, 1840 tokens, 2.1s per run; 4 tool errors
estimated cost $0.1932 ($0.00644 per run)
```

## Task format

```json
{
  "id": "sum-expenses",
  "category": "files",
  "prompt": "What is the total of the amount column in expenses.csv? Reply with just the number.",
  "files": {"expenses.csv": "item,amount\ncoffee,4.50\nlunch,12.25\n..."},
  "checks": [{"type": "answer_equals", "value": "62.5"}],
  "max_steps": 6,
  "solution": [
    {"tool": "read_file", "args": {"path": "expenses.csv"}},
    {"final": "62.5"}
  ]
}
```

Tasks are validated when loaded (unknown fields, unknown check types and missing check arguments are rejected with the file name). The optional `solution` is a list of actions that solves the task; the test suite replays every solution through the real grader, so a broken task is caught before it can mislead a benchmark. The model never sees it.

Check types: `answer_equals`, `answer_contains`, `answer_not_contains`, `file_equals`, `file_contains`, `file_unchanged`, `max_steps`.

## Project layout

```
agent_eval/
  agent.py    agent loop and action parsing
  models.py   model clients (Anthropic, OpenAI, Ollama, scripted for tests)
  tools.py    sandboxed workspace and tools
  tasks.py    task loading and checks
  runner.py   running, repeating, grading and summarizing results
  metrics.py  pass@k, pass^k and consistency metrics
  pricing.py  per-model price table and cost estimates
  cli.py      command line interface
tasks/        task suites (basic, files, reasoning, multi_step, robustness)
tests/        unit tests (run offline)
```

## Roadmap

- [x] Core agent loop, sandboxed tools, task format and checks
- [x] First task suite and offline test suite
- [x] Larger task suite (25+ tasks across files, reasoning, multi-step and robustness)
- [x] Reliability metrics: repeated runs, pass@k and consistency, cost per task
- [ ] Prompt-injection suite and an injection-resistance score
- [ ] Failure taxonomy: automatic labels for why each failed run failed
- [ ] HTML report and model leaderboard
- [ ] Mitigation experiments: measure defenses before and after
- [ ] CI with GitHub Actions
- [ ] Benchmark write-up comparing real models
