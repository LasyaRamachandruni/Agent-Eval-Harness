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

# free local option
ollama pull llama3.1
agent-eval run --model ollama:llama3.1 -v
```

Results land in `results/<timestamp>_<model>/` as `runs.jsonl` (full traces) and `summary.json`.

## Task format

```json
{
  "id": "sum-expenses",
  "category": "files",
  "prompt": "What is the total of the amount column in expenses.csv? Reply with just the number.",
  "files": {"expenses.csv": "item,amount\ncoffee,4.50\nlunch,12.25\n..."},
  "checks": [{"type": "answer_equals", "value": "62.5"}],
  "max_steps": 6
}
```

Check types: `answer_equals`, `answer_contains`, `answer_not_contains`, `file_equals`, `file_contains`, `file_unchanged`, `max_steps`.

## Project layout

```
agent_eval/
  agent.py    agent loop and action parsing
  models.py   model clients (Anthropic, OpenAI, Ollama, scripted for tests)
  tools.py    sandboxed workspace and tools
  tasks.py    task loading and checks
  runner.py   running, grading and saving results
  cli.py      command line interface
tasks/        task suites
tests/        unit tests (run offline)
```

## Roadmap

- [x] Core agent loop, sandboxed tools, task format and checks
- [x] First task suite and offline test suite
- [ ] Larger task suite (25+ tasks across files, reasoning, multi-step and robustness)
- [ ] Reliability metrics: repeated runs, pass@k and consistency, cost per task
- [ ] Prompt-injection suite and an injection-resistance score
- [ ] Failure taxonomy: automatic labels for why each failed run failed
- [ ] HTML report and model leaderboard
- [ ] Mitigation experiments: measure defenses before and after
- [ ] CI with GitHub Actions
- [ ] Benchmark write-up comparing real models
