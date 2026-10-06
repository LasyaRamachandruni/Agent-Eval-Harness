# Changelog

## 1.0.0 — 2026-10-06

First complete release. Everything runs offline except real model calls.

- **Core**: model-agnostic JSON action protocol, agent loop with error recovery, in-memory sandboxed tools (`list_files`, `read_file`, `write_file`, `calculator`), clients for Anthropic, OpenAI and Ollama, plus a scripted model for tests.
- **Tasks**: 40 tasks in five categories (files, reasoning, multi-step, robustness, injection), schema validation on load, a reference solution for every task, and `agent-eval validate` to replay solutions and attacks through the grader.
- **Reliability metrics**: `--repeats`, pass@k, pass^k, per-task variance, consistent and flaky tasks, average steps/tokens/time, estimated cost from a per-model price table.
- **Prompt injection**: 10 injection tasks with scripted attacks, separate injection checks, and an injection-resistance score.
- **Failure taxonomy**: every failed run is labelled (`followed_injection`, `modified_protected_file`, `model_error`, `wrong_answer`, `max_steps`, `bad_format`, `unknown_tool`, `tool_error_loop`); `agent-eval failures` groups them.
- **Reporting**: terminal leaderboard, self-contained HTML report, Markdown tables.
- **Mitigations**: `--defense hardened_prompt,tag_untrusted` and `agent-eval compare` for before/after runs.
- **Tooling**: `agent-eval check` (setup check without API calls), `agent-eval demo` (offline scripted agents), `scripts/run_benchmark.sh`, `docs/BENCHMARK.md` template, GitHub Actions CI.

Pending: benchmark write-up with real model results.

## 0.1.0 — 2026-09-29

Core loop, tools, model clients, first 5 tasks, tests and README.
