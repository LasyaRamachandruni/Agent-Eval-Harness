# Changelog

## Unreleased

- CI now also runs the test suite on Python 3.13, and the package metadata lists the supported Python versions.

## 1.1.0 — 2026-10-09

- **Agent strategies**: `--strategy direct|react|reflection|reflexion`, ported from Basic_-React_Agent_Langchain, Langgraph_Reflection_Agent and Neural-Reflexion-Agent onto the harness's own model interface. Reflexion uses the grounding check and stop-when-no-improvement score from Neural-Reflexion-Agent.
- **Native tool calling**: `--protocol native` for Anthropic, OpenAI and Ollama, next to the JSON-action protocol.
- **Benchmarks**: HotpotQA (seeded 200-question dev distractor subset, offline search over the provided paragraphs, official EM and F1) and HumanEval (pass@k with a sandboxed subprocess runner: timeout, resource limits, no network). `agent-eval download` verifies checksums; `agent-eval selfcheck` replays reference answers through the graders.
- **Runner**: concurrent runs (`--concurrency`), retries with backoff on rate limits, a response cache keyed by model, prompt, params and trial, resumable results (`--resume`, `--run-dir`), live cost and `--budget`, `--temperature` (default now 0.7) and `--max-tokens`.
- **Cost before running**: `agent-eval estimate` and `agent-eval plan`.
- **scripts/run_real_models.sh**: every model x benchmark x strategy, 5 repeats, with an estimate and confirmation first. Free by default (llama3.1 and qwen2.5:7b on Ollama); paid models are opt-in. `notebooks/colab_free_run.ipynb` runs it on Colab's free GPU.
- **Engineering**: type hints checked by mypy, ruff lint and format, pre-commit, coverage, typed settings from environment variables, JSON structured logging, CI with lint, type and test jobs.

Pending: real-model results (free to produce with the script or the Colab notebook).

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
