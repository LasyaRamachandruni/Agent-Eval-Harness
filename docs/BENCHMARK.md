# Benchmark: agent strategies and models on tool use, multi-hop QA and code

> **Status: results pending.** No model has been run yet. The benchmark is
> designed to cost $0: it runs open models locally through Ollama. Every
> table below is filled in from one command,
>
> ```bash
> scripts/run_real_models.sh
> ```
>
> and nothing on this page should be read as a measured model result until
> the placeholders are replaced. The only measured numbers here are the
> grader checks in [Harness checks](#harness-checks), which involve no model.

## Questions

1. Given identical tools, tasks and graders, which agent design gets the most
   right per dollar: one direct call, ReAct, ReAct plus a critique
   (Reflection), or retries with written lessons (Reflexion)?
2. Is a model dependable, not just successful: how wide is the gap between
   pass@5 and pass^5 at temperature 0.7?
3. Do the gains hold outside the 40 synthetic tasks, on multi-hop question
   answering (HotpotQA) and code (HumanEval)?
4. How easily can text inside a file take control of each agent design?

## Setup

| | |
|---|---|
| Harness version | `<git commit>` (also in each run's `meta.json`) |
| Date run | `<YYYY-MM-DD>` |
| Hardware | `<machine; Ollama GPU/CPU>` |
| Models | `llama3.1` (8B) and `qwen2.5:7b`, local through Ollama (no API cost) |
| Strategies | `direct`, `react`, `reflection` (2 rounds), `reflexion` (3 trials) |
| Protocol | JSON actions (`PROTOCOL=native` for provider tool calling) |
| Benchmarks | built-in 40 tasks; HotpotQA dev distractor, 200 questions (seed 1234); HumanEval, 164 problems |
| Trials per item | 5 |
| Temperature | 0.7 |
| Cost | $0 (local models); `<hours of GPU/CPU time>` |

Reproduce with:

```bash
pip install -e .
ollama serve &                         # models are pulled automatically
DRY_RUN=1 scripts/run_real_models.sh   # the plan, no calls
scripts/run_real_models.sh             # re-run to resume
```

Or open `notebooks/colab_free_run.ipynb` in Colab with a T4 GPU (free); it
keeps results and the cache on Google Drive so a disconnected session resumes.

The script downloads and checksums the data, runs `agent-eval selfcheck`,
checks every model's setup, prints `agent-eval plan` (an estimate of tokens
and cost per model, benchmark and strategy) and asks for confirmation. Results
go to `results/real/<benchmark>/<model>_<strategy>_<protocol>/`, with
`report.html` and `leaderboard.md` per benchmark.

## Harness checks

Measured on 2026-10-09 on a 2 vCPU Intel Xeon @ 2.10GHz, Python 3.13, with
`agent-eval selfcheck humaneval` (no model involved):

| Check | Result |
|---|---|
| HumanEval canonical solutions pass their hidden tests in the sandbox | 164/164 |
| Docstring examples agree with the canonical solution (Reflexion's visible self-check) | 73/75; HumanEval/47 and HumanEval/116 have wrong examples in the dataset |

Pending (the build environment could not download HotpotQA):
`agent-eval download hotpotqa && agent-eval selfcheck hotpotqa`, which reports
whether every gold answer scores exact match 1.0 through the grader, and how
often `search(question)` returns both supporting paragraphs in its top 3.

## Results

### Built-in tasks (40 tasks x 5 trials)

<!-- paste results/real/builtin/leaderboard.md -->

_Pending._

### HotpotQA-200 (exact match, F1)

<!-- paste results/real/hotpotqa/leaderboard.md -->

_Pending._

### HumanEval (pass@1, pass^5)

<!-- paste results/real/humaneval/leaderboard.md -->

_Pending._

### Effect of the injection defenses

<!-- scripts/run_benchmark.sh: paste results/benchmark/compare_<model>.md -->

_Pending._

## Findings

<!-- 3-5 short points, each backed by a number from the tables above:
- Which strategy gives the best success per dollar on each benchmark?
- Does reflexion beat react by more than the run-to-run noise (pass@5 vs pass^5)?
- Does direct (all paragraphs inline) beat react's search on HotpotQA?
- Which injection tasks hijacked the most strategies? Does a critique step help?
- Most common failure label per model, and what would fix it?
-->

_Pending._

## Limitations

- Small suites: 40 synthetic tasks, 200 HotpotQA questions, 164 HumanEval
  problems. Differences of a few points are within noise at this size.
- HotpotQA retrieval is over the 10 provided paragraphs, not open Wikipedia;
  it measures multi-hop reading, not search.
- The HumanEval sandbox isolates model code written in good faith; it is not
  a security boundary.
- Costs are estimates from list prices in `agent_eval/pricing.py`.
- 7-8B open models are far weaker than frontier API models; compare
  strategies on the same model rather than against published API numbers.
