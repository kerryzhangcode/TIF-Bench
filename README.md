# TIF-Bench

TIF-Bench is the temporally controlled forecasting benchmark from *Do Typed Decision Models Forecast Better? A Controlled Evaluation on Prediction-Market Events*. It standardizes the information available to each model: every system receives the same timestamp-valid ranked news evidence, with no retrieval or market prices at prediction time. The release covers 212 resolved Polymarket markets and 848 instances at 21, 14, 7, and 3 days before closure.

## Released data

- `data/benchmark/benchmark.jsonl` / `schema.json`: frozen cohort (question, criteria, cutoff, ranked evidence, Polymarket probability, outcome).
- `results/forecasts.jsonl`: forecasts from Jev 1.13, Kev, GPT-Astra, Claude Opus 5.5, Gemma-3-12B, and Qwen3.5-9B. Main evaluation: top-3 articles on all 848 instances. Evidence-budget evaluation: `k ∈ {3,5,7,10}` on 171 three-day instances with at least ten articles.
- `results/metrics.csv`: Brier score, log loss, and accuracy (Polymarket included as an external baseline).

Join on `instance_id`. `top_k` is the evidence budget. Forecast rows store `probability_yes` and usage; outcomes and market prices remain in the benchmark for scoring.

## Reproduce the scores

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m tifbench score
```

Run these commands from the cloned repository root; the benchmark and model outputs are repository files, not bundled into the Python wheel.

Checks evidence prefixes and cohort membership before writing `results/metrics.csv`. Overall values match Table 3 of the paper (four decimal places).

## Run models on the benchmark

```bash
export OPENAI_API_KEY=...
export OPENROUTER_API_KEY=...
python -m tifbench run-formal --top-k 3
```

Models see only the question, resolution criteria, cutoff, and ranked news prefix. For the evidence-budget cohort, use `--horizon-hours 72 --min-evidence 10` with each `--top-k` in `3 5 7 10`. New runs write to `data/derived/formal/`.

## Benchmark construction

```text
sample-benchmark --collect → sample-benchmark --curate → construct-formal → export-benchmark
```

`data/benchmark/benchmark.jsonl` is the fixed paper cohort. Re-running collection or construction against live services may produce a different cohort.

## License

MIT. See [LICENSE](LICENSE).
