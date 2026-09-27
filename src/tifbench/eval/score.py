"""Score released forecasts against the frozen benchmark."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from tifbench.benchmark import load_benchmark_rows

MODELS = (
    "jev", "kev", "gpt_astra", "claude_opus_5_5", "gemma_3_12b", "qwen3_5_9b"
)
HORIZONS = (21, 14, 7, 3)
LOG_LOSS_EPS = 1e-12


def _metrics(pairs: list[tuple[float, int]]) -> tuple[float, float, float]:
    n = len(pairs)
    if not n:
        raise ValueError("Empty scoring group")
    brier = sum((p - y) ** 2 for p, y in pairs) / n
    log_loss = sum(
        -(y * math.log(min(1 - LOG_LOSS_EPS, max(LOG_LOSS_EPS, p)))
          + (1 - y) * math.log(1 - min(1 - LOG_LOSS_EPS, max(LOG_LOSS_EPS, p))))
        for p, y in pairs
    ) / n
    accuracy = sum(int((p >= 0.5) == bool(y)) for p, y in pairs) / n
    return brier, log_loss, accuracy


def score_forecasts(benchmark_path: Path, forecasts_path: Path) -> list[dict]:
    benchmark_rows = load_benchmark_rows(benchmark_path)
    benchmark = {row["instance_id"]: row for row in benchmark_rows}
    if len(benchmark) != 848 or len({row["market"]["market_id"] for row in benchmark_rows}) != 212:
        raise ValueError("Expected the released 212-market, 848-instance benchmark")
    groups: dict[tuple[str, str, int], list[tuple[float, int]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    main_counts = defaultdict(int)
    sweep_counts = defaultdict(int)
    with forecasts_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            instance_id, model, k = row["instance_id"], row["model"], row["top_k"]
            key = (instance_id, model, k)
            if key in seen or instance_id not in benchmark or model not in MODELS or k not in (3, 5, 7, 10):
                raise ValueError(f"Unknown or duplicate forecast: {key}")
            seen.add(key)
            item = benchmark[instance_id]
            if row["evidence_ids"] != [e["article_id"] for e in item["evidence"][:k]]:
                raise ValueError(f"Evidence mismatch: {key}")
            p = row["probability_yes"]
            if not isinstance(p, (int, float)) or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError(f"Invalid probability: {key}")
            y = int(item["outcome"] == "YES")
            horizon = int(item["forecast"]["horizon_days"])
            if k == 3:
                main_counts[model] += 1
                groups[("overall", model, 3)].append((p, y))
                groups[(f"{horizon}d", model, 3)].append((p, y))
                if horizon == 3 and len(item["evidence"]) >= 10:
                    groups[("evidence_budget", model, 3)].append((p, y))
            else:
                if horizon != 3 or len(item["evidence"]) < 10:
                    raise ValueError(f"Forecast outside evidence-budget cohort: {key}")
                sweep_counts[(model, k)] += 1
                groups[("evidence_budget", model, k)].append((p, y))
    if any(main_counts[model] != 848 for model in MODELS):
        raise ValueError(f"Incomplete main forecasts: {dict(main_counts)}")
    if any(sweep_counts[(model, k)] != 171 for model in MODELS for k in (5, 7, 10)):
        raise ValueError(f"Incomplete evidence-budget forecasts: {dict(sweep_counts)}")
    for item in benchmark_rows:
        y = int(item["outcome"] == "YES")
        p = item["polymarket"]["yes_probability"]
        horizon = int(item["forecast"]["horizon_days"])
        groups[("overall", "polymarket", 3)].append((p, y))
        groups[(f"{horizon}d", "polymarket", 3)].append((p, y))
    output = []
    for (cohort, model, k), pairs in sorted(groups.items()):
        brier, log_loss, accuracy = _metrics(pairs)
        output.append({"cohort": cohort, "model": model, "top_k": k, "n": len(pairs),
                       "brier": brier, "log_loss": log_loss, "accuracy": accuracy})
    return output


def write_scores(benchmark_path: Path, forecasts_path: Path, output_path: Path) -> None:
    rows = score_forecasts(benchmark_path, forecasts_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("cohort", "model", "top_k", "n", "brier", "log_loss", "accuracy"))
        writer.writeheader()
        writer.writerows(rows)
