"""Thin CLI wiring for the historical evaluation pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from tifbench.benchmark import default_benchmark_dir, export_benchmark_from_instances
from tifbench.config import load_config
from tifbench.eval.formal import run_formal_forecasts
from tifbench.eval.score import write_scores
from tifbench.evidence.formal import CURATED_SUBDIR, construct_formal_dataset
from tifbench.io import ensure_dir, project_root
from tifbench.polymarket.sample_benchmark import (
    DEFAULT_MAX_PAGES_PER_TAG,
    parse_sample_months,
)
from tifbench.polymarket.sample_curate import run_sample_benchmark


def cmd_init_schema(_: argparse.Namespace) -> None:
    cfg = load_config()
    root = project_root()
    for key in ("raw", "curated", "derived", "benchmark"):
        if key in cfg["paths"]:
            ensure_dir(Path(cfg["paths"][key]))
    ensure_dir(Path(cfg["paths"]["curated"]) / "formal")
    ensure_dir(Path(cfg["paths"]["raw"]) / "markets")
    print(f"Initialized data directories under {root}")


def cmd_construct_formal(args: argparse.Namespace) -> None:
    filters = [
        name
        for name, val in (
            ("--event-id", args.event_id),
            ("--limit", args.limit),
            ("--months", args.months),
        )
        if val is not None
    ]
    if len(filters) > 1:
        raise SystemExit(
            "Use only one of --event-id, --limit, and --months "
            f"(got {', '.join(filters)})"
        )
    cfg = load_config()
    try:
        months = parse_sample_months(args.months, cfg=cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    construct_formal_dataset(
        cfg,
        from_sample=Path(args.from_sample),
        event_id=args.event_id,
        limit=args.limit,
        months=months,
        phase=args.phase,
        gpt_mode=args.gpt_mode,
    )


def cmd_export_benchmark(args: argparse.Namespace) -> None:
    """Re-export benchmark.jsonl from an existing instances.json (no news/API)."""
    cfg = load_config()
    instances = (
        Path(args.instances)
        if args.instances
        else Path(cfg["paths"]["curated"]) / CURATED_SUBDIR / "instances.json"
    )
    out_dir = Path(args.out) if args.out else default_benchmark_dir(cfg)
    path = export_benchmark_from_instances(
        instances, out_dir, expected_n=args.expected_n,
        required_horizons=tuple(cfg["experiment"]["horizon_hours"]),
        min_articles=int(cfg["formal_news"]["min_articles"]),
    )
    print(f"Wrote {path}")


def cmd_run_formal(args: argparse.Namespace) -> None:
    run_formal_forecasts(
        load_config(),
        top_k=args.top_k,
        resume_run_id=args.resume,
        forecast_mode=args.forecast_mode,
        horizon_hours=args.horizon_hours,
        min_evidence=args.min_evidence,
    )


def cmd_score(args: argparse.Namespace) -> None:
    cfg = load_config()
    write_scores(
        default_benchmark_dir(cfg) / "benchmark.jsonl",
        Path(args.forecasts),
        Path(args.out),
    )


def cmd_sample_benchmark(args: argparse.Namespace) -> None:
    if not args.dry_run and not args.collect and not args.curate:
        raise SystemExit("Specify --dry-run, --collect, and/or --curate")
    if args.curate and not args.collect and not args.from_collect and not args.dry_run:
        raise SystemExit("--curate requires --collect or --from-collect MANIFEST")
    if args.expand_months and not args.collect:
        raise SystemExit("--expand-months requires --collect")
    if args.expand_months and not args.from_collect:
        raise SystemExit("--expand-months requires --from-collect MANIFEST")
    if args.refresh_months and not args.curate:
        raise SystemExit("--refresh-months requires --curate")
    cfg = load_config()
    try:
        only_months = parse_sample_months(args.months, cfg=cfg)
        expand_months = parse_sample_months(args.expand_months, cfg=cfg)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    run_sample_benchmark(
        cfg,
        collect=bool(args.collect),
        curate=bool(args.curate),
        dry_run=bool(args.dry_run),
        max_pages=args.max_pages,
        page_limit=args.page_limit,
        out_dir=Path(args.out) if args.out else None,
        collect_manifest_path=Path(args.from_collect) if args.from_collect else None,
        only_months=only_months,
        refresh_months=bool(args.refresh_months),
        expand_months=expand_months,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tifbench",
        description="Historical binary forecasting benchmark pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-schema", help="Create data directories").set_defaults(
        func=cmd_init_schema
    )

    p_formal = sub.add_parser(
        "construct-formal",
        help="Build formal instances from sample-benchmark --from-sample DIR",
    )
    p_formal.add_argument(
        "--from-sample",
        required=True,
        help="Directory with selected.json + curate_manifest.json (quota-complete)",
    )
    p_formal.add_argument("--event-id", default=None)
    p_formal.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Build only the first N markets in selected.json order (pilot then scale up)",
    )
    p_formal.add_argument(
        "--months",
        default=None,
        help=(
            "Build only these sample months (comma-separated, e.g. "
            "2025-10,2025-11). Mutually exclusive with --event-id / --limit; "
            "partial builds skip publishing benchmark.jsonl"
        ),
    )
    p_formal.add_argument(
        "--phase",
        default="all",
        choices=["queries", "retrieve", "relevance", "summary", "assemble", "all"],
        help="Run one construct stage or all (default: all)",
    )
    p_formal.add_argument(
        "--gpt-mode",
        default=None,
        choices=["batch", "live"],
        help="Override formal_news.gpt_mode (batch=OpenAI Batch API, live=concurrent)",
    )
    p_formal.set_defaults(func=cmd_construct_formal)

    p_export = sub.add_parser(
        "export-benchmark",
        help="Offline: export publishable benchmark.jsonl from existing instances.json",
    )
    p_export.add_argument(
        "--instances",
        default=None,
        help="Path to curated instances.json (default: data/curated/formal/instances.json)",
    )
    p_export.add_argument(
        "--out",
        default=None,
        help="Output directory (default: data/benchmark)",
    )
    p_export.add_argument(
        "--expected-n",
        type=int,
        default=None,
        help="Fail closed unless exactly this many rows are written",
    )
    p_export.set_defaults(func=cmd_export_benchmark)

    p_run_formal = sub.add_parser(
        "run-formal",
        help="Forecast from published data/benchmark/benchmark.jsonl",
    )
    p_run_formal.add_argument("--top-k", type=int, choices=(3, 5, 7, 10), required=True)
    p_run_formal.add_argument(
        "--horizon-hours",
        type=int,
        default=None,
        help=(
            "Only forecast instances at this horizon "
            "(e.g. 72 for 3d). Default: all horizons in the JSONL"
        ),
    )
    p_run_formal.add_argument(
        "--min-evidence",
        type=int,
        default=None,
        help=(
            "Require at least this many frozen articles per instance "
            "(default: equal to --top-k). Use e.g. 10 with several --top-k "
            "values to hold the market cohort fixed"
        ),
    )
    p_run_formal.add_argument(
        "--resume",
        default=None,
        help="Resume an incomplete formal run_id (refuses cohort/model mismatches)",
    )
    p_run_formal.add_argument(
        "--forecast-mode",
        default=None,
        choices=["batch", "live"],
        help=(
            "Override experiment.forecast_mode "
            "(batch=OpenAI/OpenRouter Batch where available; live=concurrent)"
        ),
    )
    p_run_formal.set_defaults(func=cmd_run_formal)

    p_score = sub.add_parser("score", help="Score released forecasts against the frozen benchmark")
    p_score.add_argument("--forecasts", default="results/forecasts.jsonl")
    p_score.add_argument("--out", default="results/metrics.csv")
    p_score.set_defaults(func=cmd_score)

    p_sample = sub.add_parser(
        "sample-benchmark",
        help="Sole candidate-entry path: stratified sample (quotas from config)",
    )
    p_sample.add_argument(
        "--dry-run",
        action="store_true",
        help="Record protocol criteria only; no Gamma/CLOB/OpenAI calls",
    )
    p_sample.add_argument(
        "--collect",
        action="store_true",
        help=(
            "Stage 1: Gamma /events/keyset cursor pagination + mechanical filters "
            "+ horizon price gates"
        ),
    )
    p_sample.add_argument(
        "--curate",
        action="store_true",
        help="Stage 2: GPT-6 Sol pre-screen + month select from a collect manifest",
    )
    p_sample.add_argument(
        "--from-collect",
        default=None,
        help="Path to collect_manifest.json when running --curate without --collect",
    )
    p_sample.add_argument(
        "--months",
        default=None,
        help=(
            "Curate only these months (comma-separated, e.g. 2026-04,2026-09). "
            "Merges prior selected/reports for other months from --out curate_manifest."
        ),
    )
    p_sample.add_argument(
        "--refresh-months",
        action="store_true",
        help=(
            "With --curate: delete freeze files for --months (or all months) "
            "before calling the model"
        ),
    )
    p_sample.add_argument(
        "--expand-months",
        default=None,
        help=(
            "With --collect --from-collect: re-scan/sample/price only these months "
            "at the current tag×month cap; keep other months' qualified rows"
        ),
    )
    p_sample.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES_PER_TAG,
        help=(
            "Max Gamma /events/keyset pages per retrieval tag (default 50); "
            "capped scans are a bounded, non-exhaustive candidate frame"
        ),
    )
    p_sample.add_argument(
        "--page-limit",
        type=int,
        default=100,
        help="Gamma /events/keyset page size, limit<=100 (default 100)",
    )
    p_sample.add_argument(
        "--out",
        default=None,
        help="Output directory (default: data/derived/sample_benchmark/<run_id>/)",
    )
    p_sample.set_defaults(func=cmd_sample_benchmark)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
