"""Command-line entry point for the reproduction workflow."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .constants import (
    C_GRID,
    DEFAULT_ARCHIVE_PATH,
    DEFAULT_CACHE_PATH,
    DEFAULT_FOLDS_DIR,
    DEFAULT_ORIGINAL_LABELS_ARCHIVE_PATH,
    DEFAULT_ORIGINAL_LABELS_DIR,
    DEFAULT_RANDOM_SEED,
    DEFAULT_RAW_DIR,
    DEFAULT_RESULTS_DIR,
    HEADLINE_LABELS,
    SENSOR_ORDER,
)
from .data import (
    dataset_profile,
    download_dataset,
    load_official_folds,
    load_or_prepare,
    load_raw_dataset,
    save_dataset,
    write_profile,
)
from .modeling import evaluate_models, write_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download and verify the public dataset")
    download.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    download.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    download.add_argument(
        "--original-labels-archive",
        type=Path,
        default=DEFAULT_ORIGINAL_LABELS_ARCHIVE_PATH,
    )
    download.add_argument("--original-labels-dir", type=Path, default=DEFAULT_ORIGINAL_LABELS_DIR)

    prepare = subparsers.add_parser("prepare", help="Build a compact six-sensor NumPy cache")
    prepare.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    prepare.add_argument("--original-labels-dir", type=Path, default=DEFAULT_ORIGINAL_LABELS_DIR)
    prepare.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)

    profile = subparsers.add_parser("profile", help="Write data-quality and fold checks")
    profile.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    profile.add_argument("--original-labels-dir", type=Path, default=DEFAULT_ORIGINAL_LABELS_DIR)
    profile.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    profile.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    profile.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "data_profile.json")

    train = subparsers.add_parser("train", help="Run paper-style five-fold evaluation")
    train.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    train.add_argument("--original-labels-dir", type=Path, default=DEFAULT_ORIGINAL_LABELS_DIR)
    train.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    train.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    train.add_argument(
        "--models",
        default="EF",
        help=f"Comma-separated subset of EF,{','.join(SENSOR_ORDER)}",
    )
    train.add_argument(
        "--labels",
        default="headline",
        help="'headline', 'all', or comma-separated raw label names",
    )
    train.add_argument(
        "--label-policy",
        choices=("legacy-negative", "exclude-missing"),
        default="legacy-negative",
    )
    train.add_argument("--jobs", type=int, default=1)
    train.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    train.add_argument(
        "--c-grid",
        default=",".join(str(value) for value in C_GRID),
        help="Comma-separated logistic-regression C values",
    )
    train.add_argument("--max-iter", type=int, default=1000)
    train.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)

    research = subparsers.add_parser("research", help="Run leakage-safe improvement experiments")
    research_commands = research.add_subparsers(dest="research_command", required=True)

    research_run = research_commands.add_parser("run", help="Run one immutable experiment")
    research_run.add_argument("--config", type=Path, required=True)
    research_run.add_argument(
        "--stage",
        choices=("development", "promotion", "audit"),
        default="development",
    )
    research_run.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    research_run.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    research_run.add_argument(
        "--original-labels-dir",
        type=Path,
        default=DEFAULT_ORIGINAL_LABELS_DIR,
    )
    research_run.add_argument("--folds-dir", type=Path, default=DEFAULT_FOLDS_DIR)
    research_run.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "research",
    )

    research_summary = research_commands.add_parser(
        "summarize", help="Print completed ledger entries"
    )
    research_summary.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "research",
    )
    research_summary.add_argument("--experiment")

    research_sweep = research_commands.add_parser(
        "sweep", help="Run or resume a bounded config-only search"
    )
    research_sweep.add_argument("--spec", type=Path, required=True)
    research_sweep.add_argument("--max-trials", type=int)
    research_sweep.add_argument("--budget-hours", type=float)
    research_sweep.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "research",
    )

    research_promote = research_commands.add_parser(
        "promote", help="Evaluate the one frozen family champion on fold 4"
    )
    research_promote.add_argument("--spec", type=Path, required=True)
    research_promote.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "research",
    )

    research_audit = research_commands.add_parser(
        "audit", help="Evaluate the promoted champion once on fold 3"
    )
    research_audit.add_argument("--spec", type=Path, required=True)
    research_audit.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_RESULTS_DIR / "research",
    )
    return parser


def _parse_labels(argument: str, available_labels: list[str]) -> list[str]:
    if argument == "headline":
        return list(HEADLINE_LABELS)
    if argument == "all":
        return available_labels
    return [label.strip() for label in argument.split(",") if label.strip()]


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command == "download":
        download_dataset(
            args.archive,
            args.raw_dir,
            args.original_labels_archive,
            args.original_labels_dir,
        )
        print(
            f"Verified and extracted features to {args.raw_dir} and "
            f"original labels to {args.original_labels_dir}"
        )
        return

    if args.command == "prepare":
        dataset = load_raw_dataset(args.raw_dir, args.original_labels_dir)
        save_dataset(dataset, args.cache)
        print(
            f"Cached {len(dataset.timestamps):,} examples, "
            f"{len(dataset.feature_names)} core features at {args.cache}"
        )
        return

    if args.command == "research":
        from .research.runner import run_experiment, summarize_experiments
        from .research.sweep import (
            SweepSpec,
            audit_sweep,
            promote_sweep,
            run_sweep,
        )

        if args.research_command == "summarize":
            print(
                json.dumps(
                    summarize_experiments(args.output_root, experiment=args.experiment),
                    indent=2,
                    sort_keys=True,
                )
            )
            return
        if args.research_command in {"sweep", "promote", "audit"}:
            spec = SweepSpec.from_toml(args.spec)
            if args.research_command == "sweep":
                summary = run_sweep(
                    spec,
                    output_root=args.output_root,
                    max_trials=args.max_trials,
                    budget_hours=args.budget_hours,
                )
            elif args.research_command == "promote":
                summary = promote_sweep(spec, output_root=args.output_root)
            else:
                summary = audit_sweep(spec, output_root=args.output_root)
            print(json.dumps(summary, indent=2, sort_keys=True))
            return
        summary = run_experiment(
            args.config,
            args.stage,
            output_root=args.output_root,
            cache_path=args.cache,
            raw_dir=args.raw_dir,
            original_labels_dir=args.original_labels_dir,
            folds_dir=args.folds_dir,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    dataset = load_or_prepare(args.cache, args.raw_dir, args.original_labels_dir)
    folds = load_official_folds(dataset.user_ids.tolist(), args.folds_dir)

    if args.command == "profile":
        profile = dataset_profile(dataset, folds)
        write_profile(profile, args.output)
        print(json.dumps(profile, indent=2, sort_keys=True))
        return

    models = [model.strip() for model in args.models.split(",") if model.strip()]
    labels = _parse_labels(args.labels, dataset.label_names.tolist())
    c_grid = [float(value) for value in args.c_grid.split(",")]
    fold_frame, summary_frame = evaluate_models(
        dataset,
        folds,
        models=models,
        labels=labels,
        label_policy=args.label_policy,
        jobs=args.jobs,
        seed=args.seed,
        c_grid=c_grid,
        max_iter=args.max_iter,
    )
    metadata = {
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "models": models,
        "labels": labels,
        "label_policy": args.label_policy,
        "paper_label_reconstruction": [
            "OR_indoors",
            "OR_outside",
            "PHONE_IN_POCKET",
        ],
        "seed": args.seed,
        "jobs": args.jobs,
        "c_grid": c_grid,
        "max_iter": args.max_iter,
        "folds": "official cv5Folds.zip",
        "test_subset": "all six paper sensor groups complete",
        "solver": "liblinear",
    }
    write_results(fold_frame, summary_frame, args.output_dir, metadata)
    averages = (
        summary_frame.groupby("model")["balanced_accuracy"].mean().sort_values(ascending=False)
    )
    print(averages.to_string())


if __name__ == "__main__":
    main()
