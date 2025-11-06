#!/usr/bin/env python3
"""Run the PyTorch Figure 3a sweep (bursty sequences, varying class counts).

This utility mirrors the JAX `run_adamw_sweep.py` orchestrator but targets the
PyTorch training stack. It launches one training job per class-count setting,
waits for completion, and summarizes the latest holdout evaluation metrics
(`accuracy_query`, `loss_query`, etc.) into a CSV that downstream plotting code
can consume.

Only the in-context learning evaluation (Figure 3a) is covered here. Panel (b)
can be added later by extending the evaluation hooks to `no_support_zipfian`.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.pytorch_regression import parse_holdout_history


DEFAULT_CLASS_COUNTS: Sequence[int] = (100, 200, 400, 800, 1600)


@dataclass(frozen=True)
class SweepConfig:
    """Shared configuration for the sweep."""

    output_root: Path
    example_type: str
    n_common: int
    n_holdout: int
    max_steps: int
    eval_interval: int
    eval_batches: int
    eval_batch_size: int | None
    batch_size: int | None
    num_workers: int
    seed: int
    device: str
    log_interval: int
    save_interval: int
    symbolic_dataset_size: int | None
    example_encoding: str | None
    num_layers: int | None
    num_heads: int | None
    seq_len: int | None


def _compute_rare_classes(total: int, common: int, holdout: int) -> int:
    rare = total - common - holdout
    if rare <= 0:
        raise ValueError(
            f"Non-positive rare class count: total={total}, common={common}, "
            f"holdout={holdout}"
        )
    return rare


def _default_symbolic_dataset_size(total_classes: int) -> int:
    return max(4096, total_classes + 64)


def _build_training_command(
    *,
    class_count: int,
    config: SweepConfig,
    run_dir: Path,
    seed_offset: int,
) -> List[str]:
    train_dir = run_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    n_rare = _compute_rare_classes(class_count, config.n_common, config.n_holdout)
    if config.example_type == "symbolic":
        symbolic_size = (
            config.symbolic_dataset_size
            if config.symbolic_dataset_size is not None
            else _default_symbolic_dataset_size(class_count + config.n_holdout)
        )
    else:
        symbolic_size = None

    command: List[str] = [
        sys.executable,
        str(REPO_ROOT / "analysis" / "pytorch_train.py"),
        "--config",
        "none",
        "--example-type",
        config.example_type,
        "--output-dir",
        str(train_dir),
        "--device",
        config.device,
        "--seed",
        str(config.seed + seed_offset),
        "--n-common",
        str(config.n_common),
        "--n-holdout",
        str(config.n_holdout),
        "--n-rare",
        str(n_rare),
        "--max-steps",
        str(config.max_steps),
        "--log-interval",
        str(config.log_interval),
        "--num-workers",
        str(config.num_workers),
        "--eval-holdout-interval",
        str(config.eval_interval),
        "--eval-holdout-batches",
        str(config.eval_batches),
    ]

    if symbolic_size is not None:
        command.extend(["--symbolic-dataset-size", str(symbolic_size)])
    if config.batch_size is not None:
        command.extend(["--batch-size", str(config.batch_size)])
    if config.eval_batch_size is not None:
        command.extend(["--eval-batch-size", str(config.eval_batch_size)])
    if config.save_interval > 0:
        command.extend(["--save-interval", str(config.save_interval)])
    if config.example_encoding is not None:
        command.extend(["--example-encoding", config.example_encoding])
    if config.num_layers is not None:
        command.extend(["--num-layers", str(config.num_layers)])
    if config.num_heads is not None:
        command.extend(["--num-heads", str(config.num_heads)])
    if config.seq_len is not None:
        command.extend(["--seq-len", str(config.seq_len)])

    return command


def _run_subprocess(command: Sequence[str], log_path: Path) -> None:
    """Run a subprocess while streaming output to stdout and a log file."""
    import subprocess

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"Command exited with code {return_code}: {' '.join(command)}"
            )


def _summarize_case(
    *,
    class_count: int,
    run_dir: Path,
) -> Tuple[dict, List[Tuple[int, Dict[str, float]]]]:
    progress_path = run_dir / "train" / "progress.txt"
    history = parse_holdout_history(progress_path)
    step, metrics = history[-1]
    result = {
        "class_count": class_count,
        "step": step,
        **metrics,
        "progress_path": str(progress_path),
    }
    return result, history


def _write_results_csv(results: Iterable[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = None
    rows = list(results)
    if not rows:
        raise RuntimeError("No results available to write.")
    # Keep deterministic column order: class_count, step, metrics..., progress_path.
    metric_keys = sorted(
        key for key in rows[0].keys() if key not in {"class_count", "step", "progress_path"}
    )
    fieldnames = ["class_count", "step"] + metric_keys + ["progress_path"]
    with destination.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_history_csv(history_rows: Iterable[dict], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = list(history_rows)
    if not rows:
        raise RuntimeError("No history rows available to write.")
    metric_keys = sorted(
        key for key in rows[0].keys() if key not in {"class_count", "step", "progress_path"}
    )
    fieldnames = ["class_count", "step"] + metric_keys + ["progress_path"]
    with destination.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _save_plot(
    history_map: Dict[int, List[Tuple[int, Dict[str, float]]]],
    destination: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required to save the Figure 3a plot. "
            "Install it with `pip install matplotlib` and re-run with --plot."
        ) from exc

    destination.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    for class_count in sorted(history_map.keys()):
        steps = [step for step, _ in history_map[class_count]]
        accuracies = [metrics.get("accuracy_query", 0.0) for _, metrics in history_map[class_count]]
        label = f"{class_count} classes"
        plt.plot(steps, accuracies, marker="o", linewidth=2, label=label)
    plt.xlabel("Training step")
    plt.ylabel("Holdout accuracy (accuracy_query)")
    plt.title("Figure 3a (PyTorch) – Bursty p=0.9")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--class-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_CLASS_COUNTS),
        help="Total number of training classes to sweep over.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs" / "figure3a_torch",
        help="Directory where run outputs and summary CSV will be written.",
    )
    parser.add_argument(
        "--example-type",
        choices=("symbolic", "omniglot"),
        default="omniglot",
        help="Type of examples to train on (symbolic integers or Omniglot images).",
    )
    parser.add_argument(
        "--n-common",
        type=int,
        default=10,
        help="Number of common classes (kept constant across runs).",
    )
    parser.add_argument(
        "--n-holdout",
        type=int,
        default=10,
        help="Number of holdout classes (kept constant across runs).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50000,
        help="Number of training steps per run.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for training batch size.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=1000,
        help="Evaluate holdout accuracy every N steps.",
    )
    parser.add_argument(
        "--eval-batches",
        type=int,
        default=32,
        help="Number of batches per holdout evaluation.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=None,
        help="Optional override for evaluation batch size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (0 = synchronous).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device passed to the trainer (e.g. 'cuda', 'cuda:0', 'cpu').",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="How often to print training metrics.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=0,
        help="Save checkpoints every N steps (0 disables checkpointing).",
    )
    parser.add_argument(
        "--symbolic-dataset-size",
        type=int,
        default=None,
        help="Override symbolic dataset size; defaults to max(4096, classes+64).",
    )
    parser.add_argument(
        "--example-encoding",
        choices=("resnet", "linear", "embedding"),
        default=None,
        help="Optional override for input encoding mode.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Override transformer depth (num layers).",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=None,
        help="Override attention head count.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override training sequence length.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed; each run uses seed + index.",
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip runs that already have a progress log.",
    )
    parser.add_argument(
        "--results-csv-name",
        default="figure3a_torch_metrics.csv",
        help="Filename for the aggregated metrics CSV (stored under output root).",
    )
    parser.add_argument(
        "--timeseries-csv-name",
        default="figure3a_torch_timeseries.csv",
        help="Filename for the holdout evaluation history CSV.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a Figure 3a-style plot using the aggregated results.",
    )
    parser.add_argument(
        "--figure-name",
        default="figure3a_torch.png",
        help="Filename for the saved plot (stored under output root).",
    )
    args = parser.parse_args(argv)

    example_encoding = args.example_encoding
    if example_encoding is None and args.example_type == "omniglot":
        example_encoding = "resnet"

    config = SweepConfig(
        output_root=args.output_root,
        example_type=args.example_type,
        n_common=args.n_common,
        n_holdout=args.n_holdout,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        symbolic_dataset_size=args.symbolic_dataset_size,
        example_encoding=example_encoding,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        seq_len=args.seq_len,
    )

    config.output_root.mkdir(parents=True, exist_ok=True)

    summaries: List[dict] = []
    history_rows: List[dict] = []
    history_by_class: Dict[int, List[Tuple[int, Dict[str, float]]]] = {}
    for index, class_count in enumerate(args.class_counts):
        run_dir = config.output_root / f"class_{class_count:05d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        train_dir = run_dir / "train"
        progress_path = train_dir / "progress.txt"

        if args.skip_completed and progress_path.exists():
            print(f"[skip] Found existing progress log for class_count={class_count}: {progress_path}")
            summary, history = _summarize_case(class_count=class_count, run_dir=run_dir)
            summaries.append(summary)
            history_by_class[class_count] = history
            history_rows.extend(
                {
                    "class_count": class_count,
                    "step": step,
                    **metrics,
                    "progress_path": str(progress_path),
                }
                for step, metrics in history
            )
            continue

        print("=" * 80)
        print(f"Starting PyTorch Figure 3a run for class_count={class_count}")
        print(f"Output directory: {run_dir}")
        print("=" * 80)
        sys.stdout.flush()

        command = _build_training_command(
            class_count=class_count,
            config=config,
            run_dir=run_dir,
            seed_offset=index,
        )
        log_path = run_dir / "logs" / "train.log"
        _run_subprocess(command, log_path=log_path)

        summary, history = _summarize_case(class_count=class_count, run_dir=run_dir)
        summaries.append(summary)
        history_by_class[class_count] = history
        history_rows.extend(
            {
                "class_count": class_count,
                "step": step,
                **metrics,
                "progress_path": str(progress_path),
            }
            for step, metrics in history
        )

    if not summaries:
        raise RuntimeError("No runs executed; aborting before writing results.")

    results_csv = config.output_root / args.results_csv_name
    _write_results_csv(summaries, results_csv)
    print(f"Wrote Figure 3a metrics to {results_csv}")

    timeseries_csv = config.output_root / args.timeseries_csv_name
    _write_history_csv(history_rows, timeseries_csv)
    print(f"Wrote Figure 3a holdout history to {timeseries_csv}")

    if args.plot:
        figure_path = config.output_root / args.figure_name
        _save_plot(history_by_class, figure_path)
        print(f"Saved Figure 3a plot to {figure_path}")


if __name__ == "__main__":
    main()
