#!/usr/bin/env python3
"""Lightweight regression harness for the PyTorch ICL trainer.

This script runs a couple of short symbolic training jobs that mirror the
Figure 3a settings (varying the number of training classes with p_bursty=0.9)
and checks that holdout evaluation metrics are emitted as expected. The goal is
to provide a fast signal that recent code changes did not break the training
loop, logging format, or evaluation hooks.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis import pytorch_train


@dataclass(frozen=True)
class RegressionCase:
    """Configuration for a single short regression run."""

    label: str
    total_train_classes: int
    n_rare: int = 4
    n_holdout: int = 2
    max_steps: int = 5
    eval_interval: int = 3
    batch_size: int = 4
    seed: int = 123

    @property
    def n_common(self) -> int:
        value = self.total_train_classes - self.n_rare - self.n_holdout
        if value < 0:
            raise ValueError(
                f"Regression case '{self.label}' has invalid class split: "
                f"total={self.total_train_classes}, n_rare={self.n_rare}, "
                f"n_holdout={self.n_holdout}"
            )
        return value

    @property
    def symbolic_dataset_size(self) -> int:
        # Keep the symbolic vocabulary comfortably larger than the number of
        # training classes to emulate the JAX setup.
        return max(4096, self.total_train_classes + 64)


def build_training_args(case: RegressionCase, output_dir: Path) -> argparse.Namespace:
    """Construct an argparse.Namespace compatible with pytorch_train.train."""
    # Mirror defaults from pytorch_train.parse_args() while overriding the
    # fields we care about. Using a dict keeps the setup explicit and makes it
    # easier to spot regressions if the trainer adds new required arguments.
    args = {
        "config": "none",
        "debug": False,
        "example_type": "symbolic",
        "example_encoding": None,
        "n_rare": case.n_rare,
        "n_common": case.n_common,
        "n_holdout": case.n_holdout,
        "seq_len": None,
        "num_layers": None,
        "num_heads": None,
        "batch_size": case.batch_size,
        "max_steps": case.max_steps,
        "learning_rate": None,
        "num_workers": 0,
        "seed": case.seed,
        "image_channels": None,
        "device": "cpu",
        "output_dir": str(output_dir),
        "save_interval": 0,
        "log_interval": 1,
        "symbolic_dataset_size": case.symbolic_dataset_size,
        "eval_holdout_interval": case.eval_interval,
        "eval_holdout_batches": 2,
        "eval_batch_size": None,
    }
    return argparse.Namespace(**args)


def parse_holdout_history(progress_path: Path) -> List[Tuple[int, Dict[str, float]]]:
    """Extract the full holdout evaluation history from progress.txt."""
    if not progress_path.exists():
        raise FileNotFoundError(f"Expected progress log at {progress_path}")

    history: List[Tuple[int, Dict[str, float]]] = []
    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("[eval_holdout"):
                continue
            parts = line.strip().replace("[", "").replace("]", "").split()
            if len(parts) < 4:
                raise RuntimeError(f"Unexpected progress line format: {line}")
            if parts[1] != "step":
                raise RuntimeError(f"Unexpected progress line format: {line}")
            step_value = int(parts[2])
            metrics: Dict[str, float] = {}
            for token in parts[3:]:
                name, value = token.split("=")
                metrics[name] = float(value)
            history.append((step_value, metrics))

    if not history:
        raise RuntimeError(
            f"No holdout evaluation entries found in {progress_path}. "
            "Ensure --eval-holdout-interval > 0."
        )

    return history


def parse_holdout_metrics(progress_path: Path) -> Tuple[int, Dict[str, float]]:
    """Extract the latest holdout metrics from progress.txt."""
    history = parse_holdout_history(progress_path)
    return history[-1]


def run_regression_case(case: RegressionCase, output_root: Path, verbose: bool = True) -> Dict[str, object]:
    """Run a single regression case and return its summary."""
    case_dir = output_root / case.label
    case_dir.mkdir(parents=True, exist_ok=True)

    args = build_training_args(case, case_dir)
    if verbose:
        print(f"Running regression case '{case.label}' with {case.total_train_classes} training classes...")

    pytorch_train.train(args)

    progress_path = case_dir / "progress.txt"
    step, metrics = parse_holdout_metrics(progress_path)

    # Basic sanity checks to make sure metrics are in a plausible range.
    for key, value in metrics.items():
        if not (value == value):  # NaN check
            raise RuntimeError(f"Metric '{key}' is NaN for case '{case.label}'")
        if key.startswith("accuracy") and not (0.0 <= value <= 1.0):
            raise RuntimeError(
                f"Metric '{key}'={value} is outside [0,1] for case '{case.label}'"
            )

    summary = {
        "case": case.label,
        "step": step,
        "total_train_classes": case.total_train_classes,
        "progress_path": str(progress_path),
        "metrics": metrics,
    }

    if verbose:
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        print(f"Completed case '{case.label}' at step {step}: {metrics_str}")
        print(f"Progress log: {progress_path}")

    return summary


def run(
    cases: Iterable[RegressionCase],
    output_root: Path,
    *,
    verbose: bool = True,
) -> List[Dict[str, object]]:
    """Run all regression cases and return their summaries."""
    output_root.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []
    for case in cases:
        summary = run_regression_case(case, output_root, verbose=verbose)
        results.append(summary)
    return results


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regression harness for PyTorch ICL training.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "runs" / "torch_regression",
        help="Directory where per-case outputs (progress logs, checkpoints) will be stored.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-case progress printing (summary only).",
    )
    parser.add_argument(
        "--dump-json",
        type=Path,
        default=None,
        help="Optional path to dump the aggregated results as JSON.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)

    cases = [
        RegressionCase(label="classes_0100", total_train_classes=100),
        RegressionCase(label="classes_1600", total_train_classes=1600),
    ]

    summaries = run(cases, args.output_root, verbose=not args.quiet)
    if not args.quiet:
        print("\nRegression results:")
        for summary in summaries:
            metrics = summary["metrics"]
            metrics_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
            print(f"- {summary['case']}: step={summary['step']} {metrics_str}")

    if args.dump_json:
        payload = {"results": summaries}
        with args.dump_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        if not args.quiet:
            print(f"Saved JSON summary to {args.dump_json}")


if __name__ == "__main__":
    main()
