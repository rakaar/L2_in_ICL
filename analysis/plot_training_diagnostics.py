#!/usr/bin/env python3
"""Plot training diagnostics (losses + weight norms) from progress logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt


STEP_PATTERN = re.compile(r"^\[(?:(?P<prefix>[a-zA-Z0-9_]+)\s+)?step\s+(?P<step>\d+)\]")
KV_PATTERN = re.compile(r"([a-zA-Z0-9_\/]+)=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _parse_progress(path: Path) -> Dict[str, List[Tuple[int, Dict[str, float]]]]:
    """Return per-prefix metric history from a `progress.txt` log."""

    prefix_to_entries: Dict[str, List[Tuple[int, Dict[str, float]]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("["):
                continue
            match = STEP_PATTERN.match(line)
            if not match:
                continue
            prefix = match.group("prefix") or "train"
            step = int(match.group("step"))
            metrics: Dict[str, float] = {}
            for key, value in KV_PATTERN.findall(line):
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue
            prefix_to_entries.setdefault(prefix, []).append((step, metrics))
    if not prefix_to_entries:
        raise ValueError(f"No step entries found in {path}")
    return prefix_to_entries


def _extract_series(
    entries: Iterable[Tuple[int, Dict[str, float]]], key: str
) -> Tuple[List[int], List[float]]:
    steps, values = [], []
    for step, metrics in entries:
        if key in metrics:
            steps.append(step)
            values.append(metrics[key])
    return steps, values


def _plot_series(
    ax,
    histories: Sequence[Tuple[str, List[int], List[float]]],
    title: str,
    ylabel: str,
) -> None:
    for label, steps, values in histories:
        if steps:
            ax.plot(steps, values, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if sum(bool(steps) for _, steps, _ in histories) > 1:
        ax.legend()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot train/eval losses and weight norms from progress logs."
    )
    parser.add_argument(
        "--progress-logs",
        nargs="+",
        required=True,
        help="Path(s) to progress.txt files produced by pytorch_train.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination PNG (defaults to '<log_dir>/training_diagnostics.png').",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively in addition to saving it.",
    )
    args = parser.parse_args(argv)

    log_paths = [Path(p).expanduser().resolve() for p in args.progress_logs]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    train_histories: List[Tuple[str, List[int], List[float]]] = []
    eval_histories: List[Tuple[str, List[int], List[float]]] = []
    weight_histories: List[Tuple[str, List[int], List[float]]] = []

    for path in log_paths:
        histories = _parse_progress(path)
        train_entries = histories.get("train", [])
        eval_entries = histories.get("eval_holdout", [])

        label = path.parent.parent.name if path.parent.parent != path.parent else path.stem

        train_histories.append((label, *_extract_series(train_entries, "loss")))
        eval_histories.append((label, *_extract_series(eval_entries, "loss")))
        weight_histories.append((label, *_extract_series(train_entries, "weight_l2")))

    _plot_series(axes[0], train_histories, "Training loss", "loss")
    _plot_series(axes[1], eval_histories, "Holdout loss", "loss")
    _plot_series(axes[2], weight_histories, "Weight L2", r"$||W||_2$")

    axes[2].set_xlabel("Training step")
    fig.tight_layout()

    output_path = args.output
    if output_path is None:
        first_log = log_paths[0]
        output_path = first_log.parent / "training_diagnostics.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    print(f"Saved diagnostics plot to {output_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
