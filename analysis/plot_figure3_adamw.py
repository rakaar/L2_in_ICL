#!/usr/bin/env python3
"""Evaluate checkpoints and plot Figure 3-style curves for AdamW sweeps.

This script runs the evaluation loops defined in
`emergent_in_context_learning.experiment.experiment.Experiment` on a collection
of checkpoints, aggregates the resulting metrics, and produces the two plots
shown in Figure 3 of the paper:
  * Panel (a): in-context accuracy on holdout classes.
  * Panel (b): in-weights accuracy on the training classes (no-support sequences).

Usage overview (see README/Docs for detailed instructions):
  1. Prepare a CSV describing the runs you want to aggregate. Each row should
     contain:
       class_count,weight_decay,checkpoint[,config_module,n_common_classes,
       n_holdout_classes]
     where `checkpoint` points either to a directory containing
     `checkpoint.dill` or to the file itself.
  2. Install matplotlib (not part of the base requirements):
       pip install matplotlib
  3. Execute:
       python analysis/plot_figure3_adamw.py \
         --runs_csv /path/to/runs.csv \
         --output_dir /path/to/output \
         --config_module experiment.configs.images_all_exemplars
  4. The script evaluates each checkpoint on the two sequence types, writes the
     raw metrics to `figure3_adamw_metrics.csv`, and saves a PNG/PDF plot.
"""

from __future__ import annotations

import argparse
import collections
import csv
import importlib
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import dill
import jax
import numpy as np
from absl import logging
from jaxline import utils as jaxline_utils
from ml_collections import config_dict

# Use a headless backend so the script works on servers.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (import after backend selection)

from emergent_in_context_learning.experiment import experiment as experiment_module


RunSpec = collections.namedtuple(
    "RunSpec",
    [
        "class_count",
        "weight_decay",
        "checkpoint_path",
        "config_module",
        "n_common_classes",
        "n_holdout_classes",
    ],
)


def _load_checkpoint(restore_path: str) -> Dict[str, object]:
    """Loads a checkpoint produced by the training script."""
    checkpoint_file = restore_path
    if os.path.isdir(checkpoint_file):
        checkpoint_file = os.path.join(checkpoint_file, "checkpoint.dill")
    if not os.path.exists(checkpoint_file):
        raise FileNotFoundError(
            f"Could not locate checkpoint at '{checkpoint_file}'. "
            "If you passed a directory, ensure it contains checkpoint.dill."
        )
    with open(checkpoint_file, "rb") as ckpt_file:
        return dill.load(ckpt_file)


def _prepare_experiment(
    exp_config: config_dict.ConfigDict,
    run: RunSpec,
) -> experiment_module.Experiment:
    """Instantiates the Experiment in evaluation mode."""
    exp_config.optimizer.name = "adamw"
    exp_config.optimizer.kwargs = dict(exp_config.optimizer.kwargs)
    exp_config.optimizer.kwargs["weight_decay"] = float(run.weight_decay)

    generator_cfg = exp_config.data.generator_config
    if run.n_common_classes is not None:
        generator_cfg.n_common_classes = int(run.n_common_classes)
    if run.n_holdout_classes is not None:
        generator_cfg.n_holdout_classes = int(run.n_holdout_classes)

    if run.class_count is not None:
        n_common = generator_cfg.n_common_classes
        n_holdout = generator_cfg.n_holdout_classes
        rare = int(run.class_count) - n_common - n_holdout
        if rare <= 0:
            raise ValueError(
                "Computed non-positive number of rare classes. Check the "
                "class_count/common/holdout values."
            )
        generator_cfg.n_rare_classes = rare

    init_rng = jax.random.PRNGKey(0)
    experiment = experiment_module.Experiment(
        mode="eval_fewshot_holdout", init_rng=init_rng, config=exp_config
    )
    return experiment


def _broadcast_checkpoint_to_devices(
    experiment: experiment_module.Experiment, checkpoint: Dict[str, object]
) -> None:
    """Populates the experiment's params/state from a loaded checkpoint."""
    for attribute, key in experiment_module.Experiment.CHECKPOINT_ATTRS.items():
        if key not in checkpoint:
            raise KeyError(
                f"Missing '{key}' in checkpoint. Available keys: {list(checkpoint)}"
            )
        value = checkpoint[key]
        setattr(experiment, attribute, jaxline_utils.bcast_local_devices(value))


def _evaluate_mode(
    experiment: experiment_module.Experiment,
    rng_key: jax.random.KeyArray,
    mode: str,
    max_batches: Optional[int],
) -> Dict[str, float]:
    """Runs evaluation for a specific mode and aggregates scalar metrics."""
    experiment.mode = mode
    experiment.seq_type = mode.replace("eval_", "")
    experiment._eval_batch = jax.jit(experiment._eval_batch)  # pylint: disable=protected-access

    params = jaxline_utils.get_first(experiment._params)
    state = jaxline_utils.get_first(experiment._state)

    totals = collections.defaultdict(float)
    total_sequences = 0.0

    for batch_index, batch in enumerate(experiment._build_eval_input()):  # pylint: disable=protected-access
        batch_rng = jax.random.fold_in(rng_key, batch_index)
        loss_acc, _, _ = experiment._eval_batch(params, state, batch, batch_rng)  # pylint: disable=protected-access
        for key, value in loss_acc.items():
            totals[key] += float(value)
        total_sequences += batch["examples"].shape[0]

        if max_batches is not None and (batch_index + 1) >= max_batches:
            break

    if total_sequences == 0:
        raise RuntimeError("Evaluation yielded zero sequences; check data pipeline.")

    return {key: value / total_sequences for key, value in totals.items()}


def _format_weight_decay(weight: float) -> str:
    """Formats weight decay values for plotting."""
    if weight == 0:
        return "0"
    if weight < 1e-2:
        return f"{weight:.0e}"
    return f"{weight:.3f}".rstrip("0").rstrip(".")


def _read_run_specs(
    csv_path: str, default_config_module: str
) -> List[RunSpec]:
    """Parses the CSV describing the evaluation runs."""
    runs: List[RunSpec] = []
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"class_count", "weight_decay", "checkpoint"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing_columns))}"
            )
        for row in reader:
            config_module = row.get("config_module", default_config_module).strip()
            class_count = int(row["class_count"])
            weight_decay = float(row["weight_decay"])
            checkpoint_path = row["checkpoint"].strip()
            n_common = row.get("n_common_classes")
            n_holdout = row.get("n_holdout_classes")
            runs.append(
                RunSpec(
                    class_count=class_count,
                    weight_decay=weight_decay,
                    checkpoint_path=checkpoint_path,
                    config_module=config_module,
                    n_common_classes=int(n_common) if n_common else None,
                    n_holdout_classes=int(n_holdout) if n_holdout else None,
                )
            )
    if not runs:
        raise ValueError("No runs found in the CSV.")
    return runs


def _plot_results(
    results: Sequence[Dict[str, float]],
    weight_decays: Sequence[float],
    output_dir: str,
) -> None:
    """Creates the Figure 3-style plots and writes them to disk."""
    class_counts = sorted({int(r["class_count"]) for r in results})
    wd_sorted = sorted(weight_decays)
    index_lookup = {wd: idx for idx, wd in enumerate(wd_sorted)}
    x_positions = np.arange(len(wd_sorted))
    xticklabels = [_format_weight_decay(wd) for wd in wd_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    panel_defs = [
        ("accuracy_holdout", "In-context accuracy (holdout)"),
        ("accuracy_no_support", "In-weights accuracy (no support)"),
    ]

    for panel_index, (metric_key, title) in enumerate(panel_defs):
        ax = axes[panel_index]
        for class_count in class_counts:
            subset = [
                r for r in results if int(r["class_count"]) == class_count
            ]
            if not subset:
                continue
            series = [np.nan] * len(wd_sorted)
            for entry in subset:
                series[index_lookup[entry["weight_decay"]]] = entry[metric_key]
            ax.plot(
                x_positions,
                series,
                marker="o",
                label=f"{class_count} classes",
            )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(xticklabels, rotation=45)
        ax.set_title(title)
        ax.set_xlabel("Weight decay (AdamW λ)")
        ax.grid(True, linestyle="--", alpha=0.3)
        if panel_index == 0:
            ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.legend()

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, "figure3_adamw.png")
    pdf_path = os.path.join(output_dir, "figure3_adamw.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    logging.info("Saved plots to %s and %s", png_path, pdf_path)


def _write_csv(
    results: Sequence[Dict[str, float]],
    output_dir: str,
) -> str:
    """Persists the aggregated metrics as a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "figure3_adamw_metrics.csv")
    base_fieldnames = [
        "class_count",
        "n_rare_classes",
        "n_common_classes",
        "n_holdout_classes",
        "weight_decay",
        "global_step",
        "accuracy_holdout",
        "accuracy_no_support",
        "loss_holdout",
        "loss_no_support",
    ]
    dynamic_fields = sorted({
        key for row in results for key in row.keys()
    } - set(base_fieldnames))
    fieldnames = base_fieldnames + dynamic_fields
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            results, key=lambda r: (int(r["class_count"]), float(r["weight_decay"]))
        ):
            writer.writerow(row)
    logging.info("Wrote aggregated metrics to %s", csv_path)
    return csv_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs_csv",
        required=True,
        help="CSV describing the runs to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where plots and CSV will be written.",
    )
    parser.add_argument(
        "--config_module",
        default="experiment.configs.images_all_exemplars",
        help="Config module path used for the runs (can be overridden per-row).",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help=(
            "Optional cap on evaluation batches per checkpoint. "
            "Leave unset to match the 10k batches used in the paper."
        ),
    )
    args = parser.parse_args(argv)

    runs = _read_run_specs(args.runs_csv, args.config_module)
    results: List[Dict[str, float]] = []

    for run in runs:
        module = importlib.import_module(run.config_module)
        root_config = module.get_config()
        exp_config = root_config.experiment_kwargs.config.copy_and_resolve_references()

        experiment = _prepare_experiment(exp_config, run)
        checkpoint = _load_checkpoint(run.checkpoint_path)
        _broadcast_checkpoint_to_devices(experiment, checkpoint)

        rng = jax.random.PRNGKey(42)

        metrics_holdout = _evaluate_mode(
            experiment=experiment,
            rng_key=rng,
            mode="eval_fewshot_holdout",
            max_batches=args.max_eval_batches,
        )

        metrics_no_support = _evaluate_mode(
            experiment=experiment,
            rng_key=jax.random.fold_in(rng, 1),
            mode="eval_no_support_zipfian",
            max_batches=args.max_eval_batches,
        )

        generator_cfg = experiment.config.data.generator_config
        row = {
            "class_count": run.class_count,
            "n_rare_classes": int(generator_cfg.n_rare_classes),
            "n_common_classes": int(generator_cfg.n_common_classes),
            "n_holdout_classes": int(generator_cfg.n_holdout_classes),
            "weight_decay": run.weight_decay,
            "global_step": int(checkpoint.get("global_step", -1)),
            "accuracy_holdout": float(metrics_holdout["accuracy_query"]),
            "accuracy_no_support": float(metrics_no_support["accuracy_query"]),
            "loss_holdout": float(metrics_holdout["loss"]),
            "loss_no_support": float(metrics_no_support["loss"]),
        }
        for metrics, suffix in (
            (metrics_holdout, "holdout"),
            (metrics_no_support, "no_support"),
        ):
            for key, value in metrics.items():
                if key.startswith("weight_l2"):
                    sanitized = key.replace("/", "__")
                    row[f"{sanitized}_{suffix}"] = float(value)
        results.append(row)

    _write_csv(results, args.output_dir)
    weight_decays = {float(r["weight_decay"]) for r in results}
    _plot_results(results, sorted(weight_decays), args.output_dir)


if __name__ == "__main__":
    main()
