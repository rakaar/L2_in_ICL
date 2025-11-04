#!/usr/bin/env python3
"""Orchestrate AdamW + L2 sweeps for the Figure 3 replication.

This utility sequentially launches the training job for every combination of
class count and weight-decay value, captures the training logs (with loss values
mirrored into a progress text file), records the resulting checkpoints, and
finally calls `analysis/plot_figure3_adamw.py` to regenerate the Figure 3
panels.

Typical usage (from the repository root, ideally inside tmux):

```bash
python analysis/run_adamw_sweep.py \
  --base_output_dir ./runs/adamw_sweep \
  --training_steps 50000
```

Key outputs per run:
  * `train.log` – full stdout/stderr stream from the training process.
  * `progress.txt` – subset of log lines containing "loss".
  * `checkpoints/` – directory passed to `--config.checkpoint_dir`.

After all runs complete, the script writes `runs.csv` describing the sweep and
invokes the plotting helper. See `python analysis/run_adamw_sweep.py --help`
for additional options (custom class counts, weight decays, max eval batches,
etc.).
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


DEFAULT_CLASS_COUNTS = (100, 1600)
DEFAULT_WEIGHT_DECAYS = (0.0, 3e-4, 1e-3, 3e-3, 1e-2)
DEFAULT_CONFIG_PATH = "experiment/configs/images_all_exemplars.py"


def _format_weight_decay_for_flag(weight_decay: float) -> str:
    """Formats weight decay for command-line overrides."""
    if weight_decay == 0.0:
        return "0"
    value = f"{weight_decay:.8f}".rstrip("0")
    if value.endswith("."):
        value = value[:-1]
    return value


def _format_weight_decay_for_tag(weight_decay: float) -> str:
    """Formats weight decay for run directory names."""
    if weight_decay == 0.0:
        return "0"
    if weight_decay < 1e-2:
        return f"{weight_decay:.0e}".replace("+", "")
    return _format_weight_decay_for_flag(weight_decay).replace(".", "p")


def _compute_rare_classes(
    class_count: int, n_common: int, n_holdout: int
) -> int:
    rare = class_count - n_common - n_holdout
    if rare <= 0:
        raise ValueError(
            f"Non-positive rare class count for total={class_count}, "
            f"common={n_common}, holdout={n_holdout}."
        )
    return rare


def _run_subprocess(
    command: Sequence[str],
    log_path: Path,
    progress_path: Path | None = None,
) -> None:
    """Runs a command while tee-ing output to files and stdout."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if progress_path is not None:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        if progress_path.exists():
            progress_path.unlink()

    with log_path.open("w", encoding="utf-8") as log_file:
        progress_file = (
            progress_path.open("a", encoding="utf-8")
            if progress_path is not None
            else None
        )
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Failed to launch command: {command}") from exc

        assert process.stdout is not None  # for mypy
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
            if progress_file and any(
                token in line.lower()
                for token in ("loss", "weight_l2", "grad_l2")
            ):
                progress_file.write(line)
                progress_file.flush()

        return_code = process.wait()
        if progress_file:
            progress_file.close()
        if return_code != 0:
            raise RuntimeError(
                f"Command exited with code {return_code}: {' '.join(command)}"
            )


def _find_latest_checkpoint(checkpoint_dir: Path) -> Path:
    """Returns the newest checkpoint directory produced by training."""
    models_dir = checkpoint_dir / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(
            f"No 'models' directory found under {checkpoint_dir}. "
            "Ensure training completed successfully."
        )
    latest_dir = models_dir / "latest"
    if not latest_dir.is_dir():
        raise FileNotFoundError(
            f"No 'latest' directory found under {models_dir}. "
            "Check that checkpoints were saved."
        )

    candidate_dirs = sorted(
        d for d in latest_dir.iterdir()
        if d.is_dir() and d.name.startswith("step_")
    )
    if not candidate_dirs:
        raise FileNotFoundError(
            f"No step_* checkpoint directories found in {latest_dir}."
        )
    newest = max(candidate_dirs, key=lambda path: path.stat().st_mtime)
    checkpoint_file = newest / "checkpoint.dill"
    if not checkpoint_file.exists():
        raise FileNotFoundError(
            f"Expected checkpoint.dill at {checkpoint_file}, but it is missing."
        )
    return newest


def _build_training_command(
    config_path: str,
    class_count: int,
    weight_decay: float,
    checkpoint_dir: Path,
    n_common: int,
    n_holdout: int,
    training_steps: int | None,
) -> List[str]:
    rare_classes = _compute_rare_classes(class_count, n_common, n_holdout)
    weight_flag = _format_weight_decay_for_flag(weight_decay)

    command = [
        sys.executable,
        "-m",
        "emergent_in_context_learning.experiment.experiment",
        "--config",
        config_path,
        "--config.experiment_kwargs.config.optimizer.max_lr=0.0003",
        "--config.experiment_kwargs.config.optimizer.warmup_steps=4000",
        "--config.experiment_kwargs.config.optimizer.clip_level=0.25",
        "--config.experiment_kwargs.config.training.learning_rate=0.0001",
        "--config.optimizer.name=adamw",
        f"--config.optimizer.kwargs.weight_decay={weight_flag}",
        f"--config.data.generator_config.n_common_classes={n_common}",
        f"--config.data.generator_config.n_holdout_classes={n_holdout}",
        f"--config.data.generator_config.n_rare_classes={rare_classes}",
        f"--config.checkpoint_dir={checkpoint_dir}",
        "--jaxline_mode",
        "train",
        "--logtostderr",
    ]
    if training_steps is not None:
        command.append(f"--config.training_steps={training_steps}")
    return command


def _write_runs_csv(
    rows: Iterable[Tuple[int, float, Path]],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["class_count", "weight_decay", "checkpoint"])
        for class_count, weight_decay, checkpoint in rows:
            writer.writerow([class_count, weight_decay, str(checkpoint)])


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config_path",
        default=DEFAULT_CONFIG_PATH,
        help="Config file passed via --config (default: %(default)s).",
    )
    parser.add_argument(
        "--class_counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_CLASS_COUNTS),
        help="Total class counts to sweep over.",
    )
    parser.add_argument(
        "--weight_decays",
        nargs="+",
        type=float,
        default=list(DEFAULT_WEIGHT_DECAYS),
        help="AdamW weight-decay (lambda) values to sweep over.",
    )
    parser.add_argument(
        "--n_common_classes",
        type=int,
        default=10,
        help="Number of common classes (used to compute rare classes).",
    )
    parser.add_argument(
        "--n_holdout_classes",
        type=int,
        default=10,
        help="Number of holdout classes (used to compute rare classes).",
    )
    parser.add_argument(
        "--training_steps",
        type=int,
        default=None,
        help="Override config.training_steps (default uses config file).",
    )
    parser.add_argument(
        "--base_output_dir",
        type=Path,
        default=Path("./runs/adamw_sweep"),
        help="Root directory where logs/checkpoints/plots will be written.",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip runs whose checkpoint already exists.",
    )
    parser.add_argument(
        "--max_eval_batches",
        type=int,
        default=None,
        help="Optional cap on evaluation batches for plotting script.",
    )
    args = parser.parse_args(argv)

    args.base_output_dir.mkdir(parents=True, exist_ok=True)

    run_records: List[Tuple[int, float, Path]] = []

    for class_count in args.class_counts:
        for weight_decay in args.weight_decays:
            weight_tag = _format_weight_decay_for_tag(weight_decay)
            run_name = f"class{class_count}_wd{weight_tag}"
            run_dir = args.base_output_dir / run_name
            checkpoint_dir = run_dir / "checkpoints"
            log_dir = run_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            print("=" * 80)
            print(f"Starting run: classes={class_count}, weight_decay={weight_decay}")
            print(f"Output directory: {run_dir}")
            print("=" * 80)
            sys.stdout.flush()

            if args.skip_completed and checkpoint_dir.exists():
                try:
                    latest_checkpoint = _find_latest_checkpoint(checkpoint_dir)
                    print(
                        f"Checkpoint already present, skipping training: {latest_checkpoint}"
                    )
                    run_records.append((class_count, weight_decay, latest_checkpoint))
                    continue
                except FileNotFoundError:
                    print("Existing directory missing checkpoints; retraining.")

            training_command = _build_training_command(
                config_path=args.config_path,
                class_count=class_count,
                weight_decay=weight_decay,
                checkpoint_dir=checkpoint_dir,
                n_common=args.n_common_classes,
                n_holdout=args.n_holdout_classes,
                training_steps=args.training_steps,
            )

            train_log_path = log_dir / "train.log"
            progress_path = log_dir / "progress.txt"

            _run_subprocess(
                training_command,
                log_path=train_log_path,
                progress_path=progress_path,
            )

            latest_checkpoint = _find_latest_checkpoint(checkpoint_dir)
            print(f"Latest checkpoint: {latest_checkpoint}")
            run_records.append((class_count, weight_decay, latest_checkpoint))

    if not run_records:
        raise RuntimeError("No runs were executed; nothing to aggregate.")

    runs_csv = args.base_output_dir / "runs.csv"
    _write_runs_csv(run_records, runs_csv)
    print(f"Wrote sweep description to {runs_csv}")

    plot_script = Path(__file__).with_name("plot_figure3_adamw.py")
    plot_command = [
        sys.executable,
        str(plot_script),
        "--runs_csv",
        str(runs_csv),
        "--output_dir",
        str(args.base_output_dir),
        "--config_module",
        "experiment.configs.images_all_exemplars",
    ]
    if args.max_eval_batches is not None:
        plot_command.append(f"--max_eval_batches={args.max_eval_batches}")

    plot_log_path = args.base_output_dir / "plot.log"
    print("Running plotting script to generate Figure 3 panels...")
    _run_subprocess(plot_command, log_path=plot_log_path, progress_path=None)
    print(f"Plotting complete. See {args.base_output_dir} for results.")


if __name__ == "__main__":
    main()
