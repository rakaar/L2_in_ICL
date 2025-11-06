#!/usr/bin/env python3
"""PyTorch training harness for the Omniglot sequences experiment."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import random
from ml_collections import ConfigDict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pytorch_port import (
    EmbeddingConfig,
    GeneratorConfig,
    SequenceConfig,
    OmniglotConfig,
    SymbolicConfig,
    TransformerConfig,
    TransformerICL,
    create_seq_generator,
    create_sequence_dataloader,
)

DEFAULT_BASE_CONFIG: Dict[str, Any] = {
    "data": {
        "train_seqs": "bursty",
        "example_type": "symbolic",
        "generator_config": {
            "n_rare_classes": 6,
            "n_common_classes": 4,
            "n_holdout_classes": 2,
            "zipf_exponent": 0.0,
            "use_zipf_for_common_rare": False,
            "noise_scale": 0.0,
            "preserve_ordering_every_n": None,
        },
        "seq_config": {
            "seq_len": 9,
            "fs_shots": 4,
            "bursty_shots": 3,
            "ways": 2,
            "p_bursty": 0.9,
            "p_bursty_common": 0.0,
            "p_bursty_zipfian": 1.0,
            "p_fewshot": 0.1,
            "non_bursty_type": "zipfian",
            "labeling_common": "ordered",
            "labeling_rare": "ordered",
            "randomly_generate_rare": False,
            "grouped": False,
        },
        "omniglot_config": {
            "omniglot_split": "all",
            "exemplars": "all",
            "augment_images": False,
            "n_to_keep": None,
        },
    },
    "preproc": {
        "downsample": False,
    },
    "embedding": {
        "emb_dim": 64,
        "example_encoding": "embedding",
        "flatten_superpixels": False,
        "example_dropout_prob": 0.1,
        "concatenate_labels": False,
        "use_positional_encodings": True,
        "positional_dropout_prob": 0.0,
    },
    "transformer": {
        "num_layers": 4,
        "num_heads": 4,
        "dropout_prob": 0.1,
        "self_att_init_scale": 1.0,
        "dense_init_scale": 1.0,
    },
    "training": {
        "batch_size": 8,
        "learning_rate": 1e-4,
        "w_interim_predictions": 0.0,
    },
    "optimizer": {
        "name": "adamw",
        "max_lr": 3e-4,
        "warmup_steps": 4000,
        "clip_level": 0.25,
        "kwargs": {"weight_decay": 0.0},
    },
}


def _load_config(path: str, debug: bool = False) -> ConfigDict:
    """Dynamically import the ml_collections config."""
    if path.endswith(".py") and Path(path).is_file():
        spec = importlib.util.spec_from_file_location("pytorch_cfg_module", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import config from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(path.replace("/", ".").rstrip(".py"))
    if not hasattr(module, "get_config"):
        raise AttributeError(f"{path} does not define get_config()")
    return module.get_config(debug=debug)


def _config_dict_to_dict(cfg: ConfigDict) -> Dict[str, Any]:
    return json.loads(cfg.to_json())


def _resolve_generator_config(cfg: Dict[str, Any], overrides: argparse.Namespace) -> GeneratorConfig:
    gen = cfg["data"]["generator_config"]
    n_rare = overrides.n_rare if overrides.n_rare is not None else gen["n_rare_classes"]
    n_common = overrides.n_common if overrides.n_common is not None else gen["n_common_classes"]
    n_holdout = overrides.n_holdout if overrides.n_holdout is not None else gen["n_holdout_classes"]
    return GeneratorConfig(
        n_rare_classes=n_rare,
        n_common_classes=n_common,
        n_holdout_classes=n_holdout,
        zipf_exponent=gen.get("zipf_exponent", 1.0),
        use_zipf_for_common_rare=gen.get("use_zipf_for_common_rare", False),
        noise_scale=gen.get("noise_scale", 0.0),
        preserve_ordering_every_n=gen.get("preserve_ordering_every_n"),
        random_seed=gen.get("random_seed", 1337),
    )


def _resolve_sequence_config(cfg: Dict[str, Any], overrides: argparse.Namespace) -> SequenceConfig:
    seq = cfg["data"]["seq_config"]
    return SequenceConfig(
        seq_len=overrides.seq_len or seq.get("seq_len", 9),
        fs_shots=seq.get("fs_shots", 4),
        bursty_shots=seq.get("bursty_shots", 3),
        ways=seq.get("ways", 2),
        p_bursty=seq.get("p_bursty", 0.0),
        p_bursty_common=seq.get("p_bursty_common", 0.0),
        p_bursty_zipfian=seq.get("p_bursty_zipfian", 0.0),
        p_fewshot=seq.get("p_fewshot", 0.0),
        non_bursty_type=seq.get("non_bursty_type", "zipfian"),
        labeling_common=seq.get("labeling_common", "ordered"),
        labeling_rare=seq.get("labeling_rare", "ordered"),
        randomly_generate_rare=seq.get("randomly_generate_rare", False),
        grouped=seq.get("grouped", False),
    )


def _resolve_embedding_config(
    cfg: Dict[str, Any],
    num_classes: int,
    overrides: argparse.Namespace,
    symbolic_vocab: int | None = None,
) -> EmbeddingConfig:
    emb = cfg["embedding"]
    example_encoding = overrides.example_encoding or emb.get("example_encoding", "resnet")
    vocab_size = symbolic_vocab or num_classes
    return EmbeddingConfig(
        num_classes=vocab_size,
        emb_dim=emb["emb_dim"],
        example_encoding=example_encoding,
        flatten_superpixels=emb.get("flatten_superpixels", False),
        example_dropout_prob=emb.get("example_dropout_prob", 0.0),
        concatenate_labels=emb.get("concatenate_labels", False),
        use_positional_encodings=emb.get("use_positional_encodings", True),
        positional_dropout_prob=emb.get("positional_dropout_prob", 0.0),
        input_channels=overrides.image_channels or emb.get("input_channels", 1),
    )


def _resolve_transformer_config(cfg: Dict[str, Any], overrides: argparse.Namespace) -> TransformerConfig:
    tr = cfg["transformer"]
    return TransformerConfig(
        num_layers=overrides.num_layers or tr.get("num_layers", 8),
        num_heads=overrides.num_heads or tr.get("num_heads", 8),
        dropout_prob=tr.get("dropout_prob", 0.0),
        widening_factor=4,
        self_att_init_scale=tr.get("self_att_init_scale", 1.0),
        dense_init_scale=tr.get("dense_init_scale", 1.0),
    )


def _build_optimizer(
    model: torch.nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )


def _schedule_lr(step: int, max_lr: float, warmup_steps: int) -> float:
    step = max(step, 1)
    warmup = max_lr * float(step) / float(max(warmup_steps, 1))
    decay = max_lr * (warmup_steps ** 0.5) * (step ** -0.5)
    return min(warmup, decay)


def _compute_loss_and_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    w_interim: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch, seq_len, num_classes = logits.shape
    losses_all = torch.nn.functional.cross_entropy(
        logits.view(batch * seq_len, num_classes),
        targets.view(batch * seq_len),
        reduction="none",
    ).view(batch, seq_len)

    device = logits.device
    query_mask = torch.zeros_like(losses_all, dtype=torch.float32, device=device)
    query_mask[:, -1] = 1.0
    interim_mask = torch.zeros_like(losses_all, dtype=torch.float32, device=device)
    interim_mask[:, :-1:2] = 1.0
    if w_interim > 0:
        n_interim = max(int((seq_len - 1) // 2), 1)
        weight_mask = interim_mask * (w_interim / n_interim) + query_mask * (1.0 - w_interim)
    else:
        weight_mask = query_mask

    loss = (losses_all * weight_mask).sum() / weight_mask.sum()

    losses_query = (losses_all * query_mask).sum() / query_mask.sum()
    if w_interim > 0:
        losses_interim = (losses_all * interim_mask).sum() / interim_mask.sum()
    else:
        losses_interim = torch.tensor(0.0, device=device)

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = (preds == targets).float()
        accuracy_query = (correct * query_mask).sum() / query_mask.sum()
        if w_interim > 0 and interim_mask.sum() > 0:
            accuracy_interim = (correct * interim_mask).sum() / interim_mask.sum()
        else:
            accuracy_interim = torch.tensor(0.0, device=device)

    metrics = {
        "loss": float(loss.detach().cpu()),
        "loss_query": float(losses_query.detach().cpu()),
        "loss_interim": float(losses_interim.detach().cpu()),
        "accuracy_query": float(accuracy_query.detach().cpu()),
        "accuracy_interim": float(accuracy_interim.detach().cpu()),
    }
    return loss, metrics


def _compute_weight_norms(model: torch.nn.Module) -> Dict[str, float]:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    total_all = torch.tensor(0.0, device=total.device)
    exclude = ("bias", "norm", "layernorm", "layer_norm", "bn", "embedding")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        total_all += param.pow(2).sum()
        if param.dim() <= 1 or any(token in name.lower() for token in exclude):
            continue
        total += param.pow(2).sum()

    return {
        "weight_l2": total.sqrt().item(),
        "weight_l2_all": total_all.sqrt().item(),
    }


def _compute_grad_norm(model: torch.nn.Module) -> float:
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.grad is not None:
            total += param.grad.pow(2).sum()
    return total.sqrt().item()


def _log_to_progress(progress_path: Optional[Path], message: str) -> None:
    if progress_path is None:
        return
    with progress_path.open("a", encoding="utf-8") as progress_file:
        progress_file.write(message + "\n")


def _evaluate_sequence_type(
    model: torch.nn.Module,
    *,
    seq_type: str,
    example_type: str,
    generator_cfg: GeneratorConfig,
    sequence_cfg: SequenceConfig,
    embed_config: EmbeddingConfig,
    symbolic_config: SymbolicConfig | None,
    omniglot_config: OmniglotConfig | None,
    device: torch.device,
    batch_size: int,
    num_batches: int,
    num_workers: int,
    seed: int,
    w_interim: float,
    downsample: bool,
) -> Dict[str, float]:
    eval_seed = seed + 1337
    eval_generator_cfg = replace(generator_cfg, random_seed=eval_seed)

    eval_seq_generator = create_seq_generator(
        example_type,
        eval_generator_cfg,
        symbolic_config=symbolic_config,
        omniglot_config=omniglot_config,
    )
    max_sequences = batch_size * max(num_batches, 1)
    eval_loader = create_sequence_dataloader(
        eval_seq_generator,
        seq_type=seq_type,
        sequence_config=sequence_cfg,
        example_type=example_type,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=eval_seed,
        interleave_targets=not embed_config.concatenate_labels,
        downsample=downsample,
        max_sequences=max_sequences,
    )

    was_training = model.training
    model.eval()

    totals: Dict[str, float] = defaultdict(float)
    total_sequences = 0

    with torch.no_grad():
        for batch in eval_loader:
            examples = batch["examples"].to(device)
            labels = batch["labels"].to(device)
            targets = batch["target"].to(device)
            logits = model(examples, labels, attention_mask=batch.get("mask"))
            _, metrics = _compute_loss_and_metrics(logits, targets, w_interim)
            batch_size_actual = examples.shape[0]
            total_sequences += batch_size_actual
            for key, value in metrics.items():
                totals[key] += value * batch_size_actual

    if was_training:
        model.train(True)

    if total_sequences == 0:
        return {key: 0.0 for key in totals}

    return {key: value / total_sequences for key, value in totals.items()}


def train(args: argparse.Namespace) -> None:
    if args.config.lower() == "none":
        cfg = json.loads(json.dumps(DEFAULT_BASE_CONFIG))
    else:
        try:
            raw_config = _load_config(args.config, debug=args.debug)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                f"Failed to import dependencies for config '{args.config}'. "
                "Re-run with --config=none or install the JAX stack."
            ) from exc
        cfg = _config_dict_to_dict(raw_config.experiment_kwargs["config"])

    example_type = args.example_type or cfg["data"]["example_type"]

    generator_cfg = _resolve_generator_config(cfg, args)
    sequence_cfg = _resolve_sequence_config(cfg, args)

    total_classes = (
        generator_cfg.n_rare_classes
        + generator_cfg.n_common_classes
        + generator_cfg.n_holdout_classes
    )

    symbolic_cfg: SymbolicConfig | None = None
    omniglot_cfg: OmniglotConfig | None = None
    if example_type == "symbolic":
        symbolic_cfg = SymbolicConfig(
            dataset_size=max(args.symbolic_dataset_size, total_classes)
        )
    elif example_type == "omniglot":
        omni_raw = cfg["data"].get("omniglot_config", {})
        omniglot_cfg = OmniglotConfig(
            omniglot_split=omni_raw.get("omniglot_split", "all"),
            exemplars=omni_raw.get("exemplars", "all"),
            augment_images=omni_raw.get("augment_images", False),
            n_to_keep=omni_raw.get("n_to_keep"),
        )
    else:
        raise ValueError(f"Unsupported example_type '{example_type}'")

    vocab_override = symbolic_cfg.dataset_size if symbolic_cfg else None
    embed_cfg = _resolve_embedding_config(cfg, total_classes, args, symbolic_vocab=vocab_override)
    transformer_cfg = _resolve_transformer_config(cfg, args)

    training_cfg = cfg["training"]
    optimizer_cfg = cfg["optimizer"]
    w_interim = training_cfg.get("w_interim_predictions", 0.0)

    device = torch.device(args.device)
    downsample = bool(cfg.get("preproc", {}).get("downsample", False))
    train_batch_size = args.batch_size or training_cfg["batch_size"]

    seq_generator = create_seq_generator(
        example_type,
        generator_cfg,
        omniglot_config=omniglot_cfg,
        symbolic_config=symbolic_cfg,
    )
    dataloader = create_sequence_dataloader(
        seq_generator,
        seq_type=cfg["data"]["train_seqs"],
        sequence_config=sequence_cfg,
        example_type=example_type,
        batch_size=train_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        interleave_targets=not embed_cfg.concatenate_labels,
        downsample=downsample,
    )

    model = TransformerICL(
        embed_cfg,
        transformer_cfg,
        num_classes=total_classes,
    ).to(device)

    optimizer = _build_optimizer(
        model,
        learning_rate=args.learning_rate or training_cfg["learning_rate"],
        weight_decay=optimizer_cfg["kwargs"].get("weight_decay", 0.0),
    )

    save_dir = Path(args.output_dir) if args.output_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
    progress_path: Optional[Path] = None
    if save_dir:
        progress_path = save_dir / "progress.txt"
        if progress_path.exists():
            progress_path.unlink()

    model.train()
    step = 0
    data_iter = iter(dataloader)

    while step < args.max_steps:
        step += 1
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        examples = batch["examples"].to(device)
        labels = batch["labels"].to(device)
        targets = batch["target"].to(device)

        lr = _schedule_lr(step, optimizer_cfg["max_lr"], optimizer_cfg["warmup_steps"])
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        logits = model(examples, labels)
        loss, metrics = _compute_loss_and_metrics(
            logits,
            targets,
            w_interim,
        )
        loss.backward()
        grad_norm = _compute_grad_norm(model)
        optimizer.step()

        weight_norms = _compute_weight_norms(model)
        metrics.update(weight_norms)
        metrics["grad_l2"] = grad_norm
        metrics["lr"] = lr

        if step % args.log_interval == 0 or step == 1:
            metric_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
            log_line = f"[step {step:05d}] {metric_str}"
            print(log_line, flush=True)
            _log_to_progress(progress_path, log_line)

        if args.eval_holdout_interval and step % args.eval_holdout_interval == 0:
            eval_batch_size = args.eval_batch_size or train_batch_size
            eval_metrics = _evaluate_sequence_type(
                model,
                seq_type="fewshot_holdout",
                example_type=example_type,
                generator_cfg=generator_cfg,
                sequence_cfg=sequence_cfg,
                embed_config=embed_cfg,
                symbolic_config=symbolic_cfg,
                omniglot_config=omniglot_cfg,
                device=device,
                batch_size=eval_batch_size,
                num_batches=max(args.eval_holdout_batches, 1),
                num_workers=args.num_workers,
                seed=args.seed,
                w_interim=w_interim,
                downsample=downsample,
            )
            eval_str = " ".join(f"{k}={v:.4f}" for k, v in sorted(eval_metrics.items()))
            eval_line = f"[eval_holdout step {step:05d}] {eval_str}"
            print(eval_line, flush=True)
            _log_to_progress(progress_path, eval_line)

        if save_dir and args.save_interval and step % args.save_interval == 0:
            ckpt_dir = save_dir / "checkpoints" / "models" / "latest" / f"step_{step:07d}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                },
                ckpt_dir / "weights.pt",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch training entrypoint for Omniglot sequences.")
    parser.add_argument("--config", default="experiment/configs/images_all_exemplars.py")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--example-type", choices=("omniglot", "symbolic"), default=None)
    parser.add_argument("--example-encoding", choices=("resnet", "linear", "embedding"), default=None)
    parser.add_argument("--n-rare", type=int, default=None)
    parser.add_argument("--n-common", type=int, default=None)
    parser.add_argument("--n-holdout", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-channels", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-interval", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--symbolic-dataset-size", type=int, default=2048)
    parser.add_argument("--eval-holdout-interval", type=int, default=0)
    parser.add_argument("--eval-holdout-batches", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    train(args)
