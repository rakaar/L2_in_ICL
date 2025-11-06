"""Quick parity check between the JAX SeqGenerator and the PyTorch dataset."""

from __future__ import annotations

import argparse
import random
from typing import Dict, Iterable

import numpy as np
import torch

from datasets import data_generators
from pytorch_port.data import (
    GeneratorConfig,
    OmniglotConfig,
    SequenceConfig,
    SequenceDataset,
    SymbolicConfig,
    create_seq_generator,
)


def _build_generator(
    seq_generator: data_generators.SeqGenerator,
    seq_type: str,
    cfg: SequenceConfig,
) -> Iterable[Dict[str, np.ndarray]]:
    if seq_type == "bursty":
        return seq_generator.get_bursty_seq(
            cfg.seq_len,
            cfg.bursty_shots,
            cfg.ways,
            cfg.p_bursty,
            cfg.p_bursty_common,
            cfg.p_bursty_zipfian,
            cfg.non_bursty_type,
            cfg.labeling_common,
            cfg.labeling_rare,
            cfg.randomly_generate_rare,
            cfg.grouped,
        )
    if seq_type == "no_support_common":
        return seq_generator.get_no_support_seq(
            "common",
            cfg.seq_len,
            False,
            cfg.labeling_common,
            cfg.randomly_generate_rare,
        )
    if seq_type == "no_support_rare":
        return seq_generator.get_no_support_seq(
            "rare",
            cfg.seq_len,
            False,
            cfg.labeling_common,
            cfg.randomly_generate_rare,
        )
    if seq_type == "no_support_zipfian":
        return seq_generator.get_no_support_seq(
            "zipfian",
            cfg.seq_len,
            False,
            cfg.labeling_common,
            cfg.randomly_generate_rare,
        )
    if seq_type == "fewshot_rare":
        return seq_generator.get_fewshot_seq(
            "rare",
            cfg.fs_shots,
            cfg.ways,
            "unfixed",
            cfg.randomly_generate_rare,
            cfg.grouped,
        )
    if seq_type == "fewshot_common":
        return seq_generator.get_fewshot_seq(
            "common",
            cfg.fs_shots,
            cfg.ways,
            "unfixed",
            False,
            cfg.grouped,
        )
    if seq_type == "fewshot_zipfian":
        return seq_generator.get_fewshot_seq(
            "zipfian",
            cfg.fs_shots,
            cfg.ways,
            "unfixed",
            cfg.randomly_generate_rare,
            cfg.grouped,
        )
    if seq_type == "fewshot_holdout":
        return seq_generator.get_fewshot_seq(
            "holdout",
            cfg.fs_shots,
            cfg.ways,
            "unfixed",
            cfg.randomly_generate_rare,
            cfg.grouped,
        )
    if seq_type == "mixed":
        return seq_generator.get_mixed_seq(
            cfg.fs_shots,
            cfg.ways,
            cfg.p_fewshot,
        )
    raise ValueError(f"Unsupported seq_type '{seq_type}'")


def _prepare_targets(labels: np.ndarray, interleave: bool, constant: bool) -> np.ndarray:
    labels_prepared = np.ones_like(labels) if constant else labels
    if interleave:
        zeros = np.zeros_like(labels_prepared)
        stacked = np.stack((labels_prepared, zeros), axis=-1)
        return stacked.reshape(-1)[:-1]
    return labels_prepared.copy()


def compare_once(args: argparse.Namespace) -> None:
    generator_config = GeneratorConfig(
        n_rare_classes=args.n_rare,
        n_common_classes=args.n_common,
        n_holdout_classes=args.n_holdout,
        zipf_exponent=args.zipf_exponent,
        use_zipf_for_common_rare=args.use_zipf_for_common_rare,
        noise_scale=args.noise_scale,
        preserve_ordering_every_n=args.preserve_every_n,
        random_seed=args.partition_seed,
    )

    sequence_config = SequenceConfig(
        seq_len=args.seq_len,
        fs_shots=args.fs_shots,
        bursty_shots=args.bursty_shots,
        ways=args.ways,
        p_bursty=args.p_bursty,
        p_bursty_common=args.p_bursty_common,
        p_bursty_zipfian=args.p_bursty_zipfian,
        p_fewshot=args.p_fewshot,
        non_bursty_type=args.non_bursty_type,
        labeling_common=args.labeling_common,
        labeling_rare=args.labeling_rare,
        randomly_generate_rare=args.randomly_generate_rare,
        grouped=args.grouped,
    )

    if args.example_type == "symbolic":
        dataset_config = SymbolicConfig(
            dataset_size=max(
                args.symbolic_dataset_size,
                generator_config.n_rare_classes
                + generator_config.n_common_classes
                + generator_config.n_holdout_classes,
            )
        )
        seq_gen_reference = create_seq_generator(
            "symbolic",
            generator_config,
            symbolic_config=dataset_config,
        )
        seq_gen_torch = create_seq_generator(
            "symbolic",
            generator_config,
            symbolic_config=dataset_config,
        )
    elif args.example_type == "omniglot":
        dataset_config = OmniglotConfig(
            omniglot_split=args.omniglot_split,
            exemplars=args.omniglot_exemplars,
            augment_images=args.omniglot_augment,
            n_to_keep=args.omniglot_limit,
        )
        seq_gen_reference = create_seq_generator(
            "omniglot",
            generator_config,
            omniglot_config=dataset_config,
        )
        seq_gen_torch = create_seq_generator(
            "omniglot",
            generator_config,
            omniglot_config=dataset_config,
        )
    else:
        raise ValueError(f"Unsupported example_type '{args.example_type}'")

    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed % (2**32))
    jax_like_record = next(_build_generator(seq_gen_reference, args.seq_type, sequence_config))

    random.seed(args.sample_seed)
    np.random.seed(args.sample_seed % (2**32))
    torch.manual_seed(args.sample_seed)
    torch_dataset = SequenceDataset(
        seq_gen_torch,
        args.seq_type,
        sequence_config,
        args.example_type,
        interleave_targets=(not args.no_interleave),
        use_constant_labels=args.use_constant_labels,
        downsample=args.downsample,
        max_sequences=1,
    )
    torch_record = next(iter(torch_dataset))

    labels_match = np.array_equal(
        jax_like_record["label"], torch_record["labels"].numpy()
    )
    is_rare_match = np.array_equal(
        jax_like_record["is_rare"], torch_record["is_rare"].numpy()
    ) if "is_rare" in jax_like_record else True

    expected_target = _prepare_targets(
        jax_like_record["label"],
        interleave=(not args.no_interleave),
        constant=args.use_constant_labels,
    )
    target_match = np.array_equal(
        expected_target, torch_record["target"].numpy()
    )

    print("=== PyTorch vs JAX generator sanity check ===")
    print(f"Seq type            : {args.seq_type}")
    print(f"Example type        : {args.example_type}")
    print(f"Labels identical    : {labels_match}")
    print(f"Targets identical   : {target_match}")
    print(f"is_rare identical   : {is_rare_match}")

    unique_labels, counts = np.unique(jax_like_record["label"], return_counts=True)
    dist_pairs = ", ".join(
        f"{int(lbl)}:{int(cnt)}" for lbl, cnt in zip(unique_labels, counts)
    )
    print(f"Label histogram     : {dist_pairs}")

    if args.print_sequences:
        print("JAX labels          :", jax_like_record["label"])
        print("Torch labels        :", torch_record["labels"].numpy())
        print("Expected target     :", expected_target)
        print("Torch target        :", torch_record["target"].numpy())
        if args.example_type == "symbolic":
            print("JAX examples        :", jax_like_record["example"])
            print("Torch examples      :", torch_record["examples"].numpy())
        else:
            diff = np.abs(
                jax_like_record["example"] - torch_record["examples"].numpy()
            ).max()
            print(f"Max |example diff|  : {diff:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a single batch from the PyTorch dataset with the original generator.",
    )
    parser.add_argument("--seq-type", default="bursty")
    parser.add_argument("--example-type", default="symbolic", choices=("symbolic", "omniglot"))
    parser.add_argument("--seq-len", type=int, default=9)
    parser.add_argument("--fs-shots", type=int, default=4)
    parser.add_argument("--bursty-shots", type=int, default=3)
    parser.add_argument("--ways", type=int, default=2)
    parser.add_argument("--p-bursty", type=float, default=0.9)
    parser.add_argument("--p-bursty-common", type=float, default=0.0)
    parser.add_argument("--p-bursty-zipfian", type=float, default=1.0)
    parser.add_argument("--p-fewshot", type=float, default=0.1)
    parser.add_argument("--non-bursty-type", default="zipfian")
    parser.add_argument("--labeling-common", default="ordered")
    parser.add_argument("--labeling-rare", default="ordered")
    parser.add_argument("--randomly-generate-rare", action="store_true")
    parser.add_argument("--grouped", action="store_true")
    parser.add_argument("--n-rare", type=int, default=12)
    parser.add_argument("--n-common", type=int, default=6)
    parser.add_argument("--n-holdout", type=int, default=4)
    parser.add_argument("--zipf-exponent", type=float, default=0.0)
    parser.add_argument("--use-zipf-for-common-rare", action="store_true")
    parser.add_argument("--noise-scale", type=float, default=0.0)
    parser.add_argument("--preserve-every-n", type=int, default=None)
    parser.add_argument("--partition-seed", type=int, default=1337)
    parser.add_argument("--sample-seed", type=int, default=2023)
    parser.add_argument("--symbolic-dataset-size", type=int, default=1024)
    parser.add_argument("--omniglot-split", default="all")
    parser.add_argument("--omniglot-exemplars", default="all")
    parser.add_argument("--omniglot-augment", action="store_true")
    parser.add_argument("--omniglot-limit", type=int, default=None)
    parser.add_argument("--no-interleave", action="store_true")
    parser.add_argument("--use-constant-labels", action="store_true")
    parser.add_argument("--downsample", action="store_true")
    parser.add_argument("--print-sequences", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compare_once(args)


if __name__ == "__main__":
    main()

