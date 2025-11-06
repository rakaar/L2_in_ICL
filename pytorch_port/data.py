"""PyTorch data pipeline for the L2-regularised Omniglot sequences experiment."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterator, Optional
import importlib.machinery
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
try:
  from datasets import data_generators
except ModuleNotFoundError as exc:  # pragma: no cover - fallback for missing TF.
  def _unavailable(*_, **__):
    raise ModuleNotFoundError(
        "TensorFlow is required for Omniglot support. Install tensorflow>=2.9 "
        "or set example_type='symbolic'.") from exc

  tf_stub = types.ModuleType("tensorflow")
  image_stub = types.SimpleNamespace(
      convert_image_dtype=_unavailable,
      rgb_to_grayscale=_unavailable,
      flip_left_right=_unavailable,
      rot90=_unavailable,
  )
  tf_stub.image = image_stub
  tf_stub.data = types.SimpleNamespace(experimental=types.SimpleNamespace(AUTOTUNE=None))
  tf_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow", loader=None)
  sys.modules.setdefault("tensorflow", tf_stub)
  sys.modules.setdefault("tensorflow.compat", tf_stub)
  sys.modules.setdefault("tensorflow.compat.v2", tf_stub)

  tfds_stub = types.ModuleType("tensorflow_datasets")
  tfds_stub.load = _unavailable
  tfds_stub.__spec__ = importlib.machinery.ModuleSpec("tensorflow_datasets", loader=None)
  sys.modules.setdefault("tensorflow_datasets", tfds_stub)

  from datasets import data_generators  # type: ignore  # noqa: E402

ExampleBatch = Dict[str, torch.Tensor]


@dataclass
class GeneratorConfig:
  """Configuration for splitting classes into rare/common/holdout sets."""

  n_rare_classes: int
  n_common_classes: int
  n_holdout_classes: int = 0
  zipf_exponent: float = 1.0
  use_zipf_for_common_rare: bool = False
  noise_scale: float = 0.1
  preserve_ordering_every_n: Optional[int] = None
  random_seed: int = 1337


@dataclass
class OmniglotConfig:
  """Configuration for loading Omniglot exemplars."""

  omniglot_split: str = "all"
  exemplars: str = "all"
  augment_images: bool = False
  n_to_keep: Optional[int] = None


@dataclass
class SymbolicConfig:
  """Configuration for the synthetic integer dataset."""

  dataset_size: int = 1000


@dataclass
class SequenceConfig:
  """Configuration for sequence generation."""

  seq_len: int = 9
  fs_shots: int = 4
  bursty_shots: int = 3
  ways: int = 2
  p_bursty: float = 0.0
  p_bursty_common: float = 0.0
  p_bursty_zipfian: float = 0.0
  p_fewshot: float = 0.0
  non_bursty_type: str = "zipfian"
  labeling_common: str = "ordered"
  labeling_rare: str = "ordered"
  randomly_generate_rare: bool = False
  grouped: bool = False


def create_seq_generator(
    example_type: str,
    generator_config: GeneratorConfig,
    *,
    omniglot_config: Optional[OmniglotConfig] = None,
    symbolic_config: Optional[SymbolicConfig] = None,
) -> data_generators.SeqGenerator:
  """Factory that mirrors the JAX experiment's SeqGenerator creation."""
  if example_type == "omniglot":
    if omniglot_config is None:
      raise ValueError("omniglot_config must be provided for example_type='omniglot'")
    dataset_for_sampling = data_generators.OmniglotDatasetForSampling(
        **asdict(omniglot_config)
    )
  elif example_type == "symbolic":
    if symbolic_config is None:
      raise ValueError("symbolic_config must be provided for example_type='symbolic'")
    dataset_for_sampling = data_generators.SymbolicDatasetForSampling(
        **asdict(symbolic_config)
    )
  else:
    raise ValueError(f"Unsupported example_type '{example_type}'")

  return data_generators.SeqGenerator(
      dataset_for_sampling, **asdict(generator_config)
  )


class SequenceDataset(IterableDataset):
  """IterableDataset that reproduces the JAX sequence generator behaviour."""

  def __init__(
      self,
      seq_generator: data_generators.SeqGenerator,
      seq_type: str,
      sequence_config: SequenceConfig,
      example_type: str,
      *,
      interleave_targets: bool = True,
      use_constant_labels: bool = False,
      downsample: bool = False,
      max_sequences: Optional[int] = None,
  ):
    super().__init__()
    self._seq_generator = seq_generator
    self._seq_type = seq_type
    self._sequence_config = sequence_config
    self._example_type = example_type
    self._interleave_targets = interleave_targets
    self._use_constant_labels = use_constant_labels
    self._downsample = downsample
    self._max_sequences = max_sequences
    self._seq_len = self._resolve_seq_len()
    self._build_generator: Callable[[], Iterator[Dict[str, np.ndarray]]] = (
        self._make_generator_factory()
    )

  def _resolve_seq_len(self) -> int:
    seq_type = self._seq_type
    cfg = self._sequence_config
    fewshot_like = {
        "fewshot_rare",
        "fewshot_common",
        "fewshot_zipfian",
        "fewshot_holdout",
        "mixed",
    }
    if seq_type in fewshot_like:
      return cfg.fs_shots * cfg.ways + 1
    return cfg.seq_len

  def _make_generator_factory(self) -> Callable[[], Iterator[Dict[str, np.ndarray]]]:
    cfg = self._sequence_config
    seq_type = self._seq_type
    if seq_type == "bursty":
      return lambda: self._seq_generator.get_bursty_seq(
          self._seq_len,
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
      return lambda: self._seq_generator.get_no_support_seq(
          "common",
          self._seq_len,
          False,
          cfg.labeling_common,
          cfg.randomly_generate_rare,
      )
    if seq_type == "no_support_rare":
      return lambda: self._seq_generator.get_no_support_seq(
          "rare",
          self._seq_len,
          False,
          cfg.labeling_common,
          cfg.randomly_generate_rare,
      )
    if seq_type == "no_support_zipfian":
      return lambda: self._seq_generator.get_no_support_seq(
          "zipfian",
          self._seq_len,
          False,
          cfg.labeling_common,
          cfg.randomly_generate_rare,
      )
    if seq_type == "fewshot_rare":
      return lambda: self._seq_generator.get_fewshot_seq(
          "rare",
          cfg.fs_shots,
          cfg.ways,
          "unfixed",
          cfg.randomly_generate_rare,
          cfg.grouped,
      )
    if seq_type == "fewshot_common":
      return lambda: self._seq_generator.get_fewshot_seq(
          "common",
          cfg.fs_shots,
          cfg.ways,
          "unfixed",
          False,
          cfg.grouped,
      )
    if seq_type == "fewshot_zipfian":
      return lambda: self._seq_generator.get_fewshot_seq(
          "zipfian",
          cfg.fs_shots,
          cfg.ways,
          "unfixed",
          cfg.randomly_generate_rare,
          cfg.grouped,
      )
    if seq_type == "fewshot_holdout":
      return lambda: self._seq_generator.get_fewshot_seq(
          "holdout",
          cfg.fs_shots,
          cfg.ways,
          "unfixed",
          cfg.randomly_generate_rare,
          cfg.grouped,
      )
    if seq_type == "mixed":
      return lambda: self._seq_generator.get_mixed_seq(
          cfg.fs_shots,
          cfg.ways,
          cfg.p_fewshot,
      )
    raise ValueError(f"Unsupported seq_type '{seq_type}'")

  @property
  def seq_len(self) -> int:
    """Return the resolved sequence length for downstream consumers."""
    return self._seq_len

  def _prepare_targets(self, labels: np.ndarray) -> np.ndarray:
    labels_prepared = np.ones_like(labels) if self._use_constant_labels else labels
    if self._interleave_targets:
      zeros = np.zeros_like(labels_prepared)
      stacked = np.stack((labels_prepared, zeros), axis=-1)
      targets = stacked.reshape(-1)[:-1]
    else:
      targets = labels_prepared.copy()
    return labels_prepared, targets

  def _convert_example(self, record: Dict[str, np.ndarray]) -> ExampleBatch:
    labels_np = np.asarray(record["label"], dtype=np.int64)
    labels_np, targets_np = self._prepare_targets(labels_np)

    if self._example_type == "omniglot":
      examples = torch.from_numpy(
          np.asarray(record["example"], dtype=np.float32)
      )
      # Keep channel last until the model port decides otherwise.
      if self._downsample:
        examples = examples.permute(0, 3, 1, 2)
        examples = F.interpolate(
            examples,
            size=(28, 28),
            mode="bilinear",
            align_corners=False,
        )
        examples = examples.permute(0, 2, 3, 1)
    elif self._example_type == "symbolic":
      examples = torch.from_numpy(
          np.asarray(record["example"], dtype=np.int64)
      )
    else:
      raise ValueError(f"Unsupported example_type '{self._example_type}'")

    batch: ExampleBatch = {
        "examples": examples,
        "labels": torch.from_numpy(labels_np.astype(np.int64)),
        "target": torch.from_numpy(targets_np.astype(np.int64)),
    }
    if "is_rare" in record:
      batch["is_rare"] = torch.from_numpy(
          np.asarray(record["is_rare"], dtype=np.int64)
      )
    return batch

  def __iter__(self) -> Iterator[ExampleBatch]:
    generator = self._build_generator()
    produced = 0
    while self._max_sequences is None or produced < self._max_sequences:
      record = next(generator)
      yield self._convert_example(record)
      produced += 1


def _seed_worker(worker_id: int, base_seed: int) -> None:
  worker_seed = base_seed + worker_id
  np.random.seed(worker_seed % (2**32))
  random.seed(worker_seed)
  torch.manual_seed(worker_seed)


def create_sequence_dataloader(
    seq_generator: data_generators.SeqGenerator,
    seq_type: str,
    sequence_config: SequenceConfig,
    example_type: str,
    *,
    batch_size: int,
    num_workers: int = 0,
    seed: Optional[int] = None,
    interleave_targets: bool = True,
    use_constant_labels: bool = False,
    downsample: bool = False,
    max_sequences: Optional[int] = None,
    shuffle: bool = False,
    pin_memory: bool = False,
    persistent_workers: Optional[bool] = None,
) -> DataLoader:
  """Create a DataLoader with worker seeding aligned to the JAX pipeline."""
  if shuffle:
    raise ValueError("shuffle=True is not supported for IterableDataset pipelines.")

  dataset = SequenceDataset(
      seq_generator,
      seq_type,
      sequence_config,
      example_type,
      interleave_targets=interleave_targets,
      use_constant_labels=use_constant_labels,
      downsample=downsample,
      max_sequences=max_sequences,
  )

  worker_init_fn = None
  generator = None
  if seed is not None:
    generator = torch.Generator()
    generator.manual_seed(seed)
    if num_workers == 0:
      _seed_worker(0, seed)
    else:
      worker_init_fn = lambda worker_id: _seed_worker(worker_id, seed)

  if persistent_workers is None:
    persistent_workers = bool(num_workers)

  return DataLoader(
      dataset,
      batch_size=batch_size,
      num_workers=num_workers,
      worker_init_fn=worker_init_fn,
      generator=generator,
      shuffle=shuffle,
      pin_memory=pin_memory,
      persistent_workers=persistent_workers,
  )


__all__ = [
    "GeneratorConfig",
    "OmniglotConfig",
    "SequenceConfig",
    "SymbolicConfig",
    "SequenceDataset",
    "create_seq_generator",
    "create_sequence_dataloader",
]
