# PyTorch Port Plan – L2-Regularised Figure 3 Sweep

## 0. Objectives
- Reproduce the current JAX-based AdamW/L2 sweep (`analysis/run_adamw_sweep.py`) using a PyTorch training stack.
- Maintain functional parity: data generation, transformer architecture, logging/checkpoint layout, evaluation metrics, and plotting outputs (`figure3_adamw.*`, `figure3_adamw_metrics.csv`).
- Minimise disruption to downstream analysis by matching CLI flags, directory structures, and CSV schemas where practical.

## 1. Reference Inventory
- `analysis/run_adamw_sweep.py` – sweep driver, log/checkpoint conventions.
- `analysis/plot_figure3_adamw.py` – evaluation + plotting workflow and CSV schema.
- `emergent_in_context_learning/experiment/experiment.py` – JAX training entrypoint.
- `experiment/configs/images_all_exemplars.py` – default config tree (model dims, data shapes, optimiser settings).
- `datasets/` and `modules/` – data generator + Haiku modules used in the experiment.

## 2. Phase 0 – Discovery & Parity Notes
- Map config values that must be replicated (embedding dim, heads, depth, learning-rate schedule, batch size, sequence layout).
- Document data tensor shapes (context length, support/query split, label encoding) by instrumenting a short JAX run.
- Capture metric definitions: _few-shot holdout accuracy_, _trained-class accuracy_, `weight_l2`, `weight_l2_all`, `grad_l2`.
- Record checkpoint format expectations (per-step directories under `checkpoints/models/latest/step_*`).
- Decide on reproducibility knobs: RNG seeds, deterministic dataloaders, torch.backends flags.

## 3. Phase 1 – Data Pipeline Port
- Reimplement the synthetic dataset generator as a `torch.utils.data.Dataset`.
  - Mirror arguments from `generator_config` (common/rare/holdout counts, exemplar counts, support/query lengths).
  - Ensure identical sampling logic (class selection, exemplar sampling, sequence ordering) by porting relevant utilities from `datasets/`.
- Wrap the dataset in a `DataLoader` with worker seed control to match JAX’s `jax.random` behaviour.
- Add quick sanity script to compare one batch from PyTorch pipeline against JAX output (class distributions, label mappings).

## 4. Phase 2 – Model & Config Translation
- Port the Haiku transformer stack to PyTorch modules.
  - Match embedding initialisation, positional encoding, attention mask semantics, LayerNorm placement, dropout.
  - Translate optimiser hyperparameters: AdamW with cosine or constant LR as per config; weight decay applied to same parameter set.
- Replicate any auxiliary heads or readout layers (classification logits, value heads) with matching shapes.
- Implement weight-norm tracking hooks (`weight_l2`/`weight_l2_all`) for parity with JAX logging.
- Write unit tests comparing parameter counts and a forward pass on fixed tensors versus the JAX model; tolerate small numeric drift.

## 5. Phase 3 – Training Loop & Metrics
- Build a PyTorch training script (`analysis/pytorch_train.py` or similar) that:
  - Consumes the same CLI/config structure (argparse + overrides).
  - Handles gradient accumulation, mixed precision (optional), and logging every N steps.
  - Evaluates two regimes each checkpoint: holdout few-shot accuracy (ICL) and trained-class accuracy (memorisation).
  - Logs scalar metrics to stdout and to structured JSON/CSV files compatible with `progress.txt`.
  - Saves checkpoints in `checkpoints/models/latest/step_{k}/weights.pt` mirroring the JAX directory layout.
- Add resume capability from latest checkpoint (load model, optimiser, scheduler state).

## 6. Phase 4 – Sweep Harness Integration
- Option A: extend `analysis/run_adamw_sweep.py` with a `--backend {jax, torch}` flag that switches the launched command.
- Option B: create `analysis/run_adamw_sweep_torch.py` reusing helper utilities but pointing to the PyTorch trainer.
- Ensure manifest writing (`runs.csv`) records backend, checkpoint path, weight decay, class count, completed steps.
- Mirror log handling (`train.log`, `progress.txt`) so monitoring scripts remain unchanged.

## 7. Phase 5 – Evaluation & Plotting
- Update `analysis/plot_figure3_adamw.py` to accept PyTorch checkpoints/metrics:
  - Either load weights for evaluation or read cached metrics if training already exported them.
  - Introduce a backend abstraction layer; reuse evaluation routines for both JAX and PyTorch data.
- Verify that output figures/CSVs retain the same schema and filenames to avoid downstream breakage.
- If numerical differences appear, add flags to overlay JAX vs PyTorch curves for comparison.

## 8. Validation Strategy
- **Unit-level**: deterministic tests for dataset shapes, parameter counts, single-batch forward pass outputs.
- **Integration**: short (1k–2k step) training runs on CPU/GPU to confirm loss decreases and metrics populate.
- **Regression**: compare PyTorch sweep results (λ=0 and λ=3e-3) against archived JAX metrics; document deviations.
- **Performance**: profile GPU utilisation; adjust mixed precision or dataloader settings if throughput lags.

## 9. Deployment & Ops Checklist
- Provide environment instructions (`requirements-torch.txt` or poetry/conda env) that avoid JAX conflicts.
- Document CLI usage in `docs/icl_sweep_onboarding.md` (Torch section) once port is stable.
- Confirm CI or smoke scripts cover the new code paths, and add lint/format rules as needed.
- Archive baseline PyTorch runs under `runs/torch_adamw_sweep/` for reproducibility.

## 10. Open Questions & Follow-Ups
- Do we need Lightning/Accelerate integration for multi-GPU scalability?
- Should we support continued sweeps mixing JAX and PyTorch results in a single `runs.csv`?
- Is there appetite for wandb/tensorboard logging alongside plain text?
- Decide on default precision (fp32 vs bf16) and document required CUDA/cuDNN versions for PyTorch builds.

