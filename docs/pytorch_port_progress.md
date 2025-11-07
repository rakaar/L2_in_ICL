# PyTorch Port Progress Notes

_Last updated: 2025-11-06_

## Phase 1 – Data Pipeline
- Added `pytorch_port.data.SequenceDataset` wrapping the existing JAX `SeqGenerator`, reproducing sequence types, label interleaving, and per-worker seeding.
- Created `pytorch_port.data.create_sequence_dataloader` to expose deterministic `DataLoader` construction and enforce the non-shuffling iterable semantics.
- Implemented TensorFlow/TensorFlow Datasets stubs so symbolic-mode sampling works without the full TF stack; Omniglot modes still require a real TF install.
- Added `pytorch_port.sanity_check` CLI to compare a seeded PyTorch batch against the JAX generator; validated the symbolic path inside a project-local virtualenv.

## Phase 2 – Model Translation (in progress)
- Drafted `TransformerICL`, a PyTorch module capturing the Haiku transformer tower configuration (embedding, positional encodings, causal self-attention, FFN, readout).
- Ported the Haiku `SimpleResNet` to PyTorch and integrated it into the input embedder, including support for flattened superpixels and automatic projection back to the requested embedding dim.
- `analysis/pytorch_train.py` now accepts the Omniglot path (requires TensorFlow for data loading) and propagates image-channel/downsample overrides alongside the symbolic flow.
- Added `analysis/pytorch_train.py`, a CPU-friendly training harness with warmup+inverse-sqrt schedule, AdamW, checkpoint stubs, CLI overrides, `progress.txt` logging, and optional holdout-eval hooks every _N_ steps (symbolic runs out of the box; Omniglot works once TensorFlow is available).

## Dev Environment
- Introduced a project-local virtual environment (`.venv/`) with `torch==2.2.0+cpu` and `numpy==1.26.4`; ignore entry added to `.gitignore`.
- Use `source .venv/bin/activate` before running the sanity script or future PyTorch training scripts.

## Phase 3 – Validation & Sweeps
- Added `analysis/pytorch_regression.py`, a fast regression harness that exercises the symbolic trainer at 100- and 1600-class settings and validates holdout metrics from `progress.txt`.
- Extended `analysis/run_figure3a_torch.py` to drive either symbolic or Omniglot sweeps, emit both final metrics and full evaluation histories (per-step CSV), plot Figure 3a as accuracy-vs-training-step curves (`--plot`), and forward the new `--optimizer {adam,adamw}` switch to the trainer once TensorFlow/Matplotlib are installed.
- Smoke-tested both scripts via the project-local virtualenv; results land under `runs/torch_regression/` and `runs/figure3a_torch/` by default and can be re-pointed for GPU executions.

## Figure 3a PyTorch Runbook (Omniglot)
1. **Environment**
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install --upgrade pip`
   - PyTorch (pick the CUDA wheel for your GPU, e.g. `pip install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118`)
   - Supporting deps: `pip install numpy==1.26.4 ml_collections matplotlib tensorflow==2.18.0 tensorflow-datasets==4.9.4`
2. **Quick regression check (CPU-friendly)**  
   `python analysis/pytorch_regression.py --quiet`
3. **Figure 3a sweep (bursty `p_bursty = 0.9`, Omniglot, ResNet embedder)**  
   ```
   python analysis/run_figure3a_torch.py \
     --class-counts 100 200 400 800 1600 \
     --example-type omniglot \
     --optimizer adam \
     --batch-size 32 \
     --max-steps 50000 \
     --eval-interval 1000 \
     --eval-batches 64 \
     --num-workers 4 \
     --device cuda \
     --plot
   ```
   Outputs (under `runs/figure3a_torch/`):
   - `figure3a_torch_metrics.csv` – final holdout metrics per class count.
   - `figure3a_torch_timeseries.csv` – per-eval-step history.
   - `figure3a_torch.png` – Figure 3a-style holdout accuracy vs training step.

## Next Focus
1. Run the new Figure 3a sweep on GPU hardware to obtain longer-training checkpoints and compare against archived JAX metrics.
2. Extend the evaluation harness to cover the `no_support_zipfian` regime (Figure 3b) once the in-context pathway is validated.
3. Add unit tests for dataset shapes and transformer forward passes to harden the PyTorch port before wider usage.
