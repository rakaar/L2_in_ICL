# L2-Regularised Figure 3 Replication – Onboarding Guide

This note is for anyone who wants to rerun the “Figure 3” in-context learning experiment – with the additional AdamW/L2 sweep – on a fresh GPU machine. It captures the research hypothesis, the execution plan, required scripts, and where to inspect outputs.

---

## Research Hypothesis
- Original claim (Chan et al., 2022, Fig. 3): large numbers of training classes are necessary for transformers to learn in-context; otherwise they memorise in-weights.
- New hypothesis: explicit L2 regularisation (via AdamW weight decay) can suppress memorisation and preserve in-context learning even with a *small* class vocabulary.

We therefore sweep AdamW weight decay λ over `{0, 3e-4, 1e-3, 3e-3, 1e-2}` on two training regimes:
- 100 total classes (≈ “small vocabulary”)
- 1600 total classes (baseline “large vocabulary”)

Metrics: few-shot accuracy on holdout classes (ICL) and no-support accuracy on trained classes (in-weights memorisation), as in Figure 3a/b.

---

## High-Level Execution Plan
1. **Environment setup**: clone repo, create Python environment, install deps (JAX/Haiku/Optax, TFDS, matplotlib).
2. **Run sweeps**: use `analysis/run_adamw_sweep.py` to sequentially train all λ×class-count combos, capturing logs and checkpoints.
3. **Aggregate + plot**: the sweep script automatically calls `analysis/plot_figure3_adamw.py` to regenerate Figure 3-style panels and a CSV of raw metrics.
4. **Inspect outputs**: monitor training progress via `logs/progress.txt`, review `train.log`, validate checkpoint directories, and collect final figures/metrics.

---

## Prerequisites
| Requirement | Details |
|-------------|---------|
| GPU drivers | CUDA 11.x + matching cuDNN (align with your chosen JAX wheel). |
| Python      | 3.8–3.10 recommended. |
| tmux/screen | For long sweeps, run the driver script inside tmux. |

---

## Environment Setup
```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/L2_in_ICL.git
cd L2_in_ICL

# 2. Create & activate virtualenv (or use conda)
python3 -m venv venv
source venv/bin/activate

# 3. Install requirements (installs JAXline, Haiku, Optax, TF, matplotlib, etc.)
pip install --upgrade pip
pip install -r requirements.txt

# 4. (Optional) install GPU-enabled JAX wheel that matches your CUDA version
# Example for CUDA 11.8 - adjust if needed:
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

> **Check GPU**: `python -c "import jax; print(jax.devices())"` should list at least one GPU. If not, revisit the JAX wheel installation.

---

## Running the AdamW + L2 Sweep
Launch the scheduler script (recommended inside tmux):
```bash
tmux new -s icl_sweep
python analysis/run_adamw_sweep.py \
  --base_output_dir ./runs/adamw_sweep \
  --training_steps 50000 \
  --skip_completed
```

Key flags:
- `--class_counts` *(default: 100 1600)* – override if testing additional vocab sizes.
- `--weight_decays` *(default: 0 0.0003 0.001 0.003 0.01)* – customise λ list.
- `--training_steps` – shorten from the config default (5e5) for quicker experiments.
- `--max_eval_batches` – limit evaluation batches per checkpoint when plotting.
- `--config_path` – point to a different config if you diverge from `images_all_exemplars`.

The script executes each combination sequentially (total runs = |class_counts| × |weight_decays|). Logs stream live to the terminal (tmux pane), so you can detach/reattach. On failure, the script aborts unless you pass `--skip_completed` to resume later.

---

## Output Layout & Monitoring
Within `--base_output_dir` (default `runs/adamw_sweep/`):

```
runs/adamw_sweep/
  class100_wd0/
    checkpoints/           # Jaxline checkpoints (models/latest/step_*/checkpoint.dill)
    logs/
      train.log            # Full stdout/stderr from training
      progress.txt         # Filtered stream (loss + weight norm metrics) for quick monitoring
  class100_wd3e-4/
    ...
  class1600_wd0/
    ...
  runs.csv                 # Generated sweep manifest: class_count, weight_decay, checkpoint path
  plot.log                 # Output from plotting script
  figure3_adamw.png        # Figure 3-style plot (PNG)
  figure3_adamw.pdf        # Same plot in PDF format
  figure3_adamw_metrics.csv# Aggregated metrics (accuracy, loss, weight norms per run)
```

**Monitoring tips**
- `tail -f runs/adamw_sweep/class100_wd0/logs/progress.txt` – quick glance at loss/weight norms (`progress.txt` keeps lines with `loss`, `weight_l2`, or `grad_l2`).
- `less +F runs/adamw_sweep/class100_wd0/logs/train.log` – scrollable full log.
- Inside `train.log`, track `weight_l2`, `weight_l2_all`, and `grad_l2` alongside accuracy to see how norm shrinkage aligns with in-context learning gains.
- `ls runs/adamw_sweep/class100_wd0/checkpoints/models/latest/` – verify checkpoint steps.
- `nvidia-smi` – keep an eye on GPU utilisation/VRAM.

---

## Re-plotting / Fine-Grained Analysis
To regenerate the figure or compute additional stats later:
```bash
python analysis/plot_figure3_adamw.py \
  --runs_csv runs/adamw_sweep/runs.csv \
  --output_dir runs/adamw_sweep \
  --config_module experiment.configs.images_all_exemplars \
  --max_eval_batches 1000  # optional
```
This re-evaluates checkpoints (respecting the overrides recorded in `runs.csv`) and rewrites the PNG/PDF/CSV in-place.

---

## Extending / Debugging
- **Different class splits**: adjust `n_common_classes` / `n_holdout_classes` flags; the script auto-computes rare classes from the total.
- **Custom configs**: supply `--config_path` pointing to your forked config file under `experiment/configs/`.
- **Dry-run single experiment**: invoke the training module directly:
  ```bash
  python -m emergent_in_context_learning.experiment.experiment \
    --config experiment/configs/images_all_exemplars.py \
    --config.optimizer.name=adamw \
    --config.optimizer.kwargs.weight_decay=0.001 \
    --config.data.generator_config.n_rare_classes=980 \
    --config.data.generator_config.n_common_classes=10 \
    --config.data.generator_config.n_holdout_classes=10 \
    --config.checkpoint_dir /tmp/icl_debug \
    --jaxline_mode train --logtostderr
  ```
- **Resume**: rerun the sweep with `--skip_completed` and it will continue from missing checkpoints.

---

## Wrap-Up Checklist
1. ✅ Environment installed, GPU recognised.
2. ✅ Sweep script executed in tmux; progress monitored via `progress.txt`.
3. ✅ Checkpoints saved under each `class*` directory.
4. ✅ `figure3_adamw.(png|pdf)` + `figure3_adamw_metrics.csv` created.
5. ✅ Optional re-run of plotting script for verification.

Once these boxes are ticked, you have a reproducible AdamW/L2 variant of Figure 3 ready for analysis or comparison against the original claim. Happy experimenting!
