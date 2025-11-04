# Figure 3 Experiment Reference

This note summarises how Figure&nbsp;3 (“Effects of number of classes”) in *Data Distributional Properties Drive Emergent In-Context Learning in Transformers* is produced inside the repository.

## What Figure 3 Shows
- Panel (a) reports `accuracy_query` on `eval_fewshot_holdout` sequences, measuring in-context learning on Omniglot holdout classes after training on bursty sequences.
- Panel (b) reports the same metric on `eval_no_support_zipfian` sequences, measuring in-weights memorisation on the training classes when no support examples are available.
- The figure sweeps the number of training classes (≈100 → 1,600, plus the 12,800-class augmentation) while keeping burstiness fixed at `p_bursty = 0.9`.

## Where the Sweeps Are Configured
- Default settings for the baseline points (full Omniglot, no augmentation) live in `experiment/configs/images_all_exemplars.py` (`n_rare_classes=1603`, `n_common_classes=10`, `n_holdout_classes=10`).
- The augmented 12,800-class condition comes from `experiment/configs/images_augmented.py`, which turns on `augment_images=True` and sets `preserve_ordering_every_n=8` to keep transformed variants grouped.
- Sweeping the “number of classes” in Figure 3 amounts to editing the `generator_config` values above and rerunning training/evaluation.

## Data Generation Pipeline
- `datasets/data_generators.py` loads Omniglot (`OmniglotDatasetForSampling`) with the chosen exemplar/augmentation strategy, and the `SeqGenerator` class slices the shuffled class order into rare/common/holdout partitions.
- `SeqGenerator.get_bursty_seq(...)` emits the sequences used during training, embedding `k`-shot `n`-way problems into length-9 bursty contexts with `p_bursty=0.9`.

## Training and Metrics
- `experiment/experiment.py` instantiates the embedder (`InputEmbedder`) and the Transformer defined in `modules/transformer.py`, setting `model_config.num_classes` from the generator split to match the sweep size.
- `_compute_loss_and_accuracy` in the same file computes the `accuracy_query` scalar that Figure 3 plots for both in-context and in-weights evaluations.
- `_get_ds_seqs` maps evaluation modes to generator calls: `eval_fewshot_holdout` provides the few-shot setting for panel (a), and `eval_no_support_zipfian` covers panel (b) by withholding support for the query class.

## Reproducing the Figure
1. Pick (or clone) a config under `experiment/configs/` and set the desired `n_rare_classes`, `n_common_classes`, and `n_holdout_classes`.
2. Train the model:
   ```bash
   python -m emergent_in_context_learning.experiment.experiment \
     --config experiment/configs/your_config.py \
     --jaxline_mode train --logtostderr
   ```
   Save checkpoints with `Ctrl+C` when ready.
3. Evaluate in-context learning:
   ```bash
   python -m emergent_in_context_learning.experiment.experiment \
     --config experiment/configs/your_config.py \
     --config.one_off_evaluate \
     --config.restore_path $CKPT_DIR \
     --jaxline_mode eval_fewshot_holdout --logtostderr
   ```
4. Evaluate in-weights memorisation:
   ```bash
   python -m emergent_in_context_learning.experiment.experiment \
     --config experiment/configs/your_config.py \
     --config.one_off_evaluate \
     --config.restore_path $CKPT_DIR \
     --jaxline_mode eval_no_support_zipfian --logtostderr
   ```
5. Aggregate `accuracy_query` across seeds/sweeps to recreate the two curves in Figure 3.

## Extending Figure 3 with AdamW + L2 Weight Decay
To test the hypothesis that weight decay can substitute for large vocabularies, run the same pipeline with AdamW and different decay factors.

### Config Tweaks
- **Switch to AdamW**: Either edit the config or override flags to set `config.optimizer.name = 'adamw'`.
- **Specify weight decay**: Use `config.optimizer.kwargs.weight_decay` for each lambda value.
- **100 classes setup**: Set `config.data.generator_config.n_rare_classes = 90` (with `n_common_classes = 10`, `n_holdout_classes = 10`) so the transformer trains on 100 total classes.
- **1600 classes setup**: Set `n_rare_classes = 1590` while keeping the same common/holdout counts.

You can keep everything else (burstiness, sequence construction, model depth) identical to the original Figure 3 runs.

For quick sweeps you can override these counts from the command line rather than editing the config files, e.g.
add `--config.data.generator_config.n_rare_classes=90` for the 100-class setting or
`--config.data.generator_config.n_rare_classes=1590` for the 1600-class setting.

### Launch Commands
Example commands for one sweep element (replace `VALUE` with the target weight decay and pick the class-count config):

```bash
# Training
python -m emergent_in_context_learning.experiment.experiment \
  --config experiment/configs/your_config.py \
  --config.optimizer.name=adamw \
  --config.optimizer.kwargs.weight_decay=VALUE \
  --jaxline_mode train --logtostderr

# While training, keep an eye on the streamed metrics:
#   - `loss` / `accuracy_query` for performance
#   - `weight_l2`, `weight_l2_all`, and `grad_l2` for norm dynamics
# They appear in stdout and in any logs captured by the sweep script.

# Evaluate in-context learning (panel a)
python -m emergent_in_context_learning.experiment.experiment \
  --config experiment/configs/your_config.py \
  --config.optimizer.name=adamw \
  --config.optimizer.kwargs.weight_decay=VALUE \
  --config.one_off_evaluate \
  --config.restore_path $CKPT_DIR \
  --jaxline_mode eval_fewshot_holdout --logtostderr

# Evaluate in-weights memorisation (panel b)
python -m emergent_in_context_learning.experiment.experiment \
  --config experiment/configs/your_config.py \
  --config.optimizer.name=adamw \
  --config.optimizer.kwargs.weight_decay=VALUE \
  --config.one_off_evaluate \
  --config.restore_path $CKPT_DIR \
  --jaxline_mode eval_no_support_zipfian --logtostderr
```

Run the commands for each λ in `{0, 3e-4, 1e-3, 3e-3, 1e-2}` and for both class-count configurations. Collate the resulting `accuracy_query` values to build the two curves per Figure 3. While λ=0 reproduces the Adam baseline, ensure you launch it with AdamW so the optimizer configuration stays consistent across the sweep.

### Automating Evaluation & Plotting
Once the checkpoints are produced, the helper script `analysis/plot_figure3_adamw.py` can reload them, rerun the holdout/no-support evaluations, and regenerate the two Figure 3 panels (plus a CSV of raw metrics):

```bash
python analysis/plot_figure3_adamw.py \
  --runs_csv /path/to/runs.csv \
  --output_dir /path/to/output \
  --config_module experiment.configs.images_all_exemplars
```

Prepare a CSV describing your sweep (minimum header: `class_count,weight_decay,checkpoint`). Each row should point to the directory that contains `checkpoint.dill` (or directly to the file):

```csv
class_count,weight_decay,checkpoint
100,0,/path/to/checkpoints/class100/wd0/latest
100,0.0003,/path/to/checkpoints/class100/wd3e-4/latest
1600,0,/path/to/checkpoints/class1600/wd0/latest
1600,0.0003,/path/to/checkpoints/class1600/wd3e-4/latest
...
```

Optional columns `config_module`, `n_common_classes`, and `n_holdout_classes` let you override defaults on a per-row basis if your runs deviated from `experiment/configs/images_all_exemplars.py`. The script writes `figure3_adamw_metrics.csv` together with PNG/PDF plots inside `--output_dir`; the CSV now contains the tracked weight norms (`weight_l2*`) alongside accuracy/loss metrics.

> **Note**: Install `matplotlib` before invoking the plotting utility:
> ```bash
> pip install matplotlib
> ```

### One-Stop Sweep Runner
If you prefer a single entry point that launches training, collects checkpoints, and triggers the plotting pass automatically, use `analysis/run_adamw_sweep.py`. It sequentially runs every `(class_count, λ)` combination, tees the training logs to `train.log`, mirrors lines containing `loss` into `progress.txt`, and finally calls the plotting helper above.

Example (run inside tmux):

```bash
python analysis/run_adamw_sweep.py \
  --base_output_dir ./runs/adamw_sweep \
  --training_steps 50000 \
  --skip_completed
```

Flags of note:
- `--class_counts` / `--weight_decays` override the default `{100, 1600}` × `{0, 3e-4, 1e-3, 3e-3, 1e-2}` sweep.
- `--training_steps` replaces the 5e5-step default from the config (strongly recommended unless you truly want the full schedule).
- `--max_eval_batches` can speed up plotting by limiting evaluation batches per checkpoint.
- `--skip_completed` avoids re-running training when a checkpoint already exists under the designated output tree.

Results live under `--base_output_dir`:
- `class{N}_wd{λ}/logs/train.log` and `progress.txt` capture stdout/stderr (progress is tmux-friendly because the script streams logs live).
- `class{N}_wd{λ}/checkpoints/` houses the saved Jaxline checkpoints.
- `runs.csv`, `figure3_adamw_metrics.csv`, and `figure3_adamw.(png|pdf)` aggregate the sweep.

For a fully guided walkthrough (environment setup, tmux usage, output inspection) see `docs/icl_sweep_onboarding.md`.
