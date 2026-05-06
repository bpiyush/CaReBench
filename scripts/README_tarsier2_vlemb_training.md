# Tarsier2 Video-Text Contrastive Training Runbook

This runbook is for the additive-only pipeline:
- Trainer: `tasks/finetuning_tarsier2_vlemb.py`
- Launcher: `scripts/train_tarsier2_vlemb.sh`

## Staged Execution

1. Overfit on tiny rows
```bash
bash scripts/train_tarsier2_vlemb.sh overfit
```
Expected:
- Train loss should drop quickly.
- `train/sim_margin` should trend upward.
- No `checkpoint-*` directories during training.

2. Many-batch smoke run
```bash
bash scripts/train_tarsier2_vlemb.sh smoke
```
Expected:
- Stable loss curve, no NaNs/inf.
- No OOM with 8 GPUs at tiny micro-batch.
- W&B logs show `train/loss`, `train/pos_sim`, `train/neg_sim`, `train/sim_margin`.

3. Full-scale run
```bash
bash scripts/train_tarsier2_vlemb.sh full
```
Expected:
- Stable long-run training.
- End-of-training single save in output directory.

## W&B Setup

Set these env vars as needed:
```bash
export WANDB_PROJECT=CaReBench-Tarsier2-VLEmb
export WANDB_ENTITY=<your_entity>
```

## Notes

- The new trainer freezes non-LLM modules and only updates `language_model`.
- Save policy is `save_strategy="no"` with one final explicit save at completion.
- Existing text-only and legacy pipelines remain untouched.
