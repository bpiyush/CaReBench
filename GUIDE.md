* You need 8 GPUs with a lot of CPU memory. Since the peak memory usage is ~227GB, you need at least 256GB of CPU memory.

```sh
srun --pty --partition ddp-4way --cpus-per-task=24 --mem=256000 --gres=gpu:8 --time=24:00:00  --constraint=quadro_rtx_8000 bash
```

* Run the weight splitting script to save only the LLM weights from CaRe-7B-Stage-1 to CaRe-7B-Stage-1-llm.

```sh
python tasks/split_weights.py -m /work/piyush/pretrained_checkpoints/CaRe-7B-Stage-1
```

* Use smaller batch sizes: `BATCH_SIZE=16, MICRO_BATCH_SIZE=2`.

* Run the training script.

```sh
bash scripts/train_debug.sh
```


#### Next steps

1. Perhaps we can increase the batch sizes.
2. Perhaps we can `dataloader_num_workers` in the `transformers.TrainingArguments` to increase the data loading speed.
3. We can make the data mapping a bit more efficient.
4. Train on the entire NLI dataset.


#### Evaluation

First, we need to load the entire VLM and then load LLM weights from the fine-tuned model.
Then, we need to save the entire model in a new folder.

```sh
python tasks/merge_weights.py -b /work/piyush/pretrained_checkpoints/CaRe-7B-Stage-1 -f /work/piyush/experiments/CaRe/debug_run
```
