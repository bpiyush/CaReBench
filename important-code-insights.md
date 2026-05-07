# Important Code Insights — Tarsier2 VL Embedding Fine-tuning

Captured while building `tasks/finetuning_tarsier2_vlemb.py` and `scripts/train_tarsier2_vlemb.sh`. These notes describe non-obvious code paths and their pitfalls — read this **before** debugging memory, DeepSpeed, or distributed errors in this pipeline.

---

## 1. `attn_implementation` does **not** propagate to `text_config` automatically

### Symptom
Loading a Tarsier2 *base* checkpoint trains fine; loading a *fine-tuned* checkpoint (e.g. TARA, milestones) **OOMs on the first backward pass** even with the same micro-batch and identical parameter shapes.

Tell-tale memory shape:
- Base: pre-train ≈ 23 GB / GPU, first-batch peak ≈ 44 GB.
- Fine-tuned: pre-train ≈ 23 GB / GPU, **forward completes** (`pos_sim/neg_sim` print appears), but **`loss.backward()` blows past 80 GB**.

### Root cause
Tarsier2's wrapper builds the LLM directly from the sub-config:

```python
# models/tarsier2/modeling_tarsier2.py
self.language_model = Qwen2ForCausalLM(config.text_config)   # or Qwen2VLForCausalLM
```

`Qwen2ForCausalLM` reads `_attn_implementation` from **its own config (`text_config`)**, not from the outer `LlavaConfig`. In transformers 4.45, the `attn_implementation="flash_attention_2"` kwarg passed to the **outer** `from_pretrained` is set on the outer `LlavaConfig` and is **not propagated down** to `text_config`.

The base checkpoint's `config.json` happens to have `text_config.attn_implementation = "flash_attention_2"` baked in. Fine-tuned checkpoints saved later (TARA, milestones) drop that key entirely → Qwen2 falls back to **eager attention** → O(S²) attention matrices → OOM in backward (gradient checkpointing recomputes the eager attention buffer once more).

For S ≈ 2000, B = 4, H = 28: ~900 MB **per layer × 28 layers ≈ 25 GB** of pure eager-attention scratch in forward, doubled in backward.

### Fix
`models/modeling_basemodels.py::BaseModelForTarsier2.__init__`, before `MODEL_CLASS.from_pretrained`, force-propagates the resolved `attn_implementation` onto **every sub-config**:

```python
if model_config is not None:
    for _sub_name in ("text_config", "vision_config"):
        _sub = getattr(model_config, _sub_name, None)
        if _sub is None:
            continue
        try:
            _sub._attn_implementation = attn_implementation
        except Exception:
            pass
        try:
            _sub.attn_implementation = attn_implementation
        except Exception:
            pass
```

Both attribute names are written because, depending on transformers version, instantiation reads the private `_attn_implementation` while `from_pretrained` reads the public `attn_implementation` property.

A verification print is added after load:

```
[BaseModelForTarsier2] LLM attn_implementation = flash_attention_2
```

If this prints anything other than `flash_attention_2` for a given checkpoint, the LLM is silently using eager and the next backward pass will likely OOM on long video inputs.

---

## 2. DeepSpeed ZeRO-2 + grad checkpointing + multiple grad-enabled forwards = "already reduced"

### Symptom
```
AssertionError: The parameter <id> has already been reduced.
Gradient computed twice for this partition.
Multiple gradient reduction is currently not supported
```

### Root cause
ZeRO-2 reduces gradients per parameter as soon as the parameter's last grad-producing op is unwound. With gradient checkpointing, **each `CheckpointFunction` triggers an inner backward** during recompute. Doing two grad-enabled forward passes through the same shared LLM weights in a single training step (e.g. one for video, one for text) causes the same parameter to receive two `AccumulateGrad` events with two separate `reduce` calls → assertion.

### Mitigations applied (all three together are needed for stability)
1. **`use_reentrant=False`** when enabling gradient checkpointing:
   ```python
   model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
   ```
2. **`ddp_find_unused_parameters=True`** in `TrainingArguments`. Frozen `vision_tower` / `multi_modal_projector` parameters appear in the forward graph but produce no gradients; without this flag ZeRO-1/2 raises "already reduced".
3. **Alternate which modality gets gradients** each step. In `Tarsier2VLContrastiveTrainer.compute_loss`:
   ```python
   if self._modal_step % 2 == 0:
       out_v = self._forward_pair(model, video_cat)         # grad
       with torch.no_grad():
           out_t = self._forward_pair(model, text_cat)      # no grad
   else:
       with torch.no_grad():
           out_v = self._forward_pair(model, video_cat)     # no grad
       out_t = self._forward_pair(model, text_cat)          # grad
   self._modal_step += 1
   ```
   Only **one** grad-enabled forward exists per backward, so each shared parameter sees a single `AccumulateGrad` event per step. Concatenating `video0 + video1` into a single batched forward (and likewise for text) means the modality is encoded once per step, not twice.

A standalone DeepSpeed config with `overlap_comm: false` (`ds.config.tarsier2_vlemb.json`) is also used to remove a related ZeRO-2 communication race.

---

## 3. Do **not** call `dist.init_process_group` when launched via `deepspeed`

### Symptom
```
torch.distributed.DistBackendError: [N] is setting up NCCL communicator ...
... store->get('0') got error: failed to recv, got 0 bytes
```

### Root cause
The `deepspeed` launcher (and HF `Trainer` when handed `deepspeed=...`) initialize the default process group themselves. A second `dist.init_process_group("nccl")` corrupts the c10d store; downstream collectives (e.g. `all_gather` in `compute_loss`) then fail with NCCL recv errors.

### Rule
```python
using_deepspeed = bool(deepspeed) and str(deepspeed).strip() not in ("", "none", "None")

if torch.cuda.is_available():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)   # always pin device

if ddp and not using_deepspeed and not dist.is_initialized():
    dist.init_process_group("nccl")     # only when NOT using deepspeed
```

`torch.cuda.set_device(local_rank)` must still happen unconditionally **before** any model load — otherwise rank N may try to allocate on cuda:0.

---

## 4. `TarsierProcessor.max_seq_len` vs `processor_config.json`

`processor_config.json` differs across checkpoints (8192 in base, 16384 in TARA / milestones), but at runtime `init_processor` (in `models/tarsier2/dataset/tarsier_datamodule.py`) **overrides** it with `config.max_seq_len` from the YAML (`models/tarsier2/default_config.yaml` or `nframes=4.yaml`, currently `16384`):

```python
# tarsier_datamodule.py:61
self.processor.max_seq_len = (
    self.tokenizer.model_max_length if max_seq_len is None else max_seq_len
)
```

So the saved `processor_config.max_seq_len` is **inert** in this pipeline. Truncation in `tarsier_processor.py:114-118` always uses the YAML-configured 16384.

Same applies to `image_seq_length` in the saved `config.json` — Tarsier2's modeling code does not consume it; it's a Llava-style cosmetic field. Do not chase it as a memory cause.

---

## 5. EOL prompting must keep `(v, t)` forwards independent

For contrastive embeddings the video embedding and the text embedding must come from **separate** forward passes with the right EOL prompt:

- **Video-only**: `format_one_sample(media_file=video_path, prompt=EOL_PROMPTS["video"])` (template contains `<video>`).
- **Text-only**: `format_one_sample(media_file=None, prompt=EOL_PROMPTS["text"].replace("<sent>", caption))` (template contains `<sent>`, no media).

`Tarsier2ChiralPairDataset.__getitem__` returns four streams per row: `video0`, `text0`, `video1`, `text1`. The collator pads each stream independently. The 2×2 contrastive loss then builds `[sim(v0,t0), sim(v0,t1); sim(v1,t0), sim(v1,t1)]` — the diagonal is the positive pair.

Mixing video and text into one joint `(v, t)` forward (Tarsier2's standard captioning input) is **wrong** for embedding learning: the LLM produces a single conditional representation, not separable v and t embeddings.

---

## 6. Pool **the rightmost** token under left-padding, not `attention_mask.sum() - 1`

### Symptom
LR has **no effect on the loss curve**. Trying LR=1e-7, 1e-6, 1e-5 produces qualitatively identical training trajectories. `train/sim_margin` (pos − neg) jitters around 0 and does not grow during overfit on 4 rows × 512 repeat.

### Root cause
`DataCollatorForTarsier2Pairs._pad_1d` left-pads (`torch.cat([pad, v])`), so the EOL anchor (`...<|im_start|>assistant\n`) is at sequence position **`S - 1`**. The original `_pool_last_token` did:

```python
last_idx = attention_mask.long().sum(dim=1) - 1   # = V - 1
```

For a left-padded mask `[0]*P + [1]*V`, `V - 1` lands inside the **content region but at its *leftmost* edge**, not its rightmost. Different rows in the batch have different `V`, so the pool also lands at a **different position per row**. The four pooled embeddings (`zv0, zt0, zv1, zt1`) compared in the 2×2 contrastive loss are then not semantically comparable. With `max_grad_norm=1.0` rescaling whatever direction emerges, each step's update becomes ≈ `LR × unit_random_direction`, so the trajectory is LR-insensitive.

### Fix
Find the rightmost mask=1 index, robust to either padding side:

```python
seq_len = attention_mask.size(1)
rev_first_one = attention_mask.flip(dims=(-1,)).long().argmax(dim=-1)
last_idx = (seq_len - 1 - rev_first_one).clamp(min=0)
row_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
return hidden_states[row_idx, last_idx]
```

`argmax` on a 0/1 tensor returns the first occurrence of the max (i.e. the first `1`); flipping first means we get the **last** `1` in the original.

### Sanity checks after applying the fix
- `train/sim_margin` should grow toward `1/τ = 20` during overfit (i.e. `pos_sim → 20`, `neg_sim → -20` for the chiral pair).
- LR sweeps should now produce distinct curves: smaller LR → slower descent.
- `loss` should drop toward 0 within tens of steps when overfitting on a handful of rows.

---

## 7. Quick checks before re-running training

When you switch checkpoints or environments, the cheap sanity checks are:

1. Look for `[BaseModelForTarsier2] LLM attn_implementation = flash_attention_2` in rank-0 stdout.
2. Confirm `Trainable params: ...` shows only LLM parameters (`vision_tower`, `multi_modal_projector` should be excluded by `_freeze_non_llm`).
3. Watch for the first `{'loss': ..., 'grad_norm': ...}` line — its appearance means the **backward** completed (forward alone is not enough).
4. `pos_sim` / `neg_sim` magnitudes vary across checkpoints (a fine-tuned LLM has more aligned embeddings → larger pre-temperature cosines). This is **signal**, not a bug.
