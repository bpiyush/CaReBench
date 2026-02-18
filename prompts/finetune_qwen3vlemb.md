Our goal is to fine-tune Qwen3VLEmbedding model defined in @models/qwen3vl_embedding.py
similar to how Tarsier-7B is finetuned in @tasks/finetuning.py.

The task is contrastive fine-tuning only the LLM part of the model on text triplets. Each
triplet has an anchor, a positive and a hard negative sentence. The usual batch contrastive loss is used.

In Tarsier, the LLM weights are separated, only loaded LLM weights and only those are saved.
For Qwen3VLEmbedding model, I again want to fine-tune only LLM weights, but load and save
the entire model. 

Ask any clarification questions if something is not clear. Write a script @tasks/finetuning_qwen3vlemb.py to do this.