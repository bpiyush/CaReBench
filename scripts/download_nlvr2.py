"""
Download lmms-lab/NLVR2 dataset from Hugging Face and save raw images + metadata.

Dataset has splits: balanced_dev, balanced_test_public, balanced_test_unseen,
unbalanced_dev, unbalanced_test_unseen, unbalanced_test_public.
Each sample has left_image, right_image, question, answer, and other fields.
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "/scratch/shared/beegfs/piyush/datasets/CoIR/NLVR2-lmms"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")

SPLITS = [
    "balanced_dev",
    "balanced_test_public",
    "balanced_test_unseen",
    "unbalanced_dev",
    "unbalanced_test_unseen",
    "unbalanced_test_public",
]

print("Loading dataset from Hugging Face (lmms-lab/NLVR2)...")

all_metadata = {}

for split in SPLITS:
    print(f"\n--- Processing split: {split} ---")
    ds = load_dataset("lmms-lab/NLVR2", split=split)
    print(f"Loaded {len(ds)} samples.")

    split_images_dir = os.path.join(IMAGES_DIR, split)
    os.makedirs(split_images_dir, exist_ok=True)

    metadata = []

    for idx, sample in enumerate(tqdm(ds, desc=f"Processing {split}")):
        sample_id = sample.get("question_id") or sample.get("identifier") or f"{split}_{idx}"
        sample_dir = os.path.join(split_images_dir, sample_id)
        os.makedirs(sample_dir, exist_ok=True)

        image_paths = []
        for img_key in ["left_image", "right_image"]:
            img = sample.get(img_key)
            if img is not None:
                img_filename = f"{img_key}.jpg"
                img_path = os.path.join(sample_dir, img_filename)
                if not os.path.exists(img_path):
                    img.save(img_path)
                image_paths.append(os.path.join("images", split, sample_id, img_filename))

        entry = {
            "question_id": sample.get("question_id"),
            "identifier": sample.get("identifier"),
            "question": sample.get("question"),
            "answer": sample.get("answer"),
            "writer": sample.get("writer"),
            "synset": sample.get("synset"),
            "query": sample.get("query"),
            "extra_validations": sample.get("extra_validations"),
            "left_url": sample.get("left_url"),
            "right_url": sample.get("right_url"),
            "image_paths": image_paths,
        }
        metadata.append(entry)

    all_metadata[split] = metadata

    split_metadata_path = os.path.join(OUTPUT_DIR, f"metadata_{split}.json")
    with open(split_metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved {len(metadata)} samples for split '{split}'.")

combined_path = os.path.join(OUTPUT_DIR, "metadata_all.json")
with open(combined_path, "w") as f:
    json.dump(all_metadata, f, indent=2)

total = sum(len(v) for v in all_metadata.values())
print(f"\nDone! Saved {total} total samples across {len(SPLITS)} splits.")
print(f"Images directory: {IMAGES_DIR}")
print(f"Metadata directory: {OUTPUT_DIR}")
