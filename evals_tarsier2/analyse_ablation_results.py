import os
import json
import pandas as pd

BASE = "/work/piyush/experiments/CaRe/Tarsier2-7b-0115/ablations"
CSV_PATH = "notebooks/ablation_mixture.csv"

df = pd.read_csv(CSV_PATH)


def subfolder_name(row):
    return f"core.{row.n_core}-time.{row.n_time}-negs.{row.n_negation}-covr.{row.n_covr}"


def load_metrics(row):
    sub = subfolder_name(row)
    pattern = f"{BASE}/{sub}/merged_checkpoint/metrics/metrics_{sub}_nuanced_retrieval_data-validation-v1.json"
    if not os.path.exists(pattern):
        return None
    with open(pattern) as f:
        return json.load(f)


time_scores, neg_scores, mm_scores, avg_scores = [], [], [], []

for _, row in df.iterrows():
    m = load_metrics(row)
    if m is None:
        time_scores.append(None)
        neg_scores.append(None)
        mm_scores.append(None)
        avg_scores.append(None)
        continue

    # time: t2v R@1 for chiral subset of SSv2
    time_val = m["time_t2v"]["chiral"]["R@1"]

    # negation: avg of R@5 for standard and negation
    neg_std = m["negation_msrvtt"]["standard"]["standard"]["R@5"]
    neg_neg = m["negation_msrvtt"]["negation"]["negation"]["R@5"]
    neg_val = (neg_std + neg_neg) / 2

    # multimodal: covr R@1
    mm_val = m["multimodal_covr"]["covr"]["R@1"]

    # overall avg
    avg_val = (time_val + neg_val + mm_val) / 3

    time_scores.append(round(time_val, 2))
    neg_scores.append(round(neg_val, 2))
    mm_scores.append(round(mm_val, 2))
    avg_scores.append(round(avg_val, 2))

df["time"] = time_scores
df["negation"] = neg_scores
df["multimodal"] = mm_scores
df["avg"] = avg_scores

df.to_csv(CSV_PATH, index=False)
print(df.to_string(index=False))
print(f"\nSaved to {CSV_PATH}")
