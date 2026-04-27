"""
Refined analysis: what % of NLI hard negatives are adverb-based contradictions?

Key improvement: filter for structurally similar sentence pairs first,
then check if the modification is adverb-based.

Criteria for "adverb-based contradiction":
  1. Sentences share significant structure (high token overlap ratio).
  2. The differing tokens are primarily adverbs / negation words.

We also do a separate "negation insertion" analysis (e.g., adding "not"/"never")
which is the clearest form of adverb-based contradiction.
"""

import pandas as pd
import nltk
from nltk import word_tokenize, pos_tag
from difflib import SequenceMatcher
from collections import Counter
import json

ADVERB_TAGS = {"RB", "RBR", "RBS"}

NEGATION_WORDS = {
    "not", "n't", "never", "no", "neither", "nor", "barely", "hardly",
    "scarcely", "seldom", "rarely", "nowhere",
}


def token_overlap_ratio(tok1, tok2):
    """Jaccard-style overlap between two token lists."""
    if not tok1 and not tok2:
        return 1.0
    s1, s2 = set(tok1), set(tok2)
    if not s1 and not s2:
        return 1.0
    return len(s1 & s2) / len(s1 | s2)


def sequence_similarity(tok1, tok2):
    """SequenceMatcher ratio (order-aware)."""
    return SequenceMatcher(None, tok1, tok2).ratio()


def get_diff_tokens(seq1, seq2):
    """Return (removed, added) token lists."""
    sm = SequenceMatcher(None, seq1, seq2)
    removed, added = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "replace":
            removed.extend(seq1[i1:i2])
            added.extend(seq2[j1:j2])
        elif op == "delete":
            removed.extend(seq1[i1:i2])
        elif op == "insert":
            added.extend(seq2[j1:j2])
    return removed, added


def analyze_pair(text_a, text_b):
    """Analyze a pair of sentences for adverb-based contradiction."""
    tok_a = word_tokenize(text_a.lower().strip())
    tok_b = word_tokenize(text_b.lower().strip())

    overlap = token_overlap_ratio(tok_a, tok_b)
    sim = sequence_similarity(tok_a, tok_b)

    removed, added = get_diff_tokens(tok_a, tok_b)
    all_diff = removed + added
    n_diff = len(all_diff)

    if n_diff == 0:
        return {
            "overlap": overlap, "sim": sim, "n_diff": 0,
            "n_adv": 0, "n_neg": 0,
            "adv_frac": 0, "neg_frac": 0,
            "is_negation_only": False,
            "is_adverb_only": False,
            "is_adverb_dominant": False,
            "removed": removed, "added": added,
            "diff_tagged": [],
        }

    all_tagged = pos_tag(all_diff)
    n_adv = sum(1 for _, t in all_tagged if t in ADVERB_TAGS)
    n_neg = sum(1 for w, t in all_tagged if w.lower() in NEGATION_WORDS)

    adv_frac = n_adv / n_diff if n_diff else 0
    neg_frac = n_neg / n_diff if n_diff else 0

    is_negation_only = (n_diff == n_neg) and n_neg > 0
    is_adverb_only = (n_diff == n_adv) and n_adv > 0
    is_adverb_dominant = n_adv > 0 and adv_frac >= 0.5

    return {
        "overlap": overlap, "sim": sim, "n_diff": n_diff,
        "n_adv": n_adv, "n_neg": n_neg,
        "adv_frac": adv_frac, "neg_frac": neg_frac,
        "is_negation_only": is_negation_only,
        "is_adverb_only": is_adverb_only,
        "is_adverb_dominant": is_adverb_dominant,
        "removed": removed, "added": added,
        "diff_tagged": all_tagged,
    }


def main():
    csv_path = "/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/covr/chiral10k-covr10k.csv"
    df = pd.read_csv(csv_path)
    nli = df[df["source"] == "nli"].reset_index(drop=True)
    total = len(nli)
    print(f"Total NLI rows: {total}\n")

    # Analyze both sent1↔hard_neg and sent0↔hard_neg
    results_s1 = []  # sent1 vs hard_neg
    results_s0 = []  # sent0 vs hard_neg

    for i, row in nli.iterrows():
        r1 = analyze_pair(str(row["sent1"]), str(row["hard_neg"]))
        r0 = analyze_pair(str(row["sent0"]), str(row["hard_neg"]))
        r1["sent1"] = str(row["sent1"])
        r1["hard_neg"] = str(row["hard_neg"])
        r1["sent0"] = str(row["sent0"])
        r0["sent0"] = str(row["sent0"])
        r0["hard_neg"] = str(row["hard_neg"])
        results_s1.append(r1)
        results_s0.append(r0)
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{total}...")

    df1 = pd.DataFrame(results_s1)
    df0 = pd.DataFrame(results_s0)

    # =====================================================================
    # REPORT
    # =====================================================================
    print("\n" + "=" * 75)
    print("REPORT: Adverb-based contradictions in NLI hard negatives")
    print("=" * 75)

    # 1. Similarity distribution
    print("\n[1] Structural similarity between sent1 and hard_neg")
    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        n = (df1["sim"] >= thresh).sum()
        print(f"  seq_similarity >= {thresh}: {n:5d} / {total} ({100*n/total:.1f}%)")

    # 2. Among ALL rows
    print(f"\n[2] Adverb analysis over ALL {total} NLI rows (sent1 ↔ hard_neg)")
    for label, col in [
        ("Diff is ONLY negation words", "is_negation_only"),
        ("Diff is ONLY adverbs", "is_adverb_only"),
        ("Adverbs are >= 50% of diff", "is_adverb_dominant"),
    ]:
        n = df1[col].sum()
        print(f"  {label:40s}: {n:5d} / {total} ({100*n/total:.2f}%)")

    # 3. Among structurally similar rows (sim >= 0.5)
    sim_thresh = 0.5
    similar = df1[df1["sim"] >= sim_thresh]
    n_similar = len(similar)
    print(f"\n[3] Adverb analysis among SIMILAR rows (sim >= {sim_thresh}): {n_similar} rows")
    for label, col in [
        ("Diff is ONLY negation words", "is_negation_only"),
        ("Diff is ONLY adverbs", "is_adverb_only"),
        ("Adverbs are >= 50% of diff", "is_adverb_dominant"),
    ]:
        n = similar[col].sum()
        pct_of_similar = 100 * n / n_similar if n_similar else 0
        pct_of_total = 100 * n / total
        print(f"  {label:40s}: {n:5d} / {n_similar} ({pct_of_similar:.2f}% of similar, {pct_of_total:.2f}% of total)")

    # 4. "True adverb-based" = similar AND adverb-dominant
    true_adv = similar[similar["is_adverb_dominant"]]
    print(f"\n[4] TRUE adverb-based contradictions (similar + adverb-dominant)")
    print(f"  Count: {len(true_adv)} / {total} ({100*len(true_adv)/total:.2f}%)")
    if len(true_adv) > 0:
        # Sub-classify
        true_neg = true_adv[true_adv["n_neg"] > 0]
        true_non_neg = true_adv[true_adv["n_neg"] == 0]
        print(f"  ... involving negation:     {len(true_neg)} ({100*len(true_neg)/total:.2f}%)")
        print(f"  ... non-negation adverbs:   {len(true_non_neg)} ({100*len(true_non_neg)/total:.2f}%)")

    # 5. Broader definition: sim >= 0.3 and at least one adverb in diff
    sim_broad = 0.3
    broad = df1[(df1["sim"] >= sim_broad) & (df1["n_adv"] > 0)]
    print(f"\n[5] Broader: sim >= {sim_broad} AND at least 1 adverb in diff")
    print(f"  Count: {len(broad)} / {total} ({100*len(broad)/total:.2f}%)")

    # Among those, how many have adverb as a significant portion?
    broad_dom = broad[broad["adv_frac"] >= 0.33]
    print(f"  ... with adverbs >= 33% of diff: {len(broad_dom)} / {total} ({100*len(broad_dom)/total:.2f}%)")

    # 6. Examples
    print("\n" + "=" * 75)
    print("EXAMPLES of true adverb-based contradictions")
    print("=" * 75)

    if len(true_adv) > 0:
        print("\n--- Negation-based (most clear-cut) ---")
        neg_examples = true_adv[true_adv["n_neg"] > 0].sort_values("sim", ascending=False)
        for _, r in neg_examples.head(10).iterrows():
            print(f"  sent1:    {r['sent1']}")
            print(f"  hard_neg: {r['hard_neg']}")
            tagged_str = [(w, t) for w, t in r["diff_tagged"] if t in ADVERB_TAGS]
            print(f"  adverbs in diff: {tagged_str}")
            print(f"  sim={r['sim']:.2f}  diff_size={r['n_diff']}  adv_frac={r['adv_frac']:.2f}")
            print()

        print("\n--- Non-negation adverb swaps ---")
        non_neg = true_adv[true_adv["n_neg"] == 0].sort_values("sim", ascending=False)
        for _, r in non_neg.head(10).iterrows():
            print(f"  sent1:    {r['sent1']}")
            print(f"  hard_neg: {r['hard_neg']}")
            tagged_str = [(w, t) for w, t in r["diff_tagged"] if t in ADVERB_TAGS]
            print(f"  adverbs in diff: {tagged_str}")
            print(f"  sim={r['sim']:.2f}  diff_size={r['n_diff']}  adv_frac={r['adv_frac']:.2f}")
            print()

    # 7. Same analysis on sent0 ↔ hard_neg
    print("\n" + "=" * 75)
    print("CROSS-CHECK: sent0 ↔ hard_neg")
    print("=" * 75)
    similar0 = df0[df0["sim"] >= sim_thresh]
    n_similar0 = len(similar0)
    print(f"  Rows with sim >= {sim_thresh}: {n_similar0}")
    for label, col in [
        ("Diff is ONLY negation words", "is_negation_only"),
        ("Diff is ONLY adverbs", "is_adverb_only"),
        ("Adverbs are >= 50% of diff", "is_adverb_dominant"),
    ]:
        n = similar0[col].sum()
        pct = 100 * n / n_similar0 if n_similar0 else 0
        print(f"  {label:40s}: {n:5d} / {n_similar0} ({pct:.2f}% of similar)")

    # 8. Summary table
    print("\n" + "=" * 75)
    print("SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Metric':<55s} {'Count':>6s} {'% of 9000':>10s}")
    print("-" * 75)
    rows = [
        ("All NLI rows", total),
        ("sent1↔hard_neg: diff contains any adverb", int(df1["n_adv"].gt(0).sum())),
        ("sent1↔hard_neg: diff is ONLY adverbs", int(df1["is_adverb_only"].sum())),
        ("sent1↔hard_neg: adverbs >= 50% of diff", int(df1["is_adverb_dominant"].sum())),
        ("Similar (sim>=0.5) rows", n_similar),
        ("Similar + adverb-dominant", len(true_adv)),
        ("Similar + adverb-dominant + has negation", len(true_adv[true_adv["n_neg"] > 0]) if len(true_adv) > 0 else 0),
        ("Similar + adverb-dominant + no negation", len(true_adv[true_adv["n_neg"] == 0]) if len(true_adv) > 0 else 0),
        ("Broad (sim>=0.3) + any adverb in diff", len(broad)),
        ("Broad (sim>=0.3) + adverbs >= 33% of diff", len(broad_dom)),
    ]
    for label, count in rows:
        pct = 100 * count / total
        print(f"  {label:<53s} {count:>6d} {pct:>9.2f}%")


if __name__ == "__main__":
    main()
