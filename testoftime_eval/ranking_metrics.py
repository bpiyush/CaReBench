import numpy as np
import pandas as pd
from math import comb, factorial
from scipy.stats import kendalltau, spearmanr

def all_permutations(k):
    """
    Return list of permutations (tuples) in the same order you'd get from itertools.permutations.
    Useful to map sims indices -> permutation.
    
    Args:
        k: number of items to permute
    """
    from itertools import permutations
    return list(permutations(range(k)))

def kendall_tau_distance_between_perms(a, b):
    """
    Compute Kendall tau distance (number of discordant pairs) between two permutations a and b.
    a and b should be sequences of the same elements (e.g., 0..K-1) and are permutations.
    Returns integer between 0 and K*(K-1)/2.
    """
    assert len(a) == len(b)
    K = len(a)
    # map element -> position in b
    pos_in_b = {val: i for i, val in enumerate(b)}
    # create array of positions of a's elements in b
    seq = [pos_in_b[val] for val in a]
    # number of inversions in seq is the Kendall distance
    # count inversions (O(K^2) fine for small K; K=4 as in your case)
    inv = 0
    for i in range(K):
        for j in range(i+1, K):
            if seq[i] > seq[j]:
                inv += 1
    return inv

def compute_distance_vector(perms, original_perm):
    """
    perms: list of perms (tuples of indices)
    original_perm: tuple (the original order)
    returns: numpy array of distances (integers)
    """
    dists = np.array([kendall_tau_distance_between_perms(original_perm, p) for p in perms], dtype=int)
    return dists

def dcg_at_k(rels, k):
    """
    rels: list/array of relevance scores (ordered by predicted ranking)
    compute DCG@k using log2 discount (classic)
    """
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, rels.size + 2))  # positions 1.. -> denom log2(2..)
    return np.sum(rels / discounts)

def ndcg_at_k(predicted_indices, relevance_by_index, k):
    """
    predicted_indices: indices in order of predicted ranking (best -> worst)
    relevance_by_index: array-like mapping index -> relevance (higher = better)
    """
    rels_pred = [relevance_by_index[i] for i in predicted_indices[:k]]
    dcg = dcg_at_k(rels_pred, k)
    # ideal DCG: sort relevance descending
    ideal_rels = sorted(relevance_by_index, reverse=True)[:k]
    idcg = dcg_at_k(ideal_rels, k)
    return dcg / idcg if idcg > 0 else 0.0

def reciprocal_rank_of_first_relevant(predicted_indices, relevance_by_index, is_relevant_fn):
    """
    Return reciprocal rank for the first item that satisfies is_relevant_fn(index, relevance).
    If none found, returns 0.
    """
    for rank, idx in enumerate(predicted_indices, start=1):
        if is_relevant_fn(idx, relevance_by_index[idx]):
            return 1.0 / rank
    return 0.0

def compute_ranking_metrics(
    sims_matrix,
    n_captions,
    perms=None,
    topk_for_recall=(1, 3, 5),
    ndcg_k=10,
):
    """
    Compute ranking metrics comparing predicted similarity scores against ground truth ordering.
    
    Args:
        sims_matrix: numpy array shape (N, M) where M == factorial(n_captions)
                     higher score = model thinks this permutation is a better caption ordering.
        n_captions: number of captions (K). Determines the number of permutations M = K!
        perms: optional list of permutations (tuples of indices). If None, generated from n_captions.
        topk_for_recall: iterable of k values to compute Recall@k for the exact original permutation.
        ndcg_k: k used for NDCG@k
        
    Returns: 
        (per_video_df, aggregate_metrics_dict)
        per_video_df has columns:
            ['kendall_tau', 'spearman_rho', 'mrr', 'recall@k'..., 'ndcg@k', 'orig_perm_index', 'best_pred_index', ...]
    """
    sims = np.asarray(sims_matrix)
    if sims.ndim != 2:
        raise ValueError("sims_matrix must be 2D (N x M)")
    N, M = sims.shape
    K = n_captions
    
    if M != factorial(K):
        # allow perms list override
        if perms is None:
            raise ValueError(f"sims has M={M} columns but K={K} gives K!={factorial(K)}; pass perms explicitly.")
    
    # build perms if not given
    if perms is None:
        perms = all_permutations(K)
    
    # original_perm is simply (0,1,2,...) representing original order
    original_perm = tuple(range(K))
    dists = compute_distance_vector(perms, original_perm)  # length M
    max_dist = int(dists.max()) if M > 0 else 0
    # define graded relevance: higher relevance when distance smaller
    # shift so relevance >= 0; relevance = max_dist - dist
    relevance = (max_dist - dists).astype(float)

    rows = []
    for vid in range(N):
        sim_row = sims[vid]
        # predicted ranking: indices sorted by decreasing sim (best first)
        pred_order = np.argsort(-sim_row, kind='stable')
        # ground truth ranking: indices sorted by increasing distance (best = distance 0)
        gt_order = np.argsort(dists, kind='stable')
        # convert orders to rank positions (index -> rank position)
        # ranks start at 1 for best
        pred_rank_positions = np.empty(M, dtype=int)
        pred_rank_positions[pred_order] = np.arange(1, M+1)
        gt_rank_positions = np.empty(M, dtype=int)
        gt_rank_positions[gt_order] = np.arange(1, M+1)

        # Kendall and Spearman between the rank position arrays
        # scipy.stats expects two 1D arrays of the same length: use the rank positions
        try:
            kt_stat, kt_p = kendalltau(pred_rank_positions, gt_rank_positions)
            if np.isnan(kt_stat):
                kt_stat = 0.0
        except Exception:
            kt_stat = 0.0

        try:
            sp_stat, sp_p = spearmanr(pred_rank_positions, gt_rank_positions)
            if np.isnan(sp_stat):
                sp_stat = 0.0
        except Exception:
            sp_stat = 0.0

        # MRR & Recall@k for exact original permutation: distance == 0
        # find indices where dists == 0 (there will always be at least one — the original perm)
        orig_indices = np.where(dists == 0)[0]
        # if multiple identically-original permutations (should be exactly 1), we still treat any as relevant
        is_relevant_fn = lambda idx, rel: (dists[idx] == 0)
        mrr = reciprocal_rank_of_first_relevant(pred_order, relevance, is_relevant_fn)
        recalls = {}
        for k in topk_for_recall:
            topk = pred_order[:k]
            recalls[f"recall@{k}"] = int(np.any(dists[topk] == 0))  # 1 if original in top-k else 0

        ndcg = ndcg_at_k(pred_order, relevance, ndcg_k)

        # record which index is best predicted and its distance
        best_pred_idx = int(pred_order[0])
        best_pred_dist = int(dists[best_pred_idx])
        # find rank position of the true original index (if exactly one) — take first
        if orig_indices.size > 0:
            true_idx = int(orig_indices[0])
            true_rank = int(np.where(pred_order == true_idx)[0][0] + 1)
        else:
            true_idx = None
            true_rank = None

        row = {
            "video_index": vid,
            "kendall_tau": kt_stat,
            "spearman_rho": sp_stat,
            "mrr": mrr,
            "ndcg@k": ndcg,
            "best_pred_index": best_pred_idx,
            "best_pred_distance": best_pred_dist,
            "orig_perm_index": int(orig_indices[0]) if orig_indices.size > 0 else None,
            "orig_perm_pred_rank": true_rank,
        }
        row.update(recalls)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("video_index")
    # aggregated metrics
    agg = {
        "mean_kendall_tau": float(df["kendall_tau"].mean()),
        "mean_spearman_rho": float(df["spearman_rho"].mean()),
        "mean_mrr": float(df["mrr"].mean()),
        "mean_ndcg@k": float(df["ndcg@k"].mean()),
    }
    for k in topk_for_recall:
        agg[f"mean_recall@{k}"] = float(df[f"recall@{k}"].mean())

    return df, agg