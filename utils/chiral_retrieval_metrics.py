import json
import numpy as np
import pandas as pd
import torch


def compute_retrieval_metrics_wrapper(
    df: pd.DataFrame, vid_feat: dict, text_feat: dict,
    text_id_col="text_id", video_id_col="id",
):
    from utils.general_retrieval_metrics import (
        itm_eval_with_map,
    )

    # Compute scores
    zv = torch.stack([vid_feat[k] for k in vid_feat]).numpy()
    zt = torch.stack([text_feat[k] for k in text_feat]).numpy()
    scores_v2t = zv @ zt.T
    scores_t2v = zt @ zv.T

    # Get matchings
    video_keys = np.array(list(vid_feat.keys()))
    text_keys = np.array(list(text_feat.keys()))

    # Text to video
    t2v = {}
    j = 0
    for k in text_feat:
        # For each sample, pick ONLY the videos that match with the caption
        matching_video_ids = df[(df[text_id_col] == k)][video_id_col].values
        t2v[j] = [np.where(x == video_keys)[0][0] for x in matching_video_ids]
        j += 1

    # Video to text
    v2t = {}
    j = 0
    for k in vid_feat:
        row = df[df[video_id_col] == k]
        if len(row) != 1:
            import ipdb; ipdb.set_trace()
        
        row = row.iloc[0].to_dict()
        text_id = row[text_id_col]
        v2t[j] = list(np.where(text_id == text_keys)[0])
        j += 1

    # return itm_eval(scores_v2t, scores_t2v, t2v, v2t, verbose=False)
    return itm_eval_with_map(scores_v2t, scores_t2v, t2v, v2t, verbose=False)


def weighted_average(df: pd.DataFrame, weight_col: str = "frac") -> pd.Series:
    """
    Compute the weighted average of all numeric columns in a DataFrame 
    using a specified weight column.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing numeric columns and a weight column.
    weight_col : str, default="frac"
        Column to use as weights. Must contain values between [0, 1].

    Returns
    -------
    pd.Series
        A Series containing the weighted averages for each numeric column 
        (excluding the weight column).
    """
    if weight_col not in df.columns:
        raise ValueError(f"Weight column '{weight_col}' not found in DataFrame")

    weights = df[weight_col].to_numpy()
    if not np.isclose(weights.sum(), 1):
        weights = weights / weights.sum()  # normalize to sum=1 if needed

    numeric_cols = df.select_dtypes(include=[np.number]).drop(columns=[weight_col], errors="ignore")
    
    weighted_means = (numeric_cols * weights[:, None]).sum(axis=0)

    return weighted_means


def compute_retrieval_metrics_with_subsets(
    df: pd.DataFrame,
    vid_feat: dict,
    text_feat: dict,
    agg_col: str,
    video_id_col: str = 'id',
    text_id_col: str = 'text_id',
    verbose: bool = False,
):
    """
    Computes retrieval with different subsets of df and then
    aggregates the results.
    
    Args:
        df: pd.DataFrame, CSV with all the information.
        vid_feat: dict, Video features.
        text_feat: dict, Text features.
        agg_col: str, Column name for aggregation.
        verbose: bool, Whether to print progress.

    Returns:
        pd.DataFrame, Aggregated retrieval results.
    """
    assert agg_col in df.columns
    assert video_id_col in df.columns
    assert text_id_col in df.columns
    
    values = df[agg_col].unique()
    results = []
    for v in values:
        
        # Collect video and text features only for this value of agg_col
        _video_ids = df[df[agg_col] == v][video_id_col].values
        _text_ids = df[df[agg_col] == v][text_id_col].values

        _vid_feat = {k: vid_feat[k] for k in _video_ids}
        _text_feat = {k: text_feat[k] for k in _text_ids}
        if verbose:
            print(f"Computing retrieval metrics for {agg_col} = {v}")
            print(f"Number of video IDs: {len(np.unique(_video_ids))}")
            print(f"Number of text IDs: {len(np.unique(_text_ids))}")
            print('-' * 100)
        
        # Compute retrieval metrics
        _results = compute_retrieval_metrics_wrapper(
            df[df[agg_col] == v],
            _vid_feat,
            _text_feat,
            video_id_col=video_id_col,
            text_id_col=text_id_col,
        )
        _results[agg_col] = v
        _results['frac'] = len(_vid_feat) / len(df)
        results.append(_results)
    results = pd.DataFrame(results)
    
    # Average the results: weighted by the fraction of samples in each subset
    results_avg = weighted_average(results, weight_col='frac').to_dict()
    # results_avg = results.mean().to_dict()

    return results_avg


def compute_metrics(df, _video_feats, _texts_feats, show_metrics=True):
    df['base'] = 'base'
    metrics_base = compute_retrieval_metrics_with_subsets(
        df, _video_feats, _texts_feats, agg_col='base', verbose=False,
    )
    metrics_pair = compute_retrieval_metrics_with_subsets(
        df, _video_feats, _texts_feats, agg_col='chiral_triplet_id', verbose=False,
    )
    metrics_stat = compute_retrieval_metrics_with_subsets(
        df, _video_feats, _texts_feats, agg_col='chiral_label', verbose=False,
    )
    if show_metrics:
        print(json.dumps(metrics_pair, indent=4))
        print(json.dumps(metrics_stat, indent=4))
        print(json.dumps(metrics_base, indent=4))
    return dict(
        metrics_pair=metrics_pair,
        metrics_stat=metrics_stat,
        metrics_base=metrics_base,
    )


def round_off(x):
    return str(np.round(x, 1))


def print_metrics_as_latex_row(metrics, v2t_metric='txt_r1', t2v_metric='img_map'):
    string = ' & '.join(
        [
            round_off(metrics['metrics_pair'][v2t_metric]),
            round_off(metrics['metrics_stat'][v2t_metric]),
            round_off(metrics['metrics_base'][v2t_metric]),
            round_off(metrics['metrics_pair'][t2v_metric]),
            round_off(metrics['metrics_stat'][t2v_metric]),
            round_off(metrics['metrics_base'][t2v_metric]),
        ],
    )
    print(string)