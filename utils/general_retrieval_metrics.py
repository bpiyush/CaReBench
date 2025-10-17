"""
General retrieval evaluation metrics.

Based on the following code:
https://github.com/jayleicn/singularity/blob/main/tasks/retrieval_utils.py#L330
"""

import numpy as np
from typing import Dict, List, Union, Any


def itm_eval(
    scores_i2t: np.ndarray,
    scores_t2i: np.ndarray, 
    txt2img: Dict[int, Union[int, List[int]]],
    img2txt: Dict[int, Union[int, List[int]]],
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Evaluate Image-Text Matching (ITM) performance for bidirectional retrieval tasks.
    
    This function computes standard retrieval metrics (Recall@1, Recall@5, Recall@10)
    for both image-to-text and text-to-image retrieval directions.
    
    Args:
        scores_i2t: Similarity scores for image-to-text retrieval.
                   Shape: (num_images, num_texts)
                   scores_i2t[i, j] = similarity between image i and text j
        scores_t2i: Similarity scores for text-to-image retrieval.
                   Shape: (num_texts, num_images) 
                   scores_t2i[i, j] = similarity between text i and image j
        txt2img: Mapping from text index to ground truth image ID(s).
                Keys are text indices, values are either:
                - int: single ground truth image ID
                - List[int]: multiple ground truth image IDs
        img2txt: Mapping from image index to ground truth text ID(s).
                Keys are image indices, values are either:
                - int: single ground truth text ID  
                - List[int]: multiple ground truth text IDs
        verbose: Whether to print verbose output.
    
    Returns:
        Dictionary containing evaluation metrics:
        - txt_r1, txt_r5, txt_r10: Image-to-text Recall@1,5,10 (%)
        - img_r1, img_r5, img_r10: Text-to-image Recall@1,5,10 (%)
        - txt_r_mean: Mean of image-to-text recalls
        - img_r_mean: Mean of text-to-image recalls  
        - r_mean: Overall mean of all recalls
        
    Note:
        - All metrics are returned as percentages (0-100) rounded to 2 decimal places
        - For multiple ground truths, the best (lowest) rank is used
        - Recall@K = % of queries where correct answer appears in top-K results
    """
    
    # Input validation
    assert isinstance(scores_i2t, np.ndarray), \
        f"scores_i2t must be numpy array, got {type(scores_i2t)}"
    assert isinstance(scores_t2i, np.ndarray), \
        f"scores_t2i must be numpy array, got {type(scores_t2i)}"
    assert scores_i2t.ndim == 2, \
        f"scores_i2t must be 2D, got shape {scores_i2t.shape}"
    assert scores_t2i.ndim == 2, \
        f"scores_t2i must be 2D, got shape {scores_t2i.shape}"
    
    num_images, num_texts = scores_i2t.shape
    num_texts_t2i, num_images_t2i = scores_t2i.shape
    
    assert num_texts == num_texts_t2i, \
        f"Text dimension mismatch: {num_texts} vs {num_texts_t2i}"
    assert num_images == num_images_t2i, \
        f"Image dimension mismatch: {num_images} vs {num_images_t2i}"
    assert len(img2txt) == num_images, \
        f"img2txt length {len(img2txt)} != num_images {num_images}"
    assert len(txt2img) == num_texts, \
        f"txt2img length {len(txt2img)} != num_texts {num_texts}"
    
    # ==========================================
    # Part 1: Image-to-Text Retrieval (I2T)
    # ==========================================
    if verbose:
        print(f"Computing image-to-text retrieval metrics for {num_images} images...")
    
    # Array to store rank of ground truth text for each image query
    i2t_ranks = np.zeros(num_images, dtype=np.float32)
    
    for img_idx in range(num_images):
        # Get similarity scores for this image against all texts
        img_scores = scores_i2t[img_idx]  # Shape: (num_texts,)
        assert img_scores.shape == (num_texts,), \
            f"Expected shape ({num_texts},), got {img_scores.shape}"
        
        # Sort text indices by similarity score in descending order
        # inds[0] = index of text with highest similarity to this image
        sorted_text_indices = np.argsort(img_scores)[::-1]  # Shape: (num_texts,)
        
        # Get ground truth text ID(s) for this image
        gt_text_ids = img2txt[img_idx]
        
        if isinstance(gt_text_ids, int):
            # Single ground truth text
            gt_text_ids = [gt_text_ids]
        else:
            # Multiple ground truth texts - ensure it's a list
            assert isinstance(gt_text_ids, (list, tuple)), \
                f"gt_text_ids must be int or list, got {type(gt_text_ids)}"
            gt_text_ids = list(gt_text_ids)
        
        # Find the best (lowest) rank among all ground truth texts
        best_rank = float('inf')
        for gt_text_id in gt_text_ids:
            # Find position of this ground truth text in the sorted ranking
            rank_positions = np.where(sorted_text_indices == gt_text_id)[0]
            assert len(rank_positions) == 1, \
                f"Text ID {gt_text_id} appears {len(rank_positions)} times in rankings"
            
            current_rank = rank_positions[0]  # 0-indexed rank (0 = best)
            if current_rank < best_rank:
                best_rank = current_rank
        
        i2t_ranks[img_idx] = best_rank
    
    # Compute image-to-text recall metrics
    # Recall@K = percentage of queries where GT appears in top-K results
    i2t_recall_at_1 = 100.0 * np.sum(i2t_ranks < 1) / len(i2t_ranks)
    i2t_recall_at_5 = 100.0 * np.sum(i2t_ranks < 5) / len(i2t_ranks) 
    i2t_recall_at_10 = 100.0 * np.sum(i2t_ranks < 10) / len(i2t_ranks)
    
    if verbose:
        print(f"I2T Results - R@1: {i2t_recall_at_1:.2f}%, R@5: {i2t_recall_at_5:.2f}%, R@10: {i2t_recall_at_10:.2f}%")
    
    # ==========================================
    # Part 2: Text-to-Image Retrieval (T2I)  
    # ==========================================
    if verbose:
        print(f"Computing text-to-image retrieval metrics for {num_texts} texts...")
    
    # Array to store rank of ground truth image for each text query
    t2i_ranks = np.zeros(num_texts, dtype=np.float32)
    
    for text_idx in range(num_texts):
        # Get similarity scores for this text against all images
        text_scores = scores_t2i[text_idx]  # Shape: (num_images,)
        assert text_scores.shape == (num_images,), \
            f"Expected shape ({num_images},), got {text_scores.shape}"
        
        # Sort image indices by similarity score in descending order
        # inds[0] = index of image with highest similarity to this text
        sorted_image_indices = np.argsort(text_scores)[::-1]  # Shape: (num_images,)
        
        # Get ground truth image ID(s) for this text
        gt_image_ids = txt2img[text_idx]
        
        if isinstance(gt_image_ids, int):
            # Single ground truth image
            gt_image_ids = [gt_image_ids]
        else:
            # Multiple ground truth images - ensure it's a list
            assert isinstance(gt_image_ids, (list, tuple)), \
                f"gt_image_ids must be int or list, got {type(gt_image_ids)}"
            gt_image_ids = list(gt_image_ids)
        
        # Find the best (lowest) rank among all ground truth images
        best_rank = float('inf')
        for gt_image_id in gt_image_ids:
            # Find position of this ground truth image in the sorted ranking
            rank_positions = np.where(sorted_image_indices == gt_image_id)[0]
            assert len(rank_positions) == 1, \
                f"Image ID {gt_image_id} appears {len(rank_positions)} times in rankings"
            
            current_rank = rank_positions[0]  # 0-indexed rank (0 = best)
            if current_rank < best_rank:
                best_rank = current_rank
        
        t2i_ranks[text_idx] = best_rank
    
    # Compute text-to-image recall metrics
    # Recall@K = percentage of queries where GT appears in top-K results
    t2i_recall_at_1 = 100.0 * np.sum(t2i_ranks < 1) / len(t2i_ranks)
    t2i_recall_at_5 = 100.0 * np.sum(t2i_ranks < 5) / len(t2i_ranks)
    t2i_recall_at_10 = 100.0 * np.sum(t2i_ranks < 10) / len(t2i_ranks)
    
    if verbose:
        print(f"T2I Results - R@1: {t2i_recall_at_1:.2f}%, R@5: {t2i_recall_at_5:.2f}%, R@10: {t2i_recall_at_10:.2f}%")
    
    # ==========================================
    # Part 3: Aggregate Final Metrics
    # ==========================================
    
    # Compute mean recalls for each direction
    i2t_mean_recall = (i2t_recall_at_1 + i2t_recall_at_5 + i2t_recall_at_10) / 3.0
    t2i_mean_recall = (t2i_recall_at_1 + t2i_recall_at_5 + t2i_recall_at_10) / 3.0
    
    # Compute overall mean across both directions
    overall_mean_recall = (i2t_mean_recall + t2i_mean_recall) / 2.0
    
    # Prepare results dictionary with descriptive names
    eval_results = {
        # Image-to-Text retrieval metrics
        "txt_r1": i2t_recall_at_1,      # Image-to-text Recall@1
        "txt_r5": i2t_recall_at_5,      # Image-to-text Recall@5  
        "txt_r10": i2t_recall_at_10,    # Image-to-text Recall@10
        "txt_r_mean": i2t_mean_recall,  # Mean of I2T recalls
        
        # Text-to-Image retrieval metrics
        "img_r1": t2i_recall_at_1,      # Text-to-image Recall@1
        "img_r5": t2i_recall_at_5,      # Text-to-image Recall@5
        "img_r10": t2i_recall_at_10,    # Text-to-image Recall@10  
        "img_r_mean": t2i_mean_recall,  # Mean of T2I recalls
        
        # Overall performance
        "r_mean": overall_mean_recall   # Overall mean across both directions
    }
    
    if verbose:
        print(f"Final Results - I2T Mean: {eval_results['txt_r_mean']:.2f}%, "
              f"T2I Mean: {eval_results['img_r_mean']:.2f}%, "
              f"Overall Mean: {eval_results['r_mean']:.2f}%")
    
    return eval_results


def print_results(results: Dict[str, float]):
    """Print results in a nice, readable format."""
    print("-" * 100)
    print("Retrieval Results")
    print("-" * 100)
    for key, value in results.items():
        print(f"{key}: {value:.2f} %")
    print("-" * 100)


# ==============================================================
# Extended evaluation: add mean average precision (mAP)
# ==============================================================
def _compute_mean_average_precision(scores: np.ndarray, query_to_relevants: Dict[int, Union[int, List[int]]]) -> float:
    """Compute mean average precision given a scores matrix and relevance mapping.

    Args:
        scores: 2D array shaped (num_queries, num_candidates)
        query_to_relevants: dict mapping each query index to either an int id
            or a list of relevant candidate ids.

    Returns:
        Mean Average Precision across queries that have at least one relevant.
        Returned as percentage (0-100).
    """
    num_queries, num_candidates = scores.shape
    assert len(query_to_relevants) == num_queries, \
        f"Mapping length {len(query_to_relevants)} != num_queries {num_queries}"

    average_precisions: List[float] = []

    for query_idx in range(num_queries):
        query_scores = scores[query_idx]
        assert query_scores.shape == (num_candidates,), \
            f"Expected shape ({num_candidates},), got {query_scores.shape}"

        sorted_candidate_indices = np.argsort(query_scores)[::-1]

        relevants = query_to_relevants[query_idx]
        if isinstance(relevants, int):
            relevant_set = {relevants}
        else:
            assert isinstance(relevants, (list, tuple)), \
                f"Relevant ids must be int or list/tuple, got {type(relevants)}"
            relevant_set = set(relevants)

        if len(relevant_set) == 0:
            # Skip queries with no relevant items
            continue

        num_retrieved_relevant = 0
        precision_sum = 0.0

        for rank, candidate_id in enumerate(sorted_candidate_indices, start=1):
            if candidate_id in relevant_set:
                num_retrieved_relevant += 1
                precision_at_k = num_retrieved_relevant / rank
                precision_sum += precision_at_k
                if num_retrieved_relevant == len(relevant_set):
                    # All relevants have been found; can stop early
                    break

        ap = precision_sum / len(relevant_set)
        average_precisions.append(ap)

    if len(average_precisions) == 0:
        return 0.0

    return 100.0 * float(np.mean(average_precisions))


def itm_eval_with_map(
    scores_i2t: np.ndarray,
    scores_t2i: np.ndarray,
    txt2img: Dict[int, Union[int, List[int]]],
    img2txt: Dict[int, Union[int, List[int]]],
    verbose: bool = False,
) -> Dict[str, float]:
    """Extended ITM eval that also computes mAP for both directions.

    This function reuses `itm_eval` to compute recalls and then adds:
      - txt_map: Image-to-Text mAP (query=image, relevant texts)
      - img_map: Text-to-Image mAP (query=text, relevant images)
      - map_mean: Mean of txt_map and img_map
    All metrics are returned as percentages.
    """
    # Base recall metrics
    base_results = itm_eval(scores_i2t, scores_t2i, txt2img, img2txt, verbose=verbose)

    # Compute mAP for both directions
    i2t_map = _compute_mean_average_precision(scores_i2t, img2txt)
    t2i_map = _compute_mean_average_precision(scores_t2i, txt2img)
    map_mean = (i2t_map + t2i_map) / 2.0

    extended = dict(base_results)
    extended.update({
        "txt_map": i2t_map,
        "img_map": t2i_map,
        "map_mean": map_mean,
    })

    if verbose:
        print(f"I2T mAP: {i2t_map:.2f}%, T2I mAP: {t2i_map:.2f}%, Mean mAP: {map_mean:.2f}%")

    return extended

# Example usage and testing function
def test_itm_eval():
    """Test function to demonstrate usage of itm_eval with synthetic data."""
    
    # Create synthetic similarity scores
    num_images, num_texts = 100, 500
    np.random.seed(42)  # For reproducible results
    
    # Random similarity scores with some structure to make evaluation meaningful
    scores_i2t = np.random.randn(num_images, num_texts)
    scores_t2i = scores_i2t.T  # Ensure consistency
    
    # Create ground truth mappings
    # Simple case: each image has exactly one corresponding text and vice versa
    img2txt = {i: i % num_texts for i in range(num_images)}  # Image i -> Text (i % num_texts)
    txt2img = {i: i % num_images for i in range(num_texts)}  # Text i -> Image (i % num_images)
    
    # Boost scores for correct matches to make evaluation more meaningful
    for img_idx in range(num_images):
        correct_text_idx = img2txt[img_idx]
        scores_i2t[img_idx, correct_text_idx] += 2.0  # Make correct match more likely to rank high
    
    # Run evaluation
    results = itm_eval(scores_i2t, scores_t2i, txt2img, img2txt)
    
    print_results(results)
    return results


if __name__ == "__main__":
    test_itm_eval()