import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def embedding_distance(X: np.ndarray, Y: np.ndarray, regularization: float = 1e-4) -> dict:
    """
    Compute various distance/similarity and clustering quality metrics
    between two sets of embeddings.

    Args:
        X: np.ndarray of shape (n, d) — first set of embeddings
        Y: np.ndarray of shape (m, d) — second set of embeddings
        regularization: small value added to covariance diagonals for numerical stability

    Returns:
        dict with metrics grouped into:

        Centroid-based:
            - centroid_cosine_similarity
            - centroid_euclidean_distance

        Distributional:
            - emd_euclidean, emd_cosine
            - mmd_rbf
            - kl_xy, kl_yx, js_divergence
            - frechet_distance

        Clustering quality:
            - silhouette_euclidean   (-1 to 1, higher = better separated)
            - silhouette_cosine      (-1 to 1, higher = better separated)
            - calinski_harabasz      (higher = better separated)
            - davies_bouldin         (lower = better separated)
    """
    import ot

    X = np.array(X, dtype=np.float64)
    Y = np.array(Y, dtype=np.float64)
    n, d = X.shape
    m = Y.shape[0]

    results = {}

    # -------------------------------------------------------------------------
    # 1. Centroid-based metrics
    # -------------------------------------------------------------------------
    centroid_x = X.mean(axis=0, keepdims=True)
    centroid_y = Y.mean(axis=0, keepdims=True)

    results["centroid_cosine_similarity"] = cosine_similarity(centroid_x, centroid_y)[0, 0]
    results["centroid_euclidean_distance"] = np.linalg.norm(centroid_x - centroid_y)

    # -------------------------------------------------------------------------
    # 2. Earth Mover's Distance (Wasserstein)
    # -------------------------------------------------------------------------
    a_weights = np.ones(n) / n
    b_weights = np.ones(m) / m

    # Euclidean cost
    M_euclidean = ot.dist(X, Y, metric="euclidean")
    results["emd_euclidean"] = ot.emd2(a_weights, b_weights, M_euclidean)

    # Cosine distance cost
    M_cosine = cosine_distances(X, Y)
    results["emd_cosine"] = ot.emd2(a_weights, b_weights, M_cosine)

    # -------------------------------------------------------------------------
    # 3. Maximum Mean Discrepancy (RBF kernel)
    # -------------------------------------------------------------------------
    def rbf_kernel(A, B, sigma=None):
        dists = cdist(A, B, metric="sqeuclidean")
        if sigma is None:
            sigma = np.sqrt(np.median(dists))  # median heuristic
        return np.exp(-dists / (2 * sigma ** 2))

    K_xx = rbf_kernel(X, X)
    K_yy = rbf_kernel(Y, Y)
    K_xy = rbf_kernel(X, Y)

    mmd2 = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    results["mmd_rbf"] = float(np.sqrt(max(mmd2, 0)))

    # -------------------------------------------------------------------------
    # 4. Gaussian fit helpers (shared by KL, JS, Fréchet)
    # -------------------------------------------------------------------------
    def fit_gaussian(E, reg):
        mu = E.mean(axis=0)
        diff = E - mu
        cov = (diff.T @ diff) / (len(E) - 1)
        cov += np.eye(d) * reg  # regularize
        return mu, cov

    mu_x, cov_x = fit_gaussian(X, regularization)
    mu_y, cov_y = fit_gaussian(Y, regularization)

    # -------------------------------------------------------------------------
    # 5. KL Divergence between two multivariate Gaussians
    # -------------------------------------------------------------------------
    def kl_gaussian(mu_p, cov_p, mu_q, cov_q):
        """KL(P || Q) for multivariate Gaussians."""
        cov_q_inv = np.linalg.inv(cov_q)
        sign_p, logdet_p = np.linalg.slogdet(cov_p)
        sign_q, logdet_q = np.linalg.slogdet(cov_q)
        diff_mu = mu_q - mu_p

        kl = 0.5 * (
            logdet_q - logdet_p
            - d
            + np.trace(cov_q_inv @ cov_p)
            + diff_mu @ cov_q_inv @ diff_mu
        )
        return float(kl)

    results["kl_xy"] = kl_gaussian(mu_x, cov_x, mu_y, cov_y)
    results["kl_yx"] = kl_gaussian(mu_y, cov_y, mu_x, cov_x)
    results["js_divergence"] = 0.5 * results["kl_xy"] + 0.5 * results["kl_yx"]

    # -------------------------------------------------------------------------
    # 6. Fréchet Distance (FID-style)
    # -------------------------------------------------------------------------
    from scipy.linalg import sqrtm

    cov_product = sqrtm(cov_x @ cov_y)
    # sqrtm may return complex values due to numerical errors
    if np.iscomplexobj(cov_product):
        cov_product = cov_product.real

    frechet = (
        np.linalg.norm(mu_x - mu_y) ** 2
        + np.trace(cov_x + cov_y - 2 * cov_product)
    )
    results["frechet_distance"] = float(frechet)

    # -------------------------------------------------------------------------
    # 7. Clustering quality metrics
    #    Treat X as cluster 0, Y as cluster 1
    # -------------------------------------------------------------------------
    combined = np.vstack([X, Y])
    labels = np.array([0] * n + [1] * m)

    # Silhouette score: how well each point belongs to its own cluster vs the other
    # Range: -1 (wrong cluster) to +1 (well clustered)
    results["silhouette_euclidean"] = float(silhouette_score(combined, labels, metric="euclidean"))
    results["silhouette_cosine"] = float(silhouette_score(combined, labels, metric="cosine"))

    # Calinski-Harabasz index: ratio of between-cluster to within-cluster variance
    # Higher = better separated
    results["calinski_harabasz"] = float(calinski_harabasz_score(combined, labels))

    # Davies-Bouldin index: avg similarity between each cluster and its most similar one
    # Lower = better separated (0 is perfect)
    results["davies_bouldin"] = float(davies_bouldin_score(combined, labels))

    return results


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    np.random.seed(42)

    # Simulated embeddings: three clusters in 64-d space
    # Mimics: folding (X), unfolding (Y), not_folding (Z)
    d = 64
    X = np.random.randn(50, d) + 0.0   # "folding" cluster at origin
    Y = np.random.randn(50, d) + 0.5   # "unfolding" cluster shifted
    Z = np.random.randn(50, d) + 0.4   # "not folding" cluster (closer to unfolding?)

    pairs = [
        ("folding vs unfolding", X, Y),
        ("folding vs not_folding", X, Z),
        ("unfolding vs not_folding", Y, Z),
    ]

    for name, A, B in pairs:
        metrics = embedding_distance(A, B)
        print("=" * 55)
        print(f"  {name}")
        print("=" * 55)

        print("\n  Centroid-based:")
        for k in ["centroid_cosine_similarity", "centroid_euclidean_distance"]:
            print(f"    {k:.<40s} {metrics[k]:.6f}")

        print("\n  Distributional:")
        for k in ["emd_euclidean", "emd_cosine", "mmd_rbf",
                   "kl_xy", "kl_yx", "js_divergence", "frechet_distance"]:
            print(f"    {k:.<40s} {metrics[k]:.6f}")

        print("\n  Clustering quality:")
        for k in ["silhouette_euclidean", "silhouette_cosine",
                   "calinski_harabasz", "davies_bouldin"]:
            print(f"    {k:.<40s} {metrics[k]:.6f}")
        print()