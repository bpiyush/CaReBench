#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.decomposition import PCA


def bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR."""
    pvals = np.asarray(pvals)
    n = len(pvals)

    order = np.argsort(pvals)
    ranked = pvals[order]

    thresholds = alpha * np.arange(1, n + 1) / n
    passed = ranked <= thresholds

    if not np.any(passed):
        return np.zeros(n, dtype=bool)

    k = np.max(np.where(passed)[0])
    cutoff = ranked[k]

    return pvals <= cutoff


def gaussian_null_top_eigenvalues(
    n, d, nu2, n_sims=500, seed=12345
):
    """
    Simulate the largest sample covariance eigenvalue under
        E_ij ~ N(0, nu^2)
    with exactly the same N and D as the real data.
    """
    rng = np.random.default_rng(seed)

    top_eigs = np.empty(n_sims)

    for b in range(n_sims):
        X = rng.normal(
            loc=0.0,
            scale=np.sqrt(nu2),
            size=(n, d),
        )

        X -= X.mean(axis=0, keepdims=True)

        # Nonzero eigenvalues of X^T X/(n-1)
        # are the eigenvalues of X X^T/(n-1).
        gram = (X @ X.T) / (n - 1)

        vals = np.linalg.eigvalsh(gram)
        top_eigs[b] = vals[-1]

    return top_eigs


def show_stats(qi):
    median_qi = np.median(qi)
    mean_qi   = np.mean(qi)
    q90       = np.quantile(qi, 0.90)
    q95       = np.quantile(qi, 0.95)
    q99       = np.quantile(qi, 0.99)
    max_qi    = np.max(qi)

    print(f"Median: {median_qi}")
    print(f"Mean:   {mean_qi}")
    print(f"90%:    {q90}")
    print(f"95%:    {q95}")
    print(f"99%:    {q99}")
    print(f"Max:    {max_qi}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to E.npy with shape (N, D).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="e_diagnostics",
        help="Output directory.",
    )
    parser.add_argument(
        "--n-projections",
        type=int,
        default=300,
        help="Number of random projections.",
    )
    parser.add_argument(
        "--n-pc-projections",
        type=int,
        default=20,
        help="Number of principal-component projections.",
    )
    parser.add_argument(
        "--n-null-sims",
        type=int,
        default=500,
        help="Number of isotropic Gaussian simulations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
    )

    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------

    E = np.load(args.input).astype(np.float64)

    if E.ndim != 2:
        raise ValueError(f"E must be 2D, got shape {E.shape}")

    N, D = E.shape

    print(f"E shape: N={N}, D={D}")

    if not np.isfinite(E).all():
        raise ValueError("E contains NaN or inf.")

    # ------------------------------------------------------------
    # Center the data
    # ------------------------------------------------------------

    mean = E.mean(axis=0)
    Ec = E - mean

    # Coordinate-wise sample variances
    var_d = Ec.var(axis=0, ddof=1)
    std_d = np.sqrt(var_d)

    # Isotropic variance estimate:
    # MLE-like / average variance across dimensions
    nu2_hat = float(var_d.mean())
    nu_hat = float(np.sqrt(nu2_hat))

    # ------------------------------------------------------------
    # 1. Marginal Gaussianity
    # ------------------------------------------------------------

    skew_d = stats.skew(Ec, axis=0, bias=False)
    kurt_d = stats.kurtosis(Ec, axis=0, fisher=True, bias=False)

    # D'Agostino normality test per dimension
    normal_p = np.empty(D)

    for d in range(D):
        _, p = stats.normaltest(Ec[:, d])
        normal_p[d] = p

    normal_reject_fdr = bh_fdr(normal_p, alpha=0.05)

    # ------------------------------------------------------------
    # 2. Singular values / covariance spectrum
    # ------------------------------------------------------------

    # Since D > N, it is cheaper to compute SVD of N x D.
    #
    # Ec = U S V^T
    #
    # Eigenvalues of covariance = S^2/(N-1)
    U, S, Vt = np.linalg.svd(Ec, full_matrices=False)

    eigvals = (S ** 2) / (N - 1)

    # Only N-1 eigenvalues can be nonzero after centering.
    eigvals_positive = eigvals[eigvals > 1e-14]

    lambda_max = float(eigvals_positive[0])

    # Effective rank
    p = eigvals_positive / eigvals_positive.sum()
    effective_rank = float(np.exp(-(p * np.log(p)).sum()))

    participation_ratio = float(
        eigvals_positive.sum() ** 2 /
        np.sum(eigvals_positive ** 2)
    )

    # ------------------------------------------------------------
    # 3. Compare lambda_max against isotropic Gaussian null
    # ------------------------------------------------------------

    null_top = gaussian_null_top_eigenvalues(
        n=N,
        d=D,
        nu2=nu2_hat,
        n_sims=args.n_null_sims,
        seed=args.seed,
    )

    null_mean = float(null_top.mean())
    null_q50 = float(np.quantile(null_top, 0.50))
    null_q90 = float(np.quantile(null_top, 0.90))
    null_q95 = float(np.quantile(null_top, 0.95))
    null_q99 = float(np.quantile(null_top, 0.99))

    # Monte Carlo p-value
    null_p = float(
        (1 + np.sum(null_top >= lambda_max))
        / (len(null_top) + 1)
    )

    # Marchenko-Pastur approximation
    aspect = D / (N - 1)
    mp_upper = float(
        nu2_hat * (1.0 + np.sqrt(aspect)) ** 2
    )

    # ------------------------------------------------------------
    # 4. Random projection Gaussianity
    # ------------------------------------------------------------

    random_projection_skew = []
    random_projection_kurt = []
    random_projection_p = []

    projection_vectors = []

    for j in range(args.n_projections):
        u = rng.normal(size=D)
        u /= np.linalg.norm(u)

        y = Ec @ u

        projection_vectors.append(u)

        random_projection_skew.append(
            float(stats.skew(y, bias=False))
        )

        random_projection_kurt.append(
            float(stats.kurtosis(y, fisher=True, bias=False))
        )

        _, pval = stats.normaltest(y)
        random_projection_p.append(float(pval))

    random_projection_p = np.asarray(random_projection_p)

    random_projection_reject_fdr = bh_fdr(
        random_projection_p,
        alpha=0.05,
    )

    # ------------------------------------------------------------
    # 5. Gaussianity in top PCA directions
    # ------------------------------------------------------------

    pc_skew = []
    pc_kurt = []
    pc_pvals = []

    n_pc = min(args.n_pc_projections, len(eigvals_positive))

    # Vt rows correspond to principal directions.
    for j in range(n_pc):
        pc_dir = Vt[j]
        y = Ec @ pc_dir

        pc_skew.append(
            float(stats.skew(y, bias=False))
        )

        pc_kurt.append(
            float(stats.kurtosis(y, fisher=True, bias=False))
        )

        _, pval = stats.normaltest(y)
        pc_pvals.append(float(pval))

    pc_pvals = np.asarray(pc_pvals)

    # ------------------------------------------------------------
    # 6. Pairwise correlations
    #
    # Full D x D correlation matrix is manageable here but not
    # strictly needed. Instead compute normalized covariance
    # through matrix multiplication.
    # ------------------------------------------------------------

    # Covariance
    cov = (Ec.T @ Ec) / (N - 1)

    std_safe = np.maximum(std_d, 1e-15)

    corr = cov / np.outer(std_safe, std_safe)

    # Remove diagonal
    offdiag_mask = ~np.eye(D, dtype=bool)
    offdiag_corr = corr[offdiag_mask]

    corr_abs = np.abs(offdiag_corr)

    corr_rms = float(np.sqrt(np.mean(offdiag_corr ** 2)))

    corr_quantiles = {
        "abs_q50": float(np.quantile(corr_abs, 0.50)),
        "abs_q90": float(np.quantile(corr_abs, 0.90)),
        "abs_q95": float(np.quantile(corr_abs, 0.95)),
        "abs_q99": float(np.quantile(corr_abs, 0.99)),
        "abs_max": float(np.max(corr_abs)),
    }

    # ------------------------------------------------------------
    # 7. Useful summary statistics for coordinate variances
    # ------------------------------------------------------------

    std_summary = {
        "mean": float(np.mean(std_d)),
        "median": float(np.median(std_d)),
        "std_across_dimensions": float(np.std(std_d)),
        "q01": float(np.quantile(std_d, 0.01)),
        "q05": float(np.quantile(std_d, 0.05)),
        "q25": float(np.quantile(std_d, 0.25)),
        "q50": float(np.quantile(std_d, 0.50)),
        "q75": float(np.quantile(std_d, 0.75)),
        "q95": float(np.quantile(std_d, 0.95)),
        "q99": float(np.quantile(std_d, 0.99)),
        "min": float(np.min(std_d)),
        "max": float(np.max(std_d)),
    }

    # ------------------------------------------------------------
    # 8. Dimension-wise Gaussian diagnostics
    # ------------------------------------------------------------

    marginal_summary = {
        "fraction_fdr_rejected_0.05":
            float(np.mean(normal_reject_fdr)),
        "median_skew":
            float(np.median(skew_d)),
        "median_abs_skew":
            float(np.median(np.abs(skew_d))),
        "q95_abs_skew":
            float(np.quantile(np.abs(skew_d), 0.95)),
        "median_excess_kurtosis":
            float(np.median(kurt_d)),
        "median_abs_excess_kurtosis":
            float(np.median(np.abs(kurt_d))),
        "q95_abs_excess_kurtosis":
            float(np.quantile(np.abs(kurt_d), 0.95)),
    }

    # ------------------------------------------------------------
    # 9. Random-projection summaries
    # ------------------------------------------------------------

    random_projection_summary = {
        "fraction_fdr_rejected_0.05":
            float(np.mean(random_projection_reject_fdr)),
        "median_skew":
            float(np.median(random_projection_skew)),
        "median_abs_skew":
            float(np.median(np.abs(random_projection_skew))),
        "q95_abs_skew":
            float(np.quantile(np.abs(random_projection_skew), 0.95)),
        "median_excess_kurtosis":
            float(np.median(random_projection_kurt)),
        "median_abs_excess_kurtosis":
            float(np.median(np.abs(random_projection_kurt))),
        "q95_abs_excess_kurtosis":
            float(np.quantile(np.abs(random_projection_kurt), 0.95)),
    }

    # ------------------------------------------------------------
    # 10. Top-PC summaries
    # ------------------------------------------------------------

    pc_summary = {
        "n_pc": n_pc,
        "fraction_rejected":
            float(np.mean(pc_pvals < 0.05))
            if n_pc > 0 else None,
        "abs_skew_max":
            float(np.max(np.abs(pc_skew)))
            if n_pc > 0 else None,
        "abs_kurtosis_max":
            float(np.max(np.abs(pc_kurt)))
            if n_pc > 0 else None,
    }

    # ------------------------------------------------------------
    # 11. Loss-bound quantities
    #
    # IMPORTANT:
    # Here we only have E, not the text query embeddings.
    # Therefore we cannot compute q_i^T Sigma q_i yet.
    #
    # We can compute:
    #   - isotropic factor from nu
    #   - worst-case factor from lambda_max
    # for a given temperature.
    # ------------------------------------------------------------

    tau = 0.05

    isotropic_factor = float(
        np.exp(nu2_hat / tau**2)
    )

    worst_case_factor = float(
        np.exp(lambda_max / tau**2)
    )

    # ------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------

    results = {
        "shape": {
            "N": N,
            "D": D,
        },
        "basic": {
            "mean_abs_entry": float(np.mean(np.abs(E))),
            "global_std": float(np.std(E)),
            "nu_hat": nu_hat,
            "nu2_hat": nu2_hat,
        },
        "coordinate_std": std_summary,
        "marginal_gaussianity": marginal_summary,
        "random_projection_gaussianity": random_projection_summary,
        "top_pc_gaussianity": pc_summary,
        "correlations": {
            "rms_offdiag": corr_rms,
            **corr_quantiles,
        },
        "spectrum": {
            "lambda_max": lambda_max,
            "lambda_min_positive": float(eigvals_positive[-1]),
            "effective_rank": effective_rank,
            "participation_ratio": participation_ratio,
            "mp_upper_edge": mp_upper,
        },
        "isotropic_gaussian_null": {
            "simulation_count": args.n_null_sims,
            "null_mean_lambda_max": null_mean,
            "null_median_lambda_max": null_q50,
            "null_q90_lambda_max": null_q90,
            "null_q95_lambda_max": null_q95,
            "null_q99_lambda_max": null_q99,
            "monte_carlo_p_value":
                null_p,
        },
        "tau": tau,
        "loss_bound_factors": {
            "isotropic_exp_nu2_over_tau2":
                isotropic_factor,
            "worst_case_exp_lambda_max_over_tau2":
                worst_case_factor,
        },
    }

    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------
    # Save numerical arrays needed for deeper analysis
    # ------------------------------------------------------------

    np.save(out / "eigenvalues.npy", eigvals)
    np.save(out / "coordinate_std.npy", std_d)
    np.save(out / "coordinate_skew.npy", skew_d)
    np.save(out / "coordinate_excess_kurtosis.npy", kurt_d)
    np.save(out / "coordinate_normality_pvalues.npy", normal_p)
    np.save(out / "null_top_eigenvalues.npy", null_top)
    np.save(out / "pca_components.npy", Vt[:n_pc])

    # ------------------------------------------------------------
    # Plot 1: coordinate standard deviations
    # ------------------------------------------------------------

    plt.figure(figsize=(7, 5))
    plt.hist(std_d, bins=50, density=True)
    plt.xlabel("Per-dimension standard deviation")
    plt.ylabel("Density")
    plt.title("Distribution of coordinate standard deviations")
    plt.tight_layout()
    plt.savefig(out / "coordinate_std.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Plot 2: skewness across dimensions
    # ------------------------------------------------------------

    plt.figure(figsize=(7, 5))
    plt.hist(skew_d, bins=60, density=True)
    plt.xlabel("Skewness")
    plt.ylabel("Density")
    plt.title("Coordinate skewness")
    plt.tight_layout()
    plt.savefig(out / "coordinate_skewness.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Plot 3: excess kurtosis across dimensions
    # ------------------------------------------------------------

    plt.figure(figsize=(7, 5))
    plt.hist(kurt_d, bins=60, density=True)
    plt.xlabel("Excess kurtosis")
    plt.ylabel("Density")
    plt.title("Coordinate excess kurtosis")
    plt.tight_layout()
    plt.savefig(out / "coordinate_kurtosis.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Plot 4: covariance spectrum
    # ------------------------------------------------------------

    k_plot = min(100, len(eigvals))

    plt.figure(figsize=(7, 5))
    plt.semilogy(
        np.arange(1, k_plot + 1),
        eigvals[:k_plot],
        marker="o",
        markersize=2,
    )
    plt.xlabel("Eigenvalue rank")
    plt.ylabel("Eigenvalue")
    plt.title("Top covariance eigenvalues")
    plt.tight_layout()
    plt.savefig(out / "spectrum_top100.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Plot 5: null distribution of lambda_max
    # ------------------------------------------------------------

    plt.figure(figsize=(7, 5))
    plt.hist(null_top, bins=40, density=True)
    plt.axvline(
        lambda_max,
        linestyle="--",
        linewidth=2,
        label=f"Observed = {lambda_max:.6g}",
    )
    plt.xlabel("Largest sample covariance eigenvalue")
    plt.ylabel("Density")
    plt.title(
        "Null distribution under N(0, nu^2 I)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "lambda_max_null.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Plot 6: correlation histogram
    # ------------------------------------------------------------

    # Subsample for plotting to keep file size reasonable.
    n_plot_corr = min(500_000, len(offdiag_corr))
    idx = rng.choice(
        len(offdiag_corr),
        size=n_plot_corr,
        replace=False,
    )

    plt.figure(figsize=(7, 5))
    plt.hist(
        offdiag_corr[idx],
        bins=100,
        density=True,
    )
    plt.xlabel("Off-diagonal sample correlation")
    plt.ylabel("Density")
    plt.title("Distribution of off-diagonal correlations")
    plt.tight_layout()
    plt.savefig(out / "correlation_histogram.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # Print concise summary
    # ------------------------------------------------------------

    print("\n==============================")
    print("SUMMARY")
    print("==============================")
    print(f"N = {N}")
    print(f"D = {D}")
    print()
    print(f"nu_hat       = {nu_hat:.8f}")
    print(f"nu_hat^2     = {nu2_hat:.8f}")
    print(f"lambda_max   = {lambda_max:.8f}")
    print(f"MP upper edge= {mp_upper:.8f}")
    print()
    print(
        "Marginal Gaussian FDR rejection fraction = "
        f"{np.mean(normal_reject_fdr):.4f}"
    )
    print(
        "Random projection FDR rejection fraction = "
        f"{np.mean(random_projection_reject_fdr):.4f}"
    )
    print()
    print(
        "Offdiag correlation RMS = "
        f"{corr_rms:.6f}"
    )
    print(
        "Offdiag |corr| 95%     = "
        f"{corr_quantiles['abs_q95']:.6f}"
    )
    print(
        "Offdiag |corr| max     = "
        f"{corr_quantiles['abs_max']:.6f}"
    )
    print()
    print(
        "Isotropic null median lambda_max = "
        f"{null_q50:.8f}"
    )
    print(
        "Isotropic null 95% lambda_max    = "
        f"{null_q95:.8f}"
    )
    print(
        "Isotropic null 99% lambda_max    = "
        f"{null_q99:.8f}"
    )
    print(
        "Monte Carlo p-value               = "
        f"{null_p:.6f}"
    )
    print()
    print(
        "Effective rank                    = "
        f"{effective_rank:.2f}"
    )
    print(
        "Participation ratio               = "
        f"{participation_ratio:.2f}"
    )
    print()
    print(
        "Isotropic loss-bound factor       = "
        f"{isotropic_factor:.6g}"
    )
    print(
        "Worst-case spectral factor        = "
        f"{worst_case_factor:.6g}"
    )
    print()
    print(f"Results written to: {out.resolve()}")


if __name__ == "__main__":
    main()