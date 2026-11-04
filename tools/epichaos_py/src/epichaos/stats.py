"""Permutation tests for comparing heterogeneity between groups."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epichaos.jaccard import pairwise_jaccard_mean


def compare_groups(
    mat1: np.ndarray,
    mat2: np.ndarray,
    niter: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test comparing epiCHAOS scores between two groups.

    Parameters
    ----------
    mat1, mat2:
        Feature × cell binary matrices for group 1 and group 2.
    niter:
        Number of permutation iterations.
    seed:
        Random seed.

    Returns
    -------
    Dict with keys: observed_diff, permuted_diffs, pval.
    """
    rng = np.random.default_rng(seed)

    score1 = pairwise_jaccard_mean(mat1)
    score2 = pairwise_jaccard_mean(mat2)
    observed_diff = abs(score1 - score2)

    combined = np.concatenate([mat1, mat2], axis=1)
    n1 = mat1.shape[1]
    n_total = combined.shape[1]

    permuted_diffs = np.empty(niter)
    for i in range(niter):
        perm = rng.permutation(n_total)
        p1 = pairwise_jaccard_mean(combined[:, perm[:n1]])
        p2 = pairwise_jaccard_mean(combined[:, perm[n1:]])
        permuted_diffs[i] = abs(p1 - p2)

    pval = float(np.mean(permuted_diffs >= observed_diff))

    return {
        "observed_diff": observed_diff,
        "permuted_diffs": permuted_diffs,
        "pval": pval,
    }


def compute_diff_eICH(
    mat1: np.ndarray,
    mat2: np.ndarray,
    niter: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test with density plot of observed vs permuted differences.

    Returns the same result dict as ``compare_groups`` plus a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    result = compare_groups(mat1, mat2, niter=niter, seed=seed)

    fig, ax = plt.subplots()
    ax.hist(result["permuted_diffs"], bins=30, density=True, alpha=0.7, label="permuted")
    ax.axvline(result["observed_diff"], color="red", linestyle="--", label="observed")
    ax.set_xlabel("Absolute score difference")
    ax.set_ylabel("Density")
    ax.set_title(f"p = {result['pval']:.4f}")
    ax.legend()

    result["plot"] = fig
    return result
