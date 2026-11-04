"""Core epiCHAOS and transCHAOS score computation (NumPy API).

Matches the R epiCHAOS package algorithm exactly:
- epiCHAOS / compute_eITH: pairwise centered Jaccard + sparsity (avg coverage) correction
- transCHAOS / compute_tITH: mean pairwise Euclidean distance + outlier removal
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform

from epichaos.jaccard import pairwise_jaccard_mean
from epichaos.utils import create_group_matrices


def epiCHAOS(
    counts: np.ndarray,
    metadata: pd.DataFrame,
    group_by: str,
    n: int = 50,
    m: int = 20,
    subsample: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute epiCHAOS heterogeneity scores per group.

    Parameters
    ----------
    counts:
        Feature × cell binary matrix; columns must match ``metadata`` index order.
    metadata:
        DataFrame indexed by cell ID; must contain ``group_by`` column.
    group_by:
        Column in ``metadata`` used to define groups.
    n:
        Maximum cells to subsample per group (default 50, matching R vignette).
    m:
        Minimum cells required to include a group.
    subsample:
        Number of pseudoreplicates (1 = no repetition, matching R default).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    DataFrame with columns: state, het_raw, het_adj.
    """
    group_matrices = create_group_matrices(
        counts, metadata, group_by, n=n, m=m, seed=seed
    )
    return compute_eITH(group_matrices, subsample=subsample, seed=seed)


def compute_eITH(
    group_matrices: dict[str, np.ndarray],
    subsample: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute epigenetic heterogeneity scores from per-group binary matrices.

    Implements the R compute_eITH algorithm:
    1. Remove zero-count cells per group.
    2. Compute mean pairwise centered Jaccard → raw het score.
    3. Regress raw scores on avg coverage → residuals.
    4. Normalize both raw scores and residuals to [0,1] then negate
       so higher = more heterogeneous.
    """
    rng = np.random.default_rng(seed)

    # ── Step 1: compute raw het per group ────────────────────────────────────
    raw_scores: dict[str, float] = {}
    avg_counts: dict[str, float] = {}
    clean_matrices: dict[str, np.ndarray] = {}

    for state, mat in group_matrices.items():
        mat = mat.astype(float)

        if subsample > 1:
            group_scores = []
            for rep in range(subsample):
                rep_seed = int(rng.integers(1_000_000_000))
                rep_rng = np.random.default_rng(rep_seed)
                idx = rep_rng.choice(mat.shape[1], size=min(mat.shape[1], mat.shape[1]), replace=False)
                sub = mat[:, idx]
                sub = sub[:, sub.sum(axis=0) > 0]
                if sub.shape[1] < 2:
                    continue
                group_scores.append(pairwise_jaccard_mean(sub.astype(bool)))
            if not group_scores:
                continue
            raw_scores[state] = float(np.mean(group_scores))
            avg_counts[state] = float(np.mean(mat.mean(axis=0)))
            clean_matrices[state] = mat
        else:
            sub = mat[:, mat.sum(axis=0) > 0]
            if sub.shape[1] < 2 or sub.sum() == 0:
                continue
            raw_scores[state] = pairwise_jaccard_mean(sub.astype(bool))
            avg_counts[state] = float(np.mean(sub.mean(axis=0)))
            clean_matrices[state] = sub

    if not raw_scores:
        return pd.DataFrame(columns=["state", "het_raw", "het_adj"])

    states = list(raw_scores.keys())
    het = np.array([raw_scores[s] for s in states])
    avg = np.array([avg_counts[s] for s in states])

    # ── Step 2: sparsity correction via linear regression ────────────────────
    if len(states) > 1:
        slope, intercept, *_ = stats.linregress(avg, het)
        residuals = het - (intercept + slope * avg)
    else:
        residuals = np.zeros(1)

    # ── Step 3: normalize + negate both scores ────────────────────────────────
    het_raw = _norm_negate(het)
    het_adj = _norm_negate(residuals)

    return pd.DataFrame({"state": states, "het_raw": het_raw, "het_adj": het_adj})


def transCHAOS(
    counts: np.ndarray,
    metadata: pd.DataFrame,
    group_by: str,
    n: int = 100,
    m: int = 20,
    subsample: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute transcriptional heterogeneity scores per group (scRNA-seq).

    Uses mean pairwise Euclidean distance between cells (after outlier removal),
    matching R transCHAOS / compute_tITH exactly.
    """
    # binarize=True matches R default: clips values >1 to 1, leaves negatives intact
    group_matrices = create_group_matrices(
        counts, metadata, group_by, n=n, m=m, binarize=True, seed=seed
    )
    return compute_tITH(group_matrices)


def compute_tITH(group_matrices: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute transcriptional ITH via mean Euclidean distance per group.

    For each group:
    1. Compute full pairwise Euclidean distance matrix between cells.
    2. Mean distance per cell (column means of distance matrix).
    3. Remove outlier cells: mean_dist > mean + 3*SD.
    4. Group score = mean of surviving per-cell mean distances.
    Then normalize scores to [0,1] (no negation — higher = more heterogeneous).
    """
    records: list[dict] = []

    for state, mat in group_matrices.items():
        cell_mat = mat.T  # → cells × features (for distance computation)

        dist_mat = squareform(pdist(cell_mat, metric="euclidean"))
        mean_dists = dist_mat.mean(axis=0)  # mean distance per cell

        # R uses sd() which is ddof=1 (sample std)
        bound = mean_dists.mean() + 3 * mean_dists.std(ddof=1)
        surviving = mean_dists[mean_dists < bound]

        if len(surviving) == 0:
            continue

        records.append({"state": state, "het": float(surviving.mean())})

    if not records:
        return pd.DataFrame(columns=["state", "het"])

    df = pd.DataFrame(records)
    mn, mx = df["het"].min(), df["het"].max()
    df["het"] = 0.0 if mx == mn else (df["het"] - mn) / (mx - mn)
    return df[["state", "het"]]


def _norm_negate(arr: np.ndarray) -> np.ndarray:
    """Normalize array to [0,1] then negate so higher = more heterogeneous."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=float)
    normed = (arr - mn) / (mx - mn)
    return 1.0 - normed
