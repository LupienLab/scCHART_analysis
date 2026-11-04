"""Pairwise chance-corrected Jaccard index computation.

Formula matches jaccard::jaccard(center=TRUE) from R:
    centered = J - E[J]
where E[J] = px*py / (px + py - px*py) is the expected Jaccard under independence.

pairwise_jaccard_mean uses a fully vectorized matrix-multiply approach:
    - intersection via  M.T @ M  (BLAS-accelerated, sparse-aware)
    - union and E[J]   via broadcasting
    No Python-level pair loop — O(N²) pairs evaluated in one pass.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def jaccard_centered(a: np.ndarray, b: np.ndarray) -> float:
    """Centered Jaccard between two binary vectors; matches R jaccard::jaccard(center=TRUE).

    Kept for unit-testing and documentation. Production code uses pairwise_jaccard_mean.
    """
    n = len(a)
    intersection = int(np.sum(a & b))
    union = int(np.sum(a | b))

    px = float(np.sum(a)) / n
    py = float(np.sum(b)) / n
    expected_union = px + py - px * py
    e_j = (px * py) / expected_union if expected_union > 0 else 0.0

    if union == 0:
        return 0.0

    return float(intersection / union - e_j)


def pairwise_jaccard_mean(mat: np.ndarray | sp.spmatrix) -> float:
    """Mean centered Jaccard over all (N choose 2) cell pairs.

    Parameters
    ----------
    mat:
        Feature × cell binary matrix (dense ndarray or scipy sparse).
        Non-zero values are treated as 1.

    Returns
    -------
    Mean of J - E[J] across all unique cell pairs.

    Notes
    -----
    Algorithm (vectorized, no Python pair loop):
    1. Intersection matrix:  I = M_bin.T @ M_bin          shape (N, N)
    2. Union matrix:         U = s[:,None] + s[None,:] - I  (s = col sums)
    3. Jaccard matrix:       J = I / U   (0 where U==0)
    4. Expected J:           E = px*py / (px+py - px*py)   (px = s/F)
    5. Mean of upper triangle of (J - E).
    """
    F, N = mat.shape
    if N < 2:
        return 0.0

    # ── binarise and compute column sums ─────────────────────────────────────
    if sp.issparse(mat):
        mat_bin = mat.astype(np.float32).copy()
        mat_bin.data[:] = 1.0
        intersection = np.asarray((mat_bin.T @ mat_bin).todense(), dtype=np.float64)
        col_sums = np.asarray(mat_bin.sum(axis=0), dtype=np.float64).ravel()
    else:
        mat_bin = (mat != 0).astype(np.float32)
        intersection = (mat_bin.T @ mat_bin).astype(np.float64)
        col_sums = mat_bin.sum(axis=0).astype(np.float64)

    # ── union matrix ──────────────────────────────────────────────────────────
    union = col_sums[:, None] + col_sums[None, :] - intersection  # (N, N)

    # ── Jaccard and Expected Jaccard (suppress divide-by-zero for zero-coverage cells)
    px = col_sums / F                                   # (N,)
    px_outer = px[:, None] * px[None, :]               # (N, N)
    exp_union = px[:, None] + px[None, :] - px_outer   # (N, N)

    with np.errstate(invalid="ignore", divide="ignore"):
        j_mat = np.where(union > 0, intersection / union, 0.0)
        e_j_mat = np.where(exp_union > 0, px_outer / exp_union, 0.0)

    # ── mean of upper triangle ────────────────────────────────────────────────
    centered = j_mat - e_j_mat
    i_idx, j_idx = np.triu_indices(N, k=1)
    return float(centered[i_idx, j_idx].mean())
