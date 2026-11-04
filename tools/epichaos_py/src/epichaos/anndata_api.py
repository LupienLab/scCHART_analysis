"""AnnData-first API wrappers (scanpy-style).

Primary entry points for Python single-cell workflows. Results are written
into adata.obs (per-cell columns) and adata.uns (group-level summary).

AnnData convention: obs = cells, var = features.
The internal NumPy functions expect features × cells, so adata.X is transposed
before being passed down.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from epichaos.core import compute_eITH, compute_tITH
from epichaos.utils import create_group_matrices


def epiCHAOS(
    adata: AnnData,
    group_by: str,
    n: int = 50,
    m: int = 20,
    subsample: int = 1,
    seed: int = 42,
    key_added: str = "epiCHAOS",
    copy: bool = False,
) -> AnnData | None:
    """Compute epiCHAOS heterogeneity scores and store them in adata.

    Parameters
    ----------
    adata:
        AnnData object (obs = cells, var = features). adata.X should be a
        binary or binarizable counts matrix (e.g. peak accessibility).
    group_by:
        Column in adata.obs defining cell groups (cluster, cell type, etc.).
    n:
        Maximum cells to subsample per group (default 50).
    m:
        Minimum cells required to include a group.
    subsample:
        Number of pseudoreplicates (1 = compute once per group).
    seed:
        Random seed.
    key_added:
        Prefix for new columns in adata.obs and key in adata.uns.
    copy:
        If True, return a modified copy; otherwise modify in place and return None.

    Writes
    ------
    adata.obs[f"{key_added}_het_raw"]:
        Normalized raw heterogeneity score per cell (same value for all cells in group).
    adata.obs[f"{key_added}_het_adj"]:
        Sparsity-adjusted heterogeneity score per cell.
    adata.uns[key_added]:
        Group-level scores DataFrame with columns: state, het_raw, het_adj.
    """
    adata = adata.copy() if copy else adata

    X = np.array(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)
    counts = X.T  # → features × cells

    group_matrices = create_group_matrices(
        counts, adata.obs, group_by, n=n, m=m, seed=seed
    )
    scores = compute_eITH(group_matrices, subsample=subsample, seed=seed)

    adata.uns[key_added] = scores

    # Map group-level scores back to individual cells
    # R state names have "group-" prefix; strip it for matching adata.obs values
    state_to_raw = dict(zip(scores["state"].str.removeprefix("group-"), scores["het_raw"]))
    state_to_adj = dict(zip(scores["state"].str.removeprefix("group-"), scores["het_adj"]))

    adata.obs[f"{key_added}_het_raw"] = adata.obs[group_by].map(state_to_raw)
    adata.obs[f"{key_added}_het_adj"] = adata.obs[group_by].map(state_to_adj)

    return adata if copy else None


def transCHAOS(
    adata: AnnData,
    group_by: str,
    n: int = 100,
    m: int = 20,
    subsample: int = 1,
    seed: int = 42,
    key_added: str = "transCHAOS",
    copy: bool = False,
) -> AnnData | None:
    """Compute transcriptional heterogeneity scores and store them in adata.

    Parameters
    ----------
    adata:
        AnnData object with scaled/normalized expression in adata.X.
    group_by:
        Column in adata.obs defining cell groups.
    n, m, subsample, seed, key_added, copy:
        Same as :func:`epiCHAOS`.

    Writes
    ------
    adata.obs[f"{key_added}_het"]:
        Normalized transcriptional heterogeneity score per cell.
    adata.uns[key_added]:
        Group-level scores DataFrame with columns: state, het.
    """
    adata = adata.copy() if copy else adata

    X = np.array(adata.X.todense() if hasattr(adata.X, "todense") else adata.X)
    counts = X.T  # → features × cells

    # binarize=True matches R default: clips values >1 to 1, leaves negatives intact
    group_matrices = create_group_matrices(
        counts, adata.obs, group_by, n=n, m=m, binarize=True, seed=seed
    )
    scores = compute_tITH(group_matrices)

    adata.uns[key_added] = scores

    state_to_het = dict(zip(scores["state"].str.removeprefix("group-"), scores["het"]))
    adata.obs[f"{key_added}_het"] = adata.obs[group_by].map(state_to_het)

    return adata if copy else None
