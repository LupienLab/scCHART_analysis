"""Utility functions for building per-group matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_group_matrices(
    counts: np.ndarray,
    metadata: pd.DataFrame,
    group_by: str,
    n: int = 50,
    m: int = 20,
    binarize: bool = True,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Subset a feature × cell counts matrix into per-group submatrices.

    Matches R create.group.matrices exactly:
    - Groups with < m cells are skipped.
    - Groups with > n cells are randomly downsampled to n.
    - Values > 1 are clipped to 1 when binarize=True.
    - Group keys are prefixed with "group-" (matching R convention).

    Parameters
    ----------
    counts:
        Feature × cell matrix. Column order must match ``metadata`` row order.
    metadata:
        DataFrame indexed by cell ID; must contain ``group_by`` column.
    group_by:
        Column in ``metadata`` defining group membership.
    n:
        Downsample each group to at most ``n`` cells.
    m:
        Skip groups with fewer than ``m`` cells.
    binarize:
        Clip values > 1 to 1 (for epigenomics data).
    seed:
        Random seed for downsampling.

    Returns
    -------
    Dict mapping "group-<label>" → feature × cell submatrix.
    """
    rng = np.random.default_rng(seed)

    if counts.shape[1] != len(metadata):
        raise ValueError(
            f"counts has {counts.shape[1]} columns but metadata has {len(metadata)} rows"
        )

    cell_ids = list(metadata.index)
    id_to_col = {cid: i for i, cid in enumerate(cell_ids)}

    result: dict[str, np.ndarray] = {}

    for group, sub_meta in metadata.groupby(group_by, observed=True):
        ids = list(sub_meta.index)
        if len(ids) < m:
            continue

        if len(ids) > n:
            chosen_ids = list(rng.choice(ids, size=n, replace=False))
        else:
            chosen_ids = ids

        col_idx = np.array([id_to_col[cid] for cid in chosen_ids])
        sub = counts[:, col_idx].copy().astype(float)

        if binarize:
            # Match R: clip only values > 1 to 1; leave negatives intact
            sub = np.where(sub > 1, 1.0, sub)

        result[f"group-{group}"] = sub

    return result
