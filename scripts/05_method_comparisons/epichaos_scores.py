"""epiCHAOS scores from per-group summary statistics.

``epichaos.core.compute_eITH`` computes, per group, the mean pairwise centred Jaccard (``het``) and the mean
coverage, then (step 2) regresses ``het`` on coverage across all groups and (step 3) min-max normalises and
negates the raw score and the residual.

* ``het_raw_adj``: steps 2-3 applied to precomputed per-group statistics (used where hundreds of large benchmark
  groups are compared; scripts check it against ``compute_eITH`` on a subset of groups).
* ``group_scores``: per-group statistics on random subsets of n cells (``reps`` subsets when a group has more than
  n cells), then steps 2-3 across the groups.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse, stats

from epichaos.jaccard import pairwise_jaccard_mean


def _norm_negate(a):
    a = np.asarray(a, float)
    return 1.0 - (a - a.min()) / (a.max() - a.min()) if a.max() > a.min() else a * 0


def het_raw_adj(mean_cj, coverage):
    het = np.asarray(mean_cj, float)
    avg = np.asarray(coverage, float)
    if het.size > 1:
        slope, intercept, *_ = stats.linregress(avg, het)
        resid = het - (intercept + slope * avg)
    else:
        resid = het * 0
    return _norm_negate(het), _norm_negate(resid)


def group_scores(groups: dict, rng, n: int = 100, reps: int = 5) -> pd.DataFrame:
    """groups: name -> cells x features matrix. Returns raw_similarity, coverage, het_raw, het_adj per group."""
    rows = []
    for g, M in groups.items():
        M = sparse.csr_matrix(M)
        M = M[np.diff(M.indptr) > 0]
        M.data[:] = 1.0
        vals, covs = [], []
        for _ in range(reps if M.shape[0] > n else 1):
            idx = rng.choice(M.shape[0], size=min(n, M.shape[0]), replace=False)
            sub = M[idx]
            vals.append(pairwise_jaccard_mean(sub.T.tocsc()))
            covs.append(float(np.asarray(sub.mean(1)).mean()))
        rows.append({"group": g, "raw_similarity": float(np.mean(vals)), "coverage": float(np.mean(covs))})
    d = pd.DataFrame(rows).set_index("group")
    d["het_raw"], d["het_adj"] = het_raw_adj(d.raw_similarity, d.coverage)
    return d


def jaccard_dissimilarity(B, rng, n: int = 100, reps: int = 5) -> float:
    """1 - mean pairwise chance-centred Jaccard over `reps` random subsets of `n` cells."""
    B = sparse.csr_matrix(B)
    vals = []
    for _ in range(reps if B.shape[0] > n else 1):
        idx = rng.choice(B.shape[0], size=min(n, B.shape[0]), replace=False)
        vals.append(pairwise_jaccard_mean(B[idx].T.tocsc()))
    return float(1.0 - np.mean(vals))
