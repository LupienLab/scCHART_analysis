
"""Granja et al. (2019) healthy haematopoiesis atlas.

Input
-----
data/healthy_hematopoiesis_slim.h5ad
    35,038 cells x 50,000 peaks, with accessibility counts in X,
    population labels in obs["BioClassification"], and the published
    LSI embedding in obsm["X_lsi"].

Structural representation
-------------------------
Fit 50 LSI components aligned to the published embedding (seed=1).
Exclude components with absolute Pearson correlation to sequencing
depth greater than 0.30.

Programme representation
------------------------
Fit endogenous programme references using five-fold cross-fitting
(random_state=0) across the atlas populations.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr

import scchart
from scchart._scoring import count_depth


DATA = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "healthy_hematopoiesis_slim.h5ad"
)

GROUP_KEY = "BioClassification"
DEPTH_RULE = 0.30


def load():
    """Load the Granja atlas and prepare its count matrix and labels."""
    a = ad.read_h5ad(DATA)
    a.X = sparse.csr_matrix(a.X)
    a.X.sort_indices()
    a.obs[GROUP_KEY] = a.obs[GROUP_KEY].astype(str)
    return a


def representation(a):
    """Build depth-filtered and full LSI projections.

    Returns
    -------
    proj
        Projection retaining components with |r_depth| <= 0.30.
    full_proj
        Projection retaining all 50 components.
    tab
        Per-component depth correlations and exclusion decisions.
    """
    X = a.X

    full = scchart.LSIProjection.fit(
        X,
        n_components=50,
        align_to=np.asarray(a.obsm["X_lsi"], float),
        random_state=1,
    )

    Z = full(X)
    depth = count_depth(X)

    tab = pd.DataFrame({
        "component": np.arange(1, 51),
        "pearson_r_depth": [
            pearsonr(Z[:, k], depth).statistic
            for k in range(50)
        ],
    })

    tab["excluded"] = tab["pearson_r_depth"].abs() > DEPTH_RULE

    proj = full.with_components(
        np.flatnonzero(~tab["excluded"].to_numpy())
    )

    full_proj = full.with_components(np.arange(50))

    return proj, full_proj, tab


def programme_model(a):
    """Fit five-fold cross-fitted programme references on the atlas."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        model = scchart.ProgrammeHeterogeneity(
            n_folds=5,
            random_state=0,
            verbose=False,
        )
        model.fit(a, groupby=GROUP_KEY)

    return model


def fold_of_cells(refs, n_cells):
    """Return the reference fold assigned to each cell."""
    fold = np.full(n_cells, -1)

    for f, ref in enumerate(refs):
        fold[ref.score_index] = f

    assert (fold >= 0).all()

    return fold
