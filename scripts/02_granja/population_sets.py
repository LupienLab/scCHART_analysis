"""Fixed-size population sets used by the cross-dataset and donor comparisons with epiCHAOS / Jaccard.

Datasets (fragment counts, per-dataset LSI with 50 components, all kept, random_state 1):
  * buenrostro: common/buenrostro.py analysis cell set;
  * granja: Granja atlas with fragments approximated as ceil(insertions / 2); populations with >= 100 cells;
    obs['donor'] = sample (obs['Group']).
Population sets: up to 500 cells per population, drawn once with default_rng((20261024, 1)) in sorted population
order.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import scchart
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "06_buenrostro"))
import buenrostro
from pathlib import Path
DATA = Path(__file__).resolve().parents[2] / "data"

SEED = 20261024
NMAX = 500


def load_granja_fragments(min_cells=100):
    a = ad.read_h5ad(DATA / "healthy_hematopoiesis_slim.h5ad")
    X = sparse.csr_matrix(a.X, dtype=np.float64)
    X.data = np.ceil(X.data / 2.0)
    obs = pd.DataFrame({"population": a.obs.BioClassification.astype(str).values,
                        "donor": a.obs.Group.astype(str).values}, index=a.obs_names)
    vc = obs.population.value_counts()
    keep = obs.population.isin(vc[vc >= min_cells].index).values
    X, obs = X[keep], obs[keep].copy()
    obs["n_frag"] = np.asarray(X.sum(1)).ravel()
    obs["n_open"] = np.diff(X.indptr)
    proj = scchart.LSIProjection.fit(X, n_components=50, random_state=1)
    return X, obs, proj


def load(name):
    """Returns (X, obs, proj, sel) with sel[population] = sorted indices of up to 500 cells."""
    X, obs, proj = buenrostro.load() if name == "buenrostro" else load_granja_fragments()
    rng = np.random.default_rng((SEED, 1))
    sel = {g: np.sort(rng.choice(idx, min(NMAX, idx.size), replace=False))
           for g, idx in obs.groupby("population").indices.items()}
    return X, obs, proj, sel
