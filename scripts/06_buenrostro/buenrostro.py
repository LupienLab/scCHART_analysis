"""Buenrostro et al. (2018) FACS-sorted haematopoietic scATAC-seq: analysis cell set and representation.

Input: processed ``buenrostro2018.h5ad`` (GEO GSE96769 counts in peaks; obs: facs_population, donor, plate,
n_fragments_in_peaks).
Analysis cell set: cells with >= 1,000 fragments in peaks and at most the 99.5th percentile; unsorted (UNK)
cells excluded; populations with >= 80 cells; feature universe = peaks accessible in >= 2% of retained cells.
Representation: ``scchart.LSIProjection.fit(X, n_components=50, random_state=1)``, all components kept.
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

import scchart
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data" / "buenrostro2018.h5ad"

QC = dict(min_frag=1000, max_frag_quantile=0.995, min_cell_frac=0.02, min_cells=80, n_components=50, random_state=1)


def load(with_peaks: bool = False):
    """Returns (X counts, obs with population/donor/plate, LSI projection fitted on all retained cells)
    and, with ``with_peaks``, the retained peak names (chrom_start_end, hg19)."""
    a = ad.read_h5ad(DATA)
    o = a.obs
    hi = np.quantile(o.n_fragments_in_peaks, QC["max_frag_quantile"])
    keep = (o.n_fragments_in_peaks >= QC["min_frag"]) & (o.n_fragments_in_peaks <= hi) & (o.facs_population != "UNK")
    a = a[keep.values].copy()
    vc = a.obs.facs_population.value_counts()
    a = a[a.obs.facs_population.isin(vc[vc >= QC["min_cells"]].index).values].copy()
    X = sparse.csr_matrix(a.X)
    frac = np.asarray((X > 0).mean(0)).ravel()
    fk = frac >= QC["min_cell_frac"]
    X = sparse.csr_matrix(X[:, fk], dtype=np.float64)
    X.sort_indices()
    v = a.var.loc[fk]
    peaks = (v.chrom.astype(str) + "_" + v.start.astype(np.int64).astype(str) + "_" + v.end.astype(np.int64).astype(str)).to_numpy()
    obs = pd.DataFrame({"population": a.obs.facs_population.astype(str).values,
                        "donor": a.obs.donor.astype(str).values, "plate": a.obs.plate.astype(str).values},
                       index=a.obs_names)
    obs["n_frag"] = np.asarray(X.sum(1)).ravel()
    obs["n_open"] = np.diff(X.indptr)
    proj = scchart.LSIProjection.fit(X, n_components=QC["n_components"], random_state=QC["random_state"])
    return (X, obs, proj, peaks) if with_peaks else (X, obs, proj)
