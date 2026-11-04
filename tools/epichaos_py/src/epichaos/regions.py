"""Per-genomic-region epiCHAOS computation using pyranges."""

from __future__ import annotations

import numpy as np
import pandas as pd

from epichaos.jaccard import pairwise_jaccard_mean
from epichaos.core import _norm_negate


def compare_regions(
    counts: np.ndarray,
    region_names: list[str],
    region_sets: dict[str, "pr.PyRanges"],  # noqa: F821
    feature_ranges: "pr.PyRanges",  # noqa: F821
    m: int = 10,
) -> pd.DataFrame:
    """Compute epiCHAOS scores restricted to each genomic region set.

    Parameters
    ----------
    counts:
        Feature × cell binary matrix.
    region_names:
        Name for each feature row (genomic coordinate strings "chrN:start-end").
    region_sets:
        Mapping of region-set name → PyRanges object defining the region.
    feature_ranges:
        PyRanges object for all features in ``counts``.
    m:
        Minimum number of overlapping features required to score a region.

    Returns
    -------
    DataFrame with columns: state, het_raw, het_adj.
    """
    import pyranges as pr

    records = []
    for name, regions in region_sets.items():
        overlap = feature_ranges.overlap(regions)
        if len(overlap) < m:
            continue
        indices = _ranges_to_indices(overlap, region_names)
        sub = counts[indices, :]
        raw = pairwise_jaccard_mean(sub)
        sparsity = float(np.mean(sub > 0))
        records.append({"state": name, "het_raw": raw, "sparsity": sparsity})

    df = pd.DataFrame(records)
    if df.empty:
        return df
    from scipy import stats as _stats
    het = df["het_raw"].values
    avg = df["sparsity"].values
    if len(df) > 1:
        slope, intercept, *_ = _stats.linregress(avg, het)
        residuals = het - (intercept + slope * avg)
    else:
        residuals = np.zeros(1)
    df["het_raw"] = _norm_negate(het)
    df["het_adj"] = _norm_negate(residuals)
    return df[["state", "het_raw", "het_adj"]]


def parse_region_names(names: list[str]) -> "pr.PyRanges":
    """Parse a list of 'chr:start-end' strings into a PyRanges object."""
    import pyranges as pr

    chromosomes, starts, ends = [], [], []
    for name in names:
        # Support "chr1:1000-2000" and "chr1_1000_2000" formats
        name = name.replace("_", ":", 1).replace("_", "-", 1)
        chrom, coords = name.split(":")
        start, end = coords.split("-")
        chromosomes.append(chrom)
        starts.append(int(start))
        ends.append(int(end))

    return pr.PyRanges(
        pd.DataFrame({"Chromosome": chromosomes, "Start": starts, "End": ends})
    )


def _ranges_to_indices(overlap: "pr.PyRanges", region_names: list[str]) -> np.ndarray:
    """Return row indices in ``region_names`` corresponding to overlapping ranges."""
    name_to_idx = {n: i for i, n in enumerate(region_names)}
    hits = []
    df = overlap.as_df()
    for _, row in df.iterrows():
        key = f"{row['Chromosome']}:{row['Start']}-{row['End']}"
        if key in name_to_idx:
            hits.append(name_to_idx[key])
    return np.array(hits, dtype=int)
