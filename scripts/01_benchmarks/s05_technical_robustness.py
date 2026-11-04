
"""Technical robustness benchmarks (Figures S2a-c and S2j).

Evaluates structural and programme heterogeneity under:
1. Sequencing-depth downsampling of 1,000 GMP cells.
2. Population sizes ranging from 100 to 2,000 cells.
3. Matched-depth comparisons of two disjoint GMP cell sets.

Uses the Granja atlas, its LSI projection and cross-fitted
programme references. Scientific parameters and seeds follow
the validated RECOMB analyses.
"""

import os
import sys
import json
import hashlib
import warnings
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed

import scchart
from scchart._scoring import binarize, binomial_downsample, count_depth
from scchart.programme_entropy import programme_null_calibration
from scchart.structural_divergence import (
    structural_J,
    structural_null_calibration,
)


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

import granja


OUT = ROOT / "results" / "01_benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

N_JOBS = int(os.environ.get("N_JOBS", "20"))
BASE_SEED = 20260930

DEPTH_TARGETS = [4000, 2500, 1500, 1000, 600, 300]
DEPTH_REPLICATES = 5

CELL_NUMBERS = [100, 250, 500, 1000, 2000]
CELL_REPLICATES = 10

SHALLOW_FRACTION = 0.35
MATCHED_CELLS = 1000
MATCHED_REPLICATES = 5

VALUE_COLUMNS = [
    "median_depth", "median_n_open",
    "J_S", "E0_S", "Delta_S", "J_S_all50",
    "J_P", "E0_P", "Delta_P",
    "scored_fraction", "zero_open_removed",
]


def score_population(X, folds, seed, programme_seed, projection,
                     projection_all, references):
    """Compute calibrated structural and programme heterogeneity."""
    keep = count_depth(X) > 0
    X = X[keep]
    folds = folds[keep]

    B = binarize(X)

    structural = structural_null_calibration(
        B,
        projection,
        seed_key=(BASE_SEED, *seed),
        n_jobs=1,
    )

    parts = [
        (
            B[folds == f],
            references[f],
            30_000_000 + 1000 * programme_seed + 100 * f,
        )
        for f in range(5)
        if (folds == f).any()
    ]
    programme = programme_null_calibration(parts)

    return {
        "n_cells": int(keep.sum()),
        "zero_open_removed": int((~keep).sum()),
        "median_depth": float(np.median(count_depth(X))),
        "median_n_open": float(
            np.median(np.asarray(B.sum(axis=1)).ravel())
        ),
        "J_S": structural.J_obs,
        "E0_S": structural.E0_J,
        "Delta_S": structural.Delta,
        "sd0_S": structural.sd0,
        "J_S_all50": structural_J(projection_all(B))["J"],
        "margins_exact_S": structural.margins_exact_all,
        "chains_complete_S": structural.chains_complete_all,
        "J_P": programme.J_obs,
        "E0_P": programme.E0_J,
        "Delta_P": programme.Delta,
        "sd0_P": programme.sd0,
        "scored_fraction": programme.scored_fraction,
        "median_P_i": programme.median_P,
        "margins_exact_P": programme.margins_exact_all,
        "chains_complete_P": programme.chains_complete_all,
    }


def summarise(df, group_column):
    rows = []

    for group, subset in df.groupby(group_column):
        row = {
            group_column: group,
            "n_replicates": len(subset),
        }

        for column in VALUE_COLUMNS:
            values = subset[column].to_numpy()
            row[f"{column}_mean"] = values.mean()
            row[f"{column}_sd"] = (
                values.std(ddof=1)
                if len(values) > 1 else 0.0
            )

        rows.append(row)

    return pd.DataFrame(rows)


def save_table(df, filename):
    path = OUT / filename
    df.to_csv(path, sep="\t", index=False, float_format="%.10g")

    return {
        "file": filename,
        "rows": len(df),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    atlas = granja.load()
    X = atlas.X
    groups = atlas.obs[granja.GROUP_KEY].to_numpy()

    projection, projection_all, _ = granja.representation(atlas)
    references = granja.programme_model(atlas).references_
    folds = granja.fold_of_cells(references, X.shape[0])
    depth = count_depth(X)
    del atlas

    populations = pd.DataFrame({
        "n_cells": pd.Series(groups).value_counts(),
        "median_count_depth": (
            pd.Series(depth).groupby(groups).median()
        ),
    })

    candidates = (
        populations[populations.n_cells >= 2000]
        .sort_values("median_count_depth", ascending=False)
    )
    population = candidates.index[0]

    if population != "07_GMP":
        raise RuntimeError(
            f"Expected population 07_GMP, found {population}"
        )

    population_indices = np.flatnonzero(groups == population)

    def score(Xc, fold, seed, pseed):
        return score_population(
            Xc, fold, seed, pseed,
            projection, projection_all, references,
        )

    # D1: Sequencing depth
    depth_indices = np.sort(
        np.random.default_rng((BASE_SEED, 1)).choice(
            population_indices, 1000, replace=False
        )
    )
    X_depth = X[depth_indices].tocsr()
    folds_depth = folds[depth_indices]
    native_depth = float(np.median(count_depth(X_depth)))

    def depth_task(level, target, replicate):
        if target is None:
            thinned = X_depth
            fraction = 1.0
        else:
            fraction = target / native_depth
            thinned = binomial_downsample(
                X_depth,
                fraction,
                np.random.default_rng(
                    (BASE_SEED, 2, level, replicate)
                ),
            )

        return {
            "level": level,
            "target_depth": (
                target if target is not None else native_depth
            ),
            "q": fraction,
            "replicate": replicate,
            **score(
                thinned,
                folds_depth,
                (3, level, replicate),
                100 * level + replicate,
            ),
        }

    depth_tasks = [(0, None, 0)] + [
        (level + 1, target, rep)
        for level, target in enumerate(DEPTH_TARGETS)
        for rep in range(DEPTH_REPLICATES)
    ]

    depth_replicates = pd.DataFrame(
        Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(depth_task)(*task)
            for task in depth_tasks
        )
    )
    depth_summary = (
        summarise(depth_replicates, "target_depth")
        .sort_values("target_depth")
    )
    print("Depth series complete", flush=True)

    # D2: Population size
    def cell_number_task(level, n_cells, replicate):
        indices = np.sort(
            np.random.default_rng(
                (BASE_SEED, 4, level, replicate)
            ).choice(population_indices, n_cells, replace=False)
        )

        return {
            "N": n_cells,
            "replicate": replicate,
            "pool_fraction": n_cells / population_indices.size,
            **score(
                X[indices],
                folds[indices],
                (5, level, replicate),
                1000 + 100 * level + replicate,
            ),
        }

    cell_replicates = pd.DataFrame(
        Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(cell_number_task)(level, n_cells, rep)
            for level, n_cells in enumerate(CELL_NUMBERS)
            for rep in range(CELL_REPLICATES)
        )
    )
    cell_summary = (
        summarise(cell_replicates, "N").sort_values("N")
    )

    for column in ("J_S", "Delta_S", "J_P", "Delta_P"):
        baseline = cell_summary.loc[
            cell_summary.N == 2000, f"{column}_mean"
        ].iloc[0]

        cell_summary[f"{column}_rel_to_N2000"] = (
            cell_summary[f"{column}_mean"] / baseline
        )
        cell_summary[f"{column}_cv"] = (
            cell_summary[f"{column}_sd"]
            / cell_summary[f"{column}_mean"].abs()
        )

    print("Population-size series complete", flush=True)

    # D3: Matched-depth comparison
    permutation = np.random.default_rng(
        (BASE_SEED, 6)
    ).permutation(population_indices)

    indices_a = np.sort(permutation[:MATCHED_CELLS])
    indices_b = np.sort(
        permutation[MATCHED_CELLS:2 * MATCHED_CELLS]
    )

    X_b_shallow = binomial_downsample(
        X[indices_b],
        SHALLOW_FRACTION,
        np.random.default_rng((BASE_SEED, 7)),
    )

    remaining = np.setdiff1d(
        np.arange(X.shape[0]), indices_b
    )
    X_modified = sp.vstack([
        X[remaining], X_b_shallow
    ]).tocsr()

    order = np.r_[remaining, indices_b]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    X_modified = X_modified[inverse]

    unchanged_a = np.allclose(
        count_depth(X_modified[indices_a]),
        count_depth(X[indices_a]),
    )
    thinned_b = (
        count_depth(X_modified[indices_b])
        <= count_depth(X[indices_b])
    ).all()

    if not (unchanged_a and thinned_b):
        raise RuntimeError("Matched-depth input check failed.")

    labels = np.array(
        ["other"] * X.shape[0], dtype=object
    )
    pair_names = [
        "GMP a (native depth)",
        "GMP b (shallow copy, q = 0.35)",
    ]
    labels[indices_a], labels[indices_b] = pair_names

    observed = pd.DataFrame([
        {
            "population": name,
            **score(
                X_modified[indices],
                folds[indices],
                (8, j),
                2000 + j,
            ),
        }
        for j, (name, indices) in enumerate(
            zip(pair_names, [indices_a, indices_b])
        )
    ])

    target_depth = float(
        np.median(count_depth(X_modified[indices_b]))
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")

        matched_structural = (
            scchart.depth_standardized_structural_comparison(
                X_modified,
                labels,
                projection,
                target_depth,
                populations=pair_names,
                n_replicates=MATCHED_REPLICATES,
                seed=BASE_SEED,
                n_jobs=8,
            )
        )

        matched_programme = (
            scchart.depth_standardized_programme_comparison(
                X_modified,
                labels,
                references,
                target_depth,
                populations=pair_names,
                n_replicates=MATCHED_REPLICATES,
            )
        )

    warning_messages = sorted({
        str(w.message)[:200] for w in caught
    })

    comparison = pd.DataFrame({
        "population": pair_names,
        "J_S_observed": observed.J_S.to_numpy(),
        "Delta_S_observed": observed.Delta_S.to_numpy(),
        "J_P_observed": observed.J_P.to_numpy(),
        "Delta_P_observed": observed.Delta_P.to_numpy(),
        "Delta_S_Dstar_mean": (
            matched_structural.summary.Delta_G_mean
            .reindex(pair_names).to_numpy()
        ),
        "Delta_S_Dstar_sd": (
            matched_structural.summary.Delta_G_sd
            .reindex(pair_names).to_numpy()
        ),
        "Delta_P_Dstar_mean": (
            matched_programme.summary.Delta_P_mean
            .reindex(pair_names).to_numpy()
        ),
        "Delta_P_Dstar_sd": (
            matched_programme.summary.Delta_P_sd
            .reindex(pair_names).to_numpy()
        ),
        "q_to_Dstar": (
            matched_structural.summary.q
            .reindex(pair_names).to_numpy()
        ),
        "D_star": target_depth,
    })

    comparison_columns = [
        "J_S_observed", "Delta_S_observed",
        "J_P_observed", "Delta_P_observed",
        "Delta_S_Dstar_mean", "Delta_P_Dstar_mean",
    ]

    for column in comparison_columns:
        comparison[f"{column}_relative_gap_b_vs_a"] = (
            (comparison[column].iloc[1] - comparison[column].iloc[0])
            / abs(comparison[column].iloc[0])
        )

    all_scored = pd.concat(
        [depth_replicates, cell_replicates, observed],
        ignore_index=True,
    )

    numeric_comparison = comparison.drop(
        columns="population"
    )
    numeric_comparison = numeric_comparison.drop(
        columns=[
            c for c in numeric_comparison
            if c.endswith("_sd")
        ]
    )

    thinned_sd = comparison.loc[
        comparison.q_to_Dstar < 1,
        ["Delta_S_Dstar_sd", "Delta_P_Dstar_sd"],
    ]

    checks = {
        "Expected depth and cell-size replicate counts": (
            len(depth_replicates) == 31
            and len(cell_replicates) == 50
        ),
        "Null margins preserved and chains complete": (
            all_scored[
                [
                    "margins_exact_S", "margins_exact_P",
                    "chains_complete_S", "chains_complete_P",
                ]
            ].all().all()
        ),
        "Excess equals observed minus null": (
            np.allclose(
                all_scored.Delta_S,
                all_scored.J_S - all_scored.E0_S,
            )
            and np.allclose(
                all_scored.Delta_P,
                all_scored.J_P - all_scored.E0_P,
            )
        ),
        "Matched-depth means finite": (
            np.isfinite(
                numeric_comparison.to_numpy(float)
            ).all()
        ),
        "SD defined for thinned populations": (
            np.isfinite(thinned_sd.to_numpy()).all()
        ),
    }

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        raise RuntimeError("Technical robustness validation failed.")

    outputs = [
        save_table(
            depth_replicates,
            "robustness_depth_series_replicates.tsv",
        ),
        save_table(
            depth_summary,
            "robustness_depth_series_summary.tsv",
        ),
        save_table(
            cell_replicates,
            "robustness_cell_number_replicates.tsv",
        ),
        save_table(
            cell_summary,
            "robustness_cell_number_summary.tsv",
        ),
        save_table(
            observed,
            "robustness_matched_depth_observed.tsv",
        ),
        save_table(
            matched_structural.replicates,
            "robustness_matched_depth_structural_replicates.tsv",
        ),
        save_table(
            matched_programme.replicates,
            "robustness_matched_depth_programme_replicates.tsv",
        ),
        save_table(
            comparison,
            "robustness_matched_depth_comparison.tsv",
        ),
    ]

    metadata = {
        "analysis": "s05_technical_robustness",
        "scchart_version": version("scchart"),
        "python_version": sys.version.split()[0],
        "parameters": {
            "base_seed": BASE_SEED,
            "depth_targets": DEPTH_TARGETS,
            "depth_replicates": DEPTH_REPLICATES,
            "cell_numbers": CELL_NUMBERS,
            "cell_replicates": CELL_REPLICATES,
            "shallow_fraction": SHALLOW_FRACTION,
            "matched_cells": MATCHED_CELLS,
            "matched_replicates": MATCHED_REPLICATES,
            "n_jobs": N_JOBS,
        },
        "warnings": warning_messages,
        "checks": {k: bool(v) for k, v in checks.items()},
        "outputs": outputs,
    }

    (OUT / "s05_technical_robustness.run.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print("\nTechnical robustness benchmarks completed.")
    print(f"Results: {OUT}")


if __name__ == "__main__":
    main()
