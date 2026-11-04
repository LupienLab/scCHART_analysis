
"""Real-cell mixture benchmark (Figures 2c and S1g).

Constructs all 31 nonempty mixtures of five selected mature Granja
populations. Each mixture contains 300 cells sampled in equal proportions,
with 10 replicates.

Cells are matched to a common sequencing depth before scoring.
Structural and programme null calibrations use atlas-level projections
and cross-fitted references, rather than rebuilding them per mixture.
"""

import itertools
import os
import sys
import json
import hashlib
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import kruskal, spearmanr

from scchart._scoring import binarize, binomial_downsample, count_depth
from scchart.programme_entropy import (
    programme_null_calibration,
    programme_population_table,
    score_programmes,
)
from scchart.structural_divergence import (
    structural_J,
    structural_null_calibration,
)
from epichaos.core import compute_eITH
from epichaos.jaccard import pairwise_jaccard_mean


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "02_granja"))
sys.path.insert(0, str(ROOT / "scripts" / "05_method_comparisons"))

import granja
from epichaos_scores import het_raw_adj


import granja
from epichaos_scores import het_raw_adj


OUT = ROOT / "results" / "01_benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

N_JOBS = int(os.environ.get("N_JOBS", "20"))
N_TOTAL = 300
N_REP = 10
MIN_POOL = 750
BASE_SEED = 20260929

CLASS_H = {
    "11_CD14.Mono.1": "monocyte",
    "12_CD14.Mono.2": "monocyte",
    "09_pDC": "dendritic cell",
    "10_cDC": "dendritic cell",
    "17_B": "B",
    "18_Plasma": "B",
    "19_CD8.N": "T",
    "20_CD4.N1": "T",
    "21_CD4.N2": "T",
    "22_CD4.M": "T",
    "23_CD8.EM": "T",
    "24_CD8.CM": "T",
    "25_NK": "NK",
    "03_Late.Eryth": "erythroid",
}

SHORT = {
    "12_CD14.Mono.2": "Mono",
    "22_CD4.M": "CD4 T",
    "17_B": "B",
    "25_NK": "NK",
    "09_pDC": "pDC",
}

SCORES = [
    "Delta_S", "Delta_P", "epiCHAOS_het_adj",
    "J_S", "J_P", "E0_S", "E0_P", "epiCHAOS_het_raw",
    "J_S_all50", "J_S_natural", "J_P_natural",
    "epiCHAOS_het_adj_natural",
]


def eta2(values, groups):
    """Fraction of variance explained by group membership."""
    values = np.asarray(values, float)
    group_means = (
        pd.Series(values)
        .groupby(np.asarray(groups))
        .transform("mean")
        .to_numpy()
    )
    total = ((values - values.mean()) ** 2).sum()

    if total <= 0:
        return np.nan

    return float(
        ((group_means - values.mean()) ** 2).sum() / total
    )


def select_populations(groups, depth):
    """Apply the original mature-population selection rule."""
    populations = pd.DataFrame({
        "n_cells": pd.Series(groups).value_counts(),
        "median_count_depth": (
            pd.Series(depth).groupby(groups).median()
        ),
    })

    selection = populations.loc[list(CLASS_H)].assign(
        cell_class=pd.Series(CLASS_H)
    )
    selection["eligible_pool"] = selection.n_cells >= MIN_POOL

    chosen = (
        selection[selection.eligible_pool]
        .sort_values("n_cells", ascending=False)
        .groupby("cell_class")
        .head(1)
    )
    selection["selected"] = selection.index.isin(chosen.index)

    if set(chosen.index) != set(SHORT):
        raise RuntimeError(
            f"Unexpected populations: {sorted(chosen.index)}"
        )

    return populations, selection


def mixture_matrix(composition_id, positions, replicate, data):
    """Sample and depth-match one mixture."""
    rng = np.random.default_rng(
        (BASE_SEED, composition_id, replicate)
    )

    n_per = N_TOTAL // len(positions)
    indices = np.concatenate([
        rng.choice(data["pools"][t], n_per, replace=False)
        for t in positions
    ])

    natural = data["X"][indices]
    matched = binomial_downsample(
        natural,
        data["q"][data["g"][indices]],
        np.random.default_rng(
            (BASE_SEED, 1, composition_id, replicate)
        ),
    )

    keep = count_depth(matched) > 0

    return (
        indices[keep],
        natural[keep],
        matched[keep],
        int((~keep).sum()),
    )


def score_mixture(
    composition_id, positions, replicate, data,
    projection, projection_all, references,
):
    indices, natural, matched, removed = mixture_matrix(
        composition_id, positions, replicate, data
    )

    B = binarize(matched)
    B_natural = binarize(natural)
    folds = data["fold"][indices]

    result = {
        "n_cells": int(indices.size),
        "zero_open_removed": removed,
        "median_depth_after": float(
            np.median(count_depth(matched))
        ),
        "median_depth_natural": float(
            np.median(count_depth(natural))
        ),
    }

    structural = structural_null_calibration(
        B,
        projection,
        seed_key=(BASE_SEED, 2, composition_id, replicate),
        n_jobs=1,
    )

    result.update({
        "J_S": structural.J_obs,
        "E0_S": structural.E0_J,
        "Delta_S": structural.Delta,
        "sd0_S": structural.sd0,
        "median_d_S": float(np.median(structural.d_obs)),
        "margins_exact_S": structural.margins_exact_all,
        "chains_complete_S": structural.chains_complete_all,
    })

    result["J_S_all50"] = structural_J(
        projection_all(B)
    )["J"]

    parts = [
        (
            B[folds == f],
            references[f],
            10_000_000 + 100_000 * composition_id
            + 1000 * replicate + 100 * f,
        )
        for f in range(len(references))
        if (folds == f).any()
    ]
    programme = programme_null_calibration(parts)

    result.update({
        "J_P": programme.J_obs,
        "E0_P": programme.E0_J,
        "Delta_P": programme.Delta,
        "sd0_P": programme.sd0,
        "scored_fraction": programme.scored_fraction,
        "median_P_i": programme.median_P,
        "margins_exact_P": programme.margins_exact_all,
        "chains_complete_P": programme.chains_complete_all,
    })

    result["epichaos_mean_cJ"] = pairwise_jaccard_mean(
        B.T.tocsc()
    )
    result["epichaos_coverage"] = float(B.mean())

    result["J_S_natural"] = structural_J(
        projection(B_natural)
    )["J"]

    natural_composition = np.vstack([
        score_programmes(
            B_natural[folds == f], references[f]
        ).r
        for f in range(len(references))
        if (folds == f).any()
    ])

    valid = ~np.isnan(natural_composition).any(axis=1)
    result["J_P_natural"] = float(
        programme_population_table(
            natural_composition[valid],
            np.zeros(valid.sum()).astype(str),
        ).J_P_bits.iloc[0]
    )

    result["epichaos_mean_cJ_natural"] = (
        pairwise_jaccard_mean(B_natural.T.tocsc())
    )
    result["epichaos_coverage_natural"] = float(
        B_natural.mean()
    )

    return {
        "composition_id": composition_id,
        "replicate": replicate,
        **result,
    }, indices


def summarise_by_k(replicates):
    rows = []

    for score in SCORES:
        rho = spearmanr(replicates.k, replicates[score]).statistic
        r2_k = eta2(replicates[score], replicates.k)
        r2_composition = eta2(
            replicates[score], replicates.composition
        )

        for k, subset in replicates.groupby("k"):
            values = subset[score].to_numpy()
            means = subset.groupby("composition")[score].mean()

            half_width = (
                stats.t.ppf(0.975, len(values) - 1)
                * values.std(ddof=1)
                / np.sqrt(len(values))
            )

            p_value = (
                kruskal(*[
                    group[score].to_numpy()
                    for _, group in subset.groupby("composition")
                ]).pvalue
                if subset.composition.nunique() > 1
                else np.nan
            )

            rows.append({
                "score": score,
                "k": k,
                "n_compositions": subset.composition.nunique(),
                "n_replicates": len(subset),
                "mean": values.mean(),
                "sd": values.std(ddof=1),
                "ci95_lo": values.mean() - half_width,
                "ci95_hi": values.mean() + half_width,
                "median": np.median(values),
                "q25": np.quantile(values, 0.25),
                "q75": np.quantile(values, 0.75),
                "composition_mean_min": means.min(),
                "composition_mean_max": means.max(),
                "composition_mean_sd": (
                    means.std(ddof=1)
                    if len(means) > 1 else np.nan
                ),
                "eta2_composition_within_k": (
                    eta2(subset[score], subset.composition)
                    if len(means) > 1 else np.nan
                ),
                "kruskal_p_composition": p_value,
                "spearman_rho_k_all_replicates": rho,
                "R2_k": r2_k,
                "R2_composition": r2_composition,
            })

    return pd.DataFrame(rows)


def summarise_by_composition(replicates):
    rows = []

    groups = replicates.groupby(
        ["composition_id", "composition", "k"]
    )

    for (ci, composition, k), subset in groups:
        row = {
            "composition_id": ci,
            "composition": composition,
            "k": k,
            "types": subset.types.iloc[0],
            "n_replicates": len(subset),
        }

        for score in SCORES:
            values = subset[score].to_numpy()
            row.update({
                f"{score}_mean": values.mean(),
                f"{score}_sd": values.std(ddof=1),
                f"{score}_median": np.median(values),
            })

        row["median_depth_after_mean"] = (
            subset.median_depth_after.mean()
        )
        row["scored_fraction_min"] = subset.scored_fraction.min()
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["k", "composition_id"]
    )


def save_table(df, filename):
    path = OUT / filename
    compression = (
        {"method": "gzip", "mtime": 0}
        if filename.endswith(".gz") else None
    )

    df.to_csv(
        path,
        sep="\t",
        index=False,
        float_format="%.10g",
        compression=compression,
    )

    return {
        "file": filename,
        "rows": len(df),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    atlas = granja.load()
    X = atlas.X
    groups = atlas.obs[granja.GROUP_KEY].to_numpy()
    cell_names = atlas.obs_names.to_numpy()

    projection, projection_all, depth_table = (
        granja.representation(atlas)
    )
    references = granja.programme_model(atlas).references_
    folds = granja.fold_of_cells(references, X.shape[0])
    depth = count_depth(X)

    populations, selection = select_populations(groups, depth)

    types = list(SHORT)
    target_depth = float(
        populations.loc[types, "median_count_depth"].min()
    )
    thinning = {
        t: min(
            1.0,
            target_depth / populations.loc[t, "median_count_depth"],
        )
        for t in types
    }

    selected_indices = np.flatnonzero(np.isin(groups, types))
    selected_groups = groups[selected_indices]

    compositions = [
        combination
        for k in range(1, 6)
        for combination in itertools.combinations(types, k)
    ]
    labels = [
        "+".join(SHORT[t] for t in combination)
        for combination in compositions
    ]

    type_positions = {t: i for i, t in enumerate(types)}

    data = {
        "X": X[selected_indices].tocsr(),
        "g": np.array([
            type_positions[t] for t in selected_groups
        ]),
        "fold": folds[selected_indices],
        "pools": [
            np.flatnonzero(selected_groups == t)
            for t in types
        ],
        "q": np.array([thinning[t] for t in types]),
    }

    del atlas

    tasks = [
        (ci, [type_positions[t] for t in combination], rep)
        for ci, combination in enumerate(compositions)
        for rep in range(N_REP)
    ]

    scored = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(score_mixture)(
            ci, positions, rep, data,
            projection, projection_all, references
        )
        for ci, positions, rep in tasks
    )

    replicates = pd.DataFrame([row for row, _ in scored])
    replicates.insert(
        1, "composition",
        [labels[ci] for ci in replicates.composition_id],
    )
    replicates.insert(
        2, "k",
        [len(compositions[ci]) for ci in replicates.composition_id],
    )
    replicates.insert(
        3, "types",
        [
            ";".join(compositions[ci])
            for ci in replicates.composition_id
        ],
    )

    replicates["epiCHAOS_het_raw"], replicates[
        "epiCHAOS_het_adj"
    ] = het_raw_adj(
        replicates.epichaos_mean_cJ,
        replicates.epichaos_coverage,
    )

    replicates["epiCHAOS_het_raw_natural"], replicates[
        "epiCHAOS_het_adj_natural"
    ] = het_raw_adj(
        replicates.epichaos_mean_cJ_natural,
        replicates.epichaos_coverage_natural,
    )

    # Verify epiCHAOS against its direct implementation.
    check_indices = list(range(0, 310, 26))[:12]
    matrices = {}

    for j in check_indices:
        ci, positions, rep = tasks[j]
        indices, _, matched, _ = mixture_matrix(
            ci, positions, rep, data
        )

        assert np.array_equal(indices, scored[j][1])
        matrices[f"m{j}"] = binarize(matched).T.toarray()

    direct_scores = compute_eITH(matrices).set_index("state")
    raw, adjusted = het_raw_adj(
        replicates.epichaos_mean_cJ.iloc[check_indices],
        replicates.epichaos_coverage.iloc[check_indices],
    )

    difference = max(
        np.abs(
            direct_scores.loc[list(matrices), "het_raw"].to_numpy()
            - raw
        ).max(),
        np.abs(
            direct_scores.loc[list(matrices), "het_adj"].to_numpy()
            - adjusted
        ).max(),
    )

    cells = pd.DataFrame(
        [
            (j, cell_names[selected_indices[i]])
            for j, (_, indices) in enumerate(scored)
            for i in indices
        ],
        columns=["task", "cell"],
    )
    cells["composition_id"] = (
        replicates.composition_id.to_numpy()[cells.task]
    )
    cells["replicate"] = (
        replicates.replicate.to_numpy()[cells.task]
    )

    summary_k = summarise_by_k(replicates)
    summary_composition = summarise_by_composition(replicates)

    checks = {
        "310 mixtures scored": len(replicates) == 310,
        "Null margins preserved and chains complete": (
            replicates.margins_exact_S.all()
            and replicates.margins_exact_P.all()
            and replicates.chains_complete_S.all()
            and replicates.chains_complete_P.all()
        ),
        "Excess equals observed minus null": (
            np.allclose(
                replicates.Delta_S,
                replicates.J_S - replicates.E0_S,
            )
            and np.allclose(
                replicates.Delta_P,
                replicates.J_P - replicates.E0_P,
            )
        ),
        "All scores finite": (
            np.isfinite(replicates[SCORES].to_numpy()).all()
        ),
        "epiCHAOS direct comparison": difference < 1e-10,
    }

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        raise RuntimeError("Mixture benchmark validation failed.")

    outputs = [
        save_table(
            selection.reset_index(names="population"),
            "mixture_population_selection.tsv",
        ),
        save_table(replicates, "mixture_replicates.tsv"),
        save_table(summary_k, "mixture_summary_by_k.tsv"),
        save_table(
            summary_composition,
            "mixture_summary_by_composition.tsv",
        ),
        save_table(
            cells.drop(columns="task"),
            "mixture_cells.tsv.gz",
        ),
    ]

    metadata = {
        "analysis": "s04_real_cell_mixtures",
        "scchart_version": version("scchart"),
        "python_version": sys.version.split()[0],
        "parameters": {
            "n_total": N_TOTAL,
            "n_replicates": N_REP,
            "minimum_population_size": MIN_POOL,
            "base_seed": BASE_SEED,
            "target_depth": target_depth,
            "selected_populations": types,
            "n_jobs": N_JOBS,
        },
        "checks": {k: bool(v) for k, v in checks.items()},
        "outputs": outputs,
    }

    (OUT / "s04_real_cell_mixtures.run.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"\nCompleted: {len(replicates)} mixtures")
    print(f"Results: {OUT}")


if __name__ == "__main__":
    main()
