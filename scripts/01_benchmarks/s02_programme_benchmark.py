
"""Programme heterogeneity benchmark (Figure 2b).

Tests how programme heterogeneity changes as individual cells become
increasingly specialised, while population-average programme usage
remains balanced.

600 cells, three programmes, five specialisation levels and
20 replicates per level. Uses an independent external reference
and the scCHART fixed-margin null calibration.
"""

from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats
from scipy.special import softmax

import scchart
import synthetic_model as sm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "01_benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

N_CELLS = 600
N_PROGRAMMES = 3
N_REP = 20
DP_GRID = [0.0, 0.5, 1.0, 2.0, 3.0]

PROGRAMMES = ["P1", "P2", "P3"]
FEATURE_LABELS = sm.feature_labels(PROGRAMMES)


def beta_offset(delta):
    """Match structural peak accessibility to the reference mean."""
    rng = np.random.default_rng(
        (sm.BASE_SEED_P, 4, sm.milli(delta))
    )
    Z, _ = sm.two_state_latent(delta, rng, 20000)
    return sm.beta_for(Z)


def programme_truth(delta_p, rng, n):
    """Generate balanced programme preferences with varying specialisation."""
    favoured = np.tile(
        np.arange(N_PROGRAMMES),
        int(np.ceil(n / N_PROGRAMMES)),
    )[:n]

    rng.shuffle(favoured)

    composition = softmax(
        delta_p * np.eye(N_PROGRAMMES)[favoured],
        axis=1,
    )
    return composition, favoured


def build_reference(beta):
    """Build the independent, labelled programme reference atlas."""
    rng = np.random.default_rng((sm.BASE_SEED_P, 5))
    matrices = []
    labels = []

    for k in range(N_PROGRAMMES):
        base = np.full(N_PROGRAMMES, 0.1 / N_PROGRAMMES)
        base[k] += 0.9

        composition = rng.dirichlet(30 * base, 400)
        Z, _ = sm.two_state_latent(0.0, rng, 400)

        matrices.append(sm.observe(Z, composition, beta, rng))
        labels.extend([PROGRAMMES[k]] * 400)

    return scchart.external_programme_reference(
        FEATURE_LABELS,
        sp.vstack(matrices).tocsr(),
        np.array(labels),
        programmes=PROGRAMMES,
    )


def summarise(df, group_columns, score_columns):
    """Summarise replicates with mean, SD, CI and quantiles."""
    rows = []

    for key, subset in df.groupby(group_columns):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_columns, key))
        row["n_replicates"] = len(subset)

        for column in score_columns:
            values = subset[column].to_numpy()
            mean = values.mean()
            sd = values.std(ddof=1)
            half_width = (
                stats.t.ppf(0.975, len(values) - 1)
                * sd / np.sqrt(len(values))
            )

            row.update({
                f"{column}_mean": mean,
                f"{column}_sd": sd,
                f"{column}_ci95_lo": mean - half_width,
                f"{column}_ci95_hi": mean + half_width,
                f"{column}_median": np.median(values),
                f"{column}_q25": np.quantile(values, 0.25),
                f"{column}_q75": np.quantile(values, 0.75),
            })

        rows.append(row)

    return pd.DataFrame(rows)


def run_benchmark(reference, beta):
    """Score programme heterogeneity across all specialisation levels."""
    rows = []

    for delta_p in DP_GRID:
        for rep in range(N_REP):
            seed = (sm.BASE_SEED_P, 10, sm.milli(delta_p), rep)
            rng = np.random.default_rng(seed)

            composition, favoured = programme_truth(
                delta_p, rng, N_CELLS
            )
            Z, _ = sm.two_state_latent(0.0, rng, N_CELLS)
            X = sm.observe(Z, composition, beta, rng)

            adata = ad.AnnData(X)
            adata.obs["G"] = "G"

            model = scchart.Heterogeneity(
                structural=False,
                calibrate=True,
                verbose=False,
            ).fit(
                adata,
                groupby="G",
                reference=reference,
            )

            score = model.programme_.iloc[0]
            estimated = adata.obsm["scchart_programme_composition"]
            scored = adata.obs["scchart_programme_scored"].to_numpy(bool)
            mean_composition = estimated[scored].mean(axis=0)

            rows.append({
                "delta_p": delta_p,
                "replicate": rep,
                "seed": str(seed),
                "n_cells": N_CELLS,
                "true_dominant_weight": float(
                    composition.max(axis=1).mean()
                ),
                "true_mean_pi": ";".join(
                    f"{x:.4f}" for x in composition.mean(axis=0)
                ),
                "scored_fraction": score.scored_fraction,
                "J_P_obs": score.J_P_obs,
                "E0_J_P": score.E0_J_P,
                "Delta_P": score.Delta_P,
                "sd0_diagnostic": score.sd0_diagnostic,
                "median_P": score.median_P,
                "mean_P": score.mean_P,
                "rbar_P1": mean_composition[0],
                "rbar_P2": mean_composition[1],
                "rbar_P3": mean_composition[2],
                "median_depth": float(
                    np.median(np.asarray(X.sum(axis=1)).ravel())
                ),
                "margins_exact_all_nulls": bool(
                    score.margins_exact_all_nulls
                ),
            })

        print(f"delta_p = {delta_p}: complete", flush=True)

    return pd.DataFrame(rows)


def validate(replicates):
    """Verify the expected replicates and null-model properties."""
    checks = {
        "Expected replicate count": (
            len(replicates) == len(DP_GRID) * N_REP
        ),
        "All nulls preserve margins": (
            replicates["margins_exact_all_nulls"].all()
        ),
        "Calibrated excess equals observed minus null": (
            np.allclose(
                replicates["Delta_P"],
                replicates["J_P_obs"] - replicates["E0_J_P"],
                atol=1e-12,
            )
        ),
        "All programme scores finite": (
            np.isfinite(
                replicates[
                    ["J_P_obs", "E0_J_P", "Delta_P", "median_P"]
                ].to_numpy()
            ).all()
        ),
    }

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        raise RuntimeError("Programme benchmark validation failed.")

    return {name: bool(value) for name, value in checks.items()}


def save_table(df, filename):
    """Write a TSV and record its checksum."""
    path = OUT / filename
    df.to_csv(path, sep="\t", index=False, float_format="%.10g")

    return {
        "file": filename,
        "rows": len(df),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    beta = beta_offset(0.0)
    reference = build_reference(beta)

    replicates = run_benchmark(reference, beta)

    summary = summarise(
        replicates,
        ["delta_p"],
        [
            "Delta_P", "J_P_obs", "E0_J_P", "median_P",
            "rbar_P1", "rbar_P2", "rbar_P3",
        ],
    )

    for column in ("Delta_P", "median_P"):
        result = stats.spearmanr(
            replicates["delta_p"],
            replicates[column],
        )
        summary[f"spearman_rho_{column}"] = result.statistic
        summary[f"spearman_p_{column}"] = result.pvalue

    checks = validate(replicates)

    outputs = [
        save_table(replicates, "programme_benchmark_replicates.tsv"),
        save_table(summary, "programme_benchmark_summary.tsv"),
    ]

    metadata = {
        "analysis": "s02_programme_benchmark",
        "scchart_version": version("scchart"),
        "python_version": sys.version.split()[0],
        "parameters": {
            "n_cells": N_CELLS,
            "n_programmes": N_PROGRAMMES,
            "n_replicates": N_REP,
            "specialisation_levels": DP_GRID,
            "base_seed": sm.BASE_SEED_P,
            "reference_seed": [sm.BASE_SEED_P, 5],
            "calibrate": True,
            "structural": False,
            "null_settings": "scchart package defaults",
        },
        "checks": checks,
        "outputs": outputs,
    }

    (OUT / "s02_programme_benchmark.run.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"\nCompleted: {len(replicates)} simulations")
    print(f"Results: {OUT}")


if __name__ == "__main__":
    main()
