
"""Structural heterogeneity benchmarks (Figure 2a and Figure S1a).

Two synthetic experiments test whether observed structural heterogeneity
responds to:

1. Increasing continuous dispersion within a population.
2. Increasing separation between two equally represented cell states.

Each condition contains 500 cells in a 10-dimensional latent space,
with 20 independent replicates. No null calibration is applied.

Outputs are saved to results/01_benchmarks/.
"""

from pathlib import Path
from importlib.metadata import version
import hashlib
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scchart import StructuralHeterogeneity
import synthetic_model as sm


# Configuration
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "01_benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

N_CELLS = 500
K_DIM = 10
N_REP = 20
BASE_SEED = 20260925

H_GRID = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
DELTA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]

SIGMA_2S = 0.05
FRAC_B = 0.5

BENCH_ID = {
    "continuous_dispersion": 1,
    "two_state": 2,
}


def seed_for(benchmark, level, replicate):
    """Return the fixed seed used for one simulation replicate."""
    return (
        BASE_SEED,
        BENCH_ID[benchmark],
        sm.milli(level),
        replicate,
    )


def generate_continuous(h, rng):
    """Generate cells with increasing dispersion around a common centre."""
    return (
        sm.MU[None, :]
        + h * rng.standard_normal((N_CELLS, K_DIM)) / np.sqrt(K_DIM)
    )


def generate_two_state(delta, rng):
    """Generate two equally represented states separated by delta."""
    return sm.two_state_latent(
        delta, rng, N_CELLS, frac_b=FRAC_B, sigma=SIGMA_2S
    )


def score_structural(Z):
    """Calculate population heterogeneity and per-cell contributions."""
    adata = ad.AnnData(np.zeros((Z.shape[0], 1), dtype=np.float32))
    adata.obsm["Z"] = Z
    adata.obs["G"] = "G"

    model = StructuralHeterogeneity(
        adapter="signed_channel",
        store_probabilities=False,
        verbose=False,
    )
    scores = model.fit(adata, groupby="G", use_rep="Z")

    per_cell = scores["scchart_structural_d"].to_numpy()
    J_obs = float(model.population_summary_["J_obs"].iloc[0])

    return per_cell, J_obs


def run_benchmarks():
    """Score all simulated conditions and retain per-cell results."""
    replicate_rows = []
    cell_tables = []

    experiments = [
        ("continuous_dispersion", H_GRID),
        ("two_state", DELTA_GRID),
    ]

    for benchmark, levels in experiments:
        for level in levels:
            for rep in range(N_REP):
                rng = np.random.default_rng(
                    seed_for(benchmark, level, rep)
                )

                if benchmark == "continuous_dispersion":
                    Z = generate_continuous(level, rng)
                    states = None
                else:
                    Z, states = generate_two_state(level, rng)

                per_cell, J_obs = score_structural(Z)

                row = {
                    "benchmark": benchmark,
                    "level": level,
                    "replicate": rep,
                    "J_obs": J_obs,
                    "mean_scchart_structural_d": per_cell.mean(),
                    "median_scchart_structural_d": np.median(per_cell),
                    "abs_J_minus_mean_d": abs(J_obs - per_cell.mean()),
                }

                if states is not None:
                    row["mean_d_state_A"] = per_cell[states == "A"].mean()
                    row["mean_d_state_B"] = per_cell[states == "B"].mean()

                replicate_rows.append(row)

                cell_tables.append(pd.DataFrame({
                    "benchmark": benchmark,
                    "level": level,
                    "replicate": rep,
                    "scchart_structural_d": per_cell,
                    "true_state": states if states is not None else "",
                }))

    return (
        pd.DataFrame(replicate_rows),
        pd.concat(cell_tables, ignore_index=True),
    )


def generate_views():
    """Create 2D display coordinates for selected simulation conditions."""
    w = np.where(np.arange(K_DIM) % 2 == 0, 1.0, -1.0)
    w = w - (w @ sm.MU) * sm.MU - (w @ sm.V) * sm.V
    W = w / np.linalg.norm(w)

    views = []

    conditions = [
        ("continuous_dispersion", 0.05),
        ("continuous_dispersion", 0.8),
        ("two_state", 0.0),
        ("two_state", 3.0),
    ]

    for benchmark, level in conditions:
        rng = np.random.default_rng(seed_for(benchmark, level, 0))

        if benchmark == "continuous_dispersion":
            Z = generate_continuous(level, rng)
            states = None
        else:
            Z, states = generate_two_state(level, rng)

        views.append(pd.DataFrame({
            "benchmark": benchmark,
            "level": level,
            "replicate": 0,
            "axis_state": (Z - sm.MU) @ sm.V,
            "axis_orth": (Z - sm.MU) @ W,
            "true_state": states if states is not None else "",
        }))

    return pd.concat(views, ignore_index=True)


def summarize(replicates):
    """Summarize structural heterogeneity across replicate conditions."""
    summary = (
        replicates.groupby(["benchmark", "level"])
        .agg(
            n_replicates=("J_obs", "size"),
            J_obs_mean=("J_obs", "mean"),
            J_obs_sd=("J_obs", lambda x: x.std(ddof=1)),
            median_scchart_structural_d_mean=(
                "median_scchart_structural_d", "mean"
            ),
        )
        .reset_index()
    )

    for benchmark in summary["benchmark"].unique():
        sub = replicates[replicates["benchmark"] == benchmark]
        result = spearmanr(sub["level"], sub["J_obs"])

        mask = summary["benchmark"] == benchmark
        summary.loc[mask, "spearman_rho_level_J_obs"] = result.statistic
        summary.loc[mask, "spearman_p_level_J_obs"] = result.pvalue

    return summary


def validate(replicates, cells):
    """Check the decomposition identity and simulation integrity."""
    checks = {
        "Population J equals mean per-cell contribution": (
            replicates["abs_J_minus_mean_d"].max() < 1e-10
        ),
        "Expected replicate count": (
            len(replicates) == (len(H_GRID) + len(DELTA_GRID)) * N_REP
        ),
        "Finite, nonnegative heterogeneity": (
            np.isfinite(replicates["J_obs"]).all()
            and (replicates["J_obs"] >= -1e-12).all()
        ),
        "Two-state heterogeneity <= 1 bit": (
            (replicates.loc[
                replicates["benchmark"] == "two_state", "J_obs"
            ] <= 1 + 1e-9).all()
        ),
        "Expected number of per-cell rows": (
            len(cells) == len(replicates) * N_CELLS
        ),
    }

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        raise RuntimeError("Structural benchmark validation failed.")

    return {name: bool(value) for name, value in checks.items()}


def save_table(df, filename):
    """Write a result table using the original output precision."""
    path = OUT / filename

    compression = (
        {"method": "gzip", "mtime": 0}
        if filename.endswith(".gz")
        else None
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
    replicates, cells = run_benchmarks()
    views = generate_views()
    summary = summarize(replicates)

    checks = validate(replicates, cells)

    outputs = [
        save_table(views, "structural_benchmark_views.tsv"),
        save_table(replicates, "structural_benchmark_replicates.tsv"),
        save_table(summary, "structural_benchmark_summary.tsv"),
        save_table(cells, "structural_benchmark_per_cell.tsv.gz"),
    ]

    metadata = {
        "analysis": "s01_structural_benchmarks",
        "scchart_version": version("scchart"),
        "python_version": sys.version.split()[0],
        "parameters": {
            "n_cells": N_CELLS,
            "n_dimensions": K_DIM,
            "n_replicates": N_REP,
            "base_seed": BASE_SEED,
            "benchmark_ids": BENCH_ID,
            "dispersion_levels": H_GRID,
            "separation_levels": DELTA_GRID,
            "two_state_sigma": SIGMA_2S,
            "fraction_state_B": FRAC_B,
            "adapter": "signed_channel",
            "null_calibration": False,
        },
        "checks": checks,
        "outputs": outputs,
    }

    (OUT / "s01_structural_benchmarks.run.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"\nCompleted: {len(replicates)} simulations")
    print(f"Results: {OUT}")


if __name__ == "__main__":
    main()
