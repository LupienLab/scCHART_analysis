
"""Minority-substate benchmark (Figures S1b-e and S3e-f).

Tests how structural and programme heterogeneity respond to
minority-state abundance and separation.

600 cells per population, seven minority fractions, five separation
levels and 20 replicates: 700 simulated populations.

Uses fixed-margin null calibration for programme heterogeneity
and an independently fitted LSI projection for end-to-end structural
calibration. The latent structural score is reported separately.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from importlib.metadata import version

import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

import scchart
from scchart.structural_divergence import structural_null_calibration
from epichaos.jaccard import pairwise_jaccard_mean

import synthetic_model as sm


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "01_benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

N_JOBS = int(os.environ.get("N_JOBS", "20"))
N_CELLS = 600
N_REP = 20
BASE_SEED = 20260929

F_GRID = [0.0, 0.05, 0.10, 0.20, 0.25, 0.30, 0.50]
SEP_GRID = [0.0, 0.25, 0.50, 0.75, 1.0]
SEP_NAME = {
    0.0: "none",
    0.25: "low",
    0.5: "intermediate",
    0.75: "high",
    1.0: "maximal",
}

PROGRAMMES = ["R_A", "R_B", "R_common"]

SCORE_COLUMNS = [
    "J_S", "J_P", "Delta_P", "E0_P",
    "J_S_e2e", "E0_S_e2e", "Delta_S_e2e",
    "epichaos_mean_cJ", "epichaos_coverage", "median_P_i",
]


def generate_states(fraction, separation, rng, n=N_CELLS):
    """Generate majority and minority positions on the state axis."""
    n_b = int(round(fraction * n))
    n_a = n - n_b
    minority_centre = -1.0 + 2.0 * separation

    positions = np.concatenate([
        rng.normal(-1.0, sm.TAU, n_a),
        rng.normal(minority_centre, sm.TAU, n_b),
    ])
    labels = np.array(["A"] * n_a + ["B"] * n_b, dtype=object)

    order = rng.permutation(n)
    return positions[order], labels[order]


def build_offsets():
    """Match mean structural accessibility across conditions."""
    offsets = {}

    for fi, fraction in enumerate(F_GRID):
        for si, separation in enumerate(SEP_GRID):
            rng = np.random.default_rng((BASE_SEED, 90, fi, si))
            positions, _ = generate_states(
                fraction, separation, rng, 20000
            )
            Z = sm.state_axis_Z(positions, rng)
            offsets[(fraction, separation)] = sm.beta_for(Z)

    return offsets


def build_projection():
    """Fit the structural LSI projection on independent reference cells."""
    rng = np.random.default_rng((BASE_SEED, 91))
    positions = rng.uniform(-1, 1, 3000)

    Z = sm.state_axis_Z(positions, rng)
    X = sm.observe(
        Z,
        sm.state_axis_activity(positions),
        sm.beta_for(Z),
        rng,
    )

    return scchart.LSIProjection.fit(
        X, n_components=10, random_state=1
    )


def score_condition(fi, si, replicate, offsets, reference, projection):
    fraction = F_GRID[fi]
    separation = SEP_GRID[si]

    seed = (BASE_SEED, 100, fi, si, replicate)
    rng = np.random.default_rng(seed)

    positions, labels = generate_states(fraction, separation, rng)
    Z = sm.state_axis_Z(positions, rng)
    X = sm.observe(
        Z,
        sm.state_axis_activity(positions),
        offsets[(fraction, separation)],
        rng,
    )

    keep = np.asarray(X.sum(axis=1)).ravel() > 0

    adata = ad.AnnData(X)
    adata.obs["G"] = "G"
    adata.obsm["Z"] = Z

    model = scchart.Heterogeneity(
        use_rep="Z",
        calibrate=True,
        store_probabilities=False,
        verbose=False,
        programme_seed=(
            10_000_000
            + 1_000_000 * fi
            + 10_000 * si
            + 100 * replicate
        ),
    ).fit(adata, groupby="G", reference=reference)

    structural = model.structural_.iloc[0]
    programme = model.programme_.iloc[0]

    end_to_end = structural_null_calibration(
        X[keep],
        projection,
        seed_key=(BASE_SEED, 101, fi, si, replicate),
    )

    return {
        "minority_fraction": fraction,
        "separation": separation,
        "separation_name": SEP_NAME[separation],
        "replicate": replicate,
        "seed": str(seed),
        "n_cells": N_CELLS,
        "observed_minority_fraction": float((labels == "B").mean()),
        "true_sd_s": float(positions.std()),
        "J_S": structural.J_obs,
        "J_P": programme.J_P_obs,
        "E0_P": programme.E0_J_P,
        "Delta_P": programme.Delta_P,
        "sd0_P": programme.sd0_diagnostic,
        "median_P_i": programme.median_P,
        "scored_fraction": programme.scored_fraction,
        "margins_exact_P": bool(programme.margins_exact_all_nulls),
        "J_S_e2e": end_to_end.J_obs,
        "E0_S_e2e": end_to_end.E0_J,
        "Delta_S_e2e": end_to_end.Delta,
        "margins_exact_S_e2e": end_to_end.margins_exact_all,
        "chains_complete_S_e2e": end_to_end.chains_complete_all,
        "zero_open_cells": int((~keep).sum()),
        "epichaos_mean_cJ": pairwise_jaccard_mean(X.T.tocsc()),
        "epichaos_coverage": float(X.mean()),
        "median_count_depth": float(
            np.median(np.asarray(X.sum(axis=1)).ravel())
        ),
    }


def summarise(replicates):
    rows = []

    for (fraction, separation), sub in replicates.groupby(
        ["minority_fraction", "separation"]
    ):
        row = {
            "minority_fraction": fraction,
            "separation": separation,
            "n_replicates": len(sub),
        }

        for column in SCORE_COLUMNS:
            values = sub[column].to_numpy()
            row.update({
                f"{column}_mean": values.mean(),
                f"{column}_sd": values.std(ddof=1),
                f"{column}_median": np.median(values),
            })

        rows.append(row)

    return pd.DataFrame(rows)


def main_effects(replicates):
    rows = []

    for score in [
        "J_S", "J_P", "Delta_P", "J_S_e2e", "Delta_S_e2e"
    ]:
        for separation, sub in replicates.groupby("separation"):
            rows.append({
                "score": score,
                "factor": "minority_fraction",
                "held_fixed": "separation",
                "held_value": separation,
                "spearman_rho": (
                    spearmanr(sub.minority_fraction, sub[score]).statistic
                    if sub[score].std() > 0 else np.nan
                ),
                "mean_at_low": sub.loc[
                    sub.minority_fraction == 0, score
                ].mean(),
                "mean_at_high": sub.loc[
                    sub.minority_fraction == 0.5, score
                ].mean(),
            })

        for fraction, sub in replicates.groupby("minority_fraction"):
            rows.append({
                "score": score,
                "factor": "separation",
                "held_fixed": "minority_fraction",
                "held_value": fraction,
                "spearman_rho": (
                    spearmanr(sub.separation, sub[score]).statistic
                    if sub[score].std() > 0 else np.nan
                ),
                "mean_at_low": sub.loc[
                    sub.separation == 0, score
                ].mean(),
                "mean_at_high": sub.loc[
                    sub.separation == 1, score
                ].mean(),
            })

    effects = pd.DataFrame(rows)
    effects["rise_high_minus_low"] = (
        effects["mean_at_high"] - effects["mean_at_low"]
    )
    return effects


def save_table(df, filename):
    path = OUT / filename
    df.to_csv(path, sep="\t", index=False, float_format="%.10g")
    return {
        "file": filename,
        "rows": len(df),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    reference = sm.no_activity_reference(
        PROGRAMMES, (sm.BASE_SEED_P, 41)
    )
    offsets = build_offsets()
    projection = build_projection()

    tasks = [
        (fi, si, rep)
        for fi in range(len(F_GRID))
        for si in range(len(SEP_GRID))
        for rep in range(N_REP)
    ]

    replicates = pd.DataFrame(
        Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(score_condition)(
                fi, si, rep, offsets, reference, projection
            )
            for fi, si, rep in tasks
        )
    )

    summary = summarise(replicates)
    effects = main_effects(replicates)

    checks = {
        "Expected population count": len(replicates) == 700,
        "Programme null margins preserved": (
            replicates.margins_exact_P.all()
        ),
        "Structural null margins preserved and chains complete": (
            replicates.margins_exact_S_e2e.all()
            and replicates.chains_complete_S_e2e.all()
        ),
        "Programme excess equals observed minus null": (
            np.allclose(
                replicates.Delta_P,
                replicates.J_P - replicates.E0_P,
                atol=1e-12,
            )
        ),
        "All scores finite": (
            np.isfinite(replicates[SCORE_COLUMNS].to_numpy()).all()
        ),
    }

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        raise RuntimeError("Substate benchmark validation failed.")

    outputs = [
        save_table(replicates, "substate_replicates.tsv"),
        save_table(summary, "substate_summary.tsv"),
        save_table(effects, "substate_main_effects.tsv"),
    ]

    metadata = {
        "analysis": "s03_substate_mixtures",
        "scchart_version": version("scchart"),
        "python_version": sys.version.split()[0],
        "parameters": {
            "n_cells": N_CELLS,
            "n_replicates": N_REP,
            "minority_fractions": F_GRID,
            "separations": SEP_GRID,
            "base_seed": BASE_SEED,
            "n_jobs": N_JOBS,
        },
        "checks": {k: bool(v) for k, v in checks.items()},
        "outputs": outputs,
    }

    (OUT / "s03_substate_mixtures.run.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"\nCompleted: {len(replicates)} populations")
    print(f"Results: {OUT}")


if __name__ == "__main__":
    main()
