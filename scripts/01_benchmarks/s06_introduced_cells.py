
"""Introduced-cell detection benchmark (Figure S1f).

Evaluates whether per-cell structural divergence identifies a minority
population introduced into a defined reference population.

Five Buenrostro population pairs, five introduced-cell fractions,
20 replicates and 150 cells per mixture.

AUROC is calculated from uncalibrated per-cell structural
contributions, using the published benchmark's sampling seeds.
"""

import sys
import json
import hashlib
from pathlib import Path
from importlib.metadata import version

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scchart import StructuralHeterogeneity
from scchart._scoring import binarize


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "common"))

import buenrostro


OUT = ROOT / "results" / "01_benchmarks"
OUT.mkdir(parents=True, exist_ok=True)

BASE_SEED = 20261024
N_CELLS = 150
N_REP = 20

PAIRS = [
    ("HSC", "MEP"),
    ("GMP", "MEP"),
    ("CMP", "LMPP"),
    ("HSC", "mono"),
    ("HSC", "MPP"),
]

FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5]


def score_mixture(X, projection, cells):
    """Calculate per-cell structural contributions for one mixture."""
    adata = ad.AnnData(
        np.zeros((len(cells), 1), dtype=np.float32)
    )
    adata.obsm["X_lsi"] = projection(binarize(X[cells]))
    adata.obs["G"] = "G"

    model = StructuralHeterogeneity(
        adapter="signed_channel",
        store_probabilities=False,
        verbose=False,
    )

    scores = model.fit(
        adata, groupby="G", use_rep="X_lsi"
    )

    contributions = scores[
        "scchart_structural_d"
    ].to_numpy()

    J_obs = float(
        model.population_summary_["J_obs"].iloc[0]
    )

    return contributions, J_obs


def run_benchmark(X, obs, projection):
    rows = []

    for pair_index, (population_a, population_b) in enumerate(PAIRS):
        indices_a = np.flatnonzero(
            obs.population.values == population_a
        )
        indices_b = np.flatnonzero(
            obs.population.values == population_b
        )

        for replicate in range(N_REP):
            for fraction in FRACTIONS:
                rng = np.random.default_rng((
                    BASE_SEED,
                    21,
                    pair_index,
                    replicate,
                    int(round(100 * fraction)),
                ))

                n_b = int(round(fraction * N_CELLS))
                n_a = N_CELLS - n_b

                cells = np.r_[
                    rng.choice(indices_a, n_a, replace=False),
                    rng.choice(indices_b, n_b, replace=False),
                ]

                introduced = np.r_[
                    np.zeros(n_a),
                    np.ones(n_b),
                ]

                contributions, J_obs = score_mixture(
                    X, projection, cells
                )

                rows.append({
                    "A": population_a,
                    "B": population_b,
                    "f": fraction,
                    "replicate": replicate,
                    "n_introduced": n_b,
                    "J_obs": J_obs,
                    "auroc_scchart_structural_d": roc_auc_score(
                        introduced, contributions
                    ),
                })

        print(
            f"{population_a} + {population_b}: complete",
            flush=True,
        )

    return pd.DataFrame(rows)


def save_table(df, filename):
    path = OUT / filename
    df.to_csv(
        path,
        sep="\t",
        index=False,
        float_format="%.10g",
    )

    return {
        "file": filename,
        "rows": len(df),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    X, obs, projection = buenrostro.load()

    replicates = run_benchmark(X, obs, projection)

    summary = (
        replicates.groupby(["A", "B", "f"])
        .agg(
            n_replicates=("auroc_scchart_structural_d", "size"),
            auroc_mean=("auroc_scchart_structural_d", "mean"),
            auroc_sd=(
                "auroc_scchart_structural_d",
                lambda x: x.std(ddof=1),
            ),
        )
        .reset_index()
    )

    checks = {
        "Expected replicate count": (
            len(replicates) == 5 * 5 * 20
        ),
        "AUROC finite and between zero and one": (
            replicates[
                "auroc_scchart_structural_d"
            ].between(0, 1).all()
        ),
    }

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {name}")

    if not all(checks.values()):
        raise RuntimeError("Introduced-cell benchmark validation failed.")

    outputs = [
        save_table(
            replicates,
            "introduced_cells_replicates.tsv",
        ),
        save_table(
            summary,
            "introduced_cells_auroc_summary.tsv",
        ),
    ]

    metadata = {
        "analysis": "s06_introduced_cells",
        "scchart_version": version("scchart"),
        "python_version": sys.version.split()[0],
        "parameters": {
            "base_seed": BASE_SEED,
            "population_pairs": PAIRS,
            "introduced_fractions": FRACTIONS,
            "n_cells": N_CELLS,
            "n_replicates": N_REP,
            "representation": "Buenrostro LSI, 50 components",
        },
        "checks": {k: bool(v) for k, v in checks.items()},
        "outputs": outputs,
    }

    (OUT / "s06_introduced_cells.run.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    print(f"\nCompleted: {len(replicates)} mixtures")
    print(f"Results: {OUT}")


if __name__ == "__main__":
    main()
