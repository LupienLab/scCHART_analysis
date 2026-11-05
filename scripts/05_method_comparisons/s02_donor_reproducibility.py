"""Reproducibility of population rankings across Granja samples from different donors (Fig. 2e, structural,
epiCHAOS and Jaccard columns).

Granja fragment-count set with its 50-component LSI (common/population_sets.py). For each population (sorted, index gi)
and sample (index di within the population's samples) with >= 50 cells, up to 500 cells are drawn with
default_rng((20261024, 14, gi, di)). Scores per (population, sample) cell set (cells with no accessible feature removed):
  * scCHART structural: ``structural_null_calibration`` on the binary matrix, null seed key (20261024, 13, gi, di) -> Delta_G;
  * 1 - Jaccard: mean pairwise chance-centred Jaccard over 5 random 100-cell subsets, default_rng((20261024, 16, gi, di));
  * epiCHAOS (het_adj): common/epichaos_scores.group_scores across the populations of one sample (>= 3 populations),
    default_rng((20261024, 15, k)) for the k-th scored sample (samples in order of first appearance).
Kendall tau between the population rankings of every pair of samples from different donors (sample name without its
trailing 'T<digits>'), over populations present in both (>= 4).

Random draws for the Jaccard subsets use their own seed; earlier code drew them from a stream shared with other
estimators that are not part of the manuscript, so these draws differ from the earlier ones while the definition
(n = 100, 5 subsets) is unchanged.

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/05_method_comparisons/s02_donor_reproducibility.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "02_granja"))
sys.path.insert(0, str(ROOT / "scripts" / "06_buenrostro"))
DATA = ROOT / "data"
REFERENCE_RESULTS = ROOT / "results" / "05_method_comparisons"
OUTPUT_ENV = os.environ.get("SCCHART_OUTPUT_DIR")
if not OUTPUT_ENV:
    raise SystemExit("Set SCCHART_OUTPUT_DIR to a separate directory for a reproducibility rerun. Existing results are read-only.")
RES = Path(OUTPUT_ENV).expanduser().resolve()
if RES == REFERENCE_RESULTS.resolve() or RES.is_relative_to((ROOT / "results").resolve()):
    raise SystemExit("SCCHART_OUTPUT_DIR must be outside the saved results/ tree")
RES.mkdir(parents=True, exist_ok=True)

def data(key):
    paths = {'granja_h5ad': 'healthy_hematopoiesis_slim.h5ad', 'zeng_h5ad': 'andy_hematopoiesis.h5ad', 'buenrostro_h5ad': 'buenrostro2018.h5ad', 'catlas_granja_peaks': 'peak_to_catlas_programme.tsv.gz', 'catlas_buenrostro_peaks': 'F3E_buenrostro_catlas_peak_assignment.tsv.gz', 'hsc_motif_names': 'motifs.tsv', 'hsc_motif_matches': 'motif_matches.mtx', 'hsc_motif_peaks': 'peaks.txt', 'hsc_peak_gc': 'peak_gc_and_hsc_accessibility.tsv.gz', 'hsc_motif_families': 'family_definitions.tsv', 'hsc_published_da': 'HSCII_vs_HSCI_ATAC_DA_Markers.csv'}
    p = DATA / paths[key]
    if not p.is_file():
        raise FileNotFoundError(p)
    return p

def input_file(path, note=""):
    print(f"Input: {path} {note}", flush=True)

def save_tsv(df, name):
    p = RES / name
    if p.exists():
        raise FileExistsError(f"Refusing to overwrite {p}")
    df.to_csv(p, sep="\t", index=False, float_format="%.10g",
              compression={"method": "gzip", "mtime": 0} if name.endswith(".gz") else None)
    print(f"Saved {p} ({len(df)} rows)", flush=True)
    return p

def check(name, ok, detail=""):
    if not bool(ok):
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS: {name} {detail}", flush=True)

def note(name, value):
    print(f"{name}: {value}", flush=True)

def finish():
    print("Analysis completed successfully", flush=True)

N_JOBS = int(os.environ.get("N_JOBS", "20"))
PARAMS = dict(SEED=20261024, n_max=500, min_cells_per_sample=50, cell_draw="(SEED, 14, gi, di)",
              structural_seed_key="(SEED, 13, gi, di)", jaccard_rng="(SEED, 16, gi, di)", jaccard_subsets=dict(n=100, reps=5),
              epichaos_rng="(SEED, 15, k), k = index among scored samples", min_shared_populations=4, null="package defaults (proposal_clock, 20 replicates)",
              n_jobs=N_JOBS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

from scchart._scoring import binarize  # noqa: E402
from scchart.structural_divergence import structural_null_calibration  # noqa: E402
import population_sets  # noqa: E402
from epichaos_scores import group_scores, jaccard_dissimilarity  # noqa: E402

SEED = PARAMS["SEED"]
input_file(data("granja_h5ad"), "Granja counts")
X, obs, proj, _ = population_sets.load("granja")
pops = sorted(obs.population.unique())

cellsets = {}
for gi, g in enumerate(pops):
    idx = np.flatnonzero(obs.population.values == g)
    for di, (dn, didx) in enumerate(obs.iloc[idx].groupby("donor").indices.items()):
        if didx.size >= PARAMS["min_cells_per_sample"]:
            cells = np.sort(np.random.default_rng((SEED, 14, gi, di)).choice(idx[didx], min(500, didx.size), replace=False))
            cellsets[(g, dn, gi, di)] = cells


def task(Xg, g, dn, gi, di):
    Xg = Xg[np.diff(Xg.indptr) > 0]
    B = binarize(Xg)
    cs = structural_null_calibration(B, proj, seed_key=(SEED, 13, gi, di))
    return dict(population=g, sample=dn, n_cells=int(B.shape[0]), J_obs=cs.J_obs, E0_J=cs.E0_J, Delta_G=cs.Delta,
                margins_exact=cs.margins_exact_all,
                jaccard_dissimilarity=jaccard_dissimilarity(B, np.random.default_rng((SEED, 16, gi, di))))


S = pd.DataFrame(Parallel(n_jobs=N_JOBS, backend="loky")(delayed(task)(X[c], *k) for k, c in cellsets.items()))
ep = []
for dn in S["sample"].unique():
    groups = {g: X[c] for (g, d, _, _), c in cellsets.items() if d == dn}
    if len(groups) >= 3:
        e = group_scores(groups, np.random.default_rng((SEED, 15, len(ep))))
        ep.append(e.reset_index().rename(columns={"group": "population"}).assign(sample=dn))
ep = pd.concat(ep)
S = S.merge(ep[["population", "sample", "het_adj", "het_raw"]], on=["population", "sample"], how="left").rename(
    columns={"het_adj": "epiCHAOS_het_adj", "het_raw": "epiCHAOS_het_raw"})


def donor_of(s):
    return re.sub(r"T\d+$", "", s)


samples = list(S["sample"].unique())
pairs, summ = [], []
for est in ["Delta_G", "J_obs", "jaccard_dissimilarity", "epiCHAOS_het_adj"]:
    piv = S.pivot_table(index="population", columns="sample", values=est)
    taus = []
    for i, s1 in enumerate(samples):
        for s2 in samples[i + 1:]:
            if donor_of(s1) == donor_of(s2):
                continue
            ok = piv[[s1, s2]].dropna()
            if len(ok) >= PARAMS["min_shared_populations"]:
                tau = stats.kendalltau(ok[s1], ok[s2]).statistic
                taus.append(tau)
                pairs.append(dict(estimator=est, sample_a=s1, sample_b=s2, n_shared=len(ok), tau=tau))
    summ.append(dict(estimator=est, n_pairs=len(taus), mean_tau=np.mean(taus), sd_tau=np.std(taus), median_tau=np.median(taus)))
save_tsv(S, "donor_sample_scores.tsv")
save_tsv(pd.DataFrame(pairs), "donor_tau_pairs.tsv")
save_tsv(pd.DataFrame(summ), "donor_tau_summary.tsv")
note("samples", samples)
check("all nulls preserved margins", bool(S.margins_exact.all()))
check("every sample-level score finite", bool(np.isfinite(S[["Delta_G", "jaccard_dissimilarity"]].to_numpy()).all()))
check("same number of different-donor pairs for every estimator", len({r["n_pairs"] for r in summ}) == 1, [r["n_pairs"] for r in summ])
finish()
