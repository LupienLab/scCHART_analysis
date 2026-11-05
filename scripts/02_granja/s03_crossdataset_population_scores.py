"""Structural heterogeneity and epiCHAOS of fixed <= 500-cell population sets in Buenrostro and Granja
(Fig. 3e structural axis; Fig. S3d).

Population sets and per-dataset LSI: common/population_sets.py. Each population set is scored as one population with
``scchart.structural_divergence.structural_null_calibration`` on its binary matrix (cells with no accessible
feature removed): J_obs, E0_J, Delta_G.
  * native depth: null seed key (20261024, 11, gi, 100, 0), gi = position in sorted population order;
  * within-dataset matched depth: D* = lowest population median fragment count of the sets; q = min(1, D*/median);
    3 thinning replicates when q < 1 (else 1); thinning default_rng((20261024, 31, gi, t)) with the package
    binomial thinning, null seed key (20261024, 31, gi, t).
  * epiCHAOS (native depth): common/epichaos_scores.group_scores over all population sets of a dataset,
    default_rng((20261024, 12, 100, 0)).
The population sets come from atlas-wide LSIs that differ from the declared Granja representation, so each set is
calibrated with the package calibration function rather than refitted with ``Heterogeneity``.

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/02_granja/s03_crossdataset_population_scores.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "06_buenrostro"))
sys.path.insert(0, str(ROOT / "scripts" / "05_method_comparisons"))
sys.path.insert(0, str(ROOT / "scripts" / "06_buenrostro"))
sys.path.insert(0, str(ROOT / "scripts" / "05_method_comparisons"))
DATA = ROOT / "data"
REFERENCE_RESULTS = ROOT / "results" / "02_granja"
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
PARAMS = dict(SEED=20261024, n_max=500, native_seed_key="(SEED, 11, gi, 100, 0)",
              matched_seed_key="(SEED, 31, gi, t)", matched_replicates=3, epichaos_rng="(SEED, 12, 100, 0)",
              epichaos_subsets=dict(n=100, reps=5), null="package defaults (proposal_clock, 20 replicates)", n_jobs=N_JOBS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from scchart._scoring import binarize, binomial_downsample, count_depth  # noqa: E402
from scchart.structural_divergence import structural_null_calibration  # noqa: E402
import population_sets  # noqa: E402
from epichaos_scores import group_scores  # noqa: E402

SEED = PARAMS["SEED"]


def score(Xg, proj, key, q=1.0):
    if q < 1:
        Xg = binomial_downsample(Xg, q, np.random.default_rng(key))
    Xg = Xg[np.diff(Xg.indptr) > 0]
    cs = structural_null_calibration(binarize(Xg), proj, seed_key=key)
    return dict(n_cells=int(Xg.shape[0]), median_frag=float(np.median(count_depth(Xg))),
                J_obs=cs.J_obs, E0_J=cs.E0_J, Delta_G=cs.Delta, margins_exact=cs.margins_exact_all)


for ds in ["buenrostro", "granja"]:
    input_file(data(f"{ds}_h5ad"), f"{ds} counts")
    X, obs, proj, sel = population_sets.load(ds)
    pops = sorted(sel)
    nat = pd.DataFrame(Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(score)(X[sel[g]], proj, (SEED, 11, gi, 100, 0)) for gi, g in enumerate(pops)))
    nat.insert(0, "population", pops)
    ep = group_scores({g: X[sel[g]] for g in pops}, np.random.default_rng((SEED, 12, 100, 0)))
    nat["epiCHAOS_het_adj"] = nat.population.map(ep.het_adj)
    nat["epiCHAOS_het_raw"] = nat.population.map(ep.het_raw)
    med = {g: float(np.median(count_depth(X[sel[g]]))) for g in pops}
    dstar = min(med.values())
    jobs = [(g, gi, min(1.0, dstar / med[g]), t) for gi, g in enumerate(pops) for t in range(3 if med[g] > dstar else 1)]
    mrows = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(score)(X[sel[g]], proj, (SEED, 31, gi, t), q) for g, gi, q, t in jobs)
    mat = pd.DataFrame([dict(population=g, q=q, replicate=t, **r) for (g, gi, q, t), r in zip(jobs, mrows)])
    mat["D_star"] = dstar
    summ = nat.set_index("population").join(mat.groupby("population").agg(Delta_G_matched=("Delta_G", "mean"),
                                                                          Delta_G_matched_sd=("Delta_G", "std"),
                                                                          q=("q", "first")))
    summ["D_star"] = dstar
    save_tsv(nat, f"{ds}_population_sets_native.tsv")
    save_tsv(mat, f"{ds}_population_sets_matched_replicates.tsv")
    save_tsv(summ.reset_index(), f"{ds}_population_sets_summary.tsv")
    note(f"{ds}: populations, D*", dict(n=len(pops), D_star=dstar))
    check(f"{ds}: all nulls preserved margins", bool(nat.margins_exact.all() and mat.margins_exact.all()))
    check(f"{ds}: scores finite", bool(np.isfinite(summ[["J_obs", "Delta_G", "Delta_G_matched", "epiCHAOS_het_adj"]].to_numpy()).all()))
finish()
