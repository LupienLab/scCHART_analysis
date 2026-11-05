"""Reproducibility of programme-axis population rankings across Granja samples from different donors with a fixed
CATLAS B' reference (Fig. 2e, programme column).

Cell sets: the (population, sample) sets of 05_method_comparisons/scripts/s02_donor_reproducibility.py (populations with
>= 100 cells; samples with >= 50 cells of the population; up to 500 cells, default_rng((20261024, 14, gi, di))).
Reference: CATLAS B' (19 modules, column programme_heme) with cross-fitted background rates (5 folds, random_state 0),
fitted once on the whole atlas with ``scchart.ProgrammeHeterogeneity``; each cell is scored by the fold reference that
holds it out. Score per cell set: ``programme_null_calibration`` over its fold parts (20 replicates, null_cells
"scored"), base seed 30,000,000 + 10,000 f + 100 gi + di -> Delta_P. The sets are scored against references fitted on
the whole atlas, so the calibration function is called per set rather than refitting ``Heterogeneity`` on each set.
Kendall tau between every pair of samples from different donors sharing >= 4 populations.

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/03_catlas/s02_catlas_donor_programme.py
"""
from __future__ import annotations

import os
import re
import sys
import warnings
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "02_granja"))
sys.path.insert(0, str(ROOT / "scripts" / "06_buenrostro"))
sys.path.insert(0, str(ROOT / "scripts" / "05_method_comparisons"))
DATA = ROOT / "data"
REFERENCE_RESULTS = ROOT / "results" / "03_catlas"
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
PARAMS = dict(SEED=20261024, n_max=500, min_population_cells=100, min_cells_per_sample=50, cell_draw="(SEED, 14, gi, di)",
              reference="CATLAS B' programme_heme, ProgrammeHeterogeneity(n_folds=5, random_state=0) on the whole atlas",
              programme_seed="30_000_000 + 10_000*f + 100*gi + di", min_shared_populations=4,
              null="package defaults (proposal_clock, 20 replicates, null_cells='scored')", n_jobs=N_JOBS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402

import scchart  # noqa: E402
from scchart._scoring import binarize  # noqa: E402
from scchart.programme_entropy import programme_null_calibration  # noqa: E402
import granja  # noqa: E402

SEED = PARAMS["SEED"]
input_file(data("granja_h5ad"), "Granja healthy haematopoiesis atlas")
input_file(data("catlas_granja_peaks"), "per-peak CATLAS module labels (Granja peaks)")
A = granja.load()
X, lab = A.X, A.obs[granja.GROUP_KEY].to_numpy()
sample = A.obs["Group"].astype(str).to_numpy()
ref = pd.read_csv(data("catlas_granja_peaks"), sep="\t")
check("peak order of the CATLAS table equals the atlas", bool((ref.peak.to_numpy() == A.var_names.to_numpy()).all()))
A.var["Bheme"] = ref["programme_heme"].astype(str).to_numpy()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ph = scchart.ProgrammeHeterogeneity(programme_key="Bheme", n_folds=5, random_state=0, verbose=False)
    ph.fit(A, groupby=granja.GROUP_KEY)
refs = ph.references_
check("B' reference has 19 modules", len(refs[0].programmes) == 19, len(refs[0].programmes))
B = binarize(X)
del A

vc = pd.Series(lab).value_counts()
keep = np.flatnonzero(pd.Series(lab).isin(vc[vc >= PARAMS["min_population_cells"]].index).to_numpy())
lab_k, samp_k = lab[keep], sample[keep]
pops = sorted(set(lab_k))
sets = {}
for gi, g in enumerate(pops):
    idx = np.flatnonzero(lab_k == g)
    for di, (dn, didx) in enumerate(pd.Series(samp_k[idx]).groupby(samp_k[idx]).indices.items()):
        if didx.size >= PARAMS["min_cells_per_sample"]:
            cells = np.sort(np.random.default_rng((SEED, 14, gi, di)).choice(idx[didx], min(500, didx.size), replace=False))
            sets[(g, dn)] = (keep[cells], gi, di)

parts = {}
for key, (cells, gi, di) in sets.items():
    parts[key] = [(B[np.intersect1d(r.score_index, cells)], r, 30_000_000 + 10_000 * (r.fold or 0) + 100 * gi + di)
                  for r in refs if np.intersect1d(r.score_index, cells).size]


def task(key, pl):
    s = programme_null_calibration(pl, null_cells="scored").summary()
    return dict(population=key[0], sample=key[1], **s)


keys = sorted(parts)
sc = pd.DataFrame(Parallel(n_jobs=N_JOBS, backend="loky")(delayed(task)(k, parts[k]) for k in keys))


def donor_of(s):
    return re.sub(r"T\d+$", "", s)


samples = list(dict.fromkeys(k[1] for k in sets))
piv = sc.pivot_table(index="population", columns="sample", values="Delta_P")
pairs = []
for i, s1 in enumerate(samples):
    for s2 in samples[i + 1:]:
        if donor_of(s1) == donor_of(s2) or s1 not in piv or s2 not in piv:
            continue
        ok = piv[[s1, s2]].dropna()
        if len(ok) >= PARAMS["min_shared_populations"]:
            pairs.append(dict(estimator="Delta_P_catlas_Bheme", sample_a=s1, sample_b=s2, n_shared=len(ok),
                              tau=stats.kendalltau(ok[s1], ok[s2]).statistic))
pairs = pd.DataFrame(pairs)
summ = pd.DataFrame([dict(estimator="Delta_P_catlas_Bheme", n_pairs=len(pairs), mean_tau=pairs.tau.mean(),
                          sd_tau=pairs.tau.std(ddof=0), median_tau=pairs.tau.median())])
save_tsv(sc, "donor_programme_scores.tsv")
save_tsv(pairs, "donor_programme_tau_pairs.tsv")
save_tsv(summ, "donor_programme_tau_summary.tsv")
check("all nulls preserved margins", bool(sc.margins_exact_all_nulls.all()))
check("Delta_P finite for every cell set", bool(np.isfinite(sc.Delta_P).all()))
finish()
