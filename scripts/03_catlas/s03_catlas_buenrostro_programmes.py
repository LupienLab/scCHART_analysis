"""Programme heterogeneity of the Buenrostro FACS populations with the CATLAS B' reference (Fig. 3e programme axis,
Buenrostro side).

Cells and features: the Buenrostro analysis cell set (common/buenrostro.py); all retained cells are scored.
Reference: per-peak CATLAS B' labels for the Buenrostro peaks (common/paths.toml catlas_buenrostro_peaks, column
programme; the same 19 modules as for Granja, each kept if >= 25 Buenrostro peaks map to it), background rates
cross-fitted on Buenrostro (5 folds, random_state 0).
Native depth: ``scchart.Heterogeneity(structural=False, programme_key='Bheme', programme_seed=10_000_000)``.
Matched depth: ``scchart.depth_standardized_programme_comparison`` with the fitted ``references_`` at D* = lowest
Buenrostro population median count depth, 5 replicates, package-default seeds.

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/03_catlas/s03_catlas_buenrostro_programmes.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path


import os

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

PARAMS = dict(n_folds=5, random_state=0, programme_seed=10_000_000,
              matched_depth="depth_standardized_programme_comparison, 5 replicates, package-default seeds",
              null="package defaults (proposal_clock, 20 replicates)")

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scchart  # noqa: E402
from scchart._scoring import count_depth  # noqa: E402
import buenrostro  # noqa: E402

input_file(data("buenrostro_h5ad"), "Buenrostro 2018 processed counts")
input_file(data("catlas_buenrostro_peaks"), "per-peak CATLAS B' labels (Buenrostro peaks)")
X, obs, _, peaks = buenrostro.load(with_peaks=True)
lab = obs.population.to_numpy()
pops = sorted(set(lab))
ref = pd.read_csv(data("catlas_buenrostro_peaks"), sep="\t").sort_values("feature_index")
check("peak order of the CATLAS table equals the analysis feature universe", bool((ref.peak.to_numpy() == peaks).all()),
          f"{len(ref)} vs {len(peaks)}")
A = ad.AnnData(X=X, obs=pd.DataFrame({"population": lab}, index=obs.index),
               var=pd.DataFrame({"Bheme": ref.programme.astype(str).to_numpy()}, index=peaks))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = scchart.Heterogeneity(structural=False, programme_key="Bheme", n_folds=5, random_state=0,
                                  programme_seed=10_000_000, verbose=True).fit(A, groupby="population")
refs = model.references_
K = len(refs[0].programmes)
depth = count_depth(X)
D_bu = float(pd.Series(depth).groupby(lab).median().min())
md = scchart.depth_standardized_programme_comparison(X, lab, refs, D_bu, populations=pops, n_replicates=5)
pt = model.programme_
out = pd.DataFrame({"n_cells": pd.Series(lab).value_counts(), "median_count_depth": pd.Series(depth).groupby(lab).median(),
                    "J_P_obs": pt.J_P_obs, "E0_J_P": pt.E0_J_P, "Delta_P": pt.Delta_P, "scored_fraction": pt.scored_fraction,
                    "Delta_P_Dstar": md.summary.Delta_P_mean, "Delta_P_Dstar_sd": md.summary.Delta_P_sd}).reindex(pops)
out["D_star"], out["K"] = D_bu, K
save_tsv(out.reset_index(names="population"), "buenrostro_catlas_Bheme_population_summary.tsv")
save_tsv(md.replicates, "buenrostro_catlas_Bheme_matched_depth_replicates.tsv")
note("cells x peaks, K, D*", dict(cells=int(X.shape[0]), peaks=int(X.shape[1]), K=K, D_star=D_bu))
check("retained B' modules between 1 and 19", 0 < K <= 19, K)
check("nulls preserved margins", bool(pt.margins_exact_all_nulls.all() and md.replicates.margins_exact_all_nulls.all()))
check("Delta finite for every population", bool(np.isfinite(out[["Delta_P", "Delta_P_Dstar"]].to_numpy()).all()))
finish()
