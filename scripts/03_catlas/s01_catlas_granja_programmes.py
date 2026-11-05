"""Programme heterogeneity of the 26 Granja populations with CATLAS module references (Fig. S4i,j; Fig. 3e programme
axis, Granja side).

References: per-peak CATLAS module labels (common/paths.toml catlas_granja_peaks), peaks not assigned to a retained
module are background ("Shared"):
  * Bheme: 19 haematopoiesis-enriched CATLAS modules (column programme_heme), primary;
  * B: the native CATLAS module vocabulary (column programme_all), sensitivity.
Background rates are cross-fitted (5 folds, random_state 0) around the fixed modules.

Native depth: ``scchart.Heterogeneity(structural=False, programme_key=<variant>, programme_seed=10_000_000)``.
Matched depth: ``scchart.depth_standardized_programme_comparison`` with the fitted ``references_`` at D* = lowest
population median count depth, 5 replicates, package-default seeds.

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/03_catlas/s01_catlas_granja_programmes.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path


VARIANTS = sys.argv[1:] or ["Bheme", "B"]
COLUMN = {"Bheme": "programme_heme", "B": "programme_all"}
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

PARAMS = dict(variants={v: COLUMN[v] for v in VARIANTS}, n_folds=5, random_state=0, programme_seed=10_000_000,
              matched_depth="depth_standardized_programme_comparison, 5 replicates, package-default seeds",
              null="package defaults (proposal_clock, 20 replicates)")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scchart  # noqa: E402
from scchart._scoring import count_depth  # noqa: E402
import granja  # noqa: E402

input_file(data("granja_h5ad"), "Granja healthy haematopoiesis atlas")
input_file(data("catlas_granja_peaks"), "per-peak CATLAS module labels (Granja peaks)")
A = granja.load()
X, lab = A.X, A.obs[granja.GROUP_KEY].to_numpy()
order = sorted(set(lab))
ref = pd.read_csv(data("catlas_granja_peaks"), sep="\t")
check("peak order of the CATLAS table equals the atlas", bool((ref.peak.to_numpy() == A.var_names.to_numpy()).all()))
D_P = float(pd.Series(count_depth(X)).groupby(lab).median().min())
note("D_star", D_P)

for v in VARIANTS:
    A.var[v] = ref[COLUMN[v]].astype(str).to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = scchart.Heterogeneity(structural=False, programme_key=v, n_folds=5, random_state=0,
                                      programme_seed=10_000_000, verbose=True).fit(A, groupby=granja.GROUP_KEY)
    refs = model.references_
    K = len(refs[0].programmes)
    if v == "Bheme":
        check("B' reference has 19 modules", K == 19, K)
    note(f"{v}: K", K)
    md = scchart.depth_standardized_programme_comparison(X, lab, refs, D_P, populations=order, n_replicates=5)
    pt = model.programme_
    out = pd.DataFrame({"J_P_obs": pt.J_P_obs, "E0_J_P": pt.E0_J_P, "Delta_P": pt.Delta_P, "scored_fraction": pt.scored_fraction,
                        "Delta_P_Dstar": md.summary.Delta_P_mean, "Delta_P_Dstar_sd": md.summary.Delta_P_sd,
                        "q": md.summary.q}).reindex(order)
    out["D_star"], out["K"] = D_P, K
    save_tsv(out.reset_index(names="population"), f"granja_catlas_{v}_population_summary.tsv")
    save_tsv(md.replicates, f"granja_catlas_{v}_matched_depth_replicates.tsv")
    check(f"{v}: nulls preserved margins", bool(pt.margins_exact_all_nulls.all() and md.replicates.margins_exact_all_nulls.all()))
    check(f"{v}: Delta finite for all 26 populations", bool(np.isfinite(out[["Delta_P", "Delta_P_Dstar"]].to_numpy()).all()))
finish()
