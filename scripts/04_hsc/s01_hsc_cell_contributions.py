"""Per-cell structural and programme contributions in the Zeng et al. HSPC xenograft multiome (input to Fig. 4, S5, S6).

Input: andy_hematopoiesis.h5ad (27,492 cells x 187,329 ATAC peaks, counts; obs Cluster, Treatment, Donor; author LSI).
Declared populations: author clusters, with HSC and HSC-II collapsed into one broad population "HSC".
Structural axis: LSI (scchart.LSIProjection.fit, 50 components, random_state 1, aligned to the author X_lsi); components
  with Pearson |r(depth)| > 0.30 excluded; ``scchart.StructuralHeterogeneity`` (signed-channel) on cells with a non-zero
  representation -> scchart_structural_d.
Programme axis: endogenous cross-fitted programmes (5 folds, random_state 0) over the broad populations,
  ``scchart.ProgrammeHeterogeneity`` on all cells -> scchart_programme_d, scchart_programme_P.
The two axes are fitted separately because the structural axis excludes cells with a zero representation vector while
the programme axis scores every cell. No null model (Fig. 4 uses per-cell contributions only).

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/04_hsc/s01_hsc_cell_contributions.py
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path


import os

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "02_granja"))
sys.path.insert(0, str(ROOT / "scripts" / "06_buenrostro"))
sys.path.insert(0, str(ROOT / "scripts" / "05_method_comparisons"))
DATA = ROOT / "data"
REFERENCE_RESULTS = ROOT / "results" / "04_hsc"
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

PARAMS = dict(broad_population="author clusters HSC + HSC-II -> HSC", lsi="50 components, random_state 1, aligned to X_lsi",
              depth_rule="|r(component, count depth)| > 0.30 excluded", n_folds=5, random_state=0, null="none")

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

import scchart  # noqa: E402
from scchart._scoring import count_depth  # noqa: E402

input_file(data("zeng_h5ad"), "Zeng et al. HSPC xenograft multiome (ATAC peak counts)")
A = ad.read_h5ad(data("zeng_h5ad"))
A.X = sparse.csr_matrix(A.X)
A.X.sort_indices()
X = A.X
subtype = A.obs["Cluster"].astype(str).to_numpy()
hsc_named = sorted(l for l in set(subtype) if re.fullmatch(r"HSC(\b.*)?", l))
check("author HSC clusters are HSC and HSC-II", hsc_named == ["HSC", "HSC-II"], hsc_named)
A.obs["broad_population"] = np.where(np.isin(subtype, ["HSC", "HSC-II"]), "HSC", subtype)
depth = count_depth(X)

full = scchart.LSIProjection.fit(X, n_components=50, align_to=np.asarray(A.obsm["X_lsi"], float), random_state=1)
Zf = full(X)
DT = pd.DataFrame({"component": np.arange(1, 51), "pearson_r_depth": [pearsonr(Zf[:, k], depth).statistic for k in range(50)]})
DT["excluded"] = DT.pearson_r_depth.abs() > 0.30
proj = full.with_components(np.flatnonzero(~DT.excluded.to_numpy()))
Z = proj(X)
ok_z = np.isfinite(Z).all(1) & ((Z ** 2).sum(1) > 0)
note("LSI components excluded", DT.component[DT.excluded].tolist())
note("cells with a zero representation (no structural score)", int((~ok_z).sum()))

S = ad.AnnData(obs=A.obs[["broad_population"]][ok_z].copy())
S.obsm["Z"] = Z[ok_z]
sh = scchart.StructuralHeterogeneity(verbose=False)
sh.fit(S, groupby="broad_population", use_rep="Z")
d_S = np.full(A.n_obs, np.nan)
d_S[ok_z] = S.obs["scchart_structural_d"].to_numpy()

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ph = scchart.ProgrammeHeterogeneity(n_folds=5, random_state=0, verbose=False)
    ph.fit(A, groupby="broad_population")
scored = A.obs["scchart_programme_scored"].to_numpy(bool)

cells = pd.DataFrame({"cell": A.obs_names, "broad_population": A.obs.broad_population.to_numpy(), "author_subtype": subtype,
                      "treatment": A.obs["Treatment"].astype(str).to_numpy(), "donor": A.obs["Donor"].astype(str).to_numpy(),
                      "count_depth": depth, "scchart_structural_d": d_S,
                      "scchart_programme_d": A.obs["scchart_programme_d"].to_numpy(),
                      "scchart_programme_P": np.where(scored, A.obs["scchart_programme_P"].to_numpy(), np.nan),
                      "umap_wnn_1": A.obsm["X_umap_wnn"][:, 0], "umap_wnn_2": A.obsm["X_umap_wnn"][:, 1],
                      "umap_atac_1": A.obsm["X_umap_ATAC"][:, 0], "umap_atac_2": A.obsm["X_umap_ATAC"][:, 1]})
save_tsv(DT, "hsc_atlas_lsi_component_depth.tsv")
save_tsv(cells, "hsc_atlas_cell_contributions.tsv.gz")
save_tsv(sh.population_summary_.reset_index(names="population"), "hsc_atlas_structural_population_summary.tsv")
save_tsv(ph.population_summary_.reset_index(names="population"), "hsc_atlas_programme_population_summary.tsv")

h = (cells.broad_population == "HSC") & np.isfinite(cells.scchart_structural_d) & np.isfinite(cells.scchart_programme_d)
note("broad HSC cells / scored on both axes", [int((cells.broad_population == "HSC").sum()), int(h.sum())])
note("Spearman scchart_structural_d vs scchart_programme_d within HSC",
         float(spearmanr(cells.scchart_structural_d[h], cells.scchart_programme_d[h]).statistic))
check("J_obs = mean structural d per population",
          np.allclose(cells[ok_z].groupby("broad_population").scchart_structural_d.mean().reindex(sh.population_summary_.index),
                      sh.population_summary_.J_obs))
check("broad HSC has cells scored on both axes", int(h.sum()) > 0, int(h.sum()))
finish()
