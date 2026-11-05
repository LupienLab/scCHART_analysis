"""Structural and programme heterogeneity of the 26 Granja populations (Fig. 3a-d, S2d-g, S4a-f).

Input: Granja atlas (common/granja.py): declared LSI representation (50 components aligned to the author LSI,
components with |r(depth)| > 0.30 excluded) and endogenous cross-fitted programmes (5 folds, random_state 0).

Stage 1, native depth: ``scchart.Heterogeneity(projection=LSI, structural_seed=(20260928, 130),
    programme_seed=10_000_000)`` on all cells, groupby BioClassification (sorted population order) -> per-cell
    scchart_structural_d / scchart_programme_d, J_obs, E0_J, Delta_G, J_P_obs, E0_J_P, Delta_P.
Stage 2, structural matched depth: ``scchart.depth_standardized_structural_comparison`` at D* = lowest population
    median count depth, 5 thinning replicates, seed 20260925.
Stage 3, programme matched depth: ``scchart.depth_standardized_programme_comparison`` at D* (populations that define
    a programme), 5 replicates, package-default seeds (thinning 900,000; null 20,000,000).
Each stage writes its tables when it finishes; a rerun skips stages whose tables exist (delete them to recompute).

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/02_granja/s01_granja_population_scoring.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "healthy_hematopoiesis_slim.h5ad"

# Use a separate directory for independent reproducibility runs.
if not os.environ.get("SCCHART_OUTPUT_DIR"):
    raise SystemExit("Set SCCHART_OUTPUT_DIR for an independent rerun")
RES = Path(os.environ["SCCHART_OUTPUT_DIR"]).expanduser().resolve()

if not DATA.is_file():
    raise FileNotFoundError(DATA)

# Never overwrite existing results.
if RES == (ROOT / "results" / "02_granja").resolve() or RES.is_relative_to((ROOT / "results").resolve()):
    raise RuntimeError("Output directory must be outside the reference results tree")
if RES.exists() and any(RES.iterdir()):
    raise RuntimeError(f"Output directory is not empty: {RES}")
RES.mkdir(parents=True, exist_ok=True)


def save_tsv(df, name):
    path = RES / name
    df.to_csv(
        path, sep="\t", index=False, float_format="%.10g",
        compression={"method": "gzip", "mtime": 0}
        if name.endswith(".gz") else None,
    )
    print(f"Saved: {path}", flush=True)


def check(name, condition, detail=""):
    if not bool(condition):
        raise AssertionError(f"{name}: {detail}")
    print(f"PASS: {name}", flush=True)


def note(name, value):
    print(f"{name}: {value}", flush=True)

N_JOBS = int(os.environ.get("N_JOBS", "24"))
PARAMS = dict(structural_seed=[20260928, 130], programme_seed=10_000_000, matched_depth_structural_seed=20260925,
              matched_depth_programme_seed="package defaults (seed 900_000, null_seed 20_000_000)",
              n_matched_replicates=5, n_folds=5, random_state=0, population_order="sorted labels",
              null="package defaults (proposal_clock, 20 replicates)", n_jobs=N_JOBS)


import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import scchart  # noqa: E402
from scchart._scoring import binarize, count_depth  # noqa: E402
import granja  # noqa: E402

print(f"Input: {DATA}")
A = granja.load()
X, lab = A.X, A.obs[granja.GROUP_KEY].to_numpy()
order = sorted(set(lab))
PROJ, _, DT = granja.representation(A)
A.obsm["X_scchart_lsi"] = PROJ(binarize(X))
depth = count_depth(X)
pop_depth = pd.Series(depth).groupby(lab).median().reindex(order)
D_S = float(pop_depth.min())
note("LSI components excluded (|r| > 0.30)", DT.component[DT.excluded].tolist())
save_tsv(DT, "granja_lsi_component_depth.tsv")


def done(*names):
    return all((RES / n).exists() for n in names)


# ---------------------------------------------------------------- stage 1: native depth, both axes
S1 = ("granja_structural_native.tsv", "granja_programme_native.tsv", "granja_cells.tsv.gz", "granja_mean_programme_composition.tsv")
if not done(*S1):
    model = scchart.Heterogeneity(use_rep="X_scchart_lsi", projection=PROJ, structural_seed=(20260928, 130),
                                  programme_seed=10_000_000, n_folds=5, random_state=0, n_jobs=N_JOBS,
                                  store_probabilities=False, verbose=True)
    model.fit(A, groupby=granja.GROUP_KEY)
    check("population order is sorted labels", model.population_order_ == order)
    save_tsv(model.structural_.reset_index(names="population"), S1[0])
    save_tsv(model.programme_.reset_index(names="population"), S1[1])
    cells = pd.DataFrame({"cell": A.obs_names, "population": lab,
                          "umap1": A.obsm["X_umap"][:, 0], "umap2": A.obsm["X_umap"][:, 1], "count_depth": depth,
                          "scchart_structural_d": A.obs["scchart_structural_d"].to_numpy(),
                          "scchart_programme_d": A.obs["scchart_programme_d"].to_numpy(),
                          "scchart_programme_P": A.obs["scchart_programme_P"].to_numpy(),
                          "scchart_programme_scored": A.obs["scchart_programme_scored"].to_numpy()})
    save_tsv(cells, S1[2])
    comp = pd.DataFrame(A.obsm["scchart_programme_composition"], columns=model.programme_model_.programmes_)
    ok = cells.scchart_programme_scored.to_numpy(bool)
    rbar = comp[ok].groupby(lab[ok]).mean().reindex(order)
    save_tsv(rbar.reset_index(names="population"), S1[3])
    note("stage 1 timings (s)", model.timings_)
else:
    print("  stage 1 tables present: skipped", flush=True)

# ---------------------------------------------------------------- stage 2: structural matched depth
S2 = ("granja_structural_matched_depth_replicates.tsv", "granja_structural_matched_depth_summary.tsv")
if not done(*S2):
    mds = scchart.depth_standardized_structural_comparison(X, lab, PROJ, D_S, populations=order, n_replicates=5,
                                                           seed=20260925, n_jobs=N_JOBS)
    save_tsv(mds.replicates, S2[0])
    save_tsv(mds.summary.reset_index(names="population") if "population" not in mds.summary.columns else mds.summary, S2[1])
else:
    print("  stage 2 tables present: skipped", flush=True)

# ---------------------------------------------------------------- stage 3: programme matched depth
S3 = ("granja_programme_matched_depth_replicates.tsv", "granja_programme_matched_depth_summary.tsv")
refs = granja.programme_model(A).references_
progs = list(refs[0].programmes)
elig = [g for g in order if g in progs]
D_P = float(pop_depth.loc[elig].min())
if not done(*S3):
    mdp = scchart.depth_standardized_programme_comparison(X, lab, refs, D_P, populations=elig, n_replicates=5)
    save_tsv(mdp.replicates, S3[0])
    save_tsv(mdp.summary.reset_index(names="population") if "population" not in mdp.summary.columns else mdp.summary, S3[1])
else:
    print("  stage 3 tables present: skipped", flush=True)

# ---------------------------------------------------------------- population summary
st = pd.read_csv(RES / S1[0], sep="\t").set_index("population")
pr = pd.read_csv(RES / S1[1], sep="\t").set_index("population")
sds = pd.read_csv(RES / S2[1], sep="\t").set_index("population")
sdr = pd.read_csv(RES / S2[0], sep="\t")
pds = pd.read_csv(RES / S3[1], sep="\t").set_index("population")
pdr = pd.read_csv(RES / S3[0], sep="\t")
cells = pd.read_csv(RES / S1[2], sep="\t")
pop = pd.DataFrame(index=pd.Index(order, name="population"))
pop["n_cells"] = pd.Series(lab).value_counts()
pop["median_count_depth"] = pop_depth
pop["median_n_open"] = pd.Series(np.diff(X.indptr)).groupby(lab).median().reindex(order)
pop["J_obs"], pop["E0_J"], pop["Delta_G"] = st.J_obs, st.E0_J, st.Delta_G
pop["Delta_G_Dstar"], pop["Delta_G_Dstar_sd"], pop["q_structural"] = sds.Delta_G_mean, sds.Delta_G_sd, sds.q
pop["J_P_obs"], pop["E0_J_P"], pop["Delta_P"] = pr.J_P_obs, pr.E0_J_P, pr.Delta_P
pop["scored_fraction"], pop["median_P"] = pr.scored_fraction, pr.median_P
pop["Delta_P_Dstar"], pop["Delta_P_Dstar_sd"], pop["q_programme"] = pds.Delta_P_mean, pds.Delta_P_sd, pds.q
pop["median_scchart_structural_d"] = cells.groupby("population").scchart_structural_d.median()
pop["median_scchart_programme_d"] = cells.groupby("population").scchart_programme_d.median()
pop["D_star_structural"], pop["D_star_programme"] = D_S, D_P
save_tsv(pop.reset_index(), "granja_population_summary.tsv")

check("26 populations", len(pop) == 26, len(pop))
check("35,038 cells scored", len(cells) == 35038, len(cells))
check("nulls preserved margins (native and matched depth, both axes)",
          bool(st.margins_exact_all_nulls.all() and pr.margins_exact_all_nulls.all()
               and sdr.margins_exact_all_nulls.all() and pdr.margins_exact_all_nulls.all()))
check("Delta = J - E0 at native depth", np.allclose(st.Delta_G, st.J_obs - st.E0_J) and np.allclose(pr.Delta_P, pr.J_P_obs - pr.E0_J_P))
check("J_obs = mean per-cell d (structural)",
          np.allclose(cells.groupby("population").scchart_structural_d.mean().reindex(order), st.J_obs.reindex(order)))
check("matched-depth estimates finite for every population",
          bool(np.isfinite(pop[["Delta_G_Dstar", "Delta_P_Dstar"]].to_numpy()).all()))
note("D_star", dict(structural=D_S, programme=D_P))
print("Analysis completed successfully.")
