"""Synthetic benchmarks scored by scCHART and epiCHAOS on identical accessibility matrices (Fig. S3a-c).

Observation model: 01_synthetic_benchmarks/scripts/synthetic_model.py. Each population is one declared population:
  * latent structural J_S (``scchart.Heterogeneity`` on the latent Z, no projection) and calibrated programme Delta_P with a
    fixed external no-activity reference (same fit);
  * end-to-end structural Delta_S_e2e: ``structural_null_calibration`` on the observed matrix with an LSI (10 components,
    all kept) fitted on 3,000 independent reference cells (cells with no accessible feature removed);
  * epiCHAOS inputs: mean pairwise centred Jaccard and coverage on the same matrix (epiCHAOS port).
``Heterogeneity`` calibrates only a representation equal to projection(B) of the same fit, so the end-to-end estimator is
computed with the package calibration function.

F2 continuous dispersion: Z = mu + h eps / sqrt(D), N = 500, h in {0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8}, 20 replicates; latent
   draws use the structural-benchmark seeds (20260925, 1, round(1000 h), rep); flat programme activity (0.5, 0.5, 0.8);
   accessibility default_rng((20260929, 302, hi, rep)); programme seed 40,000,000 + 10,000 hi + 100 rep; e2e seed key
   (20260929, 303, hi, rep); reference programmes R_A, R_B, R_common, rng (20260928, 41).
F4 multistate: k = 1..5 compact equidistant states (pairwise distance 2, within-state sd 0.1), N = 600, 20 replicates;
   programme P_m active (0.95, others 0.05, R_common 0.8) in state m; default_rng((20260929, 404, k, rep)); programme seed
   50,000,000 + 10,000 k + 100 rep; e2e seed key (20260929, 405, k, rep); reference P1-P5, R_common, rng (20260929, 400).

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/05_method_comparisons/s01_comparator_synthetic_benchmarks.py
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

ROOT_ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_ / "scripts" / "01_benchmarks"))


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
PARAMS = dict(SEED_F=20260929, F2=dict(h_grid=[0.0, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80], n_cells=500, n_rep=20),
              F4=dict(k=[1, 2, 3, 4, 5], separation=2.0, n_cells=600, n_rep=20), e2e_lsi="10 components, all kept, random_state 1",
              null="package defaults (proposal_clock, 20 replicates)", n_jobs=N_JOBS)

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

import scchart  # noqa: E402
from scchart.structural_divergence import structural_null_calibration  # noqa: E402
from epichaos.jaccard import pairwise_jaccard_mean  # noqa: E402
import synthetic_model as sm  # noqa: E402

SEED_F = PARAMS["SEED_F"]


def score_replicate(Z, X, REF, PROJ_E2E, pseed, sseed):
    a = ad.AnnData(X)
    a.obs["G"] = "G"
    a.obsm["Z"] = Z
    m = scchart.Heterogeneity(use_rep="Z", calibrate=True, store_probabilities=False, verbose=False,
                              programme_seed=pseed).fit(a, groupby="G", reference=REF)
    st, pr = m.structural_.iloc[0], m.programme_.iloc[0]
    keep = np.asarray(X.sum(1)).ravel() > 0
    ce = structural_null_calibration(X[keep], PROJ_E2E, seed_key=sseed)
    return dict(J_S=st.J_obs, J_P=pr.J_P_obs, E0_P=pr.E0_J_P, Delta_P=pr.Delta_P, scored_fraction=pr.scored_fraction,
                margins_exact_P=bool(pr.margins_exact_all_nulls), J_S_e2e=ce.J_obs, E0_S_e2e=ce.E0_J, Delta_S_e2e=ce.Delta,
                margins_exact_S_e2e=ce.margins_exact_all, chains_complete_S_e2e=ce.chains_complete_all,
                zero_open_cells=int((~keep).sum()), epichaos_mean_cJ=pairwise_jaccard_mean(X.T.tocsc()),
                epichaos_coverage=float(X.mean()))


# ---------------------------------------------------------------- F2 continuous dispersion
REF3 = sm.no_activity_reference(["R_A", "R_B", "R_common"], (sm.BASE_SEED_P, 41))
H_GRID, N_F2, N_REP_F2 = PARAMS["F2"]["h_grid"], PARAMS["F2"]["n_cells"], PARAMS["F2"]["n_rep"]


def w_flat(n):
    return np.tile([0.5, 0.5, sm.W_COMMON], (n, 1))


BETA_F2 = {}
for h in H_GRID:
    r_ = np.random.default_rng((SEED_F, 300, sm.milli(h)))
    BETA_F2[h] = sm.beta_for(sm.MU[None] + h * r_.standard_normal((20000, sm.K_DIM)) / np.sqrt(sm.K_DIM))
r_ = np.random.default_rng((SEED_F, 301))
hh = r_.uniform(0, 0.8, 3000)
Zl = sm.MU[None] + hh[:, None] * r_.standard_normal((3000, sm.K_DIM)) / np.sqrt(sm.K_DIM)
PROJ_F2 = scchart.LSIProjection.fit(sm.observe(Zl, w_flat(3000), sm.beta_for(Zl), r_), n_components=10, random_state=1)


def f2_task(hi, h, rep):
    Z = sm.MU[None] + h * np.random.default_rng((sm.BASE_SEED_S, 1, sm.milli(h), rep)).standard_normal((N_F2, sm.K_DIM)) / np.sqrt(sm.K_DIM)
    X = sm.observe(Z, w_flat(N_F2), BETA_F2[h], np.random.default_rng((SEED_F, 302, hi, rep)))
    return dict(h=h, replicate=rep, **score_replicate(Z, X, REF3, PROJ_F2, 40_000_000 + 10_000 * hi + 100 * rep, (SEED_F, 303, hi, rep)))


F2 = pd.DataFrame(Parallel(n_jobs=N_JOBS, backend="loky")(delayed(f2_task)(hi, h, r) for hi, h in enumerate(H_GRID) for r in range(N_REP_F2)))
print("  F2 done", flush=True)

# ---------------------------------------------------------------- F4 multistate complexity
REF6 = sm.no_activity_reference(["P1", "P2", "P3", "P4", "P5", "R_common"], (SEED_F, 400))
SEP_F4, N_F4, N_REP_F4 = PARAMS["F4"]["separation"], PARAMS["F4"]["n_cells"], PARAMS["F4"]["n_rep"]
Q = np.linalg.qr(np.c_[sm.MU, np.random.default_rng((SEED_F, 401)).standard_normal((sm.K_DIM, 5))])[0][:, 1:6]
CENTRES = sm.MU[None] + (SEP_F4 / np.sqrt(2)) * (np.eye(5) - 0.2) @ Q.T
dists = [np.linalg.norm(CENTRES[a] - CENTRES[b]) for a, b in itertools.combinations(range(5), 2)]
check("F4 state centres equidistant and orthogonal to mu", np.allclose(dists, SEP_F4) and np.allclose((CENTRES - sm.MU) @ sm.MU, 0))


def f4_cells(k, rng, n=N_F4):
    st = np.repeat(np.arange(k), n // k)
    rng.shuffle(st)
    n = st.size
    Z = CENTRES[st] + sm.SIG_Z * rng.standard_normal((n, sm.K_DIM)) / np.sqrt(sm.K_DIM)
    W = np.full((n, 6), 0.05)
    W[np.arange(n), st] = 0.95
    W[:, 5] = sm.W_COMMON
    return Z, W, st


BETA_F4 = {k: sm.beta_for(f4_cells(k, np.random.default_rng((SEED_F, 402, k)), 20000)[0]) for k in range(1, 6)}
r_ = np.random.default_rng((SEED_F, 403))
Zl, Wl, _ = f4_cells(5, r_, 3000)
PROJ_F4 = scchart.LSIProjection.fit(sm.observe(Zl, Wl, sm.beta_for(Zl), r_), n_components=10, random_state=1)


def f4_task(k, rep):
    rng = np.random.default_rng((SEED_F, 404, k, rep))
    Z, W, _ = f4_cells(k, rng)
    X = sm.observe(Z, W, BETA_F4[k], rng)
    return dict(k_states=k, replicate=rep, **score_replicate(Z, X, REF6, PROJ_F4, 50_000_000 + 10_000 * k + 100 * rep, (SEED_F, 405, k, rep)))


F4 = pd.DataFrame(Parallel(n_jobs=N_JOBS, backend="loky")(delayed(f4_task)(k, r) for k in range(1, 6) for r in range(N_REP_F4)))

save_tsv(F2, "comparator_dispersion_replicates.tsv")
save_tsv(F4, "comparator_multistate_replicates.tsv")
both = pd.concat([F2, F4])
check("replicate counts", len(F2) == 140 and len(F4) == 100, (len(F2), len(F4)))
check("nulls preserved margins and completed", bool(both.margins_exact_P.all() and both.margins_exact_S_e2e.all() and both.chains_complete_S_e2e.all()))
check("scores finite", bool(np.isfinite(both[["J_S", "Delta_P", "Delta_S_e2e", "epichaos_mean_cJ"]].to_numpy()).all()))
finish()
