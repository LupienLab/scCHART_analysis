"""Depth and sparsity validation on the Granja atlas (Fig. S2h,i; Fig. S4h conditions B and C).

Fixed declared LSI projection and fixed atlas-level cross-fitted programme references (common/granja.py), never
refitted on thinned data. Every cell set is scored as one population with the fixed-margin null:
structural ``structural_null_calibration``, programme ``programme_null_calibration`` over the fold parts.
These cell sets use per-population thinning schemes that the package's matched-depth functions do not take,
so the calibration functions are called directly.

Part 1 -- controlled depth series (S2h,i). Populations 11_CD14.Mono.1, 17_B, 24_CD8.CM, 01_HSC, 16_Pre.B,
  05_CMP.LMPP, 07_GMP, 06_CLP.1; one fixed set of 500 cells each (default_rng((4241, pi))). Binomial thinning
  (package ``binomial_downsample``) to median 250 / 400 / 789 / 1,200 counts (q = target / median of the 500
  cells), 5 replicates (default_rng((4242, pi, 0, level_index, t))), plus native depth. Cells left with no
  accessible feature are removed. Null seeds: code = ((3 pi) * 10 + level_index) * 10 + t;
  structural seed key (4243, code); programme base seeds 40,000,000 + 1,000 code + 100 f.

Part 2 -- matching targets for all 26 populations (S4h). Condition A (count-matched) is the manuscript matched-depth
  computation and is produced by s01_granja_population_scoring. Here:
  B fragment-matched: same q_G = min(1, D* / median count depth_G), fragment-level thinning (each count c is
    ceil(c/2) fragments, each kept with probability q; returns 2 x kept fragments, capped at c);
  C open-matched: binomial thinning with q_G such that the median expected number of detected regions per cell
    equals the native median of the shallowest population (20_CD4.N1).
  Programme: 5 replicates; thinning default_rng((31337, ci, t)) with ci = 1 (B), 2 (C), applied fold by fold;
    null base seeds 20,000,000 + 1,000,000 t + 10,000 f + 100 gi.
  Structural: 3 replicates per population with q < 1 (1 otherwise); thinning default_rng((20260925, 160, k, 10 + ci, t)),
    null seed key (20260925, 161, k, 10 + ci, t) with ci = 0 (B), 1 (C); cells with no accessible feature removed.

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/02_granja/s02_depth_sparsity_validation.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]
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

N_JOBS = int(os.environ.get("N_JOBS", "24"))
POPS = ["11_CD14.Mono.1", "17_B", "24_CD8.CM", "01_HSC", "16_Pre.B", "05_CMP.LMPP", "07_GMP", "06_CLP.1"]
LEVELS = [250, 400, 789, 1200, None]
PARAMS = dict(series_populations=POPS, series_n_cells=500, series_levels=LEVELS, series_null_replicates=5,
              matching_conditions=["B_fragment_matched", "C_open_matched"], matching_programme_replicates=5,
              matching_structural_replicates=3, open_target="native median detected regions of 20_CD4.N1",
              null="package defaults (proposal_clock, 20 replicates)", n_jobs=N_JOBS)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from scchart._scoring import binarize, binomial_downsample, count_depth  # noqa: E402
from scchart.programme_entropy import programme_null_calibration  # noqa: E402
from scchart.structural_divergence import structural_null_calibration  # noqa: E402
import granja  # noqa: E402

input_file(data("granja_h5ad"), "Granja healthy haematopoiesis atlas")
A = granja.load()
X, lab = A.X.astype(np.float64), A.obs[granja.GROUP_KEY].to_numpy()
order = sorted(set(lab))
PROJ, _, _ = granja.representation(A)
REFS = granja.programme_model(A).references_
FOLD = granja.fold_of_cells(REFS, X.shape[0])
del A
depth = count_depth(X)
med = pd.Series(depth).groupby(lab).median().reindex(order)
D_STAR = float(med.min())
n_open = pd.Series(np.diff(X.indptr)).groupby(lab).median().reindex(order)
OPEN_TARGET = float(n_open[med.idxmin()])
note("D_star and open-region target", dict(D_star=D_STAR, shallowest=med.idxmin(), open_target=OPEN_TARGET))


def fragment_thin(Xc, q, rng):
    Xc = sp.csr_matrix(Xc, copy=True)
    q = np.asarray(q, float)
    qq = np.full(Xc.data.size, float(q)) if q.ndim == 0 else np.repeat(q, np.diff(Xc.indptr))
    c = Xc.data.astype(np.int64)
    fk = rng.binomial((c + 1) // 2, np.clip(qq, 0, 1))
    Xc.data = np.minimum(2 * fk, c).astype(np.float64)
    Xc.eliminate_zeros()
    return Xc


def q_for_open(Xg, target):
    c = Xg.data
    rows = np.repeat(np.arange(Xg.shape[0]), np.diff(Xg.indptr))

    def med_open(q):
        return np.median(np.bincount(rows, weights=1 - (1 - q) ** c, minlength=Xg.shape[0]))
    if med_open(1.0) <= target:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(50):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if med_open(m) < target else (lo, m)
    return 0.5 * (lo + hi)


def score_set(B, fold, s_key, p_base):
    cs = structural_null_calibration(B, PROJ, seed_key=s_key, n_jobs=1)
    parts = [(B[fold == k], REFS[k], p_base + 100 * k) for k in range(len(REFS)) if (fold == k).any()]
    cp = programme_null_calibration(parts)
    return dict(J_S=cs.J_obs, E0_S=cs.E0_J, Delta_S=cs.Delta, margins_exact_S=cs.margins_exact_all,
                J_P=cp.J_obs, E0_P=cp.E0_J, Delta_P=cp.Delta, scored_fraction=cp.scored_fraction,
                margins_exact_P=cp.margins_exact_all)


# ---------------------------------------------------------------- part 1: controlled depth series
sub = {}
for pi, G in enumerate(POPS):
    idx = np.sort(np.random.default_rng((4241, pi)).choice(np.flatnonzero(lab == G), 500, replace=False))
    sub[pi] = (X[idx].tocsr(), FOLD[idx])


def series_task(pi, li, level, t, Xs, f):
    nat = float(np.median(count_depth(Xs)))
    if level is None:
        Xt, q = Xs, 1.0
    else:
        q = level / nat
        Xt = binomial_downsample(Xs, q, np.random.default_rng((4242, pi, 0, li, t)))
    keep = count_depth(Xt) > 0
    B = binarize(Xt)
    code = ((pi * 3) * 10 + li) * 10 + t
    out = dict(population=POPS[pi], level="native" if level is None else str(level), replicate=t, q=q,
               median_counts=float(np.median(count_depth(Xt))), median_open=float(np.median(np.asarray(B.sum(1)).ravel())),
               zero_open=int((~keep).sum()))
    out.update(score_set(B[keep], f[keep], (4243, code), 40_000_000 + 1000 * code))
    return out


tasks = [(pi, li, lev, t) for pi in range(len(POPS)) for li, lev in enumerate(LEVELS) for t in range(1 if lev is None else 5)]
SER = pd.DataFrame(Parallel(n_jobs=N_JOBS, backend="loky")(delayed(series_task)(pi, li, lev, t, *sub[pi]) for pi, li, lev, t in tasks))
save_tsv(SER, "depth_series_replicates.tsv")
print("  part 1 done", flush=True)

# ---------------------------------------------------------------- part 2: matching targets, 26 populations
q_count = np.minimum(1.0, D_STAR / med)
q_open = pd.Series({G: q_for_open(X[lab == G].tocsr(), OPEN_TARGET) for G in order})
CONDS = {"B_fragment_matched": ("frag", q_count, 1, 0), "C_open_matched": ("ins", q_open, 2, 1)}

# programme: thinning fold by fold, then one calibration per (condition, replicate, population)
prog_parts, prog_extra = {}, {}
for cond, (mode, qp, ci_p, _) in CONDS.items():
    for t in range(5):
        rng = np.random.default_rng((31337, ci_p, t))
        for G in order:
            prog_parts[(cond, t, G)], prog_extra[(cond, t, G)] = [], []
        for ref in REFS:
            te = ref.score_index
            qc = np.array([qp[G] for G in lab[te]])
            Xt = binomial_downsample(X[te], qc, rng) if mode == "ins" else fragment_thin(X[te], qc, rng)
            Bte = binarize(Xt)
            for gi, G in enumerate(order):
                loc = np.flatnonzero(lab[te] == G)
                prog_parts[(cond, t, G)].append((Bte[loc], ref, 20_000_000 + 1_000_000 * t + 10_000 * (ref.fold or 0) + 100 * gi))
                prog_extra[(cond, t, G)].append(np.c_[count_depth(Xt[loc]), np.asarray(Bte[loc].sum(1)).ravel()])


def prog_task(key, parts):
    cp = programme_null_calibration(parts)
    return key, dict(J_P=cp.J_obs, E0_P=cp.E0_J, Delta_P=cp.Delta, scored_fraction=cp.scored_fraction,
                     margins_exact_P=cp.margins_exact_all)


res = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(prog_task)(k, v) for k, v in prog_parts.items())
rows = []
for (cond, t, G), r in res:
    e = np.vstack(prog_extra[(cond, t, G)])
    rows.append(dict(condition=cond, replicate=t, population=G, q=float(CONDS[cond][1][G]),
                     native_median_counts=float(med[G]), median_counts_after=float(np.median(e[:, 0])),
                     median_open_after=float(np.median(e[:, 1])), **r))
PRG = pd.DataFrame(rows).sort_values(["condition", "population", "replicate"])
save_tsv(PRG, "matching_programme_replicates.tsv")
del prog_parts
print("  part 2 programme done", flush=True)


def struct_task(cond, mode, q, ci_s, t, k, G, Xg):
    rng = np.random.default_rng((20260925, 160, k, 10 + ci_s, t))
    Xt = Xg if q >= 1 else (fragment_thin(Xg, q, rng) if mode == "frag" else binomial_downsample(Xg, q, rng))
    keep = count_depth(Xt) > 0
    cs = structural_null_calibration(binarize(Xt[np.flatnonzero(keep)]), PROJ, seed_key=(20260925, 161, k, 10 + ci_s, t))
    return dict(condition=cond, replicate=t, population=G, q=q, native_median_counts=float(med[G]), zero_open=int((~keep).sum()),
                median_counts_after=float(np.median(count_depth(Xt[keep]))),
                median_open_after=float(np.median(np.diff(binarize(Xt[keep]).indptr))),
                J_S=cs.J_obs, E0_S=cs.E0_J, Delta_S=cs.Delta, margins_exact_S=cs.margins_exact_all)


keys = [(cond, mode, float(qp[G]), ci_s, t, k, G) for cond, (mode, qp, _, ci_s) in CONDS.items()
        for k, G in enumerate(order) for t in range(3 if qp[G] < 1 else 1)]
STR = pd.DataFrame(Parallel(n_jobs=N_JOBS, backend="loky")(delayed(struct_task)(*key, X[lab == key[-1]].tocsr()) for key in keys))
STR = STR.sort_values(["condition", "population", "replicate"])
save_tsv(STR, "matching_structural_replicates.tsv")

check("depth series rows", len(SER) == len(POPS) * (4 * 5 + 1), len(SER))
check("matching rows", len(PRG) == 2 * 5 * 26, (len(PRG), len(STR)))
check("all nulls preserved margins",
          bool(SER.margins_exact_S.all() and SER.margins_exact_P.all() and PRG.margins_exact_P.all() and STR.margins_exact_S.all()))
check("all Delta finite", bool(np.isfinite(SER[["Delta_S", "Delta_P"]].to_numpy()).all()
                                   and np.isfinite(PRG.Delta_P).all() and np.isfinite(STR.Delta_S).all()))
finish()
