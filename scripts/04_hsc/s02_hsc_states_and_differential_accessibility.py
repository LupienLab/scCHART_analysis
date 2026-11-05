"""Heterogeneity-associated states and differential accessibility within the broad HSC population
(Fig. 4a-d, Fig. S5a-d, Fig. S6b inputs).

Cells: broad-HSC cells scored on both axes (s01_hsc_cell_contributions). d_S = scchart_structural_d, d_P = scchart_programme_d.
Strata: treatment x decile of log10 count depth within treatment.

1. Joint score z_J: within each stratum d_S and d_P become mid-ranks u in (0, 1); z_J = [Phi^-1(u_S) + Phi^-1(u_P)] / sqrt(2).
2. Cell groups: single-axis strata (bottom/top 20 % of d_S or d_P) and joint strata (d_S and d_P both in the top / bottom
   25 %; q increased by 0.05 until both groups have >= 150 cells). 1:1 matching within treatment x depth decile
   (default_rng(20260929), drawn stratum by stratum).
3. Differential accessibility (matched groups, binarised peaks): peaks open in >= 1 % of compared cells are tested with a
   two-sided hypergeometric test, BH q < 0.05, |log2 OR| >= log2 1.5 (OR with +0.5), |difference in fraction| >= 0.02 and
   the same sign in PBS, TNF and LPS.
4. HSC-II composition of the matched joint groups (Fisher's exact test).
5. Joint-tail threshold sensitivity, q in {0.10, 0.15, 0.20, 0.25, 0.30, 1/3}: DA as in 3; GATA-family (JASPAR 2020
   GATA1-6) enrichment among closing regions by Cochran-Mantel-Haenszel against tested peaks, strata = GC decile x
   HSC-accessibility decile x width tertile (ranks over all atlas peaks); overlap with the published HSC-I/HSC-II markers
   (padj < 0.05, sign of log2FC).
6. Continuous region association with z_J: for peaks open in >= 1 % of HSC cells, stratified score statistic
   T_j = sum_k sum_{i in k} B_ij (z_i - zbar_k) with its within-stratum permutation variance, Z_j, BH q; GATA enrichment of
   the 355 and 1,000 most negative Z_j (CMH as in 5).
7. Accessibility of the joint closing/opening region sets (split by overlap with the published sets) relative to each cell's
   overall accessibility, by z_J decile within HSC-I and HSC-II; HSC-II share by z_J decile within treatment.

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/04_hsc/s02_hsc_states_and_differential_accessibility.py
"""
from __future__ import annotations

import sys
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

PARAMS = dict(SEED=20260929, depth_bins=10, q_single=0.20, q_joint=0.25, min_joint_cells=150, min_detection=0.01,
              q_DA=0.05, min_abs_log2_OR="log2(1.5)", min_abs_diff=0.02, thresholds=[0.10, 0.15, 0.20, 0.25, 0.30, 1 / 3],
              gata_motifs=["GATA1", "GATA2", "GATA3", "GATA4", "GATA5", "GATA6"],
              cmh_strata="GC decile x HSC-accessibility decile x width tertile", published_markers="padj < 0.05")

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io  # noqa: E402
from scipy import sparse, stats  # noqa: E402
from scipy.stats import fisher_exact, hypergeom, norm  # noqa: E402
from statsmodels.stats.contingency_tables import StratifiedTable  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

SEED, NBIN, MIN_DET = PARAMS["SEED"], PARAMS["depth_bins"], PARAMS["min_detection"]
Q_DA, MIN_LOG2OR, MIN_DIFF = PARAMS["q_DA"], np.log2(1.5), PARAMS["min_abs_diff"]
QS = PARAMS["thresholds"]
TREATMENTS = ("PBS", "TNF", "LPS")

contrib = RES / "hsc_atlas_cell_contributions.tsv.gz"
input_file(contrib, "per-cell contributions (s01)")
C = pd.read_csv(contrib, sep="\t", index_col="cell")
H = C[(C.broad_population == "HSC") & np.isfinite(C.scchart_structural_d) & np.isfinite(C.scchart_programme_d)].copy()
H = H.rename(columns={"scchart_structural_d": "d_S", "scchart_programme_d": "d_P"})
H["state"] = H.author_subtype.map({"HSC": "HSC-I", "HSC-II": "HSC-II"})
H["log10_depth"] = np.log10(H.count_depth)
H["bin"] = H.groupby("treatment").log10_depth.transform(lambda v: pd.qcut(v, NBIN, labels=False, duplicates="drop"))
H["stratum"] = H.treatment + "|" + H["bin"].astype(str)
note("HSC cells scored on both axes", len(H))

input_file(data("zeng_h5ad"), "Zeng et al. ATAC peak counts")
A = ad.read_h5ad(data("zeng_h5ad"))
var_names = pd.Index(A.var_names.astype(str))
VAR = A.var[["chromosome", "start", "end"]].copy()
rows = pd.Index(A.obs_names).get_indexer(H.index)
assert (rows >= 0).all()
Bcsr = (sparse.csr_matrix(A.X)[rows] > 0).astype(np.float32).tocsr()
del A
ROW = pd.Series(np.arange(len(H)), index=H.index)

# ---------------------------------------------------------------- 1. joint score
def stratum_rank(col):
    return H.groupby("stratum")[col].rank(method="average").div(H.groupby("stratum")[col].transform("size") + 1)


H["u_S"], H["u_P"] = stratum_rank("d_S"), stratum_rank("d_P")
H["z_J"] = (norm.ppf(H.u_S) + norm.ppf(H.u_P)) / np.sqrt(2)
H["z_decile"] = pd.qcut(H.z_J, 10, labels=False)

# ---------------------------------------------------------------- 2-3. groups, matching, DA
def open_counts(cells):
    return np.asarray(Bcsr[ROW[cells].to_numpy()].sum(0)).ravel()


def match(hi, lo):
    rng = np.random.default_rng(SEED)
    d = H.loc[hi.union(lo), ["treatment", "log10_depth"]].copy()
    d["grp"] = np.where(d.index.isin(hi), "high", "low")
    d["bin"] = d.groupby("treatment").log10_depth.transform(lambda v: pd.qcut(v, NBIN, labels=False, duplicates="drop"))
    kh, kl = [], []
    for _, s in d.groupby(["treatment", "bin"]):
        a, b = s.index[s.grp == "high"], s.index[s.grp == "low"]
        m = min(len(a), len(b))
        kh += list(rng.choice(a, m, replace=False))
        kl += list(rng.choice(b, m, replace=False))
    return pd.Index(sorted(kh)), pd.Index(sorted(kl))


def da_table(hi, lo):
    nh, nl = len(hi), len(lo)
    kh, kl = open_counts(hi), open_counts(lo)
    fh, fl = kh / nh, kl / nl
    lor = np.log2(((kh + .5) / (nh - kh + .5)) / ((kl + .5) / (nl - kl + .5)))
    K, N = kh + kl, nh + nl
    tested = (K / N) >= MIN_DET
    p = np.full(K.size, np.nan)
    pl = hypergeom.cdf(kh[tested], N, K[tested], nh)
    pu = hypergeom.sf(kh[tested] - 1, N, K[tested], nh)
    p[tested] = np.minimum(1, 2 * np.minimum(pl, pu))
    q = np.full(K.size, np.nan)
    q[tested] = multipletests(p[tested], method="fdr_bh")[1]
    T = pd.DataFrame({"chrom": VAR.chromosome.to_numpy(), "start": VAR.start.to_numpy(), "end": VAR.end.to_numpy(),
                      "n_high": nh, "n_low": nl, "open_high": kh, "open_low": kl, "frac_high": fh, "frac_low": fl,
                      "diff_frac": fh - fl, "log2_OR": lor, "tested": tested, "p": p, "q": q}, index=var_names)
    for t in TREATMENTS:
        h_, l_ = hi[H.loc[hi, "treatment"] == t], lo[H.loc[lo, "treatment"] == t]
        T[f"diff_{t}"] = open_counts(h_) / max(len(h_), 1) - open_counts(l_) / max(len(l_), 1)
    T["reproducible_sign"] = np.sign(T[[f"diff_{t}" for t in TREATMENTS]]).eq(np.sign(T.diff_frac), axis=0).all(1) & (T.diff_frac != 0)
    T["DA"] = T.tested & (T.q < Q_DA) & (T.log2_OR.abs() >= MIN_LOG2OR) & (T.diff_frac.abs() >= MIN_DIFF) & T.reproducible_sign
    T["direction"] = np.where(T.DA, np.where(T.diff_frac > 0, "up_in_high", "down_in_high"), "")
    return T


qS = H.d_S.quantile([0.20, 0.80]).to_numpy()
qP = H.d_P.quantile([0.20, 0.80]).to_numpy()
H["S_group"] = np.select([H.d_S <= qS[0], H.d_S >= qS[1]], ["S_low", "S_high"], "mid")
H["P_group"] = np.select([H.d_P <= qP[0], H.d_P >= qP[1]], ["P_low", "P_high"], "mid")
qj = PARAMS["q_joint"]
while True:
    jS, jP = H.d_S.quantile([qj, 1 - qj]).to_numpy(), H.d_P.quantile([qj, 1 - qj]).to_numpy()
    jl, jh = (H.d_S <= jS[0]) & (H.d_P <= jP[0]), (H.d_S >= jS[1]) & (H.d_P >= jP[1])
    if min(jl.sum(), jh.sum()) >= PARAMS["min_joint_cells"] or qj >= 0.4:
        break
    qj += 0.05
note("joint-tail quantile used", qj)
H["J_group"] = np.select([jl, jh], ["J_low", "J_high"], "mid")
CONTRASTS = {"structural": ("S_group", "S_high", "S_low"), "programme": ("P_group", "P_high", "P_low"),
             "joint": ("J_group", "J_high", "J_low")}
summary, SETS = [], {}
for c, (col, gh, gl) in CONTRASTS.items():
    hi, lo = match(H.index[H[col] == gh], H.index[H[col] == gl])
    SETS[c] = (hi, lo)
    H[f"in_matched_{c}"] = H.index.isin(hi.union(lo))
    T = da_table(hi, lo)
    save_tsv(T.rename_axis("peak").reset_index(), f"hsc_da_{c}_high_vs_low.tsv.gz")
    summary.append(dict(contrast=c, n_high=len(hi), n_low=len(lo), n_tested=int(T.tested.sum()), n_DA=int(T.DA.sum()),
                        n_open_in_high=int((T.direction == "up_in_high").sum()),
                        n_closed_in_high=int((T.direction == "down_in_high").sum())))
    if c == "joint":
        TJ = T
save_tsv(pd.DataFrame(summary), "hsc_da_summary.tsv")

hi, lo = SETS["joint"]
a_, b_ = int((H.loc[hi, "state"] == "HSC-II").sum()), int((H.loc[hi, "state"] == "HSC-I").sum())
c_, d_ = int((H.loc[lo, "state"] == "HSC-II").sum()), int((H.loc[lo, "state"] == "HSC-I").sum())
orr, pf = fisher_exact([[a_, b_], [c_, d_]])
save_tsv(pd.DataFrame([dict(group="HH", n=a_ + b_, n_HSCII=a_, frac_HSCII=a_ / (a_ + b_)),
                           dict(group="LL", n=c_ + d_, n_HSCII=c_, frac_HSCII=c_ / (c_ + d_))]).assign(fisher_OR=orr, fisher_p=pf),
             "hsc_hh_ll_state_composition.tsv")
matched = pd.concat([H.loc[hi].assign(group="HH"), H.loc[lo].assign(group="LL")])
save_tsv(matched.reset_index(names="cell"), "hsc_hh_ll_matched_cells.tsv")
da_regions = TJ.index[TJ.DA]
Bm = Bcsr[ROW[matched.index].to_numpy()][:, var_names.get_indexer(da_regions)].toarray().astype(np.uint8).T
save_tsv(pd.DataFrame(Bm, index=da_regions, columns=matched.index).rename_axis("peak").reset_index(),
             "hsc_joint_da_binary_regions_x_cells.tsv.gz")

# ---------------------------------------------------------------- 5. threshold sensitivity with GATA content
for k in ("hsc_motif_names", "hsc_motif_matches", "hsc_motif_peaks", "hsc_peak_gc", "hsc_published_da"):
    input_file(data(k), k)
MOT = pd.read_csv(data("hsc_motif_names"), sep="\t")
MM = scipy.io.mmread(data("hsc_motif_matches")).tocsc()
mpeaks = pd.Index(open(data("hsc_motif_peaks")).read().split())
gcols = np.flatnonzero(MOT.motif_name.isin(PARAMS["gata_motifs"]).to_numpy())
gata = pd.Series(np.asarray(MM[:, gcols].sum(1)).ravel() > 0, index=mpeaks).reindex(var_names).fillna(False).to_numpy()
GC = pd.read_csv(data("hsc_peak_gc"), sep="\t").set_index("peak").reindex(var_names)
width = np.array([int(v.split("-")[2]) - int(v.split("-")[1]) for v in var_names])
strata_peaks = (pd.qcut(GC.gc.rank(method="first"), 10, labels=False).astype(str) + "_" +
                pd.qcut(GC.frac_cells_open_hsc.rank(method="first"), 10, labels=False).astype(str) + "_" +
                pd.qcut(pd.Series(width).rank(method="first"), 3, labels=False).astype(str).to_numpy()).to_numpy()


def cmh(fg, feat, universe):
    tabs = []
    for s in np.unique(strata_peaks[universe]):
        m = universe & (strata_peaks == s)
        a = int((fg & m & feat).sum()); b = int((fg & m & ~feat).sum())
        c = int((~fg & m & feat).sum()); d = int((~fg & m & ~feat).sum())
        if a + b > 0 and c + d > 0:
            tabs.append(np.array([[a, b], [c, d]]) + 0.0)
    st = StratifiedTable(tabs)
    return float(st.oddsratio_pooled), float(st.test_null_odds(correction=True).pvalue)


AU = pd.read_csv(data("hsc_published_da")).set_index("Peak")
pub_II = set(AU.index[(AU.padj < 0.05) & (AU.log2FoldChange > 0)])
pub_I = set(AU.index[(AU.padj < 0.05) & (AU.log2FoldChange < 0)])
sens = []
for qth in QS:
    jS, jP = H.d_S.quantile([qth, 1 - qth]).to_numpy(), H.d_P.quantile([qth, 1 - qth]).to_numpy()
    lo_ = H.index[(H.d_S <= jS[0]) & (H.d_P <= jP[0])]
    hi_ = H.index[(H.d_S >= jS[1]) & (H.d_P >= jP[1])]
    mh, ml = match(hi_, lo_)
    D = da_table(mh, ml)
    op, cl = D.index[D.direction == "up_in_high"], D.index[D.direction == "down_in_high"]
    univ = D.tested.to_numpy()
    fg_cl = var_names.isin(cl)
    add_cl = [r for r in cl if r not in pub_I and r not in pub_II]
    or_g, p_g = cmh(fg_cl, gata, univ) if len(cl) > 10 else (np.nan, np.nan)
    or_ga, p_ga = cmh(var_names.isin(add_cl), gata, univ) if len(add_cl) > 10 else (np.nan, np.nan)
    sens.append(dict(q=round(qth, 3), n_HH=len(hi_), n_LL=len(lo_), n_matched=len(mh),
                     frac_HSCII_HH=float((H.loc[mh, "state"] == "HSC-II").mean()),
                     frac_HSCII_LL=float((H.loc[ml, "state"] == "HSC-II").mean()), n_tested=int(D.tested.sum()),
                     n_open=len(op), n_closed=len(cl), open_in_published_HSCII=int(sum(r in pub_II for r in op)),
                     closed_in_published_HSCI=int(sum(r in pub_I for r in cl)), closed_additional=len(add_cl),
                     open_additional=int(sum((r not in pub_I and r not in pub_II) for r in op)),
                     GATA_frac_closed=float(gata[fg_cl].mean()) if len(cl) else np.nan, GATA_frac_tested=float(gata[univ].mean()),
                     GATA_MH_OR_closed=or_g, GATA_p_closed=p_g, GATA_MH_OR_closed_additional=or_ga, GATA_p_closed_additional=p_ga))
    if np.isclose(qth, qj):
        same = set(op) == set(TJ.index[TJ.direction == "up_in_high"]) and set(cl) == set(TJ.index[TJ.direction == "down_in_high"])
        check("threshold analysis at the joint quantile reproduces the joint DA sets", same)
save_tsv(pd.DataFrame(sens), "hsc_threshold_sensitivity.tsv")

# ---------------------------------------------------------------- 6. continuous association with z_J
Bc = Bcsr.tocsc()
tested_mask = np.asarray(Bc.mean(0)).ravel() >= MIN_DET
z, strat = H.z_J.to_numpy(), H.stratum.to_numpy()
T_ = np.zeros(Bc.shape[1]); V_ = np.zeros(Bc.shape[1]); SS = 0.0; MM_ = np.zeros(Bc.shape[1])
for s in np.unique(strat):
    idx = np.flatnonzero(strat == s); n = idx.size
    zc = z[idx] - z[idx].mean(); ss = float((zc ** 2).sum()); SS += ss
    Bs = Bcsr[idx]
    m = np.asarray(Bs.sum(0)).ravel()
    T_ += np.asarray(Bs.T @ zc).ravel()
    V_ += m * (n - m) / (n * (n - 1)) * ss
    MM_ += m * (n - m) / n
Zst = np.where(V_ > 0, T_ / np.sqrt(np.where(V_ > 0, V_, 1)), 0.0)
r_pb = np.where(MM_ > 0, T_ / np.sqrt(SS * np.where(MM_ > 0, MM_, 1)), 0.0)
pz = 2 * norm.sf(np.abs(Zst))
qz = np.full(Zst.size, np.nan)
qz[tested_mask] = multipletests(pz[tested_mask], method="fdr_bh")[1]
CONT = pd.DataFrame({"Z": Zst, "r_pb": r_pb, "p": pz, "q": qz, "tested": tested_mask, "frac_open": np.asarray(Bc.mean(0)).ravel()},
                    index=var_names)
save_tsv(CONT.rename_axis("peak").reset_index(), "hsc_zJ_region_association.tsv.gz")
order = np.argsort(Zst)
cont = []
for ntop in (355, 1000):
    top_neg = np.zeros(len(var_names), bool)
    top_neg[[i for i in order if tested_mask[i]][:ntop]] = True
    o_, p_ = cmh(top_neg, gata, tested_mask)
    cont.append(dict(selection=f"{ntop} most negative Z", GATA_MH_OR=o_, GATA_p=p_, GATA_frac=float(gata[top_neg].mean()),
                     GATA_frac_tested=float(gata[tested_mask].mean())))
save_tsv(pd.DataFrame(cont), "hsc_zJ_top_regions_gata.tsv")

# ---------------------------------------------------------------- 7. region-set gradients and HSC-II share along z_J
closed, opened = set(TJ.index[TJ.direction == "down_in_high"]), set(TJ.index[TJ.direction == "up_in_high"])
sets = {"closed, in published HSC-I": [r for r in closed if r in pub_I],
        "closed, additional": [r for r in closed if r not in pub_I and r not in pub_II],
        "open, in published HSC-II": [r for r in opened if r in pub_II],
        "open, additional": [r for r in opened if r not in pub_I and r not in pub_II]}
f_all = np.asarray(Bcsr.sum(1)).ravel() / Bcsr.shape[1]
grad = []
for nm, rs in sets.items():
    if not rs:
        continue
    cols = var_names.get_indexer(rs)
    H[f"rel__{nm}"] = (np.asarray(Bcsr[:, cols].sum(1)).ravel() / len(cols)) / f_all
    for st in ["all", "HSC-I", "HSC-II"]:
        sub = H if st == "all" else H[H.state == st]
        for dec, r in sub.groupby("z_decile")[f"rel__{nm}"].agg(["mean", "sem", "size"]).iterrows():
            grad.append(dict(region_set=nm, n_regions=len(rs), state=st, z_decile=int(dec), mean_rel=r["mean"], sem=r["sem"], n_cells=int(r["size"])))
save_tsv(pd.DataFrame(grad), "hsc_regionset_gradient_by_zJ_decile.tsv")
share = H.groupby(["treatment", "z_decile"]).state.apply(lambda s: (s == "HSC-II").mean()).rename("frac_HSCII").reset_index()
share_all = H.groupby("z_decile").state.apply(lambda s: (s == "HSC-II").mean()).rename("frac_HSCII").reset_index().assign(treatment="all")
save_tsv(pd.concat([share_all, share], ignore_index=True), "hsc_HSCII_share_by_zJ_decile.tsv")
keep_cols = ["author_subtype", "state", "treatment", "donor", "count_depth", "stratum", "d_S", "d_P", "u_S", "u_P", "z_J", "z_decile",
             "S_group", "P_group", "J_group", "in_matched_structural", "in_matched_programme", "in_matched_joint"]
save_tsv(H[keep_cols].reset_index(names="cell"), "hsc_cell_joint_scores.tsv.gz")

j = pd.DataFrame(summary).set_index("contrast").loc["joint"]
note("joint DA regions (closing / opening)", [int(j.n_closed_in_high), int(j.n_open_in_high)])
note("HSC-II share HH / LL", [a_ / (a_ + b_), c_ / (c_ + d_)])
note("Spearman d_S vs d_P", float(stats.spearmanr(H.d_S, H.d_P).statistic))
check("matched groups have equal size", all(len(SETS[c][0]) == len(SETS[c][1]) for c in SETS))
check("DA regions satisfy every criterion", bool((TJ[TJ.DA].q < Q_DA).all() and TJ[TJ.DA].reproducible_sign.all()))
finish()
