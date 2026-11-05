"""Heterogeneity-associated regions versus the published HSC-I / HSC-II region sets, and their accessibility within each
substate (Fig. 4e,f; Fig. S6a; region table for S6c).

Inputs: differential-accessibility tables and matched groups (s02); published HSC-II vs HSC-I ATAC markers
(HSC-II set: padj < 0.05 and log2FC > 0; HSC-I set: padj < 0.05 and log2FC < 0; tested: non-missing padj).

1. Overlap statistics: for the structural, programme and joint contrasts, regions opening / closing in high-contribution
   cells against the published HSC-II / HSC-I sets, within peaks tested by both analyses; Fisher's exact test; adjusted
   P over the 12 tests computed as p * 12 / rank(p) (as in the original analysis; no monotonicity step).
2. Overlap category of each joint DA region: open/closed in HH x published HSC-II / HSC-I / neither.
3. Within-substate accessibility of the joint DA regions: accessible fractions in matched HH and LL cells within HSC-I and
   HSC-II (same cells, substates not re-matched), deltas, direction agreement; background = the same deltas over every
   tested peak (5th, 50th, 95th percentiles); treatment-standardised deltas (equal-weight mean over treatments with >= 10
   cells in both groups); regions beyond the background 5th-95th percentile range in the HH-LL direction in both substates.
4. Ranking of the additional regions (not in either published set) for display: by q, then |difference|, then the
   smallest per-treatment |difference|; regions within 250 kb of a higher-ranked region are merged into its locus.

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/04_hsc/s03_hsc_published_sets_and_within_substate.py
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

PARAMS = dict(published_sets="padj < 0.05; sign of log2FoldChange (HSC-II vs HSC-I)", min_cells_per_treatment_arm=10,
              background="all tested peaks, 5th/50th/95th percentiles", locus_merge_bp=250_000)

import anndata as ad  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.stats import fisher_exact, spearmanr  # noqa: E402

RES = RES
D = {k: pd.read_csv(RES / f"hsc_da_{k}_high_vs_low.tsv.gz", sep="\t", low_memory=False).set_index("peak")
     for k in ("joint", "structural", "programme")}
for k in D:
    input_file(RES / f"hsc_da_{k}_high_vs_low.tsv.gz", f"{k} DA table (s02)")
CS = pd.read_csv(RES / "hsc_cell_joint_scores.tsv.gz", sep="\t", index_col="cell")
input_file(RES / "hsc_cell_joint_scores.tsv.gz", "HSC cells, groups and matched flags (s02)")
input_file(data("hsc_published_da"), "published HSC-II vs HSC-I ATAC markers")
AU = pd.read_csv(data("hsc_published_da")).set_index("Peak")
AU["pub_tested"] = AU.padj.notna()
AU["pub_HSCII"] = (AU.padj < 0.05) & (AU.log2FoldChange > 0)
AU["pub_HSCI"] = (AU.padj < 0.05) & (AU.log2FoldChange < 0)
check("published peak names equal the atlas peaks", set(AU.index) == set(D["joint"].index))

# ---------------------------------------------------------------- 1. overlap statistics
P = AU[["pub_tested", "pub_HSCII", "pub_HSCI", "log2FoldChange", "padj"]].copy()
for k, d in D.items():
    P[f"{k}_tested"] = d.tested.reindex(P.index).fillna(False).astype(bool)
    da = d.DA.reindex(P.index).fillna(False).astype(bool)
    P[f"{k}_open_high"] = da & (d.direction.reindex(P.index) == "up_in_high")
    P[f"{k}_closed_high"] = da & (d.direction.reindex(P.index) == "down_in_high")
stat = []
for k in D:
    U = P[P[f"{k}_tested"] & P.pub_tested]
    for sc in ("open_high", "closed_high"):
        for pub in ("pub_HSCII", "pub_HSCI"):
            a, b = U[f"{k}_{sc}"].to_numpy(), U[pub].to_numpy()
            t = np.array([[np.sum(a & b), np.sum(a & ~b)], [np.sum(~a & b), np.sum(~a & ~b)]])
            orr, p = fisher_exact(t)
            stat.append(dict(contrast=k, region_set=sc.replace("_high", "_in_high"), published_set=pub.replace("pub_", "published_"),
                             universe=len(U), n_regions=int(a.sum()), n_published=int(b.sum()), intersection=int(t[0, 0]),
                             frac_regions_in_published=t[0, 0] / max(a.sum(), 1), published_rate_in_universe=b.mean(),
                             odds_ratio=orr, fisher_p=p))
S = pd.DataFrame(stat)
S["fisher_p_adjusted"] = np.minimum(1, S.fisher_p * len(S) / S.fisher_p.rank())
save_tsv(S, "hsc_published_overlap_statistics.tsv")

# ---------------------------------------------------------------- 2. categories of joint DA regions
J = D["joint"].join(AU[["pub_tested", "pub_HSCII", "pub_HSCI", "log2FoldChange", "padj"]].rename(
    columns={"log2FoldChange": "published_log2FC_HSCII_vs_HSCI", "padj": "published_padj"}))
J["DA"], J["tested"] = J.DA.astype(bool), J.tested.astype(bool)


def category(r):
    if not r.DA:
        return "not_DA" if r.tested else "not_tested"
    if r.direction == "up_in_high":
        return "open_HH__published_HSCII" if r.pub_HSCII else ("open_HH__published_HSCI" if r.pub_HSCI else "open_HH__neither")
    return "closed_HH__published_HSCI" if r.pub_HSCI else ("closed_HH__published_HSCII" if r.pub_HSCII else "closed_HH__neither")


J["overlap_category"] = J.apply(category, axis=1)
save_tsv(J[J.DA].rename_axis("peak").reset_index(), "hsc_joint_da_region_categories.tsv")

# ---------------------------------------------------------------- 3. within-substate accessibility
input_file(data("zeng_h5ad"), "Zeng et al. ATAC peak counts")
A = ad.read_h5ad(data("zeng_h5ad"))
var_names = pd.Index(A.var_names.astype(str))
rows = pd.Index(A.obs_names).get_indexer(CS.index)
X = sparse.csr_matrix(A.X)[rows]
del A
TESTED = J.index[J.tested]
XT = (X[:, var_names.get_indexer(TESTED)] > 0).astype(np.float32).tocsr()
del X
RIX = pd.Series(np.arange(len(CS)), index=CS.index)


def frac(cells, cols=None):
    f = pd.Series(np.asarray(XT[RIX[cells].to_numpy()].mean(0)).ravel(), index=TESTED)
    return f if cols is None else f.loc[cols]


M = CS[CS.in_matched_joint]
HH, LL = M.index[M.J_group == "J_high"], M.index[M.J_group == "J_low"]
SUB = CS.state
DAP = J.index[J.DA]
FR = pd.DataFrame({"frac_LL": frac(LL, DAP), "frac_HH": frac(HH, DAP),
                   "frac_allHSC_HSCI": frac(CS.index[SUB == "HSC-I"], DAP), "frac_allHSC_HSCII": frac(CS.index[SUB == "HSC-II"], DAP),
                   "frac_LL_HSCI": frac(LL[SUB[LL] == "HSC-I"], DAP), "frac_HH_HSCI": frac(HH[SUB[HH] == "HSC-I"], DAP),
                   "frac_LL_HSCII": frac(LL[SUB[LL] == "HSC-II"], DAP), "frac_HH_HSCII": frac(HH[SUB[HH] == "HSC-II"], DAP)})
check("recomputed HH/LL fractions equal the DA table", np.allclose(FR.frac_HH, J.loc[DAP, "frac_high"]) and np.allclose(FR.frac_LL, J.loc[DAP, "frac_low"]))
W = J.loc[DAP].join(FR.drop(columns=["frac_LL", "frac_HH"]))
W["delta_all"] = W.frac_high - W.frac_low
W["delta_HSCI"] = W.frac_HH_HSCI - W.frac_LL_HSCI
W["delta_HSCII"] = W.frac_HH_HSCII - W.frac_LL_HSCII
W["delta_published_state"] = W.frac_allHSC_HSCII - W.frac_allHSC_HSCI
sgn = np.sign(W.delta_all)
W["same_direction_within_HSCI"] = np.sign(W.delta_HSCI) == sgn
W["same_direction_within_HSCII"] = np.sign(W.delta_HSCII) == sgn
W["same_direction_within_both_states"] = W.same_direction_within_HSCI & W.same_direction_within_HSCII

GRP4 = {"HH x HSC-I": HH[SUB[HH] == "HSC-I"], "HH x HSC-II": HH[SUB[HH] == "HSC-II"],
        "LL x HSC-I": LL[SUB[LL] == "HSC-I"], "LL x HSC-II": LL[SUB[LL] == "HSC-II"]}
CNT = pd.DataFrame({g: {"n_cells": len(c), **CS.loc[c, "treatment"].value_counts().reindex(["PBS", "TNF", "LPS"]).fillna(0).astype(int).add_prefix("n_").to_dict(),
                        "median_count_depth": CS.loc[c, "count_depth"].median()} for g, c in GRP4.items()}).T
BGD = pd.DataFrame({"delta_HSCI": frac(GRP4["HH x HSC-I"]) - frac(GRP4["LL x HSC-I"]),
                    "delta_HSCII": frac(GRP4["HH x HSC-II"]) - frac(GRP4["LL x HSC-II"])})
BG = BGD.describe(percentiles=[0.05, 0.5, 0.95]).T[["mean", "5%", "50%", "95%"]]
BG["frac_peaks_negative"] = (BGD < 0).mean()
arms = {}
for st, lab in (("HSC-I", "HSCI"), ("HSC-II", "HSCII")):
    hh, ll = HH[SUB[HH] == st], LL[SUB[LL] == st]
    arms[lab] = [t for t in ("PBS", "TNF", "LPS") if min((CS.loc[hh, "treatment"] == t).sum(), (CS.loc[ll, "treatment"] == t).sum()) >= PARAMS["min_cells_per_treatment_arm"]]
    W[f"delta_{lab}_trt_std"] = np.mean([frac(hh[CS.loc[hh, "treatment"] == t], DAP) - frac(ll[CS.loc[ll, "treatment"] == t], DAP) for t in arms[lab]], axis=0)
    W[f"delta_{lab}_minus_bg_median"] = W[f"delta_{lab}"] - BG.loc[f"delta_{lab}", "50%"]
    W[f"beyond_bg_5_95_{lab}"] = np.where(sgn > 0, W[f"delta_{lab}"] > BG.loc[f"delta_{lab}", "95%"], W[f"delta_{lab}"] < BG.loc[f"delta_{lab}", "5%"])
W["same_direction_both_trt_std"] = (np.sign(W.delta_HSCI_trt_std) == sgn) & (np.sign(W.delta_HSCII_trt_std) == sgn)
W["same_direction_both_bg_adjusted"] = (np.sign(W.delta_HSCI_minus_bg_median) == sgn) & (np.sign(W.delta_HSCII_minus_bg_median) == sgn)
W["beyond_bg_5_95_both"] = W.beyond_bg_5_95_HSCI & W.beyond_bg_5_95_HSCII
note("treatment arms used", arms)

ADD = W[W.overlap_category.isin(["open_HH__neither", "closed_HH__neither"])]
rows_ = []
for lab, d in (("additional open in HH", ADD[ADD.direction == "up_in_high"]), ("additional closed in HH", ADD[ADD.direction == "down_in_high"]),
               ("all additional", ADD), ("open_HH x published HSC-II", W[W.overlap_category == "open_HH__published_HSCII"]),
               ("closed_HH x published HSC-I", W[W.overlap_category == "closed_HH__published_HSCI"])):
    al = np.sign(d.delta_all)
    rows_.append(dict(set=lab, n=len(d), n_same_direction_both=int(d.same_direction_within_both_states.sum()),
                      n_same_direction_HSCI=int(d.same_direction_within_HSCI.sum()), n_same_direction_HSCII=int(d.same_direction_within_HSCII.sum()),
                      n_same_direction_both_trt_std=int(d.same_direction_both_trt_std.sum()),
                      n_same_direction_both_bg_adjusted=int(d.same_direction_both_bg_adjusted.sum()),
                      n_beyond_bg_5_95_both=int(d.beyond_bg_5_95_both.sum()),
                      spearman_HSCI_vs_HSCII=spearmanr(d.delta_HSCI, d.delta_HSCII).statistic if len(d) > 2 else np.nan,
                      median_aligned_delta_HSCI=(d.delta_HSCI * al).median(), median_aligned_delta_HSCII=(d.delta_HSCII * al).median(),
                      spearman_delta_all_vs_published_state=spearmanr(d.delta_all, d.delta_published_state).statistic if len(d) > 2 else np.nan))
SUMW = pd.DataFrame(rows_)

# ---------------------------------------------------------------- 4. ranking of additional regions for display
ranked = []
for d, key in (("up_in_high", "O"), ("down_in_high", "C")):
    C = ADD[ADD.direction == d].copy()
    C["abs_diff"] = C.diff_frac.abs()
    C["min_abs_arm_diff"] = C[["diff_PBS", "diff_TNF", "diff_LPS"]].abs().min(1)
    C = C.sort_values(["q", "abs_diff", "min_abs_arm_diff"], ascending=[True, False, False])
    C["stat_rank"] = np.arange(1, len(C) + 1)
    C["locus_rank"], C["same_locus_as"] = np.nan, ""
    reps = []
    for pk, r in C.iterrows():
        mid = (int(r.start) + int(r.end)) // 2
        hit = [p for p in reps if r.chrom == C.at[p, "chrom"] and abs(mid - (int(C.at[p, "start"]) + int(C.at[p, "end"])) // 2) < PARAMS["locus_merge_bp"]]
        if hit:
            C.at[pk, "same_locus_as"] = hit[0]
            continue
        reps.append(pk)
        C.at[pk, "locus_rank"] = len(reps)
    C["example_id"] = np.where(C.locus_rank.notna(), key + C.locus_rank.fillna(0).astype(int).astype(str), "")
    ranked.append(C)
ADDR = pd.concat(ranked)

save_tsv(CNT.reset_index(names="group"), "hsc_within_substate_group_counts.tsv")
save_tsv(BG.reset_index(names="delta"), "hsc_within_substate_background.tsv")
save_tsv(W.rename_axis("peak").reset_index(), "hsc_joint_da_within_substate.tsv")
save_tsv(SUMW, "hsc_within_substate_summary.tsv")
save_tsv(ADDR.rename_axis("peak").reset_index(), "hsc_additional_regions.tsv")
note("joint DA categories", W.overlap_category.value_counts().to_dict())
check("every joint DA region categorised", int(W.overlap_category.str.startswith(("open", "closed")).sum()) == len(DAP))
check("substate groups partition the matched cells", int(CNT.n_cells.sum()) == len(HH) + len(LL))
finish()
