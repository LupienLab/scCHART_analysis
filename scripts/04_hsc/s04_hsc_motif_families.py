"""Transcription-factor motif-family enrichment in the heterogeneity-associated regions (Fig. 4g).

Universe: peaks tested in the joint HH-LL comparison (s02). Region sets: all regions opening (A) / closing (B) in HH; their
overlap with the published HSC-II (C) / HSC-I (D) sets; additional regions (E open, F closed); the published HSC-II (G) and
HSC-I (H) sets within the universe, and the published sets without HH-LL regions (G', H').
Motif matches: JASPAR 2020 matches per peak (from the author object). A region carries a family if it matches >= 1
member motif. Families: the curated family definitions (family_definitions.tsv) plus the additional families listed below
(defined before testing).
Test: Cochran-Mantel-Haenszel, each set against the rest of the universe, strata = GC decile x HSC-accessibility decile
x width tertile (ranks within the universe); Mantel-Haenszel OR with Robins-Breslow-Greenland 95 % CI; CMH p with
continuity correction; BH q within set. Sensitivity: GC decile x accessibility decile strata.

Run: SCCHART_OUTPUT_DIR=/absolute/output/path python scripts/04_hsc/s04_hsc_motif_families.py
"""
from __future__ import annotations

import sys
from pathlib import Path


EXTRA_FAMILIES = {
    "FOS/JUN (TRE)": ["FOS", "FOSL1", "FOSL2", "JUN(var.2)", "JUNB", "JUND", "JDP2", "FOS::JUN", "FOSB::JUN", "FOSL1::JUN", "FOSL2::JUN", "JUN::JUNB",
                      "FOS::JUNB", "FOSB::JUNB", "FOSL1::JUNB", "FOSL2::JUNB", "FOS::JUND", "FOSL1::JUND", "FOSL2::JUND"],
    "BATF": ["BATF", "BATF3", "BATF::JUN"],
    "BACH/NFE2 (CNC)": ["BACH1", "BACH2", "NFE2", "NFE2L1", "MAF::NFE2"],
    "ATF/CREB (CRE)": ["ATF2", "ATF3", "ATF7", "CREB1", "CREM", "JUN", "JUND(var.2)", "JDP2(var.2)", "FOS::JUN(var.2)", "FOSL1::JUN(var.2)",
                       "FOSL2::JUN(var.2)", "JUN::JUNB(var.2)"],
    "GATA1::TAL1": ["GATA1::TAL1"],
    "posterior HOX (9-13)": ["HOXA9", "HOXA10", "HOXA13", "HOXB9", "HOXB13", "HOXC9", "HOXC10", "HOXC11", "HOXC12", "HOXC13", "HOXD9", "HOXD10",
                             "HOXD11", "HOXD12", "HOXD13"],
    "anterior/central HOX (1-8)": ["HOXA1", "HOXA2", "HOXA4", "HOXA5", "HOXA6", "HOXA7", "HOXB2", "HOXB3", "HOXB4", "HOXB5", "HOXB6", "HOXB7",
                                   "HOXB8", "HOXC4", "HOXC8", "HOXD3", "HOXD4", "HOXD8"],
    "CDX": ["CDX1", "CDX2", "CDX4"],
    "TALE (MEIS/PKNOX/TGIF/PBX)": ["MEIS1", "MEIS1(var.2)", "MEIS2", "MEIS2(var.2)", "MEIS3", "PKNOX1", "PKNOX2", "TGIF1", "TGIF2", "TGIF2LX",
                                   "TGIF2LY", "PBX1", "PBX2", "PBX3"],
    "SPI/PU.1": ["SPI1", "SPIB", "SPIC"],
    "RUNX": ["RUNX2", "RUNX3"],
    "KLF/SP": ["KLF2", "KLF3", "KLF4", "KLF5", "KLF6", "KLF9", "KLF10", "KLF11", "KLF13", "KLF14", "KLF15", "KLF16", "KLF17", "SP1", "SP2", "SP3",
               "SP4", "SP8", "SP9"],
    "NF-Y": ["NFYA", "NFYB", "NFYC"],
}
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

PARAMS = dict(strata_main="GC decile x accessibility decile x width tertile (ranks within universe)",
              strata_sensitivity="GC decile x accessibility decile", extra_families=EXTRA_FAMILIES)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.io  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

RES = RES
input_file(RES / "hsc_da_joint_high_vs_low.tsv.gz", "joint DA table (s02)")
J = pd.read_csv(RES / "hsc_da_joint_high_vs_low.tsv.gz", sep="\t", low_memory=False).set_index("peak")
J["DA"], J["tested"] = J.DA.astype(bool), J.tested.astype(bool)
AU = pd.read_csv(data("hsc_published_da")).set_index("Peak")
aII = set(AU.index[(AU.padj < 0.05) & (AU.log2FoldChange > 0)])
aI = set(AU.index[(AU.padj < 0.05) & (AU.log2FoldChange < 0)])
UNIV = J.index[J.tested]
OPEN, CLOSED = J.index[J.DA & (J.direction == "up_in_high")], J.index[J.DA & (J.direction == "down_in_high")]
S = {"A: all HH-open": OPEN, "B: all HH-closed": CLOSED,
     "C: HH-open in published HSC-II": pd.Index([p for p in OPEN if p in aII]),
     "D: HH-closed in published HSC-I": pd.Index([p for p in CLOSED if p in aI]),
     "E: additional HH-open": pd.Index([p for p in OPEN if p not in aI and p not in aII]),
     "F: additional HH-closed": pd.Index([p for p in CLOSED if p not in aI and p not in aII]),
     "G: published HSC-II": pd.Index(sorted(aII)).intersection(UNIV), "H: published HSC-I": pd.Index(sorted(aI)).intersection(UNIV)}
DA_ALL = set(OPEN) | set(CLOSED)
S["G': published HSC-II, not HH-LL DA"] = pd.Index([p for p in S["G: published HSC-II"] if p not in DA_ALL])
S["H': published HSC-I, not HH-LL DA"] = pd.Index([p for p in S["H: published HSC-I"] if p not in DA_ALL])

for k in ("hsc_motif_names", "hsc_motif_matches", "hsc_motif_peaks", "hsc_peak_gc", "hsc_motif_families", "hsc_published_da"):
    input_file(data(k), k)
MOT = pd.read_csv(data("hsc_motif_names"), sep="\t").set_index("motif_id")
PEAKS = pd.Index(open(data("hsc_motif_peaks")).read().split())
MM = scipy.io.mmread(data("hsc_motif_matches")).tocsr().astype(bool)
check("motif matrix dimensions", MM.shape == (len(PEAKS), len(MOT)), MM.shape)
GCA = pd.read_csv(data("hsc_peak_gc"), sep="\t").set_index("peak")
Gu = GCA.loc[UNIV].copy()
Gu["width"] = UNIV.str.split("-").str[2].astype(int) - UNIV.str.split("-").str[1].astype(int)


def q10(x):
    return pd.qcut(x.rank(method="first"), 10, labels=False).to_numpy()


STRAT_MAIN = q10(Gu.gc) * 100 + q10(Gu.frac_cells_open_hsc) * 10 + pd.qcut(Gu.width.rank(method="first"), 3, labels=False).to_numpy()
STRAT_GCACC = q10(Gu.gc) * 10 + q10(Gu.frac_cells_open_hsc)
MU = MM[PEAKS.get_indexer(UNIV)].tocsc()

FD = pd.read_csv(data("hsc_motif_families"), sep="\t")
FAM = {r.label: [m.strip() for m in r.members_primary.split(",")] for r in FD.itertuples()}
FAM.update(EXTRA_FAMILIES)
name2id = pd.Series(MOT.index, index=MOT.motif_name)
missing = {k: [m for m in v if m not in name2id] for k, v in FAM.items() if any(m not in name2id for m in v)}
check("every family member is a matched motif", not missing, missing)
FAMS = list(FAM)
FAMX = sp.csc_matrix(np.column_stack([np.asarray(MU[:, MOT.index.get_indexer(name2id[FAM[f]].tolist())].sum(1)).ravel() > 0 for f in FAMS]))


def cmh(fg, X, strata):
    a_, E_, V_, R_, S_, PR, PSQR, QS = [np.zeros(X.shape[1]) for _ in range(8)]
    for s in np.unique(strata[fg]):
        ix = strata == s
        f = fg[ix]
        Xs = X[ix]
        nn, nf = ix.sum(), f.sum()
        nb = nn - nf
        if nb == 0:
            continue
        m1 = np.asarray(Xs.sum(0)).ravel().astype(float)
        a = np.asarray(Xs[f].sum(0)).ravel().astype(float)
        b, c = nf - a, m1 - a
        d = nb - c
        a_ += a
        E_ += nf * m1 / nn
        if nn > 1:
            V_ += nf * nb * m1 * (nn - m1) / (nn ** 2 * (nn - 1))
        r, s_ = a * d / nn, b * c / nn
        P, Q = (a + d) / nn, (b + c) / nn
        R_ += r; S_ += s_; PR += P * r; PSQR += P * s_ + Q * r; QS += Q * s_
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = (np.maximum(np.abs(a_ - E_) - 0.5, 0) ** 2) / V_
        p = np.where(V_ > 0, stats.chi2.sf(chi, 1), 1.0)
        orr = R_ / S_
        se = np.sqrt(PR / (2 * R_ ** 2) + PSQR / (2 * R_ * S_) + QS / (2 * S_ ** 2))
        lo, hi = np.exp(np.log(orr) - 1.96 * se), np.exp(np.log(orr) + 1.96 * se)
    return dict(fg_hits=a_.astype(int), expected=E_, obs_over_exp=a_ / E_, MH_OR=orr, OR_ci_lo=lo, OR_ci_hi=hi, p=np.nan_to_num(p, nan=1.0))


def run_sets(X, labels, strata, kind):
    out = []
    for k, v in S.items():
        fg = UNIV.isin(v)
        d = pd.DataFrame(cmh(fg, X, strata))
        d.insert(0, kind, labels)
        d.insert(0, "region_set", k)
        d.insert(2, "n_regions", int(fg.sum()))
        d["bg_regions"] = int((~fg).sum())
        d["bg_hits"] = np.asarray(X[~fg].sum(0)).ravel().astype(int)
        d["fg_frac"], d["bg_frac"] = d.fg_hits / d.n_regions, d.bg_hits / d.bg_regions
        d["q"] = multipletests(d.p, method="fdr_bh")[1]
        out.append(d)
    return pd.concat(out, ignore_index=True)


ENR_F = run_sets(FAMX, np.array(FAMS), STRAT_MAIN, "family")
sensF = run_sets(FAMX, np.array(FAMS), STRAT_GCACC, "family")
ENR_F["sens_GCxAcc_MH_OR"], ENR_F["sens_GCxAcc_q"] = sensF.MH_OR.values, sensF.q.values
ENR_F["n_member_motifs"] = ENR_F.family.map(lambda f: len(FAM[f]))
ENR_M = run_sets(MU, MOT.index.to_numpy(), STRAT_MAIN, "motif_id")
ENR_M.insert(2, "motif", MOT.motif_name.reindex(ENR_M.motif_id).values)
save_tsv(ENR_F, "hsc_motif_family_enrichment.tsv")
save_tsv(ENR_M, "hsc_motif_enrichment.tsv.gz")
save_tsv(pd.DataFrame({"region_set": list(S), "n_regions": [len(v) for v in S.values()]}), "hsc_motif_region_sets.tsv")
g = ENR_F[(ENR_F.region_set == "B: all HH-closed") & (ENR_F.family == "GATA")].iloc[0]
note("GATA in HH-closed regions", dict(fg_hits=int(g.fg_hits), n=int(g.n_regions), MH_OR=float(g.MH_OR), q=float(g.q)))
check("family results finite for the HH-closed set", bool(np.isfinite(ENR_F[ENR_F.region_set == "B: all HH-closed"].MH_OR).all()))
finish()
