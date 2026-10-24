"""Figure 2 and its Source Data, generated together from the public measurements.

Every plotted value is recomputed here from the released measurements with the analysis code's own estimators and must
agree with the approved analysis outputs to 1e-9 before anything is drawn:
  a, b  T of each molecule in each array (the Phase 1 estimator, on the condition patterns of anatomy.pattern) against
        05_DATA/Results/directional/phase1_T.csv; the molecule differences against phase1_D.csv; means, the 95% interval
        and the counts against phase1_summary.json
  c     the two SNAr role shares against Anatomy_arrays.csv, and the class subset that selects the N-H structures
        (anatomy.class_pattern on amplitude's nucleophile classes) against Anatomy.json
The structures follow 07_METHODS_AND_PROVENANCE/FIGURE_STRUCTURES_PROMOTION_READY.csv; their selection rules are applied
again to the released rows and must give the same molecules.

Public inputs, and nothing else with a measurement:
  05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz, A1: rows of the 15 arrays with a numeric yield (177,039;
    Hiyama C46 excluded). Rows of the process-development (CMC) screens are dropped as the file is read.
  05_DATA/Analysis/directional_cohorts/DESIGN_COUNTS.csv, DC1: the frozen cohorts (identities and counts, no yields).
Writes 02_FIGURES/Main/Figure2.{pdf,svg,png}, 04_SOURCE_DATA/Figure2/Figure2_SourceData.xlsx and
Figure2_SourceData_provenance.json, and 08_REVIEW_AND_QC/figure2/figure2_plot_readback.json (what each panel draws,
read back from the drawn artists).
    python 06_CODE/Figures/figure2.py             all outputs in place
    python 06_CODE/Figures/figure2.py --out DIR   all outputs into DIR, for comparison with the delivered files
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.colors import to_hex                                         # noqa: E402
from matplotlib.patches import Polygon, Rectangle                            # noqa: E402
from matplotlib.text import Text                                             # noqa: E402
from rdkit import Chem, RDLogger                                             # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "06_CODE/Core/00_common/ctb"))
import fig_style as fs                                                       # noqa: E402
import structure_art as sa                                                   # noqa: E402
from figure1_data import matrix, public_units, write_xlsx                         # noqa: E402
from anatomy import FORMS, Shares, class_pattern, pattern                    # noqa: E402
import amplitude                                                             # noqa: E402  (after anatomy, which sets the path)
from history import ROLE                                                     # noqa: E402

RDLogger.DisableLog("rdApp.*")
RAW = ROOT / "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"
COHORTS = ROOT / "05_DATA/Analysis/directional_cohorts/DESIGN_COUNTS.csv"
RES, DI = ROOT / "05_DATA/Results/directional", ROOT / "05_DATA/Results/display_inputs"
APPROVED = {"phase1_T.csv": RES / "phase1_T.csv", "phase1_D.csv": RES / "phase1_D.csv",
            "phase1_summary.json": RES / "phase1_summary.json", "Anatomy.json": DI / "Anatomy.json",
            "Anatomy_arrays.csv": DI / "Anatomy_arrays.csv"}
CHEMDRAW = ROOT / "02_FIGURES/ChemDraw"
SPEC = ROOT / "07_METHODS_AND_PROVENANCE/FIGURE_STRUCTURES_PROMOTION_READY.csv"
OUT = dict(pdf=ROOT / "02_FIGURES/Main/Figure2.pdf", svg=ROOT / "02_FIGURES/Main/Figure2.svg",
           png=ROOT / "02_FIGURES/Main/Figure2.png", xlsx=ROOT / "04_SOURCE_DATA/Figure2/Figure2_SourceData.xlsx",
           provenance=ROOT / "04_SOURCE_DATA/Figure2/Figure2_SourceData_provenance.json",
           readback=ROOT / "08_REVIEW_AND_QC/figure2/figure2_plot_readback.json")
AMIDE, BH1, BH2, SNAR = "Acylation|1", "Buchwald-Hartwig|1", "Buchwald-Hartwig|2", "SNAr|1"
SEED_BOOT, NB, SEED_CLASS = 20261004, 10000, 20261008
TOL = 1e-9
LABEL_PT = 7.0          # atom-label size of every structure in the figure, as in Figure 1
PNG_DPI = 1200
NAME = {"Cc1ccc(N)cc1": "4-methylaniline", "Cc1ccc(N)nc1": "5-methylpyridin-2-amine", "Cc1ccccc1N": "2-methylaniline",
        "COc1ccc(N)cc1": "4-methoxyaniline", "CSc1ccc(N)cc1": "4-(methylthio)aniline", "CSc1cccc(N)c1": "3-(methylthio)aniline",
        "Cc1ccc(N)c(C)c1": "2,4-dimethylaniline", "Nc1ccc2nccnc2c1": "quinoxalin-6-amine",
        "Cc1cc(F)ccc1C#N": "4-fluoro-2-methylbenzonitrile", "Cc1ccc(=O)[nH]c1": "5-methylpyridin-2(1H)-one",
        "c1cc2[nH]ccc2cn1": "1H-pyrrolo[3,2-c]pyridine", "C1CCNC1": "pyrrolidine", "CC1CCCNC1": "3-methylpiperidine"}
TITLE = dict(a="The same molecule carries more where its role owns",
             b="The same amines carry less in both C–N libraries",
             c="In SNAr the aryl halide owns")
TRANSFORM = dict(
    a="A1 rows of the arrays named in each row. In an array a pair is product-forming if its highest yield over the "
      "conditions is at least 0.10. Condition pattern = each product-forming pair's yields minus its least-squares line on "
      "the condition means (Methods). T(molecule, array) = mean, over ordered pairs of the molecule's product-forming pairs "
      "in that array, of the mean product of the two patterns over the conditions measured for both, divided by the array's "
      "mean squared pattern. Contexts from DESIGN_COUNTS.csv (frozen cohort G1, arrays whose owner holds at least twice its "
      "partner's share): owner = the array in which the molecule's role owns; non-owner = the arrays in which it does not; "
      "a context with two arrays takes the mean of their T. Difference = T(owner) - T(non-owner). 95% interval: 10,000 "
      "resamples of the 23 molecules (numpy default_rng, seed 20261004), 2.5th and 97.5th percentiles of the mean.",
    b="T as in panel a, for the eight amines measured in all three arrays (triple_context in DESIGN_COUNTS.csv), numbered "
      "M1-M8 by heavy atoms, then SMILES. C-N mean = (T in the ligand library + T in the metal library) / 2; R = T(amide) - C-N mean. The two C-N "
      "libraries share no condition identifier.",
    c="SNAr|1 A1 rows, product-forming pairs, patterns as in panel a. Shares: role shares of Figure 1F (Methods) for "
      "SNAr|1, as fractions of the pattern variance. Structures: class = amplitude.nucleophile class of the N-H nucleophile "
      "(role 1); the two most frequent classes with at least three pairs form the class subset (36 pairs), and the two "
      "molecules of each class with the fewest heavy atoms, then SMILES, are drawn.")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix()
_can = {}


def can(s):
    if s not in _can:
        m = Chem.MolFromSmiles(str(s))
        _can[s] = Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else str(s)
    return _can[s]


heavy = lambda s: Chem.MolFromSmiles(s).GetNumHeavyAtoms()


# ====================================================================== values
def condition_patterns(A1, key):
    P, a, b = matrix(A1[A1.screen == key])
    M = P.to_numpy(float)
    act = np.nanmax(M, 1) >= FORMS
    return dict(P=P, M=M, act=act, E=pattern(M[act])[0], a=a[act], b=b[act], pairs=list(P.index[act]), rows=int((A1.screen == key).sum()))


def molecule_T(E, keys):
    """T for every molecule of one role (the Phase 1 estimator): mean pattern product over its ordered pairs of pairs
    sharing a condition, over the array's mean pattern variance."""
    Mk = np.isfinite(E).astype(float); E0 = np.nan_to_num(E); den = Mk @ Mk.T
    C = np.divide(E0 @ E0.T, den, out=np.zeros_like(den), where=den > 0)
    V = (np.nansum(E ** 2, 1) / Mk.sum(1)).mean()
    out = {}
    for u in np.unique(keys):
        idx = np.flatnonzero(keys == u)
        if len(idx) < 2:
            continue
        ok = (den[np.ix_(idx, idx)] > 0) & ~np.eye(len(idx), dtype=bool)
        if ok.sum():
            out[u] = (float(C[np.ix_(idx, idx)][ok].sum() / ok.sum() / V), int(ok.sum()))
    return out


def boot_ci(x, rng):
    x = np.asarray(x, float)
    return [float(v) for v in np.percentile(x[rng.integers(0, len(x), (NB, len(x)))].mean(1), [2.5, 97.5])]


def t_values(A1, DC):
    arrays = sorted(set(DC[(DC.set == "G1") & DC.array_clear_owner].array) | set(DC[DC.triple_context].array) | {SNAR})
    X, rows = {}, []
    for key in arrays:
        X[key] = condition_patterns(A1, key)
        for r, keys in ((1, X[key]["a"]), (2, X[key]["b"])):
            rows += [dict(molecule=can(u), array=key, role=r, T=t, n_ordered_pair_pairs=n) for u, (t, n) in molecule_T(X[key]["E"], keys).items()]
    TT = pd.DataFrame(rows)
    ref = pd.read_csv(APPROVED["phase1_T.csv"])
    ref = ref[ref.array.isin(arrays)]
    J = TT.merge(ref, on=["molecule", "array", "role"], how="outer", suffixes=("", "_approved"), indicator=True)
    assert (J._merge == "both").all() and len(J) == len(ref), J._merge.value_counts().to_dict()
    dT = float((J["T"] - J["T_approved"]).abs().max())
    assert dT < TOL and (J.n_ordered_pair_pairs == J.n_ordered_pair_pairs_approved).all(), dT
    return TT, X, dict(arrays=arrays, molecule_array_role_values=len(TT), max_abs_diff_T=dT)


def panel_a(TT, DC, S, partners):
    Tmap = {(r.molecule, r.array, r.role): r.T for r in TT.itertuples()}
    Nmap = {(r.molecule, r.array, r.role): int(r.n_ordered_pair_pairs) for r in TT.itertuples()}
    prim = DC[(DC.set == "G1") & DC.array_clear_owner]
    rows, arr_rows = [], []
    for m, h in prim.groupby("molecule"):
        o, n = h[h.context == "owner"], h[h.context == "non-owner"]
        to = np.mean([Tmap[(m, r.array, r.role)] for r in o.itertuples()])
        tn = np.mean([Tmap[(m, r.array, r.role)] for r in n.itertuples()])
        rows.append(dict(molecule=m, stratum=h.stratum.iloc[0], owner_arrays="+".join(sorted(o.array)),
                         nonowner_arrays="+".join(sorted(n.array)), T_O=to, T_N=tn, D=to - tn))
        arr_rows += [dict(molecule=m, stratum=h.stratum.iloc[0], array=r.array, array_name=fs.ARRAY_LABEL[r.array], role=int(r.role),
                          role_name=r.role_label, context=r.context, T=Tmap[(m, r.array, r.role)], product_forming_partners=partners[(m, r.array, r.role)],
                          ordered_pairs_of_pairs=Nmap[(m, r.array, r.role)]) for r in h.sort_values(["context", "array"], ascending=[False, True]).itertuples()]
    PM = pd.DataFrame(rows)                                                  # molecule order, as the frozen analysis resamples it
    assert len(PM) == 23
    ci = boot_ci(PM.D, np.random.default_rng(SEED_BOOT))
    am, ac = PM[PM.stratum == "amine"], PM[PM.stratum == "acid"]
    ref = pd.read_csv(APPROVED["phase1_D.csv"])
    mine = PM.sort_values(["stratum", "D"], ascending=[False, False]).reset_index(drop=True)
    assert mine.molecule.tolist() == ref.molecule.tolist() and (mine.owner_arrays == ref.owner_arrays).all() \
        and (mine.nonowner_arrays == ref.nonowner_arrays).all()
    dD = float(np.abs(mine[["T_O", "T_N", "D"]].to_numpy() - ref[["T_O", "T_N", "D"]].to_numpy()).max())
    assert dD < TOL, dD
    A, B, Dsc = S["A"], S["B"], S["descriptives"]
    stats = dict(n=len(PM), mean_D=float(PM.D.mean()), ci=ci, positive=int((PM.D > 0).sum()), mean_T_O=float(PM.T_O.mean()),
                 mean_T_N=float(PM.T_N.mean()),
                 amine=dict(n=len(am), mean_D=float(am.D.mean()), positive=int((am.D > 0).sum()), mean_T_O=float(am.T_O.mean()),
                            mean_T_N=float(am.T_N.mean())),
                 acid=dict(n=len(ac), mean_D=float(ac.D.mean()), positive=int((ac.D > 0).sum()), mean_T_O=float(ac.T_O.mean()),
                           mean_T_N=float(ac.T_N.mean())))
    assert (abs(stats["mean_D"] - A["mean"]) < TOL and np.allclose(ci, A["ci"], atol=1e-12, rtol=0) and stats["positive"] == A["positive"]
            and stats["n"] == A["n"] and abs(stats["amine"]["mean_D"] - B["amine_mean"]) < TOL
            and stats["amine"]["positive"] == B["amine_positive"] and abs(stats["acid"]["mean_D"] - B["acid_mean"]) < TOL
            and abs(stats["mean_T_O"] - Dsc["T_O_mean"]) < TOL and abs(stats["mean_T_N"] - Dsc["T_N_mean"]) < TOL)
    plot = pd.concat([PM[PM.stratum == s].sort_values(["D", "molecule"], ascending=[False, True]) for s in ("amine", "acid")],
                     ignore_index=True)
    plot.insert(0, "plot_row", np.arange(1, len(plot) + 1))
    return plot, pd.DataFrame(arr_rows), stats, dict(max_abs_diff_D=dD)


def panel_b(TT, DC, S, spec, partners):
    Tmap = {(r.molecule, r.array, r.role): r.T for r in TT.itertuples()}
    tri = sorted(DC[DC.triple_context].molecule.unique(), key=lambda m: (heavy(m), m))
    labels = {r["smiles"]: r["label"] for r in spec if r["panel"] == "b"}
    assert [labels.get(m) for m in tri] == [f"M{i}" for i in range(1, 9)], "M1-M8 differ from the structure list"
    rows = []
    for m in tri:
        ta, t1, t2 = Tmap[(m, AMIDE, 1)], Tmap[(m, BH1, 1)], Tmap[(m, BH2, 1)]
        rows.append(dict(label=labels[m], name=NAME[m], smiles=m, heavy_atoms=heavy(m), T_amide=ta, T_BH1=t1, T_BH2=t2,
                         CN_mean=(t1 + t2) / 2, R=ta - (t1 + t2) / 2, amide_above_CN_mean=bool(ta - (t1 + t2) / 2 > 0),
                         amide_above_both=bool(ta > t1 and ta > t2), partners_amide=partners[(m, AMIDE, 1)],
                         partners_BH1=partners[(m, BH1, 1)], partners_BH2=partners[(m, BH2, 1)]))
    R = pd.DataFrame(rows)
    C = S["C"]
    ref = {r["molecule"]: r for r in C["rows"]}
    assert all(abs(round(getattr(r, k), 4) - ref[r.smiles][j]) < 1e-12 for r in R.itertuples()
               for k, j in (("T_amide", "T_amide"), ("T_BH1", "T_BH1"), ("T_BH2", "T_BH2"), ("R", "R")))
    stats = dict(mean_T=dict(amide=float(R.T_amide.mean()), BH1=float(R.T_BH1.mean()), BH2=float(R.T_BH2.mean())),
                 amide_minus_BH1=float((R.T_amide - R.T_BH1).mean()), amide_minus_BH2=float((R.T_amide - R.T_BH2).mean()),
                 mean_R=float(R.R.mean()), R_positive=int((R.R > 0).sum()), above_both=int(R.amide_above_both.sum()),
                 BH2_partners=sorted({int(v) for v in R.partners_BH2}), BH2_range=[float(R.T_BH2.min()), float(R.T_BH2.max())])
    assert (abs(stats["amide_minus_BH1"] - C["mean_amide_minus_BH1"]) < TOL and abs(stats["amide_minus_BH2"] - C["mean_amide_minus_BH2"]) < TOL
            and abs(stats["mean_R"] - C["mean_R"]) < TOL and stats["R_positive"] == C["n_R_positive"] == 6 and stats["above_both"] == 5
            and stats["BH2_partners"] == [2, 3])
    return R, stats


def panel_c(X, J, ARR, spec):
    x = X[SNAR]
    s = Shares(x["E"], x["a"], x["b"]).compute(np.ones(len(x["E"])))
    ref = ARR.set_index("screen").loc[SNAR]
    assert abs(s["1"] - ref.share1) < TOL and abs(s["2"] - ref.share2) < TOL
    assert ROLE[SNAR][0][0].startswith("N") and "halide" in ROLE[SNAR][1][0]
    classify = dict(amplitude.CLASS_ARRAYS)[SNAR]
    raw = [classify(m) for m in x["a"]]
    counts = pd.Series(raw).value_counts()
    top = [c for c in counts.index if c != "other" and counts[c] >= 3][:2]
    lab = [c if c in top else None for c in raw]
    ce = class_pattern(x["E"], x["a"], lab, np.random.default_rng(SEED_CLASS))
    approved = next(e for e in J["substrate_class"] if e["screen"] == SNAR)
    assert (abs(ce["explained"] - approved["explained"]) < TOL and abs(ce["null_median"] - approved["null_median"]) < TOL
            and abs(ce["null_p95"] - approved["null_p95"]) < TOL and ce["p"] == approved["p"] and ce["pairs"] == approved["pairs"] == 36
            and ce["molecules"] == approved["molecules"] and top == approved["classes"])
    subset = pd.DataFrame([dict(pair=p, n_h_nucleophile=a, nucleophile_class=c, aryl_halide=b)
                           for p, a, b, c in zip(x["pairs"], x["a"], x["b"], lab) if c is not None])
    # structures: the electrophile-side example and the two smallest molecules of each class (FIGURE_STRUCTURES rules)
    P = x["P"]; M = x["M"]
    full = [i for i in range(len(P)) if x["act"].tolist()[i] and np.isfinite(M[i]).all()]
    by = {}
    for i in full:
        nuc, hal = P.index[i].split(" >> ")
        by.setdefault(hal, []).append(nuc)
    best = min(by, key=lambda h: (-len(by[h]), heavy(h), h))
    c_spec = [r for r in spec if r["panel"] == "c"]
    assert M.shape[1] == 47 and can(best) == c_spec[0]["smiles"] and len(by[best]) == 4 \
        and {classify(n) for n in by[best]} == {"azole/lactam N–H"}
    smallest = {c: sorted(set(subset[subset.nucleophile_class == c].n_h_nucleophile), key=lambda m: (heavy(m), m))[:2] for c in top}
    shown = {"azole/lactam N–H": [r["smiles"] for r in c_spec if r["label"].startswith("azole")],
             "aliphatic amine": [r["smiles"] for r in c_spec if r["label"].startswith("aliphatic")]}
    assert all([can(m) for m in smallest[c]] == shown[c] for c in top)
    stats = dict(share_nucleophile=float(s["1"]), share_aryl_halide=float(s["2"]), pairs=ce["pairs"], molecules=ce["molecules"],
                 classes=top, product_forming=int(x["act"].sum()), pairs_all=len(P), conditions=int(M.shape[1]),
                 example=dict(smiles=can(best), fully_measured_product_forming_partners=len(by[best]), partners=sorted(by[best])))
    return subset, stats


# ====================================================================== drawing
W, H = fs.WIDTH_MM, 145.0                  # 6 mm less than with the two summary lines under titles a and b
AX0, AX1 = 3.0, 85.0                       # panel a axes, mm
A_YROW0, A_PITCH = 15.6, 3.2               # first amine row centre, row pitch
B0, B1, BY0, BY1 = 101.0, 163.0, 11.0, 72.0  # panel b axes
B_XLIM, B_YLIM = (-0.3, 2.3), (-0.5, 0.72)
STRIP_TOP, STRIP_X0, STRIP_X1 = 80.5, 92.5, 179.5
YC = 114.0                                 # top of panel c
XR, XS = 78.0, 122.0                       # panel c: left edge of the exact-molecule region, of its share axes
MS = 2.7                                   # marker size, pt


def a_rows():
    """Row centres (mm) of panel a: 19 amines, their mean, 4 acids, their mean."""
    am = [A_YROW0 + A_PITCH * i for i in range(19)]
    am_mean = am[-1] + 4.2
    ac0 = am_mean + 11.2
    ac = [ac0 + A_PITCH * i for i in range(4)]
    return am, am_mean, ac, ac[-1] + 4.2


def repel(values, gap):
    """Label positions near the values, at least gap apart, order kept (deterministic)."""
    order = np.argsort(values, kind="stable")
    pos = np.array(values, float)[order].copy()
    for _ in range(200):
        moved = False
        for i in range(1, len(pos)):
            if pos[i] - pos[i - 1] < gap:
                push = (gap - (pos[i] - pos[i - 1])) / 2
                pos[i - 1] -= push; pos[i] += push; moved = True
        if not moved:
            break
    out = np.empty_like(pos); out[order] = pos
    return out




# ====================================================================== read back what was drawn


# ====================================================================== Source Data


def sa_names(spec):
    import chemdraw_structures                                              # file names only; no ChemDraw needed
    return chemdraw_structures.drawing_names(spec)


# ====================================================================== main


