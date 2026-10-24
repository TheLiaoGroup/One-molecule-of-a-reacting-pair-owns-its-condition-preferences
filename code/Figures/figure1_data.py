"""Figure 1 and its Source Data, generated together from the public measurements.

Every plotted value is recomputed here from the released measurements with the analysis code's own estimators
(06_CODE/Core/00_common/ctb/anatomy.py); the per-array values of panels E and F must agree with the approved analysis
outputs to 1e-9 before anything is drawn. Two values come from the approved outputs because they need the full analysis
run: the 95% interval of the median molecular share (2,000 hierarchical resamples) and the owner named by the likelihood
estimator; rebuild_all.py step 13 recomputes both from the measurements (06_CODE/Checks/frozen_reads_recomputed.py).

Public measurements, 05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz:
  A1  numeric yields of the 15 arrays, Hiyama C46 excluded (177,039 rows): panels B-F
  A3  Hiyama C46, the duplicate of C43 (74 rows): noise calibration of panel E
Rows of the process-development (CMC) screens are dropped as the file is read; no other file of measurements is opened.
Approved outputs, 05_DATA/Results/display_inputs: Anatomy_arrays.csv, Anatomy.json. Structures: 02_FIGURES/ChemDraw
(chemdraw_structures.py 1).

Panels B and C follow one rule: in the C-N ligand library (Buchwald-Hartwig|1), the aryl (pseudo)halide and the amine whose
agreement across their product-forming partners lies closest to the median of their role, among molecules with at least
four product-forming partners. Agreement = mean pairwise Pearson correlation of the condition patterns of those pairs.

Writes 02_FIGURES/Main/Figure1.{pdf,svg,png}, 04_SOURCE_DATA/Figure1/Figure1_SourceData.xlsx and
Figure1_SourceData_provenance.json, and 08_REVIEW_AND_QC/figure1/figure1_plot_readback.json (what each panel draws,
read back from the drawn artists).
    python 06_CODE/Figures/figure1.py             all outputs in place
    python 06_CODE/Figures/figure1.py --out DIR   all outputs into DIR, for comparison with the delivered files
"""
import argparse
import csv
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.colors import Normalize, to_hex                              # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle   # noqa: E402
from matplotlib.text import Text                                             # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "06_CODE/Core/00_common/ctb"))
import fig_style as fs                                                       # noqa: E402
import structure_art as sa                                                   # noqa: E402
from anatomy import FORMS, Shares, electrophile_role, noise_variance, pattern     # noqa: E402
from history import ROLE                                                     # noqa: E402

RAW = ROOT / "05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"
DI = ROOT / "05_DATA/Results/display_inputs"
APPROVED = {n: DI / n for n in ("Anatomy_arrays.csv", "Anatomy.json")}
CHEMDRAW = ROOT / "02_FIGURES/ChemDraw"
SPEC = ROOT / "07_METHODS_AND_PROVENANCE/FIGURE_STRUCTURES_PROMOTION_READY.csv"
OUT = dict(pdf=ROOT / "02_FIGURES/Main/Figure1.pdf", svg=ROOT / "02_FIGURES/Main/Figure1.svg",
           png=ROOT / "02_FIGURES/Main/Figure1.png", xlsx=ROOT / "04_SOURCE_DATA/Figure1/Figure1_SourceData.xlsx",
           provenance=ROOT / "04_SOURCE_DATA/Figure1/Figure1_SourceData_provenance.json",
           readback=ROOT / "08_REVIEW_AND_QC/figure1/figure1_plot_readback.json")
N_A1, N_A3 = 177039, 74
TOL = 1e-9
CN, AMIDE = "Buchwald-Hartwig|1", "Acylation|1"
MIN_PARTNERS = 4        # product-forming partners a molecule needs for an agreement value
CLIP = 30.0             # colour range of panels B and C, percentage points
LABEL_PT = 7.0          # atom-label size of the panel-A structures
PNG_DPI = 1200
NAME = {  # names printed on panels A-C; the panels assert that these are exactly the molecules the rule selects
    "N#Cc1ccnc(Cl)c1": "2-chloro-4-cyanopyridine", "Cc1cc(C)nc(N)c1": "2-amino-4,6-dimethylpyridine",
    "CNc1cccc(C)c1": "N,3-dimethylaniline", "COc1ccc(N)cc1": "4-methoxyaniline", "COc1ccc(N)cc1OC": "3,4-dimethoxyaniline",
    "COc1ccc(N)cn1": "5-amino-2-methoxypyridine", "Cc1ccc(N)c(C)c1": "2,4-dimethylaniline",
    "Cc1nccn1-c1ccc(N)cc1": "4-(2-methylimidazol-1-yl)aniline",
    "CC(=O)c1ccc(Cl)c(C)c1": "4-chloro-3-methylacetophenone", "COc1ccccc1OS(=O)(=O)C(F)(F)F": "2-methoxyphenyl triflate",
    "COc1cnc(Cl)nc1": "2-chloro-5-methoxypyrimidine", "C[Si](C)(C)c1ccccc1OS(=O)(=O)C(F)(F)F": "2-(trimethylsilyl)phenyl triflate",
    "Cc1ccc(OS(=O)(=O)C(F)(F)F)cc1": "4-methylphenyl triflate", "Cc1ccccc1OS(=O)(=O)C(F)(F)F": "2-methylphenyl triflate",
    "Cc1ccnc(OS(=O)(=O)C(F)(F)F)c1": "4-methylpyridin-2-yl triflate", "FC(F)(F)c1cnc(Cl)nc1": "2-chloro-5-(trifluoromethyl)pyrimidine",
    "N#Cc1cc(Br)ccc1F": "5-bromo-2-fluorobenzonitrile"}
TITLE = dict(a="Keep one molecule, replace its partner",
             b="One aryl halide, seven amines: the patterns agree",
             c="One amine, ten aryl (pseudo)halides: the patterns differ",
             d="Halides agree in C–N, amines in amides",
             e="A third of the pattern belongs to the molecules",
             f="One partner carries most of it")
TRANSFORM = dict(
    a="Schematic. The two molecules are those of panels B and C; they form one pair of the C-N ligand library (A1 rows, "
      "best yield in this sheet). Left: conditions chosen by amine class, as practical guides do; no measurement is plotted.",
    bc="A1 rows of the C-N ligand library (Buchwald-Hartwig|1). A pair is product-forming if its highest yield over the "
       "conditions is at least 0.10. Condition mean = mean yield of a condition over the product-forming pairs. Each pair's "
       "yields are fitted by least squares to intercept + slope x condition mean over its conditions; condition pattern = "
       "100 x (yield - fitted yield), in percentage points. Cell colour = pattern clipped to +/-30 points. r = Pearson "
       "correlation of two rows' patterns over the conditions measured for both; mean r = mean over all pairs of rows. "
       "Panel B: the product-forming pairs of 2-chloro-4-cyanopyridine; panel C: those of 2-amino-4,6-dimethylpyridine; "
       "their shared pair is the first row of both. Best yield = highest yield of the pair over the conditions.",
    d="Agreement of a molecule = mean pairwise r (as in panels B and C) over its product-forming pairs, for every molecule "
      "with at least four product-forming partners in the C-N ligand library and in the amide coupling array (A1 rows). "
      "Median per group. The molecules of panels B and C are, for each role of the C-N ligand library, the molecule whose "
      "agreement lies closest to the median of its role.",
    e="For each of the 15 arrays (A1 rows), condition patterns as in panel B over the array's product-forming pairs. "
      "Share of role 1 (2) = mean, over ordered pairs of pairs that share only their role-1 (role-2) molecule, of the "
      "mean product of the two patterns over the conditions measured for both, divided by the mean over pairs of the "
      "mean squared pattern. Molecular = role 1 + role 2. Noise = mean over pairs of the mean model measurement variance "
      "left in a pattern, k x max(0.07 x clip(fitted, 0, 1), 0.012)^2 x (1 - leverage), divided by the same mean squared "
      "pattern; k makes the model reproduce the Hiyama C43/C46 duplicate (C43 from A1, C46 from A3). Unattributed "
      "= 1 - molecular - noise (pair-specific interaction and any error outside the noise model). Neither shared = the "
      "role-share ratio for pairs sharing neither molecule. All x 100 (%).",
    f="Role shares of panel E (x 100, %). Owner = role with the larger share; owner fraction = owner share / (role 1 + "
      "role 2 share); owner >= 2x partner compares the owner share with twice the partner share, a negative partner share "
      "counted as zero. Electrophilic halide = the role named '... halide' (10 arrays). Likelihood owner: approved output "
      "of the Gaussian likelihood fit (anatomy.ml_components), not rerun here.")
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix()
neg = lambda s: s.replace("-", "−")                     # typographic minus in printed numbers


# ====================================================================== public measurements
def public_units():
    cols = ["screen", "condition_id", "subpair", "sub_1_smiles", "sub_2_smiles", "yield"]
    A = pd.read_csv(RAW, usecols=cols, low_memory=False)
    A = A[~A.screen.str.contains("CMC", regex=False)]                         # process-development screens: not used
    y = pd.to_numeric(A["yield"], errors="coerce")
    c46 = (A.screen == "Hiyama|1") & (A.condition_id == "C46")
    A1 = A[y.notna() & ~c46].assign(**{"yield": y[y.notna() & ~c46]})
    A3 = A[c46].assign(**{"yield": y[c46]})
    assert len(A1) == N_A1 and len(A3) == N_A3 and A3["yield"].notna().all(), (len(A1), len(A3))
    assert set(A1.screen) == set(fs.ARRAY_LABEL)
    assert (A1.subpair == A1.sub_1_smiles + " >> " + A1.sub_2_smiles).all()
    return A1, A3


def matrix(g):
    P = g.pivot(index="subpair", columns="condition_id", values="yield")
    a = np.array([p.split(" >> ")[0] for p in P.index]); b = np.array([p.split(" >> ")[1] for p in P.index])
    return P, a, b


def noise_scale(A1, A3):
    """anatomy.hiyama_noise_scale on the released rows: Hiyama C43 from A1, its duplicate C46 from A3."""
    P, _, _ = matrix(A1[A1.screen == "Hiyama|1"])
    d46 = A3.set_index("subpair")["yield"].reindex(P.index).to_numpy(float)
    M = P.to_numpy(float); E, H, fit = pattern(M); j = list(P.columns).index("C43")
    ok = np.isfinite(d46) & np.isfinite(M[:, j])
    e43, e46 = E[ok, j], d46[ok] - fit[ok, j]
    r = np.mean(e43 * e46) / np.mean((e43 ** 2 + e46 ** 2) / 2)
    n_model = np.mean(np.maximum(0.07 * np.clip(fit[ok, j], 0, 1), 0.012) ** 2 * (1 - H[ok, j]))
    return float(r), float((1 - r) * np.mean((e43 ** 2 + e46 ** 2) / 2) / n_model), int(ok.sum()), float(np.corrcoef(e43, e46)[0, 1])


# ====================================================================== panels B-D: patterns of single molecules
def corr(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def array_patterns(A1, key):
    """Condition patterns of the product-forming pairs of one array, conditions in identifier order."""
    P, a, b = matrix(A1[A1.screen == key])
    P = P[sorted(P.columns, key=lambda c: int(c[1:]))]
    M = P.to_numpy(float); act = np.nanmax(M, 1) >= FORMS
    E, _, _ = pattern(M[act])
    return dict(key=key, E=100 * E, Y=M[act], pairs=list(P.index[act]), a=a[act], b=b[act], conds=list(P.columns),
                all_pairs=len(P))


def agreement(AP):
    """Every molecule with at least MIN_PARTNERS product-forming partners: mean pairwise r of its pairs' patterns."""
    rows = []
    for role, mol in (("1", AP["a"]), ("2", AP["b"])):
        for m in sorted(set(mol)):
            ix = np.where(mol == m)[0]
            if len(ix) >= MIN_PARTNERS:
                r = [corr(AP["E"][i], AP["E"][j]) for i, j in itertools.combinations(ix, 2)]
                rows.append(dict(screen=AP["key"], array=fs.ARRAY_LABEL[AP["key"]], role=role, role_name=ROLE[AP["key"]][int(role) - 1][0],
                                 molecule=m, product_forming_partners=len(ix), agreement=float(np.mean(r))))
    return pd.DataFrame(rows)


def examples(D):
    """For each role of the C-N ligand library, the molecule whose agreement lies closest to the role's median."""
    out = {}
    for role, g in D[D.screen == CN].groupby("role"):
        med = g.agreement.median(); d = (g.agreement - med).abs()
        assert (d == d.min()).sum() == 1, role                                  # the rule names one molecule
        out[role] = g.loc[d.idxmin(), "molecule"]
    return out


def example_block(AP, focal, side, other):
    """Rows of panel B (side 1: the halide kept) or C (side 0: the amine kept): the shared pair first, then by name."""
    mol = AP["b"] if side == 1 else AP["a"]
    partner = AP["a"] if side == 1 else AP["b"]
    ix = [int(i) for i in np.where(mol == focal)[0]]
    names = {partner[i]: NAME[partner[i]] for i in ix}
    ix.sort(key=lambda i: (partner[i] != other, names[partner[i]].casefold()))
    assert partner[ix[0]] == other
    Z = AP["E"][ix]
    r = {(p, q): corr(Z[p], Z[q]) for p, q in itertools.combinations(range(len(ix)), 2)}
    best = 100 * np.nanmax(AP["Y"][ix], 1)
    cells = [dict(row=k + 1, partner=NAME[partner[i]], partner_smiles=partner[i], pair=AP["pairs"][i], condition_id=c,
                  column=j + 1, yield_fraction=AP["Y"][i, j], pattern_points=Z[k, j],
                  colour_value_points=float(np.clip(Z[k, j], -CLIP, CLIP)) if np.isfinite(Z[k, j]) else None)
             for k, i in enumerate(ix) for j, c in enumerate(AP["conds"])]
    return dict(focal=focal, Z=Z, labels=[NAME[partner[i]] for i in ix], smiles=[partner[i] for i in ix], best=best, r=r,
                mean_r=float(np.mean(list(r.values()))), cells=pd.DataFrame(cells))


def panels_bcd(A1):
    APs = {k: array_patterns(A1, k) for k in (CN, AMIDE)}
    D = pd.concat([agreement(APs[k]) for k in (CN, AMIDE)], ignore_index=True)
    ex = examples(D)
    amine, halide = ex["1"], ex["2"]
    assert ROLE[CN][0][0].startswith("N") and "halide" in ROLE[CN][1][0], ROLE[CN]   # role 1 the amine, role 2 the halide
    Bk = example_block(APs[CN], halide, 1, amine)
    Ck = example_block(APs[CN], amine, 0, halide)
    for blk, role in ((Bk, "2"), (Ck, "1")):                                    # the panel mean is the molecule's agreement
        v = D[(D.screen == CN) & (D.role == role) & (D.molecule == blk["focal"])].agreement.item()
        assert abs(v - blk["mean_r"]) < 1e-12
    D["example"] = [("B" if m == halide else "C") if (s == CN and m in (halide, amine)) else "" for s, m in zip(D.screen, D.molecule)]
    med = D.groupby(["screen", "role"]).agreement.agg(["median", "size"])
    shared = Bk["cells"][Bk["cells"].row == 1]
    info = dict(conditions=len(APs[CN]["conds"]), pairs=APs[CN]["all_pairs"], product_forming=len(APs[CN]["pairs"]),
                shared_pair_best_yield_pct=float(Bk["best"][0]), shared_pair=shared.pair.iloc[0])
    return Bk, Ck, D, med, info


# ====================================================================== panels E, F: shares of the 15 arrays
def panels_ef(A1, A3, approved_arrays, J):
    r_dup, k, n_dup, pearson_dup = noise_scale(A1, A3)
    assert abs(r_dup - J["share"]["hiyama_duplicate_reliability"]) < TOL and abs(k - J["share"]["noise_scale"]) < TOL
    rows = []
    for key in sorted(fs.ARRAY_LABEL):
        g = A1[A1.screen == key]; P, a, b = matrix(g); M = P.to_numpy(float)
        act = np.nanmax(M, 1) >= FORMS
        E, H, fit = pattern(M[act]); sh = Shares(E, a[act], b[act]); s = sh.compute(np.ones(act.sum()))
        noise = noise_variance(M[act], fit, H, k) / sh.var.mean()
        er = electrophile_role(key)
        own = "1" if s["1"] >= s["2"] else "2"; par = "2" if own == "1" else "1"
        rows.append(dict(screen=key, array=fs.ARRAY_LABEL[key], a1_rows=len(g), conditions=M.shape[1], pairs=len(M),
                         product_forming=int(act.sum()), role1=ROLE[key][0][0], role2=ROLE[key][1][0],
                         share1=s["1"], share2=s["2"], neither=s["N"], molecular=s["1"] + s["2"], noise=noise,
                         combination=1 - s["1"] - s["2"] - noise, owner_role=own, owner=ROLE[key][int(own) - 1][0],
                         partner=ROLE[key][int(par) - 1][0], owner_share=s[own], partner_share=s[par],
                         owner_fraction=s[own] / (s["1"] + s["2"]), twice=bool(s[own] >= 2 * max(s[par], 0.0)),
                         halide=None if er is None else ROLE[key][int(er) - 1][0],
                         halide_owns=None if er is None else bool(s[er] > s["2" if er == "1" else "1"])))
    T = pd.DataFrame(rows).set_index("screen")
    ref = approved_arrays.set_index("screen").loc[T.index]
    diffs = {}
    for mine, theirs in (("pairs", "pairs"), ("product_forming", "product_forming"), ("share1", "share1"),
                         ("share2", "share2"), ("neither", "share_neither"), ("molecular", "share"), ("noise", "noise"),
                         ("combination", "combination"), ("owner_share", "owner_share"),
                         ("partner_share", "partner_share"), ("owner_fraction", "owner_fraction")):
        diffs[mine] = float(np.abs(T[mine].to_numpy(float) - ref[theirs].to_numpy(float)).max())
        assert diffs[mine] < TOL, (mine, diffs[mine])
    assert (T.owner_role.astype(int) == ref.owner_role).all() and (T.owner == ref.owner).all()
    hal = T.halide.notna()
    assert (T.halide_owns[hal].astype(bool) == ref.electrophile_carries[hal].astype(bool)).all()
    T["ml_owner_role"] = ref.ml_owner_role.astype(int).astype(str)          # approved output, not rerun
    T["ml_same_owner"] = T.ml_owner_role == T.owner_role
    med = {c: float(T[c].median()) for c in ("molecular", "combination", "noise", "neither")}
    sj = J["share"]
    for c, jk in (("molecular", "median"), ("combination", "combination_median"), ("noise", "noise_median"),
                  ("neither", "neither_median")):
        assert abs(med[c] - sj[jk]) < TOL, c
    own = dict(median_owner_fraction=float(T.owner_fraction.median()), twice=int(T.twice.sum()),
               same_owner=int(T.ml_same_owner.sum()), halide_arrays=int(hal.sum()),
               halide_owns=int(T.halide_owns[hal].astype(bool).sum()), arrays=len(T))
    o = J["ownership"]
    assert (abs(own["median_owner_fraction"] - o["median_owner_fraction"]) < TOL and own["twice"] == o["owner_at_least_twice_partner"]
            and own["same_owner"] == o["same_owner_both_estimators"] and own["arrays"] == o["arrays"] == J["n_arrays"]
            and own["halide_arrays"] == J["electrophile"]["arrays"] and own["halide_owns"] == J["electrophile"]["carries"])
    assert int(T.pairs.sum()) == J["pairs"] and int(T.product_forming.sum()) == J["product_forming_pairs"]
    halide_first, others = fs.array_order({key: electrophile_role(key) is not None for key in T.index})
    T = T.loc[halide_first + others]
    T["group"] = ["electrophilic halide" if h else "other" for h in T.halide.notna()]
    T["plot_row"] = np.arange(1, len(T) + 1)
    info = dict(noise_calibration=dict(duplicate_pairs=n_dup, duplicate_agreement=r_dup, duplicate_pattern_pearson=pearson_dup, k=k),
                max_abs_diff=diffs, arrays=len(T), pairs=int(T.pairs.sum()), product_forming=int(T.product_forming.sum()),
                a1_rows=int(T.a1_rows.sum()))
    return T, med, own, list(sj["ci"]), info


def panel_a():
    spec = {r["label"].split()[0]: r for r in csv.DictReader(open(SPEC, encoding="utf-8")) if r["figure"] == "1"}
    man = {r["name"]: r for r in csv.DictReader(open(CHEMDRAW / "structures_manifest.csv", encoding="utf-8"))}
    art, S = {}, {}
    for L in ("halide", "amine"):
        m = man[f"Fig1_a_{L}"]
        assert m["smiles"] == spec[L]["smiles"], L
        art[L] = sa.load(CHEMDRAW / f"Fig1_a_{L}.svg"); S[L] = m["smiles"]
        assert art[L]["sha256"] == m["svg_sha256"], f"{L}: SVG differs from the ChemDraw manifest"
    return art, S


# ====================================================================== drawing
W = fs.WIDTH_MM
H = 170.0
YB, YC = 37.0, 86.0                                    # tops of the B/C row and the D/E/F row
ROWH = 2.6                                             # heatmap row height (mm)
BLK = dict(b=dict(x0=0.5, hx0=35.0, hx1=79.5), c=dict(x0=91.0, hx0=128.0, hx1=172.5))   # equal heatmap widths
HTOP = YB + 10.5                                       # top of both heatmaps
YSTRIP = HTOP + 10 * ROWH + 7.5                        # the r strips of B and C
RW, R0 = 26.0, -0.4                                    # r strip: mm per unit r, r at the left edge
DX0, DW = 13.0, 37.0                                   # panel D axes
CX0, CW = 80.0, 42.0                                   # panel E axes
FX0, FW = 132.0, 34.0                                  # panel F axes
ROW0, PITCH, GROUP_GAP = YC + 13.0, 4.0, 3.0           # first array row top, row pitch, gap between the groups
DY = (-0.2, 0.8)                                       # agreement axis of panel D


def row_y(i):
    return ROW0 + PITCH / 2 + PITCH * i + (GROUP_GAP if i >= 10 else 0.0)




# ====================================================================== read back what was drawn


# ====================================================================== Source Data


def write_xlsx(sheets, path):
    from openpyxl.styles import Alignment, Font
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            ws = xw.sheets[name]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.font = Font(bold=True)
            for j, col in enumerate(df.columns, start=1):
                vals = [str(col)] + [("%.6g" % v) if isinstance(v, float) else str(v) for v in df[col].tolist()]
                width = min(max(len(v) for v in vals) + 2, 60 if name != "README" else 110)
                ws.column_dimensions[ws.cell(row=1, column=j).column_letter].width = width
            if name == "README":
                for row in ws.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(wrap_text=True, vertical="top")
    normalise_xlsx(path)


def normalise_xlsx(path):
    """No save time inside the file, so the same data give the same bytes and a delivered copy can be compared byte for byte."""
    import io
    import re
    import zipfile
    with zipfile.ZipFile(path) as src:
        items = [(i, src.read(i.filename)) for i in src.infolist()]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for info, data in items:
            if info.filename == "docProps/core.xml":
                data = re.sub(rb"<dcterms:(?:created|modified)[^>]*>[^<]*</dcterms:(?:created|modified)>", b"", data)
                assert b"dcterms:created" not in data and b"dcterms:modified" not in data
            zi = zipfile.ZipInfo(info.filename, date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type, zi.external_attr = zipfile.ZIP_DEFLATED, info.external_attr
            z.writestr(zi, data)
    Path(path).write_bytes(buf.getvalue())


# ====================================================================== main


