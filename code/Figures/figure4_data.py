"""Figure 4 and its Source Data, generated together from the public measurements.

Both panels use held-out pairs within the analysis arrays (A1); neither uses the out-of-array evaluation sets of Figure 3.
Every plotted value is recomputed here from A1 with the analysis code's own estimators and must agree with the frozen
outputs before anything is drawn:
  a  depth records (anatomy.depth_records): additive-residual response profiles of single-role histories cut to k = 1, 2,
     3, 4 or 6 training pairings drawn at random, five draws each, on the five history folds; against anatomy_depth.csv,
     then split by the role of the molecule against phase2_depth_by_role.csv. The folds are the history|<array>|cv|0
     entries of the frozen fold registry; the subsets come from one generator (seed 20261003) consumed array by array and
     fold by fold, so the draws of an array depend on every array before it. Both are replayed here and written out with
     the Source Data: the fold of every analysis pair and the five subsets of every record at every k.
  b  equal-depth transport (Phase 3, section 7.2): 908 product-forming targets in 12 arrays, two-partner histories on each
     side (SHA-256 subsets), condition means and patterns recomputed without the target, against phase3_transport.csv and
     phase3_summary.json; whole-array interval from a fresh generator (seed 20261005).
The equal-depth decision and the depth curves of Phase 2, sections 7.3 and 7.4 (decision_panels), are drawn in Extended Data
Fig. 8 (06_CODE/Figures/ed08.py), which imports them from here. The Phase 2 helpers are copied from
06_CODE/Directional_Ownership/phase2_section7.py and checked against it token for token; the other estimators are imported
from the analysis code, as Phase 2 checked its copies against them.
Inputs with a measurement: 05_DATA/Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz, A1 (177,039 rows; process-development
rows dropped on reading). Also read: the fold registry 05_DATA/Analysis/06_METADATA/frozen_folds.json (index lists only;
its fifteen history|<array>|cv|0 entries are used), the owner roles (Anatomy_arrays.csv) and the frozen outputs above.
Writes 02_FIGURES/Main/Figure4.{pdf,svg,png}, 04_SOURCE_DATA/Figure4/Figure4_SourceData.xlsx, Fig4a_history_draws.csv,
Fig4a_folds.csv and Figure4_SourceData_provenance.json, and 08_REVIEW_AND_QC/figure4/figure4_plot_readback.json.
With P2_RELEASE_MODE=public (the rebuild of the public data package) panel a is recomputed from A1 and the released
Fig4a_folds.csv and Fig4a_history_draws.csv instead of the registry replay, and the Phase 2 source is not compared.
    python 06_CODE/Figures/figure4.py             all outputs in place
    python 06_CODE/Figures/figure4.py --out DIR   all outputs into DIR, for comparison with the delivered files
"""
import argparse
import ast
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.colors import to_hex                                         # noqa: E402
from matplotlib.patches import Rectangle                                     # noqa: E402
from matplotlib.text import Text                                             # noqa: E402
from rdkit import Chem, RDLogger                                             # noqa: E402

RDLogger.DisableLog("rdApp.*")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path[:0] = [str(ROOT / "06_CODE/Core/00_common"), str(ROOT / "06_CODE/Core/00_common/ctb")]
import fig_style as fs                                                       # noqa: E402
from figure1_data import public_units, write_xlsx                                 # noqa: E402
import anatomy                                                               # noqa: E402
import common                                                                # noqa: E402
import history                                                               # noqa: E402
from figure4_bootstrap_data import write_seta_bootstrap                        # noqa: E402

DATA = ROOT / "05_DATA"
RAW = DATA / "Release/00_RAW_IMMUTABLE/DATASET_analysis.csv.gz"
FOLDS = DATA / "Analysis/06_METADATA/frozen_folds.json"
PHASE2 = ROOT / "06_CODE/Directional_Ownership/phase2_section7.py"
PUBLIC = os.environ.get("P2_RELEASE_MODE") == "public"                      # released folds and draws; no registry, no Phase 2 source
APPROVED = {"anatomy_depth.csv": DATA / "Results/primary/anatomy_depth.csv",
            "phase2_depth_by_role.csv": DATA / "Results/directional/phase2_depth_by_role.csv",
            "phase3_transport.csv": DATA / "Results/directional/phase3_transport.csv",
            "phase3_summary.json": DATA / "Results/directional/phase3_summary.json",
            "Anatomy_arrays.csv": DATA / "Results/display_inputs/Anatomy_arrays.csv"}
DECISION = {"phase2_decision.csv": DATA / "Results/directional/phase2_decision.csv",          # Fig. S8, lower panels
            "phase2_depth_curve.csv": DATA / "Results/directional/phase2_depth_curve.csv",
            "phase2_summary.json": DATA / "Results/directional/phase2_summary.json",
            "Anatomy_arrays.csv": DATA / "Results/display_inputs/Anatomy_arrays.csv"}
OUT = dict(pdf=ROOT / "02_FIGURES/Main/Figure4.pdf", svg=ROOT / "02_FIGURES/Main/Figure4.svg",
           png=ROOT / "02_FIGURES/Main/Figure4.png", xlsx=ROOT / "04_SOURCE_DATA/Figure4/Figure4_SourceData.xlsx",
           draws=ROOT / "05_DATA/Analysis/figure_display/Fig4a_history_draws.csv", folds=ROOT / "05_DATA/Analysis/figure_display/Fig4a_folds.csv",
           provenance=ROOT / "04_SOURCE_DATA/Figure4/Figure4_SourceData_provenance.json",
           readback=ROOT / "08_REVIEW_AND_QC/figure4/figure4_plot_readback.json")
FORMS, SHRINK, TOPK = 0.10, 4.0, 3                                           # Phase 2's constants
KS, NDRAW74 = (1, 2, 3, 4, 6), 5
DEPTH_ARRAYS = ("Acylation|1", "Suzuki|1", "Buchwald-Hartwig|1")
SEED_B, SEED_C, NBOOT = 20261006, 20261008, 10000
SEED_T, MIN_CELLS = 20261005, 8                                              # equal-depth transport
TOL = 1e-9
PNG_DPI = 1200
C_ORDER = ("Suzuki|1", "Acylation|1", "Buchwald-Hartwig|1")                  # depth curves: wins early, wins late, does not win
TITLE = dict(a="Owner histories accumulate with partners",
             b="At equal depth, the owner's history carries")
NOTE = dict(a="Analysis-array held-out pairs; additive residual yield profiles",
            b="Analysis-array held-out pairs, product-forming; two partners per side; slope-adjusted pattern")
EST = dict(a="additive residual response profile (retains response amplitude)",
           b="slope-adjusted condition pattern at equal depth",
           bc="condition-selection utility from preference-based history ranking")
TRANSFORM = dict(
    a="Analysis arrays (A1), five history folds per array (Fig4a_folds.csv). A record is a held-out product-forming pair "
      "with one of its molecules (owner or partner) that has at least six training pairings in that role. Additive "
      "residual of a pair = its yields minus its own mean, minus the training condition means, plus the training grand "
      "mean. For k = 1, 2, 3, 4, 6 the molecule's profile = sum of the residuals of k of its training pairings drawn at "
      "random (Fig4a_history_draws.csv) / (observed contributions + 4); SSE_k = squared error of the target's residual "
      "against that profile, averaged over five draws; SST = sum of the target's squared residuals. Explained = 1 - sum "
      "SSE_k / sum SST over the records of a side.",
    t="Product-forming targets whose two molecules each have at least two other product-forming partners, arrays with at least ten "
      "(908 targets, 12 arrays). For each target: condition means and slope-adjusted patterns of the other product-forming pairs; "
      "each side's profile from two of its molecule's other pairs (all two-pair subsets, or the 45 with the smallest SHA-256), "
      "projected with the target off an intercept and the condition means; squared errors averaged over subsets. Fraction "
      "explained = 1 - sum SSE / sum SST; D = (sum SSE partner - sum SSE owner) / sum SST. Interval: 10,000 whole-array draws, "
      "seed 20261005 (arrays in sorted identifier order).",
    b="Analysis arrays (A1). Targets: product-forming pairs whose two molecules each have at least two other "
      "product-forming partners, in arrays with at least ten such targets (908); a target is scored when every condition "
      "of its panels and of the trio was measured (728). For each target, condition means and slope-adjusted patterns are "
      "refitted on the other product-forming pairs of its array. A side's panel = the three conditions with the largest "
      "condition mean + that molecule's profile from two of its other pairs (all two-pair subsets, or the 45 with the "
      "smallest SHA-256 of the sorted partner identities), ties to the first condition. Best yield = the panel's largest "
      "measured yield, averaged over subsets. Difference = owner's panel minus partner's, in percentage points; targets "
      "weighted equally. 95% interval: 10,000 draws resampling whole arrays (seed 20261006, arrays in sorted identifier "
      "order).",
    c="As panel b, in the three arrays where at least 50 targets have an owner with at least six other product-forming "
      "partners, with 1, 2, 3, 4 or 6 history partners: five subsets per depth, subset j holding the partners with the "
      "smallest SHA-256 of 'target|j|partner'. Gain = best yield of the history-ranked panel minus that of the "
      "complementary trio (greedy, fitted on all other pairs of the array), in percentage points. Pointwise 95% "
      "intervals: 10,000 draws resampling molecules within the array (seed 20261008 for each curve, molecules in sorted "
      "order, the same draws at every depth).")
sha_file = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
rel = lambda p: Path(p).resolve().relative_to(ROOT).as_posix()


# ====================================================================== Phase 2 helpers (copied; checked in phase2_identical)
def sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_can = {}
def can(s):
    if s not in _can:
        m = Chem.MolFromSmiles(str(s))
        _can[s] = Chem.MolToSmiles(m, isomericSmiles=True) if m is not None else str(s)
    return _can[s]


def subsets_k2(ids, salt=None):
    """All two-pair subsets if at most 45, otherwise the 45 with the smallest SHA-256 of the sorted partner identities."""
    c = list(itertools.combinations(range(len(ids)), 2))
    if len(c) <= 45:
        return c
    return sorted(c, key=lambda ij: sha("\t".join(sorted((ids[ij[0]], ids[ij[1]])))))[:45]


def subsets_74(ids, K, subpair):
    """Note 2: subset j holds the K partners with the smallest SHA-256 of 'subpair|j|partner'."""
    return [tuple(sorted(range(len(ids)), key=lambda i: sha(f"{subpair}|{j}|{ids[i]}"))[:K]) for j in range(NDRAW74)]


def pooled_ci(sums, counts, idx):
    """Pooled mean over resampled units: sum of the drawn units' sums over the sum of their counts."""
    num = sums[idx].sum(1); den = counts[idx].sum(1)
    v = num / den
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def phase2_identical():
    """The helpers above are Phase 2's own, token for token; the imported estimators are the ones Phase 2 copied. (The public
    package does not carry the Phase 2 script; there the helpers are released as they stand.)"""
    if not PUBLIC:
        theirs = {n.name: ast.dump(n) for n in ast.parse(PHASE2.read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
        mine = {n.name: ast.dump(n) for n in ast.parse(Path(__file__).read_text(encoding="utf-8")).body if isinstance(n, ast.FunctionDef)}
        for name in ("sha", "can", "subsets_k2", "subsets_74", "pooled_ci"):
            assert mine[name] == theirs[name], f"{name} differs from Phase 2"
    assert (FORMS, SHRINK, TOPK) == (anatomy.FORMS, history.SHRINK, common.TOPK) and KS == anatomy.K_LIST


# ====================================================================== a: depth records, replayed
class Recorder:
    """The generator handed to depth_records: draws as it would and keeps every subset, in order."""
    def __init__(self, rng):
        self.rng, self.drawn = rng, []

    def choice(self, a, size, replace=True):
        out = self.rng.choice(a, size, replace=replace)
        self.drawn.append(np.asarray(out).copy())
        return out


def released_records(AR, owner):
    """Public mode: the records from A1 and the two released files alone. The draws file holds, record by record in the order
    of the frozen run and with the drawn pairs in drawn order, the five subsets of every k, so anatomy.depth_records'
    arithmetic is repeated on the same rows in the same order and gives the same values."""
    FO, DR = pd.read_csv(OUT["folds"]), pd.read_csv(OUT["draws"])
    nk = len(anatomy.K_LIST)
    assert len(FO) == sum(len(A["pairs"]) for A in AR.values()) and len(DR) % nk == 0
    fold = {key: FO[FO.screen == key].set_index("subpair").fold.reindex(A["pairs"]).to_numpy() for key, A in AR.items()}
    assert all(np.isfinite(f).all() for f in fold.values())
    pos = {key: {sp: i for i, sp in enumerate(A["pairs"])} for key, A in AR.items()}
    resid, recs = {}, []
    for s in range(0, len(DR), nk):
        blk = DR.iloc[s:s + nk]
        key, role, tp = blk.screen.iat[0], int(blk.role.iat[0]), blk.target_pair.iat[0]
        assert (blk.screen == key).all() and (blk.role == role).all() and (blk.target_pair == tp).all() and tuple(blk.k) == anatomy.K_LIST
        A = AR[key]
        p, f = pos[key][tp], int(fold[key][pos[key][tp]])
        if (key, f) not in resid:                                            # training pairs: the array outside the target's fold
            resid[(key, f)] = anatomy.train_residual(A["M"], np.array([q for q in range(len(A["M"])) if fold[key][q] != f]))[0]
        R = resid[(key, f)]
        idx, names = (A["ia"], A["ua"]) if role == 1 else (A["ib"], A["ub"])
        mol = names[idx[p]]
        ok = np.isfinite(R[p]); y = R[p, ok]
        rec = dict(screen=key, role=role, side="owner" if role == owner[key] else "partner", target_pair=tp, molecule=mol, fold=f,
                   pair=p, molecule_index=int(idx[p]), sst=float(np.sum(y ** 2)))
        for k, row in zip(anatomy.K_LIST, blk.itertuples()):
            sse = 0.0
            for d in range(1, anatomy.K_DRAWS + 1):
                sub = [pos[key][f"{mol} >> {o}" if role == 1 else f"{o} >> {mol}"] for o in getattr(row, f"draw_{d}").split(";")]
                assert len(sub) == k and all(fold[key][h] != f for h in sub)
                Z = R[sub][:, ok]
                prof = np.nansum(Z, 0) / (np.isfinite(Z).sum(0) + anatomy.SHRINK)
                sse += np.sum((y - prof) ** 2) / anatomy.K_DRAWS
            rec[f"sse_{k}"] = float(sse)
        recs.append(rec)
    return recs, DR, FO


def depth(A1, owner):
    AR = history.partner_arrays(A1)
    assert list(AR) == sorted(fs.ARRAY_LABEL), list(AR)
    recs, draws, folds_rows = (released_records if PUBLIC else replayed_records)(AR, owner)
    D = pd.DataFrame(recs)
    ref = pd.read_csv(APPROVED["anatomy_depth.csv"])
    assert len(D) == len(ref) and (D.screen.to_numpy() == ref.screen.to_numpy()).all() and (D.role.to_numpy() == ref.role.to_numpy()).all()
    assert (D.pair.to_numpy() == ref.pair.to_numpy()).all() and (D.molecule_index.to_numpy() == ref.molecule.to_numpy()).all()
    dev = max(float(np.abs(D[c].to_numpy() - ref[c].to_numpy()).max()) for c in ["sst"] + [f"sse_{k}" for k in anatomy.K_LIST])
    assert dev < TOL, dev
    curve = []
    for side, g in D.groupby("side"):
        curve += [dict(side=side, k=k, explained=float(1 - g[f"sse_{k}"].sum() / g.sst.sum()), records=len(g),
                       molecules=int(g.groupby(["screen", "molecule"]).ngroups)) for k in anatomy.K_LIST]
    curve = pd.DataFrame(curve)
    byrole = pd.read_csv(APPROVED["phase2_depth_by_role.csv"]).query("array == 'all'").set_index("side")
    for r in curve.itertuples():
        assert abs(r.explained - byrole.loc[r.side, f"R2_{r.k}"]) < TOL and (r.records, r.molecules) == (
            byrole.loc[r.side, "records"], byrole.loc[r.side, "molecules"])
    return D.drop(columns=["pair", "molecule_index"]), curve, pd.DataFrame(draws), pd.DataFrame(folds_rows), dev


def replayed_records(AR, owner):
    """The frozen run replayed: the registry's folds and depth_records with a recording generator."""
    registry = json.loads(FOLDS.read_text(encoding="utf-8"))
    folds_of = {key: [np.asarray(x, int) for x in registry[f"history|{key}|cv|0"]] for key in AR}
    del registry
    rng = Recorder(np.random.default_rng(anatomy.SEED))
    nk = len(anatomy.K_LIST) * anatomy.K_DRAWS
    recs, draws, folds_rows = [], [], []
    for key, A in AR.items():
        folds = folds_of[key]
        assert sorted(np.concatenate(folds).tolist()) == list(range(len(A["M"])))
        fold = np.empty(len(A["M"]), int)
        for j, f in enumerate(folds):
            fold[f] = j + 1
        folds_rows += [dict(screen=key, subpair=sp, fold=int(fold[p])) for p, sp in enumerate(A["pairs"])]
        start = len(rng.drawn)
        new = anatomy.depth_records(A, folds, np.nanmax(A["M"], 1) >= anatomy.FORMS, rng)
        assert len(rng.drawn) - start == len(new) * nk
        for i, r in enumerate(new):
            role, p = int(r["role"]), r["pair"]
            mine, other = (A["a"], A["b"]) if role == 1 else (A["b"], A["a"])
            mol = (A["ua"] if role == 1 else A["ub"])[r["molecule"]]
            assert mine[p] == mol
            recs.append(dict(screen=key, role=role, side="owner" if role == owner[key] else "partner", target_pair=A["pairs"][p],
                             molecule=mol, fold=int(fold[p]), pair=p, molecule_index=r["molecule"], sst=r["sst"],
                             **{f"sse_{k}": r[f"sse_{k}"] for k in anatomy.K_LIST}))
            for j, k in enumerate(anatomy.K_LIST):
                row = dict(screen=key, role=role, target_pair=A["pairs"][p], k=k)
                for d in range(anatomy.K_DRAWS):
                    sub = rng.drawn[start + i * nk + j * anatomy.K_DRAWS + d]
                    assert len(sub) == k == len(set(sub.tolist())) and all(mine[h] == mol and fold[h] != fold[p] for h in sub)
                    assert not any(";" in other[h] for h in sub)
                    row[f"draw_{d + 1}"] = ";".join(other[h] for h in sub)
                draws.append(row)
    return recs, draws, folds_rows


# ====================================================================== b and c: Phase 2's decision loop on A1
def decisions(A1, owner):
    """Phase 2, sections 7.3 and 7.4, on the public rows: the same loop, line for line."""
    from anatomy import pattern
    from history import train_residual
    from common import top_k, greedy_trio
    rec73, rec74, K2_LIST = [], [], []
    for key, g in A1.groupby("screen"):
        P = g.pivot(index="subpair", columns="condition_id", values="yield")
        sp = np.array(P.index); Y = P.to_numpy(float)
        a = np.array([p.split(" >> ")[0] for p in sp]); b = np.array([p.split(" >> ")[1] for p in sp])
        pf = np.nanmax(Y, 1) >= FORMS; pfi = np.flatnonzero(pf)
        own = owner[key]; okey, pkey = (a, b) if own == 1 else (b, a)
        deg_o = {u: int((okey[pfi] == u).sum()) for u in set(okey[pfi])}; deg_p = {u: int((pkey[pfi] == u).sum()) for u in set(pkey[pfi])}
        k2t = [t for t in pfi if deg_o[okey[t]] - 1 >= 2 and deg_p[pkey[t]] - 1 >= 2]
        in73 = set(k2t) if len(k2t) >= 10 else set()
        K2_LIST += [f"{key}\t{sp[t]}" for t in sorted(in73)]
        in74o = {t for t in pfi if deg_o[okey[t]] - 1 >= 6} if key in DEPTH_ARRAYS else set()
        in74p = {t for t in pfi if deg_p[pkey[t]] - 1 >= 6} if key in DEPTH_ARRAYS else set()
        for t in sorted(in73 | in74o | in74p):
            tr = pfi[pfi != t]; Mtr = Y[tr]
            s = np.nanmean(Mtr, 0)
            E = pattern(Mtr)[0]
            R = train_residual(Mtr, np.arange(len(Mtr)))[0]
            trio = greedy_trio(Y[np.arange(len(Y)) != t])
            obs = lambda idx: bool(np.isfinite(Y[t, np.unique(np.concatenate(idx))]).all())

            def panels(hist, subs, Z):
                return [top_k(s + np.nansum(Z[hist[list(S)]], 0) / (np.isfinite(Z[hist[list(S)]]).sum(0) + SHRINK)) for S in subs]
            h_o = np.flatnonzero(okey[tr] == okey[t]); h_p = np.flatnonzero(pkey[tr] == pkey[t])
            ids_o = [can(x) for x in pkey[tr][h_o]]; ids_p = [can(x) for x in okey[tr][h_p]]
            best = lambda pan: float(np.mean([np.max(Y[t, q]) for q in pan]))
            if t in in73:
                so, sp_ = subsets_k2(ids_o), subsets_k2(ids_p)
                r = dict(screen=key, subpair=sp[t], subsets_owner=len(so), subsets_partner=len(sp_), trio_best=float(np.nanmax(Y[t, trio])))
                for ver, Z in (("pref", E), ("add", R)):
                    po, pp = panels(h_o, so, Z), panels(h_p, sp_, Z)
                    ok = obs([trio] + po + pp)
                    r[f"{ver}_scored"] = ok
                    r[f"{ver}_owner_best"] = best(po) if ok else np.nan
                    r[f"{ver}_partner_best"] = best(pp) if ok else np.nan
                rec73.append(r)
            for side, inset, hist, ids in (("owner", in74o, h_o, ids_o), ("partner", in74p, h_p, ids_p)):
                if t not in inset:
                    continue
                pan = {K: panels(hist, subsets_74(ids, K, sp[t]), E) for K in KS}
                if not obs([trio] + [q for K in KS for q in pan[K]]):
                    continue
                mol = okey[t] if side == "owner" else pkey[t]
                for K in KS:
                    rec74.append(dict(screen=key, side=side, molecule=can(mol), subpair=sp[t], K=K,
                                      gain=best(pan[K]) - float(np.max(Y[t, trio]))))
    assert len(K2_LIST) == 908 and sha("\n".join(sorted(K2_LIST))) == "8ca68b30854b53a3ad1bdd4b402dcc93cece5edc51dc36838a931b3c69ccd144"
    D73, D74 = pd.DataFrame(rec73), pd.DataFrame(rec74)
    devs = {}
    for name, mine, keys in (("phase2_decision.csv", D73, ["screen", "subpair"]), ("phase2_depth_curve.csv", D74, ["screen", "side", "subpair", "K"])):
        ref = pd.read_csv(DECISION[name])
        assert list(ref.columns) == list(mine.columns) and len(ref) == len(mine)
        J = mine.merge(ref, on=keys, how="outer", suffixes=("", "_ref"), indicator=True)
        assert (J._merge == "both").all()
        num = [c for c in mine.columns if c not in keys and mine[c].dtype.kind == "f"]
        same = [c for c in mine.columns if c not in keys and c not in num]
        assert all((J[c] == J[c + "_ref"]).all() for c in same), same
        devs[name] = max(float(np.nanmax(np.abs(J[c] - J[c + "_ref"]).to_numpy())) for c in num)
        assert devs[name] < TOL and all((J[c].isna() == J[c + "_ref"].isna()).all() for c in num)
    return D73, D74, devs


def summary_b(D73):
    """Section 7.3: per-array and pooled owner-minus-partner difference (preference ranking), with the frozen interval."""
    arrays = sorted(D73.screen.unique())
    idx = np.random.default_rng(SEED_B).integers(0, len(arrays), (NBOOT, len(arrays)))
    ref = json.loads(DECISION["phase2_summary.json"].read_text(encoding="utf-8"))["7.3"]
    out = {}
    for ver in ("pref", "add"):
        d = D73[D73[f"{ver}_scored"]].copy()
        d["owner_minus_partner"] = d[f"{ver}_owner_best"] - d[f"{ver}_partner_best"]
        d["owner_minus_trio"] = d[f"{ver}_owner_best"] - d.trio_best
        o = dict(targets=len(d), arrays=int(d.screen.nunique()))
        for c in ("owner_minus_partner", "owner_minus_trio"):
            s_ = d.groupby("screen")[c].sum().reindex(arrays).fillna(0).to_numpy()
            n_ = d.groupby("screen")[c].size().reindex(arrays).fillna(0).to_numpy()
            per = d.groupby("screen")[c].mean()
            o[c] = dict(mean_pp=float(100 * d[c].mean()), ci_pp=[100 * x for x in pooled_ci(s_, n_, idx)],
                        arrays_positive=int((per > 0).sum()), per_array_pp={k: float(100 * v) for k, v in per.items()},
                        n={k: int(v) for k, v in d.groupby("screen").size().items()})
            r = ref[ver][c]
            assert o["targets"] == ref[ver]["targets"] and o["arrays"] == ref[ver]["arrays"] and o[c]["arrays_positive"] == r["arrays_positive"]
            assert abs(o[c]["mean_pp"] - r["mean_pp"]) < TOL and np.allclose(o[c]["ci_pp"], r["ci_pp"], atol=1e-10, rtol=0)
            assert o[c]["per_array_pp"].keys() == r["per_array_pp"].keys() and all(abs(o[c]["per_array_pp"][k] - v) < TOL for k, v in r["per_array_pp"].items())
        out[ver] = o
    return out


def summary_c(D74):
    """Section 7.4: per array and side, the mean gain over the trio at each depth with pointwise intervals."""
    ref = json.loads(DECISION["phase2_summary.json"].read_text(encoding="utf-8"))["7.4"]
    out = {}
    for side in ("owner", "partner"):
        for key in DEPTH_ARRAYS:
            d = D74[(D74.side == side) & (D74.screen == key)]
            mols = sorted(d.molecule.unique())
            idx = np.random.default_rng(SEED_C).integers(0, len(mols), (NBOOT, len(mols)))
            curve = []
            for K in KS:
                dk = d[d.K == K]
                s_ = dk.groupby("molecule").gain.sum().reindex(mols).to_numpy(); n_ = dk.groupby("molecule").gain.size().reindex(mols).to_numpy()
                curve.append(dict(K=K, mean_pp=float(100 * dk.gain.mean()), ci_pp=[100 * x for x in pooled_ci(s_, n_, idx)]))
            r = ref[f"{side} | {key}"]
            assert (int(d.subpair.nunique()), len(mols)) == (r["targets"], r["molecules"])
            assert all(c["K"] == q["K"] and abs(c["mean_pp"] - q["mean_pp"]) < TOL and np.allclose(c["ci_pp"], q["ci_pp"], atol=1e-10, rtol=0)
                       for c, q in zip(curve, r["curve"]))
            out[(key, side)] = dict(targets=int(d.subpair.nunique()), molecules=len(mols), curve=curve)
    return out


# ====================================================================== b: equal-depth transport, Phase 3 section 7.2 on A1
def transport(A1, owner):
    rec = []
    for key, g in A1.groupby("screen"):
        P = g.pivot(index="subpair", columns="condition_id", values="yield")
        sp, Y = np.array(P.index), P.to_numpy(float)
        a = np.array([x.split(" >> ")[0] for x in sp]); b = np.array([x.split(" >> ")[1] for x in sp])
        pfi = np.flatnonzero(np.nanmax(Y, 1) >= anatomy.FORMS)
        okey, pkey = (a, b) if owner[key] == 1 else (b, a)
        deg_o = {u: int((okey[pfi] == u).sum()) for u in set(okey[pfi])}; deg_p = {u: int((pkey[pfi] == u).sum()) for u in set(pkey[pfi])}
        k2t = [t for t in pfi if deg_o[okey[t]] - 1 >= 2 and deg_p[pkey[t]] - 1 >= 2]
        if len(k2t) < 10:
            continue
        for t in k2t:
            tr = pfi[pfi != t]; Mtr = Y[tr]
            s = np.nanmean(Mtr, 0); Ep = anatomy.pattern(Mtr)[0]
            o = np.isfinite(Y[t]) & np.isfinite(s)
            assert o.sum() >= MIN_CELLS
            Xd = np.c_[np.ones(o.sum()), s[o]]
            proj = lambda v: v - Xd @ np.linalg.lstsq(Xd, v, rcond=None)[0]
            e = proj(Y[t, o]); r = dict(screen=key, subpair=sp[t], sst=float(np.sum(e ** 2)))
            for side, mk, other in (("owner", okey, pkey), ("partner", pkey, okey)):
                h = np.flatnonzero(mk[tr] == mk[t]); ids = [can(x) for x in other[tr][h]]
                sse = []
                for S in subsets_k2(ids):
                    rows = h[list(S)]
                    prof = np.nansum(Ep[rows], 0) / (np.isfinite(Ep[rows]).sum(0) + SHRINK)
                    sse.append(float(np.sum((e - proj(prof[o])) ** 2)))
                r[f"sse_{side}"] = float(np.mean(sse)); r[f"subsets_{side}"] = len(sse)
            rec.append(r)
    T = pd.DataFrame(rec)
    ref = pd.read_csv(APPROVED["phase3_transport.csv"])
    J = T.merge(ref, on=["screen", "subpair"], how="outer", suffixes=("", "_f"), indicator=True)
    assert (J._merge == "both").all() and len(J) == 908
    dev = max(float((J[c] - J[c + "_f"]).abs().max()) for c in ("sst", "sse_owner", "sse_partner"))
    assert dev < TOL and (J.subsets_owner == J.subsets_owner_f).all() and (J.subsets_partner == J.subsets_partner_f).all()
    T["gain"] = T.sse_partner - T.sse_owner
    A = T.groupby("screen")[["gain", "sst"]].sum()
    ia = np.random.default_rng(SEED_T).integers(0, len(A), (NBOOT, len(A)))
    boot = A.gain.to_numpy()[ia].sum(1) / A.sst.to_numpy()[ia].sum(1)
    per = (A.gain / A.sst)
    out = dict(T=T, D=float(T.gain.sum() / T.sst.sum()), ci=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
               r2_owner=float(1 - T.sse_owner.sum() / T.sst.sum()), r2_partner=float(1 - T.sse_partner.sum() / T.sst.sum()),
               per=per.to_dict(), n=T.groupby("screen").size().to_dict(), dev=dev)
    s = json.loads(APPROVED["phase3_summary.json"].read_text(encoding="utf-8"))["7.2"]
    assert abs(out["D"] - s["D"]) < TOL and np.allclose(out["ci"], s["ci"], atol=1e-12, rtol=0)
    assert abs(out["r2_owner"] - s["r2_owner"]) < TOL and abs(out["r2_partner"] - s["r2_partner"]) < TOL
    assert all(abs(out["per"][k] - v) < TOL for k, v in s["per_array"].items()) and int((per > 0).sum()) == s["arrays_positive"] == 11
    return out


def owner_roles(table=APPROVED):
    return pd.read_csv(table["Anatomy_arrays.csv"]).set_index("screen").owner_role.astype(int).to_dict()


def decision_panels(A1):
    """Phase 2, sections 7.3 and 7.4 (the equal-depth decision and the depth curves), for Fig. S8."""
    phase2_identical()
    D73, D74, dev_bc = decisions(A1, owner_roles(DECISION))
    B, C = summary_b(D73), summary_c(D74)
    hal, other = fs.array_order({k: anatomy.electrophile_role(k) is not None for k in fs.ARRAY_LABEL})
    order_b = [k for k in hal + other if k in B["pref"]["owner_minus_partner"]["per_array_pp"]]
    assert len(order_b) == 12
    groups = {k: ("electrophilic halide" if k in hal else "other") for k in order_b}
    return D73, D74, B, C, order_b, groups, dev_bc


def compute(A1):
    phase2_identical()
    owner = owner_roles()
    D, curve_a, draws, folds, dev_a = depth(A1, owner)
    T = transport(A1, owner)
    info = dict(max_abs_diff=dict(depth_records=dev_a, transport_targets=T["dev"]),
                folds_registry_sha256=None if PUBLIC else sha_file(FOLDS))
    return D, curve_a, draws, folds, T, info


# ====================================================================== drawing
W, H = fs.WIDTH_MM, 62.0
XB = 84.0                                    # left edge of panel b's block (letter at XB - 2)
A_AX = (12.0, 13.0, 50.0, 40.0)              # 4a axes: x, top, width, height (mm)
BAR_AX, D_AX = (96.0, 13.0, 28.0, 40.0), (136.0, 13.0, 40.0, 40.0)    # 4b: the two bars, the per-array differences


def signed(v):
    return f"{v:+.1f}".replace("-", "−")


def offsets(vals, dy, dx):
    """Deterministic swarm (as in Figure 3): each dot, in order of value, takes the smallest sideways offset that keeps it one
    dot width from every dot already placed."""
    placed, out = [], {}
    for i in np.argsort(vals, kind="stable"):
        for off in [0.0] + [s_ * k * dx for k in range(1, 20) for s_ in (1, -1)]:
            if all(abs(vals[i] - pv) >= dy or abs(off - po) >= dx for pv, po in placed):
                placed.append((vals[i], off)); out[i] = off
                break
    return np.array([out[i] for i in range(len(vals))])




# ====================================================================== read back what was drawn


# ====================================================================== Source Data


# ====================================================================== main


