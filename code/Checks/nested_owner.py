"""Owner labels fixed without the scored pair, and ownership within connected pair networks.
The owner label of Fig. 4 came from the full array, which contains the scored pairs; and the
shares assume that a molecule's sampled partners are exchangeable while 13 of 15 pair networks are not connected.

1 History-depth analysis (Fig. 4C), nested: in each array-fold the owner is re-estimated from the training pairs of that fold alone (method-of-moments role
  shares, product-forming gate, pattern recomputed on those pairs); each depth record takes the side of that label. Positive
  control: the labels differ from the full-array owner in 3 of 75 array-folds, all in Chan-Lam coupling and reductive amination,
  as Methods states.
2 Within-array equal-depth analysis (Fig. 4B), nested: for each of the 908 targets the owner is re-estimated from the array without that target (pattern recomputed
  without it, as the transport itself does); where the label flips, the target's owner and partner errors are exchanged.
  Interval: 10,000 whole-array draws, seed 20261005, as the published panel.
3 Connected components: in each array the bipartite network of product-forming pairs is split into connected components; the role
  shares are re-estimated within the largest component alone, and, separately, with every comparison of two pairs restricted to
  pairs of the same component (the neither-shared control included).
Reads the released measurements (through figure1_data.public_units), the owner of every array (Anatomy_arrays.csv), the
Figure 4 Source Data and the frozen history folds. The workbook's Fig4a/Fig4b sheet names and the JSON's fig4a/fig4b keys
retain their established analysis identities; they are not current panel references. Writes
05_DATA/Results/directional/nested_owner_results.json, or the file named by --out; rebuild_all.py runs it on the rebuilt
Figure 4 Source Data and compares the result with the released one.
    python 06_CODE/Checks/nested_owner.py [--source-data XLSX] [--folds CSV] [--out JSON]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

threadpool_limits(1)        # the internal check and public rebuild use the same one-thread numerical path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "06_CODE/Figures")]
import figure1_data as F1                                                   # noqa: E402
from anatomy import Shares, pattern, FORMS                                  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--source-data", default=str(ROOT / "04_SOURCE_DATA/Figure4/Figure4_SourceData.xlsx"))
ap.add_argument("--folds", default=str(ROOT / "05_DATA/Analysis/figure_display/Fig4a_folds.csv"))
ap.add_argument("--out", default=str(ROOT / "05_DATA/Results/directional/nested_owner_results.json"))
ARGS = ap.parse_args()
OUT = Path(ARGS.out)
X4 = pd.read_excel(ARGS.source_data, sheet_name=None)
PUB = pd.read_csv(ROOT / "05_DATA/Results/display_inputs/Anatomy_arrays.csv").set_index("screen")
OWN = PUB.owner_role.astype(int).to_dict()
KS = (1, 2, 3, 4, 6)


def owner_of(M, a, b):
    act = np.nanmax(M, 1) >= FORMS
    E = pattern(M[act])[0]
    s = Shares(E, a[act], b[act]).compute(np.ones(act.sum()))
    return (1 if s["1"] >= s["2"] else 2), s


def arrays():
    A1, _ = F1.public_units()
    out = {}
    for key, g in A1.groupby("screen"):
        P, a, b = F1.matrix(g)
        out[key] = dict(sp=np.array(P.index), M=P.to_numpy(float), a=a, b=b)
    return out


def r2(g, k):
    return float(1 - g[f"sse_{k}"].sum() / g.sst.sum())


def main():
    AR = arrays()
    # the full-array owners reproduce the published labels
    full = {k: owner_of(v["M"], v["a"], v["b"])[0] for k, v in AR.items()}
    assert full == OWN, {k: (full[k], OWN[k]) for k in OWN if full[k] != OWN[k]}
    res = {}

    # ---------------------------------------------------------------- 1 Fig. 4C with training-fold owners
    rec = X4["Fig4a_records"].copy()
    folds = pd.read_csv(ARGS.folds)
    fo = {}
    for key, v in AR.items():
        fmap = folds[folds.screen == key].set_index("subpair").fold.reindex(v["sp"]).to_numpy()
        assert np.isfinite(fmap.astype(float)).all()
        for f in sorted(set(fmap)):
            tr = fmap != f
            fo[(key, int(f))] = owner_of(v["M"][tr], v["a"][tr], v["b"][tr])[0]
    diff = sorted(f"{k}|{f}" for (k, f), o in fo.items() if o != OWN[k])
    assert len(fo) == 75 and len(diff) == 3 and {d.rsplit("|", 1)[0] for d in diff} <= {"Chan-Lam|1", "Reductive-amination|1"}, diff
    assert (rec.side == np.where(rec.role.astype(int) == rec.screen.map(OWN), "owner", "partner")).all()
    rec["side_nested"] = np.where(rec.role.astype(int) == [fo[(s, int(f))] for s, f in zip(rec.screen, rec.fold)], "owner", "partner")
    moved = int((rec.side != rec.side_nested).sum())
    a4 = {}
    for col in ("side", "side_nested"):
        a4[col] = {side: dict(records=len(g), **{f"R2_{k}": r2(g, k) for k in KS}) for side, g in rec.groupby(col)}
        common = rec.groupby("target_pair").filter(lambda g: set(g[col]) == {"owner", "partner"})
        a4[col + "_common"] = {side: dict(records=len(g), targets=int(g.target_pair.nunique()), **{f"R2_{k}": r2(g, k) for k in KS})
                               for side, g in common.groupby(col)}
    published_a = X4["Fig4a"].set_index(["side", "k"])
    for side, values in a4["side"].items():
        for k in KS:
            reference = published_a.loc[(side, k)]
            assert int(reference.records) == values["records"]
            assert np.isclose(100 * values[f"R2_{k}"], reference.additive_profile_explained_pct, atol=1e-10, rtol=0)
    res["fig4a"] = dict(array_folds=75, labels_differing=diff, records_moved=moved, published=a4["side"], nested=a4["side_nested"],
                        published_common=a4["side_common"], nested_common=a4["side_nested_common"])

    # ---------------------------------------------------------------- 2 Fig. 4B with leave-target-out owners
    T = X4["Fig4b_targets"].copy()
    lto, flips = [], []
    for key, g in T.groupby("screen", sort=False):
        v = AR[key]
        idx = {p: i for i, p in enumerate(v["sp"])}
        for p in g.subpair:
            keep = np.arange(len(v["sp"])) != idx[p]
            o = owner_of(v["M"][keep], v["a"][keep], v["b"][keep])[0]
            lto.append(o)
            if o != OWN[key]:
                flips.append(f"{key}\t{p}")
    T = pd.concat([g for _, g in T.groupby("screen", sort=False)], ignore_index=True)
    T["owner_lto"] = lto
    flip = T.owner_lto != T.screen.map(OWN)
    T["so"] = np.where(flip, T.sse_partner, T.sse_owner)
    T["spart"] = np.where(flip, T.sse_owner, T.sse_partner)

    def pooled(d, so, sp_):
        return float((d[sp_].sum() - d[so].sum()) / d.sst.sum())

    G = {k: (g.sst.sum(), g.so.sum(), g.spart.sum(), g.sse_owner.sum(), g.sse_partner.sum()) for k, g in T.groupby("screen")}
    keys = sorted(G)
    rng = np.random.default_rng(20261005)
    draws = []
    for _ in range(10000):
        pick = rng.choice(len(keys), len(keys), replace=True)
        s = np.array([G[keys[i]] for i in pick]).sum(0)
        draws.append((s[2] - s[1]) / s[0])
    per = {k: (G[k][2] - G[k][1]) / G[k][0] for k in keys}
    res["fig4b"] = dict(targets=len(T), flipped_targets=len(flips), flipped=flips,
                        published=dict(owner=float(1 - T.sse_owner.sum() / T.sst.sum()), partner=float(1 - T.sse_partner.sum() / T.sst.sum()),
                                       D=pooled(T, "sse_owner", "sse_partner")),
                        nested=dict(owner=float(1 - T.so.sum() / T.sst.sum()), partner=float(1 - T.spart.sum() / T.sst.sum()),
                                    D=pooled(T, "so", "spart"), D_ci=[float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                                    arrays_positive=int(sum(v > 0 for v in per.values())), arrays=len(per)))
    published_b = X4["Fig4b"].set_index("row_type").loc["pooled"]
    assert int(published_b.targets) == len(T) and int(published_b.arrays) == len(keys)
    for result, column in (("owner", "owner_fraction"), ("partner", "partner_fraction"), ("D", "D")):
        assert np.isclose(res["fig4b"]["published"][result], published_b[column], atol=1e-12, rtol=0)

    # ---------------------------------------------------------------- 3 connected components
    rows = []
    for key, v in AR.items():
        act = np.nanmax(v["M"], 1) >= FORMS
        a, b, M = v["a"][act], v["b"][act], v["M"][act]
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for u, w in zip(a, b):
            parent[find("1:" + u)] = find("2:" + w)
        comp = np.array([find("1:" + u) for u in a])
        labels, counts = np.unique(comp, return_counts=True)
        big = comp == labels[counts.argmax()]
        o_big, s_big = owner_of(M[big], a[big], b[big]) if big.sum() >= 10 else (None, None)
        # all pairs, comparisons restricted to the same component
        E = pattern(M)[0]
        sh = Shares(E, a, b)
        same = comp[:, None] == comp[None, :]
        for k in sh.CR:
            sh.CR[k] = sh.CR[k] * same; sh.MR[k] = sh.MR[k] * same
        s_w = sh.compute(np.ones(len(a)))
        o_w = 1 if s_w["1"] >= s_w["2"] else 2
        rows.append(dict(array=key, owner=OWN[key], components=int(len(labels)), largest_pairs=int(big.sum()), pairs=int(len(a)),
                         largest_owner=o_big, largest_share1=None if s_big is None else float(s_big["1"]),
                         largest_share2=None if s_big is None else float(s_big["2"]), within_owner=o_w,
                         within_share1=float(s_w["1"]), within_share2=float(s_w["2"]), within_neither=float(s_w["N"])))
    C = pd.DataFrame(rows)
    res["components"] = dict(rows=C.to_dict("records"),
                             largest_same_owner=int((C.largest_owner == C.owner).sum()), largest_evaluable=int(C.largest_owner.notna().sum()),
                             within_same_owner=int((C.within_owner == C.owner).sum()),
                             within_molecular_median=float((C.within_share1 + C.within_share2).median()),
                             within_neither_median=float(C.within_neither.median()))
    OUT.write_bytes((json.dumps(res, indent=1, default=float) + "\n").encode("utf-8"))
    print(json.dumps({"4a": {k: res["fig4a"][k] for k in ("labels_differing", "records_moved")},
                      "4a_nested_R2_6": {s: round(100 * v["R2_6"], 1) for s, v in res["fig4a"]["nested"].items()},
                      "4a_nested_common_R2_6": {s: (round(100 * v["R2_6"], 1), v["targets"]) for s, v in res["fig4a"]["nested_common"].items()},
                      "4b": {k: res["fig4b"][k] for k in ("flipped_targets",)}, "4b_nested": res["fig4b"]["nested"],
                      "components": {k: v for k, v in res["components"].items() if k != "rows"}}, indent=1, default=float))


if __name__ == "__main__":
    main()
