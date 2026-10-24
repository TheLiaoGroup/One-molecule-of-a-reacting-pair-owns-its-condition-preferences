"""Fig. S5 | On 217 out-of-array amide pairs the amine again owns.

The 217 product-forming amide pairs of Set A, the first out-of-array evaluation set (SA1, released): both role shares and the share of
pairs sharing neither molecule (anatomy.pattern and anatomy.Shares, the estimator of Fig. 2A), the owner margin (amine minus
acid) and 2,000 molecule resamples (anatomy.molecule_weights). The resamples are the first draws of the frozen evaluation's
generator (seed 20261004; no other array of the set has 50 product-forming pairs), so they follow from the seed. Checked
against reserve_validation.json (e1) before anything is drawn.
Input with a measurement: SA1 (never_analysed_reserve.csv.gz; the rows outside the release rule are dropped on reading).
    python 06_CODE/Figures/ed05.py [--out DIR]
"""
import argparse
import json

import numpy as np
import pandas as pd

import ed_common as E
from ed_common import fs
import anatomy
import history

N = 5
SET_A = E.DATA / "Validation_Set_A/never_analysed_reserve.csv.gz"
TRANSFER = E.DATA / "Results/reserve_validation/record_transfer_losses.csv"
FROZEN = E.DATA / "Results/reserve_validation/reserve_validation.json"
SEED, NBOOT, MIN_PAIRS, N_SA1 = 20261004, 2000, 50, 29517
TITLE = "On 217 out-of-screen amide pairs the amine again owns"
TITLES = dict(a="Role shares on the 217 pairs", b="The amine's margin over 2,000 resamples")
NOTE = "Set A (SA1), product-forming amide pairs; estimator of Fig. 2A"
TRANSFORM = dict(
    a="Product-forming amide pairs of SA1 (best measured yield at least 10%). Condition pattern = residual after an intercept and "
      "slope on the condition means of these pairs; share of a role = mean product of the patterns of pairs sharing only that "
      "molecule / mean pattern variance; 'neither' = the same for pairs sharing no molecule.",
    b="2,000 resamples of the molecules of each role (anatomy.molecule_weights) with the generator default_rng(20261004), drawn "
      "before any other use of it; each resample's margin = amine share minus acid share. 95% interval: 2.5th and 97.5th "
      "percentiles (linear interpolation); the fraction of resamples with a positive margin.")
H = 62.0
num = lambda v, d=3: f"{v:.{d}f}".replace("-", "−")


def public_sa1():
    """SA1 by the release rule: the record-transfer targets and the product-forming amide pairs."""
    SA = pd.read_csv(SET_A, low_memory=False)
    key = SA.screen + "|" + SA.subpair
    tl = pd.read_csv(TRANSFER)
    best = pd.to_numeric(SA["yield"], errors="coerce").groupby(key).max()
    keep = set(tl.screen + "|" + tl.subpair) | {k for k, v in best.items() if k.startswith("Acylation|1|") and v >= anatomy.FORMS}
    SA = SA[key.isin(keep)].assign(**{"yield": lambda d: pd.to_numeric(d["yield"], errors="coerce")})
    assert len(SA) == N_SA1, len(SA)
    return SA


def compute(SA, owner):
    rng = np.random.default_rng(SEED)
    out = None
    for key, A in history.partner_arrays(SA).items():                # the frozen loop: only arrays with 50 product-forming pairs draw
        keep = np.nanmax(A["M"], 1) >= anatomy.FORMS
        if keep.sum() < MIN_PAIRS:
            continue
        assert key == "Acylation|1" and out is None
        E_ = anatomy.pattern(A["M"][keep])[0]
        a, b = A["a"][keep], A["b"][keep]
        S = anatomy.Shares(E_, a, b)
        s = S.compute(np.ones(int(keep.sum())))
        own = owner[key]
        draws = np.empty(NBOOT)
        for t in range(NBOOT):
            d = S.compute(anatomy.molecule_weights(a, b, rng))
            draws[t] = d[str(own)] - d[str(3 - own)]
        out = dict(pairs=int(keep.sum()), share1=float(s["1"]), share2=float(s["2"]), neither=float(s["N"]), owner=own,
                   margin=float(s[str(own)] - s[str(3 - own)]), ci=[float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))],
                   positive=float(np.nanmean(draws > 0)), draws=draws)
    ref = next(r for r in json.loads(FROZEN.read_text(encoding="utf-8"))["e1"] if r["screen"] == "Acylation|1")
    assert out["pairs"] == ref["reserve_product_forming"] == 217 and ref["paper_owner_role"] == out["owner"]
    dev = max(abs(out["share1"] - ref["share_role1"]), abs(out["share2"] - ref["share_role2"]), abs(out["neither"] - ref["share_neither"]),
              abs(out["margin"] - ref["owner_margin"]), abs(out["positive"] - ref["draws_favouring_paper_owner"]),
              *(abs(x - y) for x, y in zip(out["ci"], ref["margin_ci"])))
    assert dev < E.TOL, dev                                          # e1 was stored by pandas to_json, ten significant digits
    return out, dev


def draw(r, roles):
    fig, cv = E.figure(H, "ed5")
    fs.title(fig, E.LEFT, 0.6, TITLE, H).set_gid("title|main")
    cv.text(E.LEFT, 0.6 + 5.8, NOTE, ha="left", va="center", fontsize=fs.PT["tiny"], color=fs.C["muted"], gid="note|main")
    for L, x in (("a", 0.0), ("b", 70.0)):
        fs.letter(fig, x, 11.0, L, H).set_gid(f"letter|{L}")
        fs.title(fig, x + 4.2, 11.35, TITLES[L], H).set_gid(f"title|{L}")
    own, oth = r["owner"], 3 - r["owner"]
    bars = [("owner", f"{roles[own - 1]} (owner)", r[f"share{own}"], fs.C["owner"]),
            ("partner", roles[oth - 1], r[f"share{oth}"], fs.C["partner"]), ("neither", "sharing neither", r["neither"], fs.C["faint"])]
    ax = fs.axes_mm(fig, 13.0, 20.0, 46.0, 30.0, H); ax.set_gid("a|axes")
    ax.set_xlim(-0.6, 2.6); ax.set_ylim(-0.05, 0.4)
    ax.axhline(0, color=fs.C["text"], lw=0.5, zorder=1, gid="a|zero")
    from matplotlib.patches import Rectangle
    for i, (g, lab, v, colr) in enumerate(bars):
        ax.add_patch(Rectangle((i - 0.3, 0), 0.6, v, facecolor=colr, edgecolor="none", gid=f"a|bar|{g}"))
        ax.text(i, max(v, 0) + 0.012, num(v), ha="center", va="bottom", fontsize=fs.PT["tiny"], gid=f"a|value|{g}")
    ax.set_xticks(range(3), [b_[1] for b_ in bars], rotation=0); ax.set_yticks([0, 0.1, 0.2, 0.3])
    ax.tick_params(axis="x", labelsize=fs.PT["tiny"])
    ax.set_ylabel("Share of the condition pattern", labelpad=1.5)
    E.clean(ax)
    bx = fs.axes_mm(fig, 84.0, 20.0, 90.0, 30.0, H); bx.set_gid("b|axes")
    edges = np.linspace(-0.4, 0.8, 49)
    cnt, _ = np.histogram(r["draws"], bins=edges)
    assert cnt.sum() == NBOOT, "a resample falls outside the bins"
    for j, c in enumerate(cnt):
        if c:
            bx.add_patch(Rectangle((edges[j], 0), edges[j + 1] - edges[j], c, facecolor=fs.C["partner"], edgecolor="white", lw=0.2,
                                   gid=f"b|bin|{j}"))
    top = cnt.max() * 1.25
    bx.set_xlim(edges[0], edges[-1]); bx.set_ylim(0, top)
    bx.set_yticks([v for v in bx.get_yticks() if 0 <= v <= top])          # no tick label beyond the plotted range
    bx.plot([0, 0], [0, top], color=fs.C["text"], lw=0.5, zorder=3, gid="b|zero")
    bx.plot([r["margin"], r["margin"]], [0, top * 0.86], color=fs.C["owner"], lw=1.0, zorder=4, gid="b|margin")
    bx.plot(r["ci"], [top * 0.93, top * 0.93], color=fs.C["text"], lw=0.8, zorder=4, gid="b|ci")
    bx.text(r["ci"][1] + 0.015, top * 0.93, f"{num(r['margin'])} ({num(r['ci'][0])} to {num(r['ci'][1])})", ha="left", va="center",
            fontsize=fs.PT["tiny"], gid="b|ci_value")
    bx.text(edges[-1], top * 0.78, f"{int(round(r['positive'] * NBOOT)):,} of {NBOOT:,} resamples > 0", ha="right", va="center",
            fontsize=fs.PT["tiny"], color=fs.C["muted"], gid="b|positive")          # the count: 94.05% rounds to the legend's 94.1
    bx.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8])
    bx.set_xlabel("Amine share minus acid share", labelpad=1.5); bx.set_ylabel("Resamples", labelpad=1.5)
    E.clean(bx)
    return fig, edges, cnt


def source_data(r, roles, edges, cnt):
    own, oth = r["owner"], 3 - r["owner"]
    a = pd.DataFrame([dict(group=f"{roles[own - 1]} (owner)", share=r[f"share{own}"]), dict(group=roles[oth - 1], share=r[f"share{oth}"]),
                      dict(group="pairs sharing neither molecule", share=r["neither"]),
                      dict(group="margin (owner minus partner)", share=r["margin"], ci_lower=r["ci"][0], ci_upper=r["ci"][1],
                           fraction_of_resamples_positive=r["positive"], pairs=r["pairs"])]).astype({"pairs": "Int64"})
    b = pd.DataFrame(dict(resample=np.arange(1, NBOOT + 1), margin=r["draws"]))
    h = pd.DataFrame(dict(bin_left=edges[:-1], bin_right=edges[1:], resamples=cnt))
    readme = pd.DataFrame([
        dict(sheet="ED5a", panel="a", estimand="role shares of the condition pattern on the out-of-array pairs of Set A", rows="one per role, "
             "then the margin with its interval", units="shares", transformation=TRANSFORM["a"]),
        dict(sheet="ED5b_resamples", panel="b", estimand="owner margin under molecule resampling", rows="one per resample (2,000)",
             units="share difference", transformation=TRANSFORM["b"]),
        dict(sheet="ED5b_histogram", panel="b", estimand="as ED5b_resamples", rows="one per histogram bin (width 0.025)",
             units="counts of resamples", transformation="Counts of ED5b_resamples in each bin, as drawn."),
        dict(sheet="(all)", panel="-", estimand="-", rows="-", units="-",
             transformation="Measurements: unit SA1 (Set A) of the data release. Input hashes: "
                            "FigS5_SourceData_provenance.json.")])
    return dict(README=readme, ED5a=a, ED5b_resamples=b, ED5b_histogram=h)


def main(out=None):
    SA = public_sa1()
    owner = E.owner_roles()
    r, dev = compute(SA, owner)
    roles = (history.ROLE["Acylation|1"][0][0].replace("N–H partner", "amine"), history.ROLE["Acylation|1"][1][0])
    fig, edges, cnt = draw(r, roles)
    prov = dict(figure=f"Fig. S{N} | {TITLE}", generated_by=E.rel(__file__),
                inputs=dict(SA1=dict(file=E.rel(SET_A), sha256=E.sha_file(SET_A), rows=int(len(SA))),
                            release_rule=dict(file=E.rel(TRANSFER), sha256=E.sha_file(TRANSFER)),
                            owner_roles=dict(file=E.rel(E.ANATOMY_ARRAYS), sha256=E.sha_file(E.ANATOMY_ARRAYS)),
                            frozen=dict(file=E.rel(FROZEN), sha256=E.sha_file(FROZEN))),
                randomness="2,000 molecule resamples, default_rng(20261004), the generator's first draws", transformation=TRANSFORM,
                result={k: v for k, v in r.items() if k != "draws"}, agreement_with_frozen=float(dev))
    p = E.write(N, fig, H, source_data(r, roles, edges, cnt), prov, out)
    print(f"Fig. S{N}: shares {r['share1']:.3f}/{r['share2']:.3f}/{r['neither']:.3f}; margin {r['margin']:.3f} "
          f"({r['ci'][0]:.3f}-{r['ci'][1]:.3f}); {100 * r['positive']:.1f}% > 0; max |diff| {dev:.1e}; wrote {len(p)} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    main(ap.parse_args().out)
