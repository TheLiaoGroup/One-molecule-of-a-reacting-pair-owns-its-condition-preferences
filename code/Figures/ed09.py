"""Fig. S9 | Response amplitude and condition preference can belong to different molecules.

For the product-forming pairs of each of the fifteen arrays (A1): each pair's slope on the array's condition means (the same
least-squares fit the pattern uses), centred; the amplitude share of a role = mean product of the centred slopes of pairs
sharing only that molecule / their variance. The preference shares are those of Fig. 2A (anatomy.Shares). Checked against
phase3_amplitude.csv and Anatomy_arrays.csv before anything is drawn. No random step.
Input with a measurement: A1.
    python 06_CODE/Figures/ed09.py [--out DIR]
"""
import argparse

import numpy as np
import pandas as pd

import ed_common as E
from ed_common import fs
import anatomy
import history

N = 9
FROZEN = E.DATA / "Results/directional/phase3_amplitude.csv"
TITLE = "Response amplitude and condition preference can belong to different molecules"
TITLES = dict(a="Preference share (Fig. 2A)", b="Amplitude share")
TRANSFORM = dict(
    a="Share of the condition pattern carried by each role (estimator of Fig. 2A), product-forming pairs.",
    b="Each product-forming pair's slope on the array's condition means (least squares over its measured conditions), centred "
      "over the pairs; share of a role = mean product of the centred slopes of two pairs sharing only that role's molecule / "
      "mean squared centred slope.")
H, TOP, NAME_R, AX_W, PITCH, GAP_G = 76.0, 15.0, 33.0, 52.0, 3.0, 1.4
PANEL_X = dict(a=0.0, b=92.0)


def slopes(Y):
    s = np.nanmean(Y, 0); out = np.full(len(Y), np.nan)
    for i in range(len(Y)):
        o = np.isfinite(Y[i]) & np.isfinite(s)
        X = np.c_[np.ones(o.sum()), s[o]]
        out[i] = (np.linalg.pinv(X.T @ X) @ X.T @ Y[i, o])[1]
    return out


def compute(A1, owner):
    AR = history.partner_arrays(A1)
    ref = pd.read_csv(FROZEN).set_index("array")
    an = pd.read_csv(E.ANATOMY_ARRAYS).set_index("screen")
    rows, dev = [], 0.0
    for key, A in AR.items():
        pf = np.nanmax(A["M"], 1) >= anatomy.FORMS
        Y, a, b = A["M"][pf], A["a"][pf], A["b"][pf]
        b_ = slopes(Y); c = b_ - b_.mean(); V = float(np.mean(c ** 2))
        prod = c[:, None] * c[None, :]; off = ~np.eye(len(c), dtype=bool)
        amp = {}
        for r, keys, other in ((1, a, b), (2, b, a)):
            same = (keys[:, None] == keys[None, :]) & (other[:, None] != other[None, :]) & off
            amp[r] = float(prod[same].mean() / V) if same.any() else np.nan
        E_ = anatomy.pattern(Y)[0]
        s = anatomy.Shares(E_, a, b).compute(np.ones(len(Y)))
        f = ref.loc[key]
        dev = max(dev, abs(amp[1] - f.amplitude_share1), abs(amp[2] - f.amplitude_share2), abs(s["1"] - an.loc[key, "share1"]),
                  abs(s["2"] - an.loc[key, "share2"]))
        own = owner[key]
        rows.append(dict(screen=key, pref1=float(s["1"]), pref2=float(s["2"]), amp1=amp[1], amp2=amp[2],
                         pref_owner=float(s[str(own)]), pref_partner=float(s[str(3 - own)]), amp_owner=amp[own], amp_partner=amp[3 - own]))
    T = pd.DataFrame(rows).set_index("screen")
    assert dev < E.TOL, dev
    T["agrees"] = T.amp_owner >= T.amp_partner
    assert int(T.agrees.sum()) == int(ref.agrees.sum()) == 9 and set(T.index[~T.agrees]) == set(ref.index[~ref.agrees])
    return T, dev


def draw(T, order, group):
    fig, cv = E.figure(H, "ed9")
    fs.title(fig, E.LEFT, 0.6, TITLE, H).set_gid("title|main")
    for j, (side, colr) in enumerate((("preference owner", fs.C["owner"]), ("its partner", fs.C["partner"]))):
        x = 120.0 + 24.0 * j
        cv.plot([x + 0.7], [0.6 + 2.3], marker="o", ms=2.6, mew=0, mfc=colr, ls="none", gid=f"key|{j}")
        cv.text(x + 2.0, 0.6 + 2.3, side, ha="left", va="center", fontsize=fs.PT["tiny"], gid=f"key_label|{j}")
    ys, y_ = {}, PITCH / 2
    for i, k in enumerate(order):
        if i and group[k] != group[order[i - 1]]:
            y_ += GAP_G
        ys[k] = y_; y_ += PITCH
    hb = y_ + 0.4
    for L, (oc, pc) in (("a", ("pref_owner", "pref_partner")), ("b", ("amp_owner", "amp_partner"))):
        x0 = PANEL_X[L]
        fs.letter(fig, x0, 7.0, L, H).set_gid(f"letter|{L}")
        fs.title(fig, x0 + 4.2, 7.35, TITLES[L], H).set_gid(f"title|{L}")
        ax = fs.axes_mm(fig, x0 + NAME_R + 1.5, TOP + 3.0, AX_W, hb, H); ax.set_gid(f"axes|{L}")
        ax.set_xlim(-0.45, 0.75); ax.set_ylim(hb, 0)                     # one range for both panels; the lowest share is -0.37
        ax.plot([0, 0], [0, hb], color=fs.C["faint"], lw=0.5, zorder=0, gid=f"{L}|zero")
        for k in order:
            r = T.loc[k]
            ax.plot([r[oc], r[pc]], [ys[k], ys[k]], color=fs.C["faint"], lw=0.6, zorder=1, gid=f"{L}|link|{k}")
            ax.plot([r[oc]], [ys[k]], marker="o", ms=2.6, mew=0, mfc=fs.C["owner"], ls="none", zorder=3, gid=f"{L}|owner|{k}")
            ax.plot([r[pc]], [ys[k]], marker="o", ms=2.6, mew=0, mfc=fs.C["partner"], ls="none", zorder=3, gid=f"{L}|partner|{k}")
            bold = L == "b" and not r.agrees
            cv.text(x0 + NAME_R, TOP + 3.0 + ys[k], fs.ARRAY_LABEL[k], ha="right", va="center", fontsize=fs.PT["tiny"],
                    fontweight="bold" if bold else "normal", gid=f"{L}|name|{k}")
        ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6]); ax.set_yticks([])
        ax.set_xlabel("Share of the condition pattern" if L == "a" else "Share of the response amplitude", labelpad=1.5)
        E.clean(ax, keep=("bottom",))
    cv.text(PANEL_X["b"] + NAME_R + 1.5 + AX_W, TOP + 0.6, f"larger amplitude share on the preference owner in {int(T.agrees.sum())} of 15; bold, the other {15 - int(T.agrees.sum())}",
            ha="right", va="center", fontsize=fs.PT["tiny"], color=fs.C["muted"], gid="b|count")
    return fig


def source_data(T, order, group, roles, owner):
    rows = [dict(screen=k, array=fs.ARRAY_LABEL[k], group=group[k], owner_role=roles[k][owner[k] - 1], partner_role=roles[k][2 - owner[k]],
                 preference_share_owner=float(T.loc[k, "pref_owner"]), preference_share_partner=float(T.loc[k, "pref_partner"]),
                 amplitude_share_owner=float(T.loc[k, "amp_owner"]), amplitude_share_partner=float(T.loc[k, "amp_partner"]),
                 larger_amplitude_on_preference_owner=bool(T.loc[k, "agrees"])) for k in order]
    readme = pd.DataFrame([dict(sheet="ED9", panel="a, b", estimand="role shares of the condition pattern (a) and of the response amplitude "
                                "(b)", rows="one per array in the order of Fig. 2A; bold names in b mark the six arrays whose amplitude "
                                "is larger on the other role", units="shares", transformation=TRANSFORM["a"] + " " + TRANSFORM["b"]),
                           dict(sheet="(all)", panel="-", estimand="-", rows="-", units="-",
                                transformation="Measurements: DATASET_analysis.csv.gz (A1). Owner roles: Anatomy_arrays.csv. Input "
                                               "hashes: FigS9_SourceData_provenance.json.")])
    return dict(README=readme, ED9=pd.DataFrame(rows))


def main(out=None):
    A1, _ = E.public_units()
    owner = E.owner_roles()
    T, dev = compute(A1, owner)
    order, group = E.order()
    roles = {k: (history.ROLE[k][0][0], history.ROLE[k][1][0]) for k in order}
    fig = draw(T, order, group)
    prov = dict(figure=f"Fig. S{N} | {TITLE}", generated_by=E.rel(__file__),
                inputs=dict(A1=dict(file=E.rel(E.RAW), sha256=E.sha_file(E.RAW), rows=int(len(A1))),
                            frozen={n: dict(file=E.rel(p), sha256=E.sha_file(p)) for n, p in (("phase3_amplitude.csv", FROZEN),
                                                                                               ("Anatomy_arrays.csv", E.ANATOMY_ARRAYS))}),
                randomness="none", transformation=TRANSFORM, agree=int(T.agrees.sum()), discordant=sorted(T.index[~T.agrees]),
                agreement_with_frozen=float(dev))
    p = E.write(N, fig, H, source_data(T, order, group, roles, owner), prov, out)
    print(f"Fig. S{N}: {int(T.agrees.sum())}/15 agree; max |diff| {dev:.1e}; wrote {len(p)} files")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    main(ap.parse_args().out)
