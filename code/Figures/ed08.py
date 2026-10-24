"""Fig. S8 | Ranking conditions by histories, and a molecule's own history against a substitute.

a  Both partners' histories (history.cv_records): three repeats of five training folds (the history|<array>|transfer|<rep>
   entries of the fold registry, written out as FigS8_transfer_folds.csv); per target the best measured yield of
   the trio ranked by the fitted histories (H) or by the raw records (N) minus that of the complementary trio (C), averaged
   over the repeats; pooled over all and over product-forming targets with 5,000 resamples of building blocks within arrays
   (common.owner_interval, seed 20260920). The resampled building block is the one of the role whose single-role history
   explains more held-out residual (history.heldout_r2 on the five history folds of Fig. 4A, Fig4a_folds.csv).
b  Substituted histories (the frozen matched-control pipeline, matched_control.py with its hash-checked configuration): the
   same folds; own trio against the mean of matched donors' trios; four of its donor rules; 10,000 whole-array draws
   (seed 20260928).
c  The equal-depth decision of Phase 2, section 7.3 (preference-based ranking, two partners per side), and
d  the depth curves of Phase 2, section 7.4, both computed by figure4.decision_panels and checked there against
   phase2_decision.csv, phase2_depth_curve.csv and phase2_summary.json; whole-array interval (seed 20261006) for c,
   pointwise intervals resampling molecules within the array (seed 20261008, a fresh generator for each curve) for d.
Checked against history_arrays.csv, history_cv.csv, anatomy.json (selection), and the matched-control outputs
(fold_assignments.csv, fold_effects.csv, pooled_effects.csv) before anything is drawn.
Inputs with a measurement: A1; the released Fig4a_folds.csv; the fifteen history|<array>|transfer|0-2 entries of the fold
registry (released here). With P2_RELEASE_MODE=public (the rebuild of the public data package) the folds are read from the
released FigS8_transfer_folds.csv and served to history.cv_records and the matched-control code in place of the
registry; the panel values do not depend on the order of the pairs within a fold.
    python 06_CODE/Figures/ed08.py [--out DIR]
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd

import ed_common as E
from ed_common import fs
import anatomy
import history
from common import owner_interval
from matplotlib.patches import Rectangle
import figure4_data as F4

N = 8
PUBLIC_MODE = os.environ.get("P2_RELEASE_MODE") == "public"
FOLDS_REG = E.DATA / "Analysis/06_METADATA/frozen_folds.json"
RELEASED_FOLDS = E.SDDIR / "FigS8_transfer_folds.csv"             # read in public mode only
FIG4A_FOLDS = E.ROOT / "05_DATA/Analysis/figure_display/Fig4a_folds.csv"
MATCHED = E.DATA / "Results/validation/matched"
MC_PATH = E.ROOT / "06_CODE/Validation/06_validation/matched_control.py"
MC_CONFIG = MC_PATH.parent / "analysis_config.json"
APPROVED = dict(history_arrays=E.DATA / "Results/primary/history_arrays.csv", history_cv=E.DATA / "Results/primary/history_cv.csv",
                anatomy=E.DATA / "Results/primary/anatomy.json", fold_assignments=MATCHED / "fold_assignments.csv",
                fold_effects=MATCHED / "fold_effects.csv", pooled_effects=MATCHED / "pooled_effects.csv")
TITLE = "Ranking conditions by histories, and a molecule's own history against a substitute"
TITLES = dict(a="Both partners' histories against a fixed trio", b="A molecule's own history against a matched substitute",
              c="At equal depth, owner history chooses better panels", d="Beating a fixed panel depends on the reaction")
NOTES = dict(a="Analysis-screen held-out pairs; three repeats of five training folds; 5,000 resamples of building blocks within screens",
             b="Analysis-screen held-out pairs, fully measured; donors of the same screen, role and training fold; 10,000 whole-screen draws",
             c="Analysis-screen held-out pairs; two earlier partners per side; best of three ranked conditions",
             d="Analysis-screen held-out pairs; 0 = fixed complementary trio")
POPS = (("product_forming", "Product-forming targets"), ("all", "All targets"))
RULES = (("fitted", "H_C", "fitted histories"), ("raw", "N_C", "raw records"))
CONTROLS = (("primary", "Depth and coverage matched"), ("strict", "Exact per-condition counts"),
            ("activity", "Training-activity caliper"), ("structure", "Closest structural neighbour, Tanimoto ≥ 0.5"))
ROLES = (("p1", "role 1"), ("p2", "role 2"))
TRANSFORM = dict(
    a="Three repeats of five training folds (FigS8_transfer_folds.csv). In each fold: residual = yield - pair mean - "
      "training condition mean + training grand mean; both partners' shrunken profiles (sum / (count + 4)) fitted jointly on the "
      "training pairs (ten alternating rounds). Fitted ranking: the three largest of training condition mean + the profiles of "
      "the target's two molecules (zero for a molecule without training pairs). Raw ranking: the three largest mean raw yields "
      "of the training pairs sharing either molecule. Complementary trio: greedy on the training pairs (each step adds the "
      "condition that most raises the mean best-of-panel yield). A target is scored when one of its molecules has training "
      "pairs and every condition of the panels is measured; per target, the mean over the repeats in which it was scored of "
      "best measured yield of the ranked trio minus that of the complementary trio. Product-forming: best measured yield at "
      "least 10%. Interval: 5,000 resamples, within each array, of the building blocks of the role whose single-role history "
      "explains more held-out residual on the five history folds of Fig. 4A, each with all its targets; pooled mean = sum / "
      "count; 2.5th and 97.5th percentiles; default_rng(20260920), arrays in sorted identifier order.",
    b="The same folds. For each role and fold: training residual = yield - pair mean - condition mean + grand mean on the "
      "training pairs; each molecule's shrunken profile (sum / (count + 4)) over its training pairs; its trio = the three "
      "largest of profile + training condition mean. For a fully measured held-out pair whose molecule on that role has "
      "training pairs: own = best measured yield of its molecule's trio; substitute = mean over the eligible donors of the best "
      "measured yield of each donor's trio. Donors are other molecules of the role with training pairs. Depth and coverage "
      "matched: the same number of training pairs, support Jaccard at least 0.95, covered conditions differing by at most one, "
      "per-condition counts differing by at most one, normalized count L1 at most 0.05. Exact per-condition counts: identical "
      "per-condition counts. Training-activity caliper: matched and mean training yield within 0.05. Closest structural "
      "neighbour: matched and ECFP4 (radius 2, 2,048 bits) Tanimoto at least 0.5, the most similar (ties kept). Per target: "
      "mean over the folds in which it was scored. Interval: 10,000 whole-array draws, default_rng(20260928), arrays in sorted "
      "identifier order; pooled mean = sum / count; 2.5th and 97.5th percentiles.",
    c=F4.TRANSFORM["b"], d=F4.TRANSFORM["c"])
H, AX_X, AX_W, ROW_R, GROUP_X, VALUE_X, PITCH, GAP_G, XLIM = 166.0, 66.0, 70.0, 64.0, 4.2, 139.0, 3.6, 1.8, (-3.5, 4.5)
TOP = dict(a=18.5, b=56.0)
YL = 101.0                                                                # c and d, side by side: heading line
C_NAME_R, C_AX, N_X = 31.0, (33.0, 46.0), 88.0                            # c: names end, axes x and width, right edge of the n column
C_PITCH, C_GAP_G, C_GAP_P = 2.8, 1.4, 2.0                                 # c: row pitch, gap between the chemistry groups, before the pooled row
XD = 96.0                                                                 # d: letter
D_X, D_W, D_H = (104.5, 130.0, 155.5), 20.0, 28.0                         # d: small multiples (left edges, width, height)
OFF = 0.12                                                                # d: sideways offset of the owner and partner intervals
num = lambda v: f"{v:.2f}".replace("-", "−")


def fmt(m, lo, hi):
    return f"{'+' if m >= 0 else ''}{num(m)} ({num(lo)} to {num(hi)})" if lo < 0 else f"{'+' if m >= 0 else ''}{num(m)} ({num(lo)}–{num(hi)})"


def matched_control():
    spec = importlib.util.spec_from_file_location("matched_control", MC_PATH)
    MC = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(MC)
    return MC, MC.read_config()                                      # read_config checks the configuration's SHA-256


def released_folds(AR):
    """Public mode: the released folds, and a stand-in for common.five_folds that serves them by the generator's fold key
    (history|<array>|transfer|<repeat>), pairs in array order."""
    F = pd.read_csv(RELEASED_FOLDS)
    assert len(F) == 3 * sum(len(A["pairs"]) for A in AR.values()) and set(F.screen) == set(AR)
    fold_of = {}
    for key, A in AR.items():
        for rep in range(3):
            f = F[(F.screen == key) & (F.repeat == rep + 1)].set_index("subpair").fold.reindex(A["pairs"]).to_numpy()
            assert np.isfinite(f).all() and set(f.tolist()) == {1, 2, 3, 4, 5}
            fold_of[(key, rep)] = f

    def five_folds(n, rng):
        head, task, rep = rng.fold_key.rsplit("|", 2)
        assert head.startswith("history|") and task == "transfer", rng.fold_key
        f = fold_of[(head[len("history|"):], int(rep))]
        assert len(f) == n
        return [np.flatnonzero(f == j) for j in range(1, 6)]
    return F, five_folds


def transfer_folds(AR):
    reg = json.loads(FOLDS_REG.read_text(encoding="utf-8"))
    rows = []
    for key, A in AR.items():
        for rep in range(3):
            folds = reg[f"history|{key}|transfer|{rep}"]
            assert sorted(np.concatenate(folds).tolist()) == list(range(len(A["M"])))
            rows += [dict(screen=key, subpair=A["pairs"][p], repeat=rep + 1, fold=j + 1) for j, f in enumerate(folds) for p in f]
    del reg
    return pd.DataFrame(rows).sort_values(["screen", "subpair", "repeat"], kind="stable").reset_index(drop=True)


def history_owner(AR):
    """The role whose single-role history explains more held-out residual, on the five history folds of Fig. 4A."""
    FO = pd.read_csv(FIG4A_FOLDS)
    V = pd.read_csv(APPROVED["history_arrays"]).set_index("screen")
    own, dev = {}, 0.0
    for key, A in AR.items():
        f = FO[FO.screen == key].set_index("subpair").fold.reindex(A["pairs"]).to_numpy()
        assert np.isfinite(f).all()
        r = history.heldout_r2(A, [np.flatnonzero(f == j) for j in range(1, 6)])
        own[key] = "p1" if r["p1"] >= r["p2"] else "p2"
        assert own[key] == V.loc[key, "owner"], key
        dev = max(dev, abs(r["p1"] - V.loc[key, "r2_p1"]), abs(r["p2"] - V.loc[key, "r2_p2"]))
    assert dev < E.TOL, dev
    return own, dev


def compute_a(AR, own):
    cv = pd.DataFrame([r for key, A in AR.items() for r in history.cv_records(A, own[key])])
    ref = pd.read_csv(APPROVED["history_cv"])
    J = cv.merge(ref, on=["screen", "rep", "pair"], suffixes=("", "_f"), how="outer", indicator=True)
    assert (J._merge == "both").all() and len(J) == len(ref) == 5694
    assert all((J[c] == J[c + "_f"]).all() for c in ("owner", "known_a", "known_b"))
    dev = max(float((J[c] - J[c + "_f"]).abs().max()) for c in ("H", "B", "N", "C"))
    assert dev < E.TOL, dev
    cv["best"] = [np.nanmax(AR[k]["M"][p]) for k, p in zip(cv.screen, cv.pair)]
    cv = cv.assign(H_C=cv.H - cv.C, N_C=cv.N - cv.C)
    per = cv.groupby(["screen", "pair", "owner"], as_index=False).agg(H_C=("H_C", "mean"), N_C=("N_C", "mean"), best=("best", "mean"),
                                                                     repeats=("rep", "nunique"))
    sel = json.loads(APPROVED["anatomy"].read_text(encoding="utf-8"))["selection"]
    est = []
    for pop, _ in POPS:
        z = per if pop == "all" else per[per.best >= anatomy.FORMS]
        assert len(z) == sel[pop]["n"]
        for rule, col, _ in RULES:
            m, lo, hi = (100 * v for v in owner_interval(z, col, "owner"))
            ref_ = sel[pop][rule]
            assert max(abs(m - ref_[0]), abs(lo - ref_[1]), abs(hi - ref_[2])) < E.TOL
            est.append(dict(population=pop, ranking=rule, targets=len(z), arrays=z.screen.nunique(),
                            building_blocks=z.groupby(["screen", "owner"]).ngroups, mean_pp=m, ci_lower_pp=lo, ci_upper_pp=hi))
    T = per.assign(subpair=[AR[k]["pairs"][p] for k, p in zip(per.screen, per.pair)],
                   bb_role=[1 if own[k] == "p1" else 2 for k in per.screen],
                   building_block=[(AR[k]["ua"] if own[k] == "p1" else AR[k]["ub"])[o] for k, o in zip(per.screen, per.owner)],
                   product_forming=per.best >= anatomy.FORMS)
    return pd.DataFrame(est), T, dev, len(cv)


def compute_b(AR, MC, cfg):
    effects = []
    for key, A in AR.items():
        M = A["M"]; complete = np.isfinite(M).all(axis=1)
        sim = {role: MC.structural_similarities(names, cfg) for role, names in (("p1", A["ua"]), ("p2", A["ub"]))}
        for rep in range(cfg["repeats"]):
            for fold, te in enumerate(MC.five_folds(len(M), MC.stream(key, cfg["fold_stream"], rep))):
                tr = np.setdiff1d(np.arange(len(M)), te)
                for role, idx, names in (("p1", A["ia"], A["ua"]), ("p2", A["ib"], A["ub"])):
                    fit = MC.fit_role(M, idx, len(names), tr, cfg["profile_shrinkage"])
                    for q in te:
                        target = int(idx[q]); depth = int(fit["counts"][target])
                        if not complete[q] or depth == 0:
                            continue
                        for kind, donors in MC.control_donors(target, fit, sim[role], cfg).items():
                            if len(donors):
                                res = MC.score_comparison(M[q], fit["panels"][target], fit["panels"][donors], fit["baseline"])
                                effects.append(dict(screen=key, role=role, rep=rep, fold=fold, pair=int(q), molecule=target, control=kind,
                                                    n_history=depth, n_donors=len(donors), **res))
    FE = pd.DataFrame(effects)
    ref = pd.read_csv(APPROVED["fold_effects"])
    J = FE.merge(ref, on=["screen", "role", "rep", "fold", "pair", "control"], suffixes=("", "_f"), how="outer", indicator=True)
    assert (J._merge == "both").all() and len(J) == len(ref)
    assert all((J[c] == J[c + "_f"]).all() for c in ("molecule", "n_history", "n_donors"))
    dev = max(float((J[c] - J[c + "_f"]).abs().max()) for c in ("own", "substitute", "baseline", "difference"))
    assert dev < E.TOL, dev
    targets = MC.summarize_targets(FE)
    pooled = pd.read_csv(APPROVED["pooled_effects"]).set_index(["control", "role"])
    est = []
    for control, _ in CONTROLS:
        for role, _ in ROLES:
            z = targets[(targets.control == control) & (targets.role == role)]
            lo, hi = MC.array_bootstrap(z, cfg)
            r = dict(control=control, role=int(role[1]), targets=len(z), arrays=z.screen.nunique(),
                     mean_pp=100 * z.difference.mean(), ci_lower_pp=100 * lo, ci_upper_pp=100 * hi)
            f = pooled.loc[(control, role)]
            assert (r["targets"], r["arrays"]) == (int(f.n_targets), int(f.n_arrays))
            assert max(abs(r["mean_pp"] - f.difference_pp), abs(r["ci_lower_pp"] - f.array_ci_low_pp),
                       abs(r["ci_upper_pp"] - f.array_ci_high_pp)) < E.TOL
            est.append(r)
    keep = targets[targets.control.isin([c for c, _ in CONTROLS])]
    T = keep.assign(subpair=[AR[k]["pairs"][p] for k, p in zip(keep.screen, keep.pair)],
                    own_molecule=[(AR[k]["ua"] if r_ == "p1" else AR[k]["ub"])[m] for k, r_, m in zip(keep.screen, keep.role, keep.molecule)],
                    role_number=keep.role.str[1].astype(int))
    return pd.DataFrame(est), T, dev


def rows_y(groups, members):
    """Row centres (mm from the top of the axes) and the height that leaves half a row below the last one."""
    ys, y_ = {}, PITCH / 2
    for i, g in enumerate(groups):
        if i:
            y_ += GAP_G
        for m in members:
            ys[(g, m)] = y_; y_ += PITCH
    return ys, y_ - PITCH / 2


def draw(ea, eb, B, C, order_b, groups_b):
    fig, cv = E.figure(H, "ed8")
    text_c, muted, faint = fs.C["text"], fs.C["muted"], fs.C["faint"]
    fs.title(fig, E.LEFT, 0.6, TITLE, H).set_gid("title|main")
    for L, x, y in (("a", 0, 8.0), ("b", 0, 45.5), ("c", 0, YL), ("d", XD, YL)):
        E.heading(fig, cv, x, y, L, TITLES[L], H, NOTES[L])
    panels = (("a", ea.set_index(["population", "ranking"]), [p for p, _ in POPS], [r for r, _, _ in RULES], dict(POPS),
               {r: lab for r, _, lab in RULES}, "Best measured yield: ranked trio minus complementary trio (percentage points)"),
              ("b", eb.set_index(["control", "role"]), [c for c, _ in CONTROLS], [1, 2], dict(CONTROLS), {1: "role 1", 2: "role 2"},
               "Best measured yield: own history's trio minus the substitute's (percentage points)"))
    for L, T, groups, members, glabel, mlabel, xlabel in panels:
        ys, hb = rows_y(groups, members)
        hb += 0.4
        ax = fs.axes_mm(fig, AX_X, TOP[L], AX_W, hb, H); ax.set_gid(f"{L}|axes")
        ax.set_xlim(*XLIM); ax.set_ylim(hb, 0)
        ax.plot([0, 0], [0, hb], color=faint, lw=0.5, zorder=0, gid=f"{L}|zero")
        for g in groups:
            for j, m in enumerate(members):
                r = T.loc[(g, m)]; y = ys[(g, m)]
                ax.plot([r.ci_lower_pp, r.ci_upper_pp], [y, y], color=text_c, lw=0.8, solid_capstyle="butt", zorder=2, gid=f"{L}|ci|{g}|{m}")
                ax.plot([r.mean_pp], [y], marker="o", ms=2.6, mew=0, mfc=text_c, ls="none", zorder=3, gid=f"{L}|point|{g}|{m}")
                cv.text(ROW_R, TOP[L] + y, mlabel[m], ha="right", va="center", fontsize=fs.PT["tiny"], color=muted, gid=f"{L}|row|{g}|{m}")
                cv.text(VALUE_X, TOP[L] + y, fmt(r.mean_pp, r.ci_lower_pp, r.ci_upper_pp), ha="left", va="center", fontsize=fs.PT["tiny"],
                        gid=f"{L}|value|{g}|{m}")
                if L == "b":
                    cv.text(E.RIGHT, TOP[L] + y, f"{int(r.targets):,} targets, {int(r.arrays)} screens", ha="right", va="center",
                            fontsize=fs.PT["tiny"], color=muted, gid=f"b|n|{g}|{m}")
                if j == 0:
                    lab = f"{glabel[g]} ({int(r.targets):,})" if L == "a" else glabel[g]
                    cv.text(GROUP_X, TOP[L] + y, lab, ha="left", va="center", fontsize=fs.PT["tiny"], gid=f"{L}|group|{g}")
        ax.set_xticks(range(-3, 5)); ax.set_yticks([])
        ax.set_xlabel(xlabel, labelpad=1.5)
        E.clean(ax, keep=("bottom",))
    draw_decision(fig, cv, B, order_b, groups_b)
    draw_depth(fig, cv, C)
    return fig


def draw_decision(fig, cv, B, order_b, groups):
    """c: owner minus partner best yield at equal depth, one row per array (Figure 1F's order), then the pooled estimate."""
    owner_c, faint, muted, text_c = fs.C["owner"], fs.C["faint"], fs.C["muted"], fs.C["text"]
    tiny = dict(fontsize=fs.PT["tiny"], va="center")
    top = YL + 12.0
    o = B["pref"]["owner_minus_partner"]
    rows_y, y_ = {}, C_PITCH / 2
    for i, k in enumerate(order_b):
        if i and groups[k] != groups[order_b[i - 1]]:
            y_ += C_GAP_G
        rows_y[k] = y_
        y_ += C_PITCH
    y_pool = y_ - C_PITCH / 2 + C_GAP_P + C_PITCH / 2
    hb = y_pool + C_PITCH / 2 + 0.6
    bx = fs.axes_mm(fig, C_AX[0], top, C_AX[1], hb, H); bx.set_gid("c|axes")
    bx.set_xlim(-3.0, 18.0); bx.set_ylim(hb, 0)
    bx.axvline(0, color=faint, lw=0.5, zorder=0, gid="c|zero")
    for k in order_b:
        v = o["per_array_pp"][k]
        bx.plot([v], [rows_y[k]], marker="o", ms=2.8, mew=0, mfc=owner_c, ls="none", zorder=3, gid=f"c|array|{k}")   # identical markers
        cv.text(C_NAME_R, top + rows_y[k], fs.ARRAY_LABEL[k], gid=f"c|name|{k}", ha="right", **tiny)
        cv.text(N_X, top + rows_y[k], f"{o['n'][k]}", gid=f"c|n|{k}", ha="right", color=muted, **tiny)
    lo, hi = o["ci_pp"]
    bx.plot([lo, hi], [y_pool, y_pool], color=text_c, lw=0.8, zorder=3, gid="c|pooled_ci")
    bx.plot([o["mean_pp"]], [y_pool], marker="D", ms=4.2, mew=0, mfc=text_c, ls="none", zorder=4, gid="c|pooled")
    bx.text(hi + 0.6, y_pool, f"{F4.signed(o['mean_pp'])} ({lo:.1f}–{hi:.1f})", ha="left", va="center", fontsize=fs.PT["tiny"],
            gid="c|pooled_value")
    cv.text(C_NAME_R, top + y_pool, f"All {len(order_b)} screens", gid="c|name|pooled", ha="right", fontweight="bold", **tiny)
    cv.text(N_X, top + y_pool, f"{sum(o['n'].values())}", gid="c|n|pooled", ha="right", fontweight="bold", **tiny)
    cv.text(N_X, top - 1.6, "n", gid="c|n_head", ha="right", color=muted, **tiny)
    bx.text(18.0, -1.6, f"{o['arrays_positive']} of {len(order_b)} > 0", ha="right", va="center", fontsize=fs.PT["tiny"], color=muted,
            gid="c|positive", clip_on=False)
    bx.set_xticks([0, 5, 10, 15]); bx.set_yticks([])
    bx.set_xlabel("Owner − partner best-yield difference\n(percentage points)", labelpad=1.5)
    E.clean(bx, keep=("bottom",))


def draw_depth(fig, cv, C):
    """d: gain of each history-ranked panel over the fixed trio by depth, three reactions as small multiples."""
    owner_c, partner_c, muted = fs.C["owner"], fs.C["partner"], fs.C["muted"]
    KS = F4.KS
    top = YL + 17.0
    for j, (side, colr) in enumerate((("owner", owner_c), ("partner", partner_c))):
        x = XD + 4.2 + 22.0 * j
        cv.add_patch(Rectangle((x, YL + 10.3 - 0.7), 1.4, 1.4, facecolor=colr, edgecolor="none", gid=f"d|key|{j}"))
        cv.text(x + 2.0, YL + 10.3, f"{side}'s history", gid=f"d|key_label|{j}", ha="left", fontsize=fs.PT["tiny"], va="center")
    for n_, key in enumerate(F4.C_ORDER):
        cx = fs.axes_mm(fig, D_X[n_], top, D_W, D_H, H); cx.set_gid(f"d|axes|{key}")
        name = fs.ARRAY_LABEL[key]
        cv.text(D_X[n_], top - 2.6, name, gid=f"d|name|{key}", ha="left", fontweight="bold", fontsize=fs.PT["tiny"], va="center")
        bounds = []
        for side, colr, dx in (("owner", owner_c, -OFF), ("partner", partner_c, OFF)):
            cur = C[(key, side)]["curve"]
            x = np.array([c["K"] for c in cur], float) + dx
            y = np.array([c["mean_pp"] for c in cur])
            for c, xi in zip(cur, x):
                cx.plot([xi, xi], c["ci_pp"], color=colr, lw=0.7, zorder=2, gid=f"d|{key}|{side}|ci|{c['K']}")
                bounds += c["ci_pp"]
            cx.plot(x, y, color=colr, lw=0.9, marker="o", ms=2.4, mew=0, mfc=colr, zorder=3, gid=f"d|{key}|{side}|curve")
        cx.axhline(0, color=muted, lw=0.5, ls=(0, (2, 1.5)), zorder=1, gid=f"d|{key}|zero")
        top_, bot = max(bounds), min(bounds)
        span = top_ - bot
        cx.set_ylim(bot - 0.12 * span, top_ + 0.12 * span)
        lo_, hi_ = cx.get_ylim()
        cx.set_yticks([v for v in cx.get_yticks() if lo_ <= v <= hi_])     # no tick label below the axes
        own_cur = C[(key, "owner")]["curve"]                                 # the owner's first and last depth, printed
        clear = 1.3 * (cx.get_ylim()[1] - cx.get_ylim()[0]) / D_H                # half a label height plus a margin, in data units
        at = lambda v: v if abs(v) >= clear else (clear if v > 0 else -clear)    # a label never sits on the zero line
        cx.text(KS[0] - OFF - 0.5, at(own_cur[0]["mean_pp"]), F4.signed(own_cur[0]["mean_pp"]), ha="right", va="center",
                fontsize=fs.PT["tiny"], gid=f"d|{key}|owner|value|{KS[0]}")
        cx.text(KS[-1] + OFF + 0.5, at(own_cur[-1]["mean_pp"]), F4.signed(own_cur[-1]["mean_pp"]), ha="left", va="center",
                fontsize=fs.PT["tiny"], gid=f"d|{key}|owner|value|{KS[-1]}", clip_on=False)
        cx.set_xlim(-1.8, 8.8); cx.set_xticks(list(KS))
        if n_ == 1:
            cx.set_xlabel("Earlier partners in the history", labelpad=1.5)
        if n_ == 0:
            cx.set_ylabel("Best-yield gain over\nfixed trio (percentage points)", labelpad=1.5)
        E.clean(cx)


def source_data(ea, Ta, eb, Tb, D73, D74, B, C, order_b, groups):
    a = ea.assign(population=ea.population.map({"product_forming": "product-forming targets", "all": "all targets"}),
                  ranking=ea.ranking.map({"fitted": "fitted histories", "raw": "raw records"}))
    ta = Ta.assign(array=Ta.screen.map(fs.ARRAY_LABEL), best_measured_yield=Ta.best, fitted_minus_trio_pp=100 * Ta.H_C,
                   raw_minus_trio_pp=100 * Ta.N_C)[["screen", "array", "subpair", "bb_role", "building_block", "repeats",
                                                    "best_measured_yield", "product_forming", "fitted_minus_trio_pp", "raw_minus_trio_pp"]]
    ta = ta.rename(columns={"bb_role": "resampled_role", "building_block": "resampled_building_block"}).sort_values(
        ["screen", "subpair"], kind="stable").reset_index(drop=True)
    b = eb.assign(control=eb.control.map(dict(CONTROLS)))
    tb = Tb.assign(array=Tb.screen.map(fs.ARRAY_LABEL), control_label=Tb.control.map(dict(CONTROLS)),
                   own_minus_substitute_pp=100 * Tb.difference)[["control_label", "screen", "array", "role_number", "subpair", "own_molecule",
                                                                 "n_repeats", "own_minus_substitute_pp"]]
    tb = tb.rename(columns={"control_label": "control", "role_number": "role", "n_repeats": "repeats_scored"})
    order_c = {lab: i for i, (_, lab) in enumerate(CONTROLS)}
    tb = tb.sort_values(["control", "role", "screen", "subpair"], key=lambda s: s.map(order_c) if s.name == "control" else s,
                        kind="stable").reset_index(drop=True)
    readme = pd.DataFrame([
        dict(sheet="ED8a", panel="a", estimand="best measured yield of a ranked trio minus that of the complementary trio",
             rows="one per population and ranking", units="percentage points of yield", transformation=TRANSFORM["a"]),
        dict(sheet="ED8a_targets", panel="a", estimand="as ED8a", rows="one per target (1,964); product_forming marks the 1,372",
             units="yield fraction (best_measured_yield); percentage points", transformation=TRANSFORM["a"]),
        dict(sheet="ED8b", panel="b", estimand="best measured yield of the own molecule's trio minus the mean over matched donors' trios",
             rows="one per donor rule and role", units="percentage points of yield", transformation=TRANSFORM["b"]),
        dict(sheet="ED8b_targets", panel="b", estimand="as ED8b", rows="one per donor rule, role and target", units="percentage points",
             transformation=TRANSFORM["b"]),
        dict(sheet="ED8c", panel="c", estimand=F4.EST["bc"], rows="one per array in the order of Figure 1F (electrophilic "
             "halide arrays first), then the pooled row", units="percentage points of yield; counts", transformation=TRANSFORM["c"]),
        dict(sheet="ED8c_targets", panel="c", estimand=F4.EST["bc"], rows="one per scored target (728)", units="yield fractions "
             "(best measured yield of a panel, averaged over history subsets)", transformation=TRANSFORM["c"]),
        dict(sheet="ED8d", panel="d", estimand=F4.EST["bc"], rows="per array, side and k", units="percentage points of yield; "
             "counts", transformation=TRANSFORM["d"]),
        dict(sheet="ED8d_targets", panel="d", estimand=F4.EST["bc"], rows="one per target, side and k (the gains averaged in "
             "ED8d)", units="yield fractions", transformation=TRANSFORM["d"]),
        dict(sheet="FigS8_transfer_folds.csv (separate file)", panel="a, b", estimand="-",
             rows="one per pair and repeat: the training fold (1-5) in which the pair is held out", units="fold numbers",
             transformation="The frozen fold registry entries history|<array>|transfer|<repeat - 1>; the training pairs of a fold are "
                            "the pairs of its array in the other four folds of the same repeat."),
        dict(sheet="(all)", panel="-", estimand="-", rows="-", units="-",
             transformation="Measurements: DATASET_analysis.csv.gz (A1). Panel a also reuses Figure 4's Fig4a_folds.csv for the "
                            "resampled role. Input hashes: FigS8_SourceData_provenance.json.")])
    lab = fs.ARRAY_LABEL
    o = B["pref"]["owner_minus_partner"]
    c = pd.DataFrame([dict(row_type="array", screen=k, array=lab[k], group=groups[k], targets=o["n"][k],
                           preference_ranked_owner_minus_partner_pp=o["per_array_pp"][k]) for k in order_b]
                     + [dict(row_type="pooled", array=f"All {len(order_b)} arrays", targets=sum(o["n"].values()),
                             preference_ranked_owner_minus_partner_pp=o["mean_pp"], ci_lower_pp=o["ci_pp"][0], ci_upper_pp=o["ci_pp"][1],
                             arrays_positive=o["arrays_positive"], arrays=len(order_b))])
    c = c.astype({"targets": "Int64", "arrays_positive": "Int64", "arrays": "Int64"})
    d = D73[D73.pref_scored]
    ct = pd.DataFrame(dict(screen=d.screen, array=d.screen.map(lab), target_pair=d.subpair, owner_subsets=d.subsets_owner,
                           partner_subsets=d.subsets_partner, owner_panel_best_yield=d.pref_owner_best,
                           partner_panel_best_yield=d.pref_partner_best))
    dd = pd.DataFrame([dict(screen=key, array=lab[key], side=side, k=q["K"], preference_ranked_gain_over_trio_pp=q["mean_pp"],
                            ci_lower_pp=q["ci_pp"][0], ci_upper_pp=q["ci_pp"][1], targets=C[(key, side)]["targets"],
                            molecules=C[(key, side)]["molecules"])
                       for key in F4.C_ORDER for side in ("owner", "partner") for q in C[(key, side)]["curve"]])
    dt = pd.DataFrame(dict(screen=D74.screen, array=D74.screen.map(lab), side=D74.side, molecule=D74.molecule, target_pair=D74.subpair,
                           k=D74.K, gain_over_trio_yield=D74.gain))
    return dict(README=readme, ED8a=a, ED8a_targets=ta, ED8b=b, ED8b_targets=tb, ED8c=c, ED8c_targets=ct, ED8d=dd, ED8d_targets=dt)


def main(out=None):
    A1, _ = E.public_units()
    AR = history.partner_arrays(A1)
    if PUBLIC_MODE:                                   # released folds stand in for the registry (and its copy, fold_assignments.csv)
        folds, five_folds = released_folds(AR)
        history.five_folds = five_folds
    else:
        folds = transfer_folds(AR)
        fa = pd.read_csv(APPROVED["fold_assignments"])
        pos = {k: {sp: i for i, sp in enumerate(A["pairs"])} for k, A in AR.items()}
        mine = folds.assign(pair=[pos[k][sp] for k, sp in zip(folds.screen, folds.subpair)], rep=folds.repeat - 1, fold0=folds.fold - 1)
        assert sorted(zip(mine.screen, mine.rep, mine.fold0, mine.pair)) == sorted(zip(fa.screen, fa.rep, fa.fold, fa.pair)) and len(fa) == 6621
    own, dev_owner = history_owner(AR)
    ea, Ta, dev_a, n_records = compute_a(AR, own)
    MC, cfg = matched_control()
    if PUBLIC_MODE:
        MC.five_folds = five_folds
    eb, Tb, dev_b = compute_b(AR, MC, cfg)
    D73, D74, B, C, order_b, groups, dev_cd = F4.decision_panels(A1)
    fig = draw(ea, eb, B, C, order_b, groups)
    if PUBLIC_MODE:                                   # what the public rebuild read in place of the registry
        inputs = dict(A1=dict(file=E.rel(E.RAW), sha256=E.sha_file(E.RAW), rows=int(len(A1))),
                      released=dict(fig4a_folds=dict(file=E.rel(FIG4A_FOLDS), sha256=E.sha_file(FIG4A_FOLDS)),
                                    transfer_folds=dict(file=E.rel(RELEASED_FOLDS), sha256=E.sha_file(RELEASED_FOLDS))),
                      matched_control=dict(code=dict(file=E.rel(MC_PATH), sha256=E.sha_file(MC_PATH)),
                                           configuration=dict(file=E.rel(MC_CONFIG), sha256=E.sha_file(MC_CONFIG))),
                      frozen={k: dict(file=E.rel(p), sha256=E.sha_file(p)) for k, p in APPROVED.items() if k != "fold_assignments"},
                      decision={k: dict(file=E.rel(p), sha256=E.sha_file(p)) for k, p in F4.DECISION.items()})
    else:
        inputs = dict(A1=dict(file=E.rel(E.RAW), sha256=E.sha_file(E.RAW), rows=int(len(A1))),
                      released=dict(fig4a_folds=dict(file=E.rel(FIG4A_FOLDS), sha256=E.sha_file(FIG4A_FOLDS))),
                      fold_registry=dict(file=E.rel(FOLDS_REG), sha256=E.sha_file(FOLDS_REG), entries_used="history|<array>|transfer|0-2"),
                      registry_integrity=[dict(file=E.rel(p), sha256=E.sha_file(p)) for p in E.REGISTRY_INTEGRITY],
                      matched_control=dict(code=dict(file=E.rel(MC_PATH), sha256=E.sha_file(MC_PATH)),
                                           configuration=dict(file=E.rel(MC_CONFIG), sha256=E.sha_file(MC_CONFIG))),
                      frozen={k: dict(file=E.rel(p), sha256=E.sha_file(p)) for k, p in APPROVED.items()},
                      decision={k: dict(file=E.rel(p), sha256=E.sha_file(p)) for k, p in F4.DECISION.items()})
    prov = dict(figure=f"Fig. S{N} | {TITLE}", generated_by=E.rel(__file__),
                inputs=inputs,
                randomness=dict(folds="frozen registry (released as FigS8_transfer_folds.csv)",
                                a="common.owner_interval: default_rng(20260920), 5,000 draws, a fresh generator per interval",
                                b="matched_control.array_bootstrap: default_rng(20260928), 10,000 draws, a fresh generator per interval",
                                c="history subsets and the trio are deterministic (SHA-256 order; greedy); interval default_rng(20261006)"
                                  ".integers(0, 12, (10000, 12)) over the arrays in sorted identifier order, drawn once",
                                d="history subsets deterministic (SHA-256 of 'target|j|partner'); intervals default_rng(20261008).integers"
                                  "(0, n, (10000, n)) over the n molecules of each curve in sorted order, a fresh generator for each curve"),
                transformation=TRANSFORM, resampled_role={k: v for k, v in own.items()},
                estimates=dict(a=ea.to_dict("records"), b=eb.to_dict("records"),
                               c=dict(order=order_b, preference=B["pref"], additive=B["add"]),
                               d={f"{s} | {k}": v for (k, s), v in C.items()}),
                agreement_with_frozen=dict(history_r2=dev_owner, history_cv=dev_a, matched_fold_effects=dev_b, **dev_cd), history_records=n_records)
    p = E.write(N, fig, H, source_data(ea, Ta, eb, Tb, D73, D74, B, C, order_b, groups), prov, out, extra=(("folds", "FigS8_transfer_folds.csv"),),
                tables=(("folds", folds),))
    msg = (f"Fig. S{N}: " + "; ".join(f"{r.population}/{r.ranking} {fmt(r.mean_pp, r.ci_lower_pp, r.ci_upper_pp)}" for r in ea.itertuples())
           + " | " + "; ".join(f"{r.control}/{r.role} {fmt(r.mean_pp, r.ci_lower_pp, r.ci_upper_pp)} n={r.targets}" for r in eb.itertuples())
           + f"; wrote {len(p)} files")
    print(msg.replace("−", "-").replace("–", "-"))                   # the console may not encode typographic dashes


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    main(ap.parse_args().out)
