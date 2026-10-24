"""P5: how condition recommendation degrades as molecular identity is taken away, one regime at a time.

For a target coupling A + B, four training sets are built from the same role-specific molecule
partitions, so the four regimes differ only in which identities the model is allowed to have seen:

  warm/warm   every pair except the target pairs themselves; A and B both occur in training with other
              partners, and the target is scored only if they do
  new/known   every pair that does not contain a held first-partner molecule: A is unseen, B is seen
  known/new   the mirror image: B is unseen, A is seen
  new/new     every pair containing a held molecule in either role is removed: neither is seen

Reported for each regime, on the same targets and with the same rule:

  prediction  held-out R2 of the pair's amplitude (descriptive b_i) and of its out-of-fold preference
  decision    the three conditions of S_ij = mu_j + p_hat_ij against the mean-ranked and complementary
              trios built from the same training rows

Cross-role occurrence is checked: a molecule that appears in both roles would leak an identity the split
means to withhold, and the count is reported per array.

Writes out/fig3_generalisation_regimes.csv (per array and regime), out/fig3_regime_cases.csv (per target)
and a regimes block in out/structure.json.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from common import (load_analysis, arrays, labels, top_k, greedy_trio, whole_array_interval,
                    load_json, save_json, OUT)
import preference as PREF
import structure as S

REGIMES = ("warm/warm", "new/known", "known/new", "new/new")

# the primary target; the raw uncentred residual was checked as a sensitivity and agreed
TARGETS = ("p_oof_centred",)


def training_rows(cp, held1, held2, te, regime):
    """The rows a regime allows, given the molecules held out in each role."""
    from boundaries import frozen_folds, regime_fold_key
    return frozen_folds(regime_fold_key(cp,held1,held2,te,regime))[0]


def _historical_training_rows(cp, held1, held2, te, regime):
    """Reference definition retained for review; active execution reads precomputed indices."""
    keep = []
    for i, (m1, m2) in enumerate(cp):
        if i in te:
            continue
        if regime == "new/new" and (m1 in held1 or m2 in held2 or m1 in held2 or m2 in held1):
            continue
        if regime == "new/known" and m1 in held1:
            continue
        if regime == "known/new" and m2 in held2:
            continue
        keep.append(i)
    return np.array(keep, int)


def main():
    d = load_analysis()
    A = arrays(d)
    lab = labels(d)
    rows, cases = [], []
    for key, (pairs, conds, M) in A.items():
        # labels are rebuilt inside every outer training set (preference.split_decomposition)
        F1, F2, D = S.featurise(pairs)
        cp = [S.canonical(s) for s in pairs]
        role1 = {m for m, _ in cp}; role2 = {m for _, m in cp}
        both_roles = len(role1 & role2)
        acc = {r: dict(amp=[0.0, 0.0], n=0, pairs=set(),
                       **{k: [0.0, 0.0] for k in TARGETS}) for r in REGIMES}
        for seed in S.PARTITION_SEEDS:
            id1, map1 = S.assignment([v[0] for v in cp], seed)
            id2, map2 = S.assignment([v[1] for v in cp], seed + 1)
            for a in sorted(set(id1)):
                for b in sorted(set(id2)):
                    te = np.flatnonzero((id1 == a) & (id2 == b))
                    if not len(te):
                        continue
                    held1 = {v for v, z in map1.items() if z == a}
                    held2 = {v for v, z in map2.items() if z == b}
                    for regime in REGIMES:
                        tr = training_rows(cp, held1, held2, set(te.tolist()), regime)
                        if len(tr) < 10:
                            continue
                        seen1 = {cp[i][0] for i in tr}; seen2 = {cp[i][1] for i in tr}
                        # the regime's own definition, enforced target by target
                        ok_te = [i for i in te
                                 if ((cp[i][0] in seen1 and cp[i][1] in seen2) if regime == "warm/warm" else
                                     (cp[i][0] not in seen1 and cp[i][1] in seen2) if regime == "new/known" else
                                     (cp[i][0] in seen1 and cp[i][1] not in seen2) if regime == "known/new" else
                                     (cp[i][0] not in seen1 and cp[i][1] not in seen2))]
                        if not ok_te:
                            continue
                        ok_te = np.array(ok_te, int)
                        Xtr, Xte = S.feature_blocks(F1, F2, D, tr, ok_te)
                        sp = PREF.split_decomposition(M, tr, ok_te)
                        b_tr, b_te = sp["train_beta"], sp["test_beta"]
                        fin = np.isfinite(b_tr); good = np.isfinite(b_te)
                        pr = Ridge(alpha=100.0).fit(Xtr[fin], b_tr[fin]).predict(Xte)
                        acc[regime]["amp"][0] += float(np.sum((b_te[good] - pr[good]) ** 2))
                        acc[regime]["amp"][1] += float(np.sum((b_te[good] - b_tr[fin].mean()) ** 2))
                        for name in ("p_oof_centred",):
                            T = np.full_like(M, np.nan); T[tr] = sp["train_p"]; T[ok_te] = sp["test_p"]
                            colm, gm, pp = S.fit_predict(M, F1, F2, D, tr, ok_te, target=T)
                            t = T[ok_te]; m_ok = np.isfinite(t)
                            acc[regime][name][0] += float(np.sum((t - pp)[m_ok] ** 2))
                            acc[regime][name][1] += float(np.sum(t[m_ok] ** 2))
                            gp = top_k(colm); cpan = greedy_trio(M[tr])
                            for j, i in enumerate(ok_te):
                                sc = S.score(M[i], top_k(colm + pp[j]), gp, cpan)
                                cases.append(dict(screen=key, regime=regime, target=name, seed=seed,
                                                  subpair=pairs[i], n_train=len(tr), **sc))
                        acc[regime]["n"] += len(ok_te)
                        acc[regime]["pairs"].update(pairs[i] for i in ok_te)
        for regime in REGIMES:
            a_ = acc[regime]
            rows.append(dict(screen=key, label=lab[key], regime=regime,
                             records=a_["n"], unique_pairs=len(a_["pairs"]),
                             partitions=len(S.PARTITION_SEEDS), molecules_in_both_roles=both_roles,
                             r2_amplitude=1 - a_["amp"][0] / a_["amp"][1] if a_["amp"][1] > 0 else np.nan,
                             **{"r2_" + k: (1 - a_[k][0] / a_[k][1] if a_[k][1] > 0 else np.nan)
                                for k in TARGETS}))
        print(f"  {key:28s} " + "  ".join(
            f"{r} pairs={len(acc[r]['pairs'])}" for r in REGIMES), flush=True)

    pd.DataFrame(rows).to_csv(OUT / "fig3_generalisation_regimes.csv", index=False)
    pd.DataFrame(cases).to_csv(OUT / "fig3_regime_cases.csv", index=False)
    summarise()


def summarise():
    """The summary block of out/structure.json, from the two files the run leaves behind.

    Separate from main() so it can be recomputed after a change to what is reported without repeating the
    fitting, and so the reported counts have one implementation rather than one per caller.
    """
    R = pd.read_csv(OUT / "fig3_generalisation_regimes.csv")
    C = pd.read_csv(OUT / "fig3_regime_cases.csv")
    out = load_json("structure.json"); out["regimes"] = {}
    for name in TARGETS:
        print(f"\ntarget {name}")
        print(f"  {'regime':10s} {'arrays':>6s} {'pairs':>7s} {'records':>8s} {'R2 amp':>8s} "
              f"{'R2 pref':>8s} {'vs mean-ranked':>21s} {'vs complementary':>21s}")
        for regime in REGIMES:
            r = R[(R.regime == regime) & R.records.gt(0)]
            z = C[(C.regime == regime) & (C.target == name) & C.scorable]
            g = {c: [100 * v for v in whole_array_interval(z, "gain_" + c)] for c in ("mean", "greedy")}
            # the pairs and the records the text prints together are counted on the same rows, the
            # scored ones, and per array, so a pair measured in two arrays counts as two targets
            n_pairs = int(z.groupby("screen").subpair.nunique().sum())
            out["regimes"].setdefault(name, {})[regime] = dict(
                n_arrays=int(len(r)), n_unique_pairs=n_pairs,
                n_records=int(len(z)), n_partitions=len(S.PARTITION_SEEDS),
                median_r2_amplitude=float(r.r2_amplitude.median()),
                median_r2_preference=float(r["r2_" + name].median()), **g)
            print(f"  {regime:10s} {len(r):6d} {n_pairs:7d} {len(z):8d} "
                  f"{r.r2_amplitude.median():+8.3f} {r['r2_' + name].median():+8.3f} "
                  f"{g['mean'][0]:+7.2f} [{g['mean'][1]:+5.2f},{g['mean'][2]:+5.2f}] "
                  f"{g['greedy'][0]:+7.2f} [{g['greedy'][1]:+5.2f},{g['greedy'][2]:+5.2f}]")
    save_json("structure.json", out)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
