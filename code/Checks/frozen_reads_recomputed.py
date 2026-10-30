"""Values that the figure generators and the SI tables read from analysis outputs, recomputed from the released measurements.

1 Figure 1: the 95% interval of the median molecular share and the owner named by the likelihood estimator in each array. The
  analysis (anatomy.main) draws the interval from one stream (seed 20261003) after the history-depth records of every array have
  drawn from it, so the depth records are rebuilt here, on the released history folds (Fig4a_folds.csv), only to bring the
  stream to the same point; the interval
  is then the same 2,000 hierarchical draws (arrays resampled, then the molecules of each role within each array). The likelihood
  owner is anatomy.ml_components on each array's product-forming patterns.
2 The frozen three-condition decision on Set A (E3): every released Set A pair is planned from the training arrays alone, with the
  frozen modules' own functions in the order of reserve_validation.decision (fit_array, recommend), and scored on its released
  measurements (score_panel); the scored targets must equal results/reserve_validation/decision_targets.csv as written.
The arrays are built from the released A1 records (figure1_data.public_units), which give the analysis arrays cell for cell.
Both must equal the released outputs exactly. Writes 05_DATA/Results/directional/frozen_reads_recomputed.json, or --out.
    python 06_CODE/Checks/frozen_reads_recomputed.py --set-a CSV [--out JSON]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "06_CODE/Figures"), str(ROOT / "06_CODE/Core/00_common"), str(ROOT / "06_CODE/Core/00_common/ctb"),
                 str(ROOT / "06_CODE/Validation/06_validation")]
import figure1_data as F1                                                   # noqa: E402
import anatomy as AN                                                         # noqa: E402
import validation_core as V                                                 # noqa: E402
from history import partner_arrays                                          # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "05_DATA/Results/directional/frozen_reads_recomputed.json"))
ap.add_argument("--set-a", required=True,
                help="the released Set A measurements")
ARGS = ap.parse_args()
RES = ROOT / "05_DATA/Results"
FOLDS = ROOT / "05_DATA/Analysis/figure_display/Fig4a_folds.csv"     # the released history|<array>|cv|0 folds


def figure1(AR):
    rng = np.random.default_rng(AN.SEED)
    F = pd.read_csv(FOLDS)
    share_objs, owners = {}, {}
    for key, A in AR.items():                                               # the order of anatomy.main
        M = A["M"]
        act = np.nanmax(M, 1) >= AN.FORMS
        E = AN.pattern(M[act])[0]
        share_objs[key] = (AN.Shares(E, A["a"][act], A["b"][act]), A["a"][act], A["b"][act])
        ml1, ml2 = AN.ml_components(E, A["a"][act], A["b"][act])
        owners[key] = 1 if ml1 >= ml2 else 2
        fold = dict(zip(F[F.screen == key].subpair, F[F.screen == key].fold))
        folds = [np.array([i for i, p in enumerate(A["pairs"]) if fold[p] == j]) for j in range(1, 6)]
        assert sorted(np.concatenate(folds).tolist()) == list(range(len(M)))
        AN.depth_records(A, folds, act, rng)                               # advances the stream as the analysis did
    keys, boot, redraws = list(share_objs), [], 0
    for _ in range(AN.NBOOT):
        vals = []
        for key in rng.choice(keys, len(keys)):
            sh, a, b = share_objs[key]
            for _try in range(50):
                r = sh.compute(AN.molecule_weights(a, b, rng))
                if np.isfinite(r["1"]) and np.isfinite(r["2"]):
                    break
                redraws += 1
            vals.append(r["1"] + r["2"])
        boot.append(np.median(vals))
    ci = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    ref = json.loads((RES / "display_inputs/Anatomy.json").read_text(encoding="utf-8"))["share"]
    ref_own = pd.read_csv(RES / "display_inputs/Anatomy_arrays.csv").set_index("screen").ml_owner_role.astype(int).to_dict()
    return dict(molecular_median_interval=ci, redrawn=redraws, likelihood_owner_role=owners,
                interval_equals_released=ci == ref["ci"] and redraws == ref["redrawn"],
                likelihood_owner_equals_released=owners == {k: ref_own[k] for k in owners})


def e3(AR):
    S = pd.read_csv(ARGS.set_a)
    assert S[["screen", "subpair"]].drop_duplicates().groupby("subpair").screen.nunique().max() == 1, "a subpair occurs in multiple arrays"
    fits = {k: V.fit_array(A) for k, A in AR.items()}
    rows = []
    for c in S.drop_duplicates("subpair").itertuples():                    # the target order of the frozen manifest
        fit = fits.get(c.screen)
        if fit is None:
            continue
        try:
            rec = V.recommend(fit, V.canonical(c.sub_1_smiles), V.canonical(c.sub_2_smiles))
        except ValueError:
            continue
        conds = fit["conditions"]
        idx = {cid: j for j, cid in enumerate(conds)}
        z = S[(S.screen == c.screen) & (S.subpair == c.subpair)]
        z = z[~z.duplicated("condition_id", keep=False)].set_index("condition_id")["yield"]
        y = np.array([z.get(cid, np.nan) for cid in conds], float)
        r = V.score_panel(y, np.array([idx[conds[j]] for j in rec["history"]]), np.array([idx[conds[j]] for j in rec["complementary"]]))
        if r is None:
            continue
        a, b = c.subpair.split(" >> ")
        rows.append(dict(screen=c.screen, subpair=c.subpair, **r, product_forming=bool(np.nanmax(y) >= AN.FORMS),
                         building_block_a=a, building_block_b=b))
    text = pd.DataFrame(rows).to_csv(index=False, lineterminator="\n")
    released = (RES / "reserve_validation/decision_targets.csv").read_bytes().decode("utf-8").replace("\r\n", "\n")
    mine, ref = pd.read_csv(pd.io.common.StringIO(text)), pd.read_csv(RES / "reserve_validation/decision_targets.csv")
    return dict(targets=len(rows), product_forming=int(sum(r["product_forming"] for r in rows)),
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), bytes_equal_released=text == released,
                max_abs_difference_pp=float((mine.difference_pp - ref.difference_pp).abs().max()),
                targets_equal_released=bool(list(zip(mine.screen, mine.subpair)) == list(zip(ref.screen, ref.subpair))
                                            and (mine.product_forming == ref.product_forming).all()))


def main():
    AR = partner_arrays(F1.public_units()[0])                               # the released A1 records, as the analysis arrays
    res = dict(figure1=figure1(AR), e3=e3(AR))
    f, e = res["figure1"], res["e3"]
    ok = (len(f["likelihood_owner_role"]) == 15 and f["interval_equals_released"] and f["likelihood_owner_equals_released"]
          and e["targets"] == 396 and e["targets_equal_released"] and e["max_abs_difference_pp"] < 1e-9)
    Path(ARGS.out).write_bytes((json.dumps(res, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(res))
    if not ok:
        raise SystemExit("a recomputed value differs from the released output")


if __name__ == "__main__":
    main()
