"""The decomposition of section 2, saved as an object the later sections read.

    y_ij = a_i + b_i g_j + p_ij

  g_j        the condition signature: the additive fit's grand mean plus condition effect, estimated on
             the OTHER half of the substrate pairs, so a pair never contributes to the signature it is
             scored against
  p_oof      out-of-fold residual of the amplitude model: preference together with measurement
             noise, any nonlinearity the model does not hold and the error in a_i and b_i, so its
             variance bounds preference from above rather than measuring it. The intercept and
             amplitude come from the other condition
             folds, so a cell takes no part in its own residualisation. This is the target of the
             structural analyses.
  p_desc     descriptive preference. The pair's intercept and amplitude come from all of its observed
             conditions. Only for the class-level interpretation of Figure 2B,C, never for prediction.
  p_oof_centred  p_oof with a cross-fitted condition mean removed: the mean at each condition is taken over
             the pairs of the other half, so removing it cannot manufacture agreement between two pairs
             that are being compared.

Every observed cell appears once, with the halves, folds and fitted quantities that produced it, so a
result downstream can be traced back to the numbers it came from.

Writes out/preference_residuals.csv.gz (one row per observed cell) and out/preference_arrays.csv.
"""
import numpy as np
import pandas as pd
from common import load_analysis, arrays, labels, save_json, OUT
from amplitude import additive, slope_fit, halves, condition_folds


def decompose(Y, *, screen):
    """Returns a dict of cell-level matrices, all the shape of Y and NaN where Y is."""
    n, m = Y.shape
    out = {k: np.full((n, m), np.nan) for k in
           ("g", "alpha_oof", "beta_oof", "p_oof", "alpha_desc", "beta_desc", "p_desc", "p_oof_centred")}
    half = np.full(n, -1); sig_half = np.full(n, -1); fold = np.full((n, m), -1)
    for h, (er, cr) in enumerate(halves(n, screen=screen)):
        mu, _, b = additive(Y[cr]); c = mu + b
        E = Y[er]
        half[er] = h; sig_half[er] = 1 - h
        out["g"][er] = c[None, :]
        # descriptive: the pair's own whole row
        fit_d, a_d, s_d, _ = slope_fit(E, c, np.arange(m))
        out["alpha_desc"][er] = a_d[:, None]; out["beta_desc"][er] = s_d[:, None]
        out["p_desc"][er] = E - fit_d
        # out of fold: the pair's intercept and slope from the other condition folds
        for f, te in enumerate(condition_folds(m)):
            tr = np.setdiff1d(np.arange(m), te)
            fit_o, a_o, s_o, _ = slope_fit(E, c, tr)
            ix = np.ix_(er, te)
            out["alpha_oof"][ix] = a_o[:, None]; out["beta_oof"][ix] = s_o[:, None]
            out["p_oof"][ix] = (E - fit_o)[:, te]
            fold[ix] = f
    for k in out:
        out[k] = np.where(np.isfinite(Y), out[k], np.nan)
    # cross-fitted condition mean of the out-of-fold preference: for a pair of one half, the mean over
    # the pairs of the other half
    for h, (er, cr) in enumerate(halves(n, screen=screen)):
        with np.errstate(all="ignore"):
            cm = np.nanmean(out["p_oof"][cr], axis=0)
        out["p_oof_centred"][er] = out["p_oof"][er] - np.where(np.isfinite(cm), cm, 0.0)[None, :]
    out["p_oof_centred"] = np.where(np.isfinite(Y), out["p_oof_centred"], np.nan)
    return out, half, sig_half, fold


def split_decomposition(Y, tr, te):
    """The decomposition of an outer training set, applied to both sides of that split.

    The condition signature and the residual condition mean come from the training rows alone, so no
    quantity used for training or for ranking depends on a test row's yields. Within each side a pair's
    intercept and amplitude are still estimated on the other condition folds, so a cell takes no part in
    its own residual. Returns, for the training rows and for the test rows, the centred out-of-fold
    residual and the descriptive amplitude.
    """
    m = Y.shape[1]
    mu, _, b = additive(Y[tr])
    c = mu + b                                        # signature: training rows only
    out = {}
    for name, rows in (("train", tr), ("test", te)):
        E = Y[rows]
        p = np.full(E.shape, np.nan)
        for te_c in condition_folds(m):
            tr_c = np.setdiff1d(np.arange(m), te_c)
            fit, _, _, _ = slope_fit(E, c, tr_c)
            p[:, te_c] = (E - fit)[:, te_c]
        _, _, beta, _ = slope_fit(E, c, np.arange(m))
        out[name] = [np.where(np.isfinite(E), p, np.nan), beta]
    with np.errstate(all="ignore"):
        cm = np.nanmean(out["train"][0], axis=0)      # residual condition mean: training rows only
    cm = np.where(np.isfinite(cm), cm, 0.0)
    for name in out:
        out[name][0] = out[name][0] - cm[None, :]
    return dict(train_p=out["train"][0], train_beta=out["train"][1],
                test_p=out["test"][0], test_beta=out["test"][1])

def main():
    d = load_analysis()
    lab = labels(d)
    rows, summary = [], []
    for key, (pairs, conds, Y) in arrays(d).items():
        P, half, sig_half, fold = decompose(Y, screen=key)
        obs = np.isfinite(Y)
        yhat_oof = P["alpha_oof"] + P["beta_oof"] * P["g"]
        yhat_desc = P["alpha_desc"] + P["beta_desc"] * P["g"]
        assert np.allclose(Y[obs], (yhat_oof + P["p_oof"])[obs]), key       # the decomposition is exact
        assert np.allclose(Y[obs], (yhat_desc + P["p_desc"])[obs]), key
        assert np.isfinite(P["p_oof"][obs]).all(), key                      # every observed cell has one
        i, j = np.where(obs)
        rows.append(pd.DataFrame(dict(
            screen=key, subpair=np.asarray(pairs)[i], condition_id=np.asarray(conds)[j], y=Y[obs],
            pair_half=half[i], signature_half=sig_half[i], condition_fold=fold[obs],
            g=P["g"][obs], alpha_oof=P["alpha_oof"][obs], beta_oof=P["beta_oof"][obs],
            yhat_oof=yhat_oof[obs], p_oof=P["p_oof"][obs], p_oof_centred=P["p_oof_centred"][obs],
            alpha_desc=P["alpha_desc"][obs], beta_desc=P["beta_desc"][obs],
            yhat_desc=yhat_desc[obs], p_desc=P["p_desc"][obs])))
        v = np.nanvar
        summary.append(dict(screen=key, label=lab[key], pairs=len(pairs), conditions=len(conds),
                            cells=int(obs.sum()),
                            var_y=float(v(Y[obs])), var_p_oof=float(v(P["p_oof"][obs])),
                            var_p_desc=float(v(P["p_desc"][obs])),
                            resid_share_oof=float(v(P["p_oof"][obs]) / v(Y[obs])),
                            resid_share_desc=float(v(P["p_desc"][obs]) / v(Y[obs])),
                            median_beta_desc=float(np.nanmedian(P["beta_desc"][:, 0])),
                            iqr_beta_desc=float(np.nanpercentile(P["beta_desc"][:, 0], 75)
                                                - np.nanpercentile(P["beta_desc"][:, 0], 25))))
        print(f"  {key:28s} cells {int(obs.sum()):6d}  residual share oof {summary[-1]['resid_share_oof']:.3f} "
              f"desc {summary[-1]['resid_share_desc']:.3f}", flush=True)

    R = pd.concat(rows, ignore_index=True)
    R.to_csv(OUT / "preference_residuals.csv.gz", index=False,
             compression={"method": "gzip", "mtime": 0}, float_format="%.6g")
    S = pd.DataFrame(summary); S.to_csv(OUT / "preference_arrays.csv", index=False)
    save_json("preference.json", dict(n_arrays=int(len(S)), n_cells=int(S.cells.sum()),
                                      median_resid_share_oof=float(S.resid_share_oof.median()),
                                      median_resid_share_desc=float(S.resid_share_desc.median())))
    print(f"\n{len(R):,} cells written; median share of yield variance the amplitude model leaves (preference plus measurement noise, nonlinearity and estimation error): "
          f"out-of-fold {S.resid_share_oof.median():.3f}, descriptive {S.resid_share_desc.median():.3f}")


if __name__ == "__main__":
    main()
