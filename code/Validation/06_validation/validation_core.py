"""Training-only recommendations and fixed retrospective scoring definitions."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2] / "06_CODE/Core/00_common"), str(HERE.parents[2] / "06_CODE/Core/00_common/ctb")]
import common
import history


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        raise ValueError("Unparsable molecular identity")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def fit_array(A):
    """Fit only the pre-existing cleaned analysis matrix, never target outcomes."""
    tr = np.arange(len(A["M"]))
    R, means = history.train_residual(A["M"], tr)
    U, W = history.joint_profiles(R, A["ia"], A["ib"], len(A["ua"]), len(A["ub"]), tr)
    a = [canonical(s) for s in A["ua"]]
    b = [canonical(s) for s in A["ub"]]
    if len(set(a)) != len(a) or len(set(b)) != len(b):
        raise ValueError("Training identities collide under canonicalization")
    # One row per pair; distinct copartners also protect the minimum-history rule.
    depths_a = [len(set(A["b"][A["ia"] == i])) for i in range(len(a))]
    depths_b = [len(set(A["a"][A["ib"] == i])) for i in range(len(b))]
    return dict(conditions=A["conds"], means=means, U=U, W=W,
                a_lookup={s: i for i, s in enumerate(a)},
                b_lookup={s: i for i, s in enumerate(b)},
                depths_a=depths_a, depths_b=depths_b,
                complementary=common.greedy_trio(A["M"]),
                mean_panel=common.top_k(means),
                training_pairs={(canonical(x), canonical(y)) for x, y in zip(A["a"], A["b"])})


def recommend(fit, a, b):
    if (a, b) in fit["training_pairs"]:
        raise ValueError("Target pair is present in training")
    ia, ib = fit["a_lookup"].get(a), fit["b_lookup"].get(b)
    da = fit["depths_a"][ia] if ia is not None else 0
    db = fit["depths_b"][ib] if ib is not None else 0
    ka, kb = da >= 2, db >= 2
    if not (ka or kb):
        raise ValueError("Neither target building block has two training partners")
    u = np.zeros(len(fit["means"]))
    if ka:
        u += fit["U"][ia]
    if kb:
        u += fit["W"][ib]
    return dict(history=common.top_k(fit["means"] + u),
                complementary=fit["complementary"], mean_panel=fit["mean_panel"],
                interaction=u, means=fit["means"], depth_a=da, depth_b=db,
                known_a=ka, known_b=kb)


def score_panel(y, history_panel, comparator_panel):
    y = np.asarray(y, float)
    union = np.unique(np.r_[history_panel, comparator_panel]).astype(int)
    if not np.isfinite(y[union]).all():
        return None
    h, c = float(y[history_panel].max()), float(y[comparator_panel].max())
    return dict(history_yield=h, comparator_yield=c, difference_pp=100 * (h-c))


def residual_losses(y, training_means, predicted_interaction):
    y, m, u = (np.asarray(x, float) for x in (y, training_means, predicted_interaction))
    if not (np.isfinite(y).all() and np.isfinite(m).all() and np.isfinite(u).all()):
        raise ValueError("Residual prediction requires every locked observation")
    r = y-y.mean()-(m-m.mean())
    v = u-u.mean()
    return dict(sse=float(np.sum((r-v)**2)), sst=float(np.sum(r**2)), n_conditions=len(y))


def summarize_effects(d, value="difference_pp", seed=20260928, nboot=10000):
    if d.empty:
        return dict(n_targets=0, n_arrays=0, mean_pp=None, ci_low=None, ci_high=None)
    g = d.groupby("screen")[value].agg(["sum", "count", "mean"])
    result = dict(n_targets=len(d), n_arrays=len(g), mean_pp=float(d[value].mean()),
                  equal_array_mean_pp=float(g["mean"].mean()),
                  positive=int((d[value] > 1e-10).sum()),
                  negative=int((d[value] < -1e-10).sum()),
                  tied=int((d[value].abs() <= 1e-10).sum()),
                  ci_low=None, ci_high=None)
    if len(g) >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.integers(len(g), size=(nboot, len(g)))
        boot = g["sum"].to_numpy()[idx].sum(axis=1)/g["count"].to_numpy()[idx].sum(axis=1)
        result["ci_low"], result["ci_high"] = map(float, np.quantile(boot, [.025, .975]))
        result["leave_one_array_out"] = {
            k: float(d[d.screen != k][value].mean()) for k in g.index
        }
    return result


def summarize_residuals(d, seed=20260928, nboot=10000):
    if d.empty:
        return dict(n_targets=0, n_arrays=0, r_squared=None, ci_low=None, ci_high=None)
    g = d.groupby("screen")[["sse", "sst"]].sum()
    sst = float(g.sst.sum())
    result = dict(n_targets=len(d), n_arrays=len(g), n_conditions=int(d.n_conditions.sum()),
                  sse=float(g.sse.sum()), sst=sst,
                  r_squared=float(1-g.sse.sum()/sst) if sst > 0 else None,
                  ci_low=None, ci_high=None)
    if len(g) >= 2 and sst > 0:
        rng = np.random.default_rng(seed)
        idx = rng.integers(len(g), size=(nboot, len(g)))
        den = g.sst.to_numpy()[idx].sum(axis=1)
        num = g.sse.to_numpy()[idx].sum(axis=1)
        ok = den > 0
        result["bootstrap_undefined_draws"] = int((~ok).sum())
        if ok.all():
            result["ci_low"], result["ci_high"] = map(float, np.quantile(1-num/den, [.025, .975]))
    return result


def verify_lock(lock_path):
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    for item in lock["files"]:
        p = Path(item["path"])
        if sha256(p) != item["sha256"]:
            raise ValueError(f"Locked file changed: {p.name}")
    return lock
