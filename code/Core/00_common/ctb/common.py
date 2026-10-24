"""Shared scientific definitions and registered data boundaries.

The immutable public release is under 02_DATA/00_RAW_IMMUTABLE. Models load the physically separate
cleaned training and development-seen, analysis-fit-excluded files. Legacy `sealed` and `months`
identifiers are compatibility names, not evidence of never-seen or temporal validation.
"""
from pathlib import Path
import json
import zlib
import os, sys, copy, time, tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from boundaries import read_table, validate_training, frozen_folds, verify_file
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
DATA = ROOT / "05_DATA/Release/00_RAW_IMMUTABLE"
OUT = Path(os.environ.get("NATURE_OUT", ROOT / "05_DATA/Results/primary"))
OUT.mkdir(parents=True, exist_ok=True)
TRAIN = Path(os.environ.get("NATURE_TRAIN", ROOT / "05_DATA/Analysis/02_ANALYSIS_TRAINING/analysis_measurements.csv.gz"))
TARGET = Path(os.environ.get("NATURE_TARGET", ROOT / "05_DATA/Analysis/04_RESERVED_VALIDATION/development_seen_analysis_fit_excluded.csv.gz"))

# The Hiyama array ran one condition twice (C43 and C46, identical composition). A condition main
# effect cannot hold two identical conditions, so C46 enters no fit.
DUPLICATE = ("Hiyama|1", "C46")
# Three process-development C-N arrays pair every amine with a different aryl halide: no building
# block recurs, so partner histories are not defined there.
UNIQUE_PAIR_ARRAYS = ("Buchwald-Hartwig(CMC)|1", "Buchwald-Hartwig(CMC)|2", "Buchwald-Hartwig(CMC)|3")
USABLE, DEAD = 0.20, 0.05       # isolable yield; measurable response
TOPK = 3                        # conditions per recommended panel


def data_file(name):
    """A released file, gzipped or not. The release ships the large tables as .csv.gz, which pandas
    reads directly; a reader who gunzips them gets the same result from the same code."""
    for p in (DATA / (name + ".gz"), DATA / name):
        if p.exists():
            return verify_file(p)
    raise FileNotFoundError(DATA / name)


def load_analysis():
    d = validate_training(read_table(TRAIN))
    if os.environ.get('NATURE_TASK_INDEX'):
        index=read_table(Path(os.environ['NATURE_TASK_INDEX']))
        keys=pd.MultiIndex.from_frame(index[['screen','subpair']])
        d=d[pd.MultiIndex.from_frame(d[['screen','subpair']]).isin(keys)]
    return d


def load_sealed():
    # Historical alias retained for numerical compatibility; these records were development-seen.
    from boundaries import assert_disjoint
    d = read_table(TARGET)
    assert_disjoint(load_analysis(), d)
    if d.allowed_for_model_fit.any():
        raise ValueError("Evaluation file contains fit-eligible rows")
    return d


def load_conditions():
    return pd.read_csv(data_file("DATASET_conditions.csv"), low_memory=False)


def load_archive():
    return pd.read_csv(data_file("DATASET_plate_archive.csv"))


def matrix(d):
    """Substrate pairs (rows, sorted) x condition ids (columns, sorted) of measured yield."""
    p = d.pivot(index="subpair", columns="condition_id", values="yield")
    return list(p.index), list(p.columns), p.to_numpy(float)


def arrays(d):
    return {key: matrix(g) for key, g in d.groupby("screen", sort=True)}


def labels(d):
    return d.drop_duplicates("screen").set_index("screen").screen_label.to_dict()


def top_k(values, k=TOPK):
    """Indices of the k largest values; missing values rank last; ties go to the earlier column
    (conditions are in sorted-id order), so the choice does not depend on the sort implementation."""
    return np.argsort(-np.nan_to_num(np.asarray(values, float), nan=-9.0), kind="stable")[:k]


def greedy_trio(y, k=TOPK):
    """Complementary trio from training rows: add the condition that maximises the mean, over rows with
    at least one measured cell in the candidate panel, of each row's best measured panel yield. A missing
    cell is never read as zero; exact ties go to the first condition id."""
    best = np.full(len(y), np.nan)
    chosen = []
    for _ in range(k):
        cand = np.fmax(best[:, None], y)
        with np.errstate(all="ignore"):
            obj = np.nanmean(cand, axis=0)
        obj[chosen] = -np.inf
        obj[~np.isfinite(obj)] = -np.inf
        j = int(np.argmax(obj))
        chosen.append(j)
        best = cand[:, j]
    return np.array(chosen)


class RegisteredRandomState(np.random.RandomState):
    pass


def stream(array, task, rep=0, seed=20260905):
    rng = RegisteredRandomState((seed + zlib.crc32(f"{array}|{task}|{rep}".encode())) % (2 ** 32 - 1))
    rng.fold_key = f"history|{array}|{task}|{rep}"
    return rng


def five_folds(n, rng):
    folds = frozen_folds(rng.fold_key)
    if sorted(np.concatenate(folds).tolist()) != list(range(n)):
        raise ValueError("Fold population mismatch")
    return folds


def whole_array_interval(z, col, seed=20260920, nboot=10000):
    """Resample whole arrays with every row of an array kept together; pooled mean = sum / count."""
    a = z.groupby("screen")[col].agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(a), size=(nboot, len(a)))
    vals = a["sum"].to_numpy()[draws].sum(1) / a["count"].to_numpy()[draws].sum(1)
    lo, hi = np.quantile(vals, [.025, .975])
    return float(z[col].mean()), float(lo), float(hi)


def owner_interval(d, value, group, seed=20260920, nboot=5000):
    """Resample building blocks within arrays, each with all of its rows; pooled mean = sum / count."""
    rng = np.random.default_rng(seed)
    sums = np.zeros(nboot); counts = np.zeros(nboot)
    for _, z in d.groupby("screen"):
        t = z.groupby(group)[value].agg(["sum", "count"])
        ix = rng.integers(len(t), size=(nboot, len(t)))
        sums += t["sum"].to_numpy()[ix].sum(1); counts += t["count"].to_numpy()[ix].sum(1)
    lo, hi = np.quantile(sums / counts, [.025, .975])
    return float(d[value].mean()), float(lo), float(hi)


def bh(p):
    """Benjamini-Hochberg q values."""
    p = np.asarray(p, float); o = np.argsort(p); q = np.empty(len(p))
    q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


_loaded_json_snapshots = {}


def save_json(name, obj):
    """Atomic publication; independent analysis blocks cannot overwrite each other's updates."""
    path=OUT/name; lock=path.with_suffix(path.suffix+'.lock')
    started=time.monotonic()
    while True:
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.close(fd);break
        except FileExistsError:
            if time.monotonic()-started>60:raise TimeoutError(f'Output lock busy: {name}')
            time.sleep(.05)
    try:
        if name in _loaded_json_snapshots and path.exists():
            previous=_loaded_json_snapshots[name]
            changes={k:v for k,v in obj.items() if k not in previous or v!=previous[k]}
            current=json.loads(path.read_text(encoding='utf-8'));current.update(changes);obj=current
        fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=OUT)
        with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(json.dumps(obj,indent=1,default=float))
        os.replace(tmp,path)
    finally:
        lock.unlink()


def load_json(name):
    obj=json.loads((OUT / name).read_text(encoding="utf-8"))
    _loaded_json_snapshots[name]=copy.deepcopy(obj)
    return obj
