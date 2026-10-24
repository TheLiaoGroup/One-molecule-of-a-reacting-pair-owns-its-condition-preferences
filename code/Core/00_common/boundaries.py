"""Versioned physical data and outcome-free prediction boundaries."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
META = ROOT / '05_DATA/Analysis/06_METADATA'
_verified = {}

def logical_key(path):
    key = Path(path).resolve().relative_to(ROOT).as_posix()
    mapping_path = ROOT / '07_METHODS_AND_PROVENANCE/analysis_runtime/PATH_RELOCATIONS.json'
    mapping = json.loads(mapping_path.read_text(encoding='utf-8')) if mapping_path.exists() else {}
    if 'relocations' in mapping:
        mapping = mapping['relocations']
    if isinstance(mapping, list):
        mapping = {r.get('old_path', r.get('original_path')): r.get('new_path', r.get('local_path')) for r in mapping}
    for old, new in mapping.items():
        if isinstance(new, dict): new = new.get('new_path', new.get('local_path'))
        if key == new: return old
    return key

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_file(path, expected=None):
    path = Path(path).resolve()
    if expected is None:
        lock = json.loads((META/'input_lock.json').read_text(encoding='utf-8'))
        key = logical_key(path)
        if key not in lock: raise ValueError(f'Unregistered dataset: {key}')
        expected = lock[key]['sha256']
    stat = path.stat()
    token = (str(path), stat.st_mtime_ns, stat.st_size, expected)
    if token not in _verified:
        if sha256(path) != expected: raise ValueError(f'Unexpected dataset version: {path.name}')
        _verified[token] = True
    return path

def read_table(path):
    path = verify_file(path)
    df = pd.read_csv(path, low_memory=False)
    lock = json.loads((META/'input_lock.json').read_text(encoding='utf-8'))
    schema = lock[logical_key(path)].get('columns')
    if schema is not None and list(df.columns) != schema: raise ValueError('Unexpected schema')
    dtypes = lock[logical_key(path)].get('dtypes')
    if dtypes is not None and {c:str(t) for c,t in df.dtypes.items()} != dtypes: raise ValueError('Unexpected field types')
    return df

def validate_training(frame):
    if 'allowed_for_model_fit' not in frame or not frame.allowed_for_model_fit.astype(bool).all():
        raise ValueError('Protected or unregistered rows supplied for fitting')
    if frame[['screen','subpair','condition_id']].duplicated().any():
        raise ValueError('Duplicate analysis cells')
    return frame

def assert_disjoint(train, target):
    a = set(map(tuple, train[['screen','subpair']].drop_duplicates().to_numpy()))
    b = set(map(tuple, target[['screen','subpair']].drop_duplicates().to_numpy()))
    if a & b: raise ValueError('Illegal train/target pair overlap')

def validate_indices(tr, te):
    tr, te = np.asarray(tr, int), np.asarray(te, int)
    if len(set(tr)) != len(tr) or len(set(te)) != len(te) or set(tr) & set(te):
        raise ValueError('Illegal training/target index overlap')
    return tr, te

_folds = None
def _load_folds():
    global _folds
    if _folds is None:
        _folds = json.loads(verify_file(META/'frozen_folds.json').read_text(encoding='utf-8'))
    return _folds

def frozen_fold_keys(prefix):
    keys = [key for key in _load_folds() if key.startswith(prefix)]
    if not keys: raise ValueError(f'Unregistered fold population: {prefix}')
    return keys

def frozen_folds(key):
    _load_folds()
    if key not in _folds: raise ValueError(f'Unregistered fold specification: {key}')
    return [np.asarray(x, int) for x in _folds[key]]

def frozen_assignment(values, seed):
    key = 'assignment|' + str(seed) + '|' + hashlib.sha256(json.dumps(list(values),ensure_ascii=False).encode()).hexdigest()
    ids = frozen_folds(key)[0]
    return ids, dict(zip(values, map(int, ids)))

def regime_fold_key(cp, held1, held2, te, regime):
    payload=[cp,sorted(held1),sorted(held2),sorted(map(int,te)),regime]
    return 'regime|' + hashlib.sha256(json.dumps(payload,ensure_ascii=False).encode()).hexdigest()
