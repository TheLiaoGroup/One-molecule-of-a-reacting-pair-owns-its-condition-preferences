"""Reproduce the two external public screens used in Supplementary Note 21.

Run this optional network-enabled entry point separately from rebuild_all.py.
ORD data are CC-BY-SA-4.0 and are downloaded only to the requested output
directory, which must be outside the public package.

Use --calibrate separately to reproduce the two frozen training-only
500-simulation calibrations, with their distinct original success criteria.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import urllib.request
import warnings

sys.dont_write_bytecode = True
for _name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from ord_schema.proto import reaction_pb2 as pb
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.error')
ANATOMY = None
HF_REVISION = 'b6c89761dacf809d4082acbdbebe72042a08db91'
DATASETS = {
    'CH_arylation': dict(id='675eddcaa6674ce3ae61e79bbc1e1c08', bytes=121369,
        sha256='e95aa366e1e2fb05b8af252af83c024430203b84eb4a3caf738b3e89053f461a',
        shape=(8, 8, 24), records=1536, cells=1536,
        title='Identifying general reaction conditions by bandit optimization',
        doi='10.1038/s41586-024-07021-y'),
    'Suzuki': dict(id='3b8a2ef300e145468579027f206a3ac8', bytes=189018,
        sha256='e18e7f7afa6639bbec35c2f52ccace4ecc39eb0994814984e53cf8b3b569579e',
        shape=(12, 10, 12), records=1440, cells=1392,
        title='Miniaturization of popular reactions from the medicinal chemists\u2019 toolbox for ultrahigh-throughput experimentation',
        doi='10.1038/s44160-023-00351-1'),
}
HALIDE = Chem.MolFromSmarts('[c,n][Cl,Br,I,$(OS(=O)(=O)C(F)(F)F)]')
ZERO = 64 * np.finfo(float).eps
NBOOT, MAX_ATTEMPTS = 2000, 20000


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def save_json(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')


def verify_bytes(content, digest, size=None):
    actual = hashlib.sha256(content).hexdigest()
    if actual != digest or (size is not None and len(content) != size):
        raise ValueError(f'ORD source checksum/size mismatch: {actual}; {len(content)} bytes')
    return actual


def download_source(key, output, input_dir=None):
    spec = DATASETS[key]
    filename = f"ord_dataset-{spec['id']}.parquet"
    url = (f'https://huggingface.co/datasets/open-reaction-database/ord-data/resolve/{HF_REVISION}/'
           f"data/{spec['id'][:2]}/{filename}")
    path = Path(input_dir) / filename if input_dir else output / 'raw_ord' / filename
    if path.is_file():
        content = path.read_bytes()
    elif input_dir:
        raise FileNotFoundError(path)
    else:
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        verify_bytes(content, spec['sha256'], spec['bytes'])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    verify_bytes(content, spec['sha256'], spec['bytes'])
    return path, dict(dataset_id='ord_dataset-' + spec['id'], url=url,
        sha256=spec['sha256'], bytes=len(content), primary_paper=spec['title'], doi=spec['doi'],
        license='CC-BY-SA-4.0', source_path=str(path.resolve()))


def selected_product(reaction):
    if not reaction.outcomes:
        return None, dict(reason='no_outcome', outcomes=0, selected_outcome=None)
    units = {pb.Time.SECOND: 1, pb.Time.MINUTE: 60, pb.Time.HOUR: 3600, pb.Time.DAY: 86400}
    timed = [(i, item.reaction_time.value * units[item.reaction_time.units])
             for i, item in enumerate(reaction.outcomes)
             if item.HasField('reaction_time') and item.reaction_time.HasField('value')
             and item.reaction_time.units in units]
    index = max(timed, key=lambda pair: (pair[1], pair[0]))[0] if len(timed) == len(reaction.outcomes) else len(reaction.outcomes) - 1
    products = reaction.outcomes[index].products
    desired = [item for item in products if item.is_desired_product]
    selected = desired if desired else list(products)
    info = dict(outcomes=len(reaction.outcomes), selected_outcome=index)
    if len(selected) != 1:
        return None, dict(reason='ambiguous_target', products=len(products), desired=len(desired), **info)
    return selected[0], info


def extract_ratio(reaction):
    product, info = selected_product(reaction)
    if product is None:
        raise ValueError('Suzuki ratio endpoint is absent or ambiguous: ' + str(info))
    measurements = [item for item in product.measurements if item.details.strip() == 'Pdt-to-IS Ratio']
    if len(measurements) != 1:
        raise ValueError('Suzuki requires exactly one Pdt-to-IS Ratio measurement')
    item = measurements[0]
    if (item.type != pb.ProductMeasurement.AREA or item.WhichOneof('value') != 'float_value'
            or not item.float_value.HasField('value')):
        raise ValueError('Suzuki ratio requires AREA with a numeric float_value')
    value = item.float_value.value
    if not np.isfinite(value) or value < 0:
        raise ValueError('Suzuki ratio must be finite and nonnegative')
    return float(value), dict(reason='numeric', ratio_measurements=1, **info)


def extract_yield(reaction):
    product, info = selected_product(reaction)
    metadata = dict(yield_measurements=0, measurements_below0=0, measurements_above105=0,
                    measurements_100_to105=0, measurements_above100=0,
                    minimum_measurement=None, maximum_measurement=None, **info)
    if product is None:
        return None, metadata
    values, missing = [], None
    for item in product.measurements:
        if item.type != pb.ProductMeasurement.YIELD:
            continue
        field = item.WhichOneof('value')
        if field == 'percentage' and item.percentage.HasField('value'):
            value = item.percentage.value / 100
        elif field == 'float_value' and item.float_value.HasField('value'):
            detail = item.details.casefold()
            if '%' in detail or 'percent' in detail:
                value = item.float_value.value / 100
            elif 'fraction' in detail or '0-1' in detail or '0 to 1' in detail:
                value = item.float_value.value
            else:
                missing = missing or 'unspecified_float_yield_unit'
                continue
        else:
            missing = missing or 'unsupported_or_missing_yield_value'
            continue
        if not np.isfinite(value):
            raise ValueError('Nonfinite external YIELD; do not clip')
        values.append(value)
    metadata.update(yield_measurements=len(values), measurements_below0=sum(v < 0 for v in values),
        measurements_above105=sum(v > 1.05 for v in values),
        measurements_100_to105=sum(1 < v <= 1.05 for v in values), measurements_above100=sum(v > 1 for v in values),
        minimum_measurement=min(values) if values else None, maximum_measurement=max(values) if values else None)
    if missing or not values:
        return None, dict(reason=missing or 'no_YIELD_measurement', **metadata)
    return float(np.mean(values)), dict(reason='numeric', **metadata)


def compound_identity(compound):
    identifiers = {}
    for identifier in compound.identifiers:
        identifiers.setdefault(identifier.type, identifier.value)
    for kind in (pb.CompoundIdentifier.SMILES, pb.CompoundIdentifier.CXSMILES,
                 pb.CompoundIdentifier.NAME, pb.CompoundIdentifier.IUPAC_NAME):
        if identifiers.get(kind):
            return identifiers[kind]
    raise ValueError('Compound has no registered identity')


def structural_identity(reaction):
    components = []
    for reaction_input in reaction.inputs.values():
        for compound in reaction_input.components:
            # The frozen structural parser represented an absent ORD role as "None".
            role = ('None' if compound.reaction_role == pb.ReactionRole.UNSPECIFIED
                    else pb.ReactionRole.ReactionRoleType.Name(compound.reaction_role))
            components.append((role, compound_identity(compound)))
    reactants = [text for role, text in components if role == 'REACTANT']
    if len(reactants) != 2:
        raise ValueError('Exactly two reactants required')
    molecules = [Chem.MolFromSmiles(text) for text in reactants]
    if any(item is None for item in molecules):
        raise ValueError('Invalid reactant SMILES')
    flags = [item.HasSubstructMatch(HALIDE) for item in molecules]
    if sum(flags) != 1:
        raise ValueError('Ambiguous or absent aryl leaving-group role')
    h = flags.index(True)
    halide, partner = Chem.MolToSmiles(molecules[h]), Chem.MolToSmiles(molecules[1 - h])
    conditions = []
    for role, text in components:
        if role != 'REACTANT':
            molecule = Chem.MolFromSmiles(text)
            conditions.append((role, Chem.MolToSmiles(molecule) if molecule is not None else text))
    temperature = reaction.conditions.temperature
    temp = round(temperature.setpoint.value, 1) if temperature.HasField('setpoint') and temperature.setpoint.HasField('value') else None
    return halide, partner, json.dumps([sorted(conditions), temp], separators=(',', ':'))


def parse_source(path, key):
    spec = DATASETS[key]
    table = pq.read_table(path, columns=['reaction_id', 'reaction'], use_threads=False)
    metadata = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items() if k.startswith(b'ord.')}
    if metadata.get('ord.dataset_id') != 'ord_dataset-' + spec['id']:
        raise ValueError('ORD dataset metadata differs from requested identifier')
    records = []
    for rid, blob in zip(table['reaction_id'].to_pylist(), table['reaction'].to_pylist()):
        reaction = pb.Reaction.FromString(blob)
        if reaction.reaction_id != rid:
            raise ValueError('Parquet/protobuf reaction identifier mismatch')
        h, p, condition = structural_identity(reaction)
        value, info = extract_ratio(reaction) if key == 'Suzuki' else extract_yield(reaction)
        records.append(dict(dataset=key, reaction_id=rid, halide=h, partner=p,
                            condition_signature=condition, value=value, **info))
    frame = pd.DataFrame(records)
    columns = ['halide', 'partner', 'condition_signature']
    shape = tuple(frame[column].nunique() for column in columns)
    cells = frame.groupby(columns).size()
    if (len(frame) != spec['records'] or frame.reaction_id.duplicated().any()
            or shape != spec['shape'] or len(cells) != spec['cells']
            or len(frame[['halide', 'partner']].drop_duplicates()) != spec['shape'][0] * spec['shape'][1]):
        raise ValueError('Source design/multiplicity differs from the pinned screen')
    return frame, dict(records=len(frame), halides=shape[0], partners=shape[1], conditions=shape[2],
        unique_cells=len(cells), duplicate_extra_records=int((cells - 1).sum()), metadata=metadata)


def matrix_from_frame(frame):
    means = frame.groupby(['halide', 'partner', 'condition_signature'], sort=True, dropna=False).value.mean()
    matrix = means.unstack('condition_signature').sort_index().sort_index(axis=1)
    return matrix.to_numpy(float), np.asarray(matrix.index.get_level_values('halide')), np.asarray(matrix.index.get_level_values('partner'))


def _pearson(x, y, zero, ratio):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return np.nan
    if not ratio:
        return float(np.corrcoef(x[ok], y[ok])[0, 1]) if np.std(x[ok]) > zero and np.std(y[ok]) > zero else np.nan
    x, y = x[ok], y[ok]
    x, y = x - x.mean(), y - y.mean()
    if np.sqrt(np.mean(x * x)) <= zero or np.sqrt(np.mean(y * y)) <= zero:
        return np.nan
    return float(np.clip(np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y)), -1., 1.))


def _graph(residuals, focal, opposite, nfocal, nopposite, zero, ratio):
    adjacency = np.zeros((nfocal, nopposite), dtype=bool)
    adjacency[focal, opposite] = True
    edges = list(itertools.combinations(range(nopposite), 2))
    left, right = np.array([x[0] for x in edges], dtype=int), np.array([x[1] for x in edges], dtype=int)
    required = adjacency[:, left] & adjacency[:, right]
    row = np.full((nfocal, nopposite), -1, dtype=int)
    row[focal, opposite] = np.arange(len(residuals))
    correlations = np.full(required.shape, np.nan)
    for molecule, edge in zip(*np.nonzero(required)):
        correlations[molecule, edge] = _pearson(residuals[row[molecule, left[edge]]],
                                               residuals[row[molecule, right[edge]]], zero, ratio)
    return dict(adjacency=adjacency, left=left, right=right, required=required, correlations=correlations)


def prepare(M, a, b, *, ratio):
    M, a, b = np.asarray(M, float), np.asarray(a), np.asarray(b)
    if M.ndim != 2 or a.ndim != 1 or b.ndim != 1 or len(a) != len(M) or len(b) != len(M):
        raise ValueError('Expected one original pair per matrix row and one ID per role')
    if len(set(zip(a.tolist(), b.tolist()))) != len(M):
        raise ValueError('Duplicate original pair identities are not permitted')
    halides, hi = np.unique(a, return_inverse=True)
    partners, pi = np.unique(b, return_inverse=True)
    finite = np.isfinite(M)
    productive = ((finite & (M > 0)).sum(axis=1) >= 3 if ratio else
        finite.any(axis=1) & (np.max(np.where(finite, M, -np.inf), axis=1) >= .10))
    zero = ZERO if ratio else ZERO * max(1., float(np.max(np.abs(M[finite]))) if finite.any() else 1.)
    active = M[productive].copy()
    if ratio:
        for index, row in enumerate(active):
            observed = np.isfinite(row)
            values = row[observed]
            if np.all(values == values[0]):
                active[index, observed] = 0.
            else:
                values = values / np.max(np.abs(values))
                centered = values - values.mean()
                active[index, observed] = centered / np.sqrt(np.mean(centered * centered))
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
        E = ANATOMY.pattern(active)[0] if len(active) else active.copy()
    h, p = hi[productive], pi[productive]
    return dict(matrix=M, all_a=a, all_b=b, productive_mask=productive, standardized=active,
        residuals=E, halides=halides, partners=partners, numerical_zero=zero, ratio=ratio,
        a=a[productive], b=b[productive],
        halide_graph=_graph(E, h, p, len(halides), len(partners), zero, ratio),
        partner_graph=_graph(E, p, h, len(partners), len(halides), zero, ratio))


def weighted_median(scores, counts, eligible):
    if scores.shape[1] == 0:
        return np.full(len(scores), np.nan)
    values = np.where(eligible, scores, np.inf)
    order = np.argsort(values, axis=1)
    values = np.take_along_axis(values, order, axis=1)
    weights = np.take_along_axis(np.where(eligible, counts, 0), order, axis=1)
    cumulative, total = np.cumsum(weights, axis=1), weights.sum(axis=1)
    lower = np.argmax(cumulative > ((total - 1) // 2)[:, None], axis=1)
    upper = np.argmax(cumulative > (total // 2)[:, None], axis=1)
    result = (values[np.arange(len(scores)), lower] + values[np.arange(len(scores)), upper]) / 2
    return np.where(total > 0, result, np.nan)


def _role_summary(graph, focal_counts, opposite_counts):
    represented = (opposite_counts > 0).astype(int) @ graph['adjacency'].T.astype(int)
    weights = opposite_counts[:, graph['left']] * opposite_counts[:, graph['right']]
    denominator = weights @ graph['required'].T.astype(int)
    numerator = np.einsum('be,me->bm', weights,
        np.where(np.isfinite(graph['correlations']), graph['correlations'], 0.), optimize=False)
    undefined = (weights > 0).astype(int) @ (graph['required'] & ~np.isfinite(graph['correlations'])).T.astype(int)
    scores = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    eligible = (focal_counts > 0) & (represented >= 4) & (undefined == 0) & np.isfinite(scores)
    return weighted_median(scores, focal_counts, eligible), scores, eligible


def consistency_point(prepared):
    point = dict(pairs=len(prepared['matrix']), observed_cells=int(np.isfinite(prepared['matrix']).sum()),
                 productive_pairs=int(prepared['productive_mask'].sum()))
    records = []
    for role, opposite in (('halide', 'partners'), ('partner', 'halides')):
        labels = prepared['halides' if role == 'halide' else 'partners']
        graph = prepared[role + '_graph']
        median, scores, eligible = _role_summary(graph, np.ones((1, len(labels)), int),
                                                np.ones((1, len(prepared[opposite])), int))
        point[role + '_consistency'] = float(median[0]) if np.isfinite(median[0]) else None
        point[role + '_eligible_molecules'] = int(eligible.sum())
        for i, molecule in enumerate(labels):
            n = int(graph['adjacency'][i].sum())
            status = ('no_productive_partners' if n == 0 else
                'fewer_than_four_productive_partners' if n < 4 else
                'eligible' if eligible[0, i] else 'undefined_pairwise_correlation')
            records.append(dict(role=role, molecule=molecule, productive_partners=n,
                consistency=float(scores[0, i]) if eligible[0, i] else None,
                defined_comparisons=int((graph['required'][i] & np.isfinite(graph['correlations'][i])).sum()) if n >= 4 else 0,
                required_comparisons=int(graph['required'][i].sum()) if n >= 4 else 0, status=status))
    h, p = point['halide_consistency'], point['partner_consistency']
    point['difference'] = h - p if h is not None and p is not None else None
    return point, records


def consistency_bootstrap(prepared, *, seed, draws=NBOOT, max_attempts=MAX_ATTEMPTS, chunk_size=2000):
    for value in (draws, max_attempts, chunk_size):
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError('Draw, attempt and chunk counts must be positive integers')
    nh, npart = len(prepared['halides']), len(prepared['partners'])
    if nh == 0 or npart == 0:
        raise ValueError('Both original molecule universes must be nonempty')
    streams = [np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(2)]
    probabilities = (np.full(nh, 1 / nh), np.full(npart, 1 / npart))
    accepted, attempts, count = [], 0, 0
    rejected = dict(no_finite_halide_median=0, no_finite_partner_median=0, no_finite_both_medians=0)
    while attempts < max_attempts and count < draws:
        size = min(chunk_size, max_attempts - attempts)
        ch = streams[0].multinomial(nh, probabilities[0], size=size)
        cp = streams[1].multinomial(npart, probabilities[1], size=size)
        h, _, he = _role_summary(prepared['halide_graph'], ch, cp)
        p, _, pe = _role_summary(prepared['partner_graph'], cp, ch)
        take = np.flatnonzero(np.isfinite(h) & np.isfinite(p))[:draws - count]
        used = int(take[-1] + 1) if len(take) == draws - count else size
        no_h, no_p = ~np.isfinite(h[:used]), ~np.isfinite(p[:used])
        rejected['no_finite_halide_median'] += int((no_h & ~no_p).sum())
        rejected['no_finite_partner_median'] += int((no_p & ~no_h).sum())
        rejected['no_finite_both_medians'] += int((no_h & no_p).sum())
        if len(take):
            accepted.append(dict(attempt=attempts + take, halide_consistency=h[take], partner_consistency=p[take],
                difference=h[take] - p[take], halide_counts=ch[take], partner_counts=cp[take],
                halide_eligible_molecules=he[take].sum(axis=1), partner_eligible_molecules=pe[take].sum(axis=1)))
            count += len(take)
        attempts += used
    summary = dict(draws=int(draws), accepted_draws=int(count), attempts=int(attempts),
        rejection_count=sum(rejected.values()), rejected=rejected, undefined_accepted=0,
        acceptance_fraction=float(count / attempts), seed=int(seed), residuals_refitted_each_draw=False,
        rng='SeedSequence(seed).spawn(2); independent full-universe role multinomial streams',
        interval_condition='both role medians defined; productive membership and fitted residuals fixed')
    if count != draws:
        raise ValueError('Finite consistency draw cap exhausted: ' + str(summary))
    result = {key: np.concatenate([chunk[key] for chunk in accepted]) for key in accepted[0]}
    result['draw'] = np.arange(draws)
    summary.update(difference_ci95=np.quantile(result['difference'], [.025, .975]).tolist(),
                   fraction_positive=float(np.mean(result['difference'] > 0)))
    return result, summary


def moment_shares(E, a, b):
    if not len(E):
        return dict(halide=None, partner=None, neither=None, pattern_variance=None)
    shares = ANATOMY.Shares(E, a, b)
    variance = float(np.mean(shares.var))
    if variance <= ZERO ** 2:
        return dict(halide=None, partner=None, neither=None, pattern_variance=variance)
    with np.errstate(all='ignore'):
        values = shares.compute(np.ones(len(E)))
    return clean(dict(halide=values['1'], partner=values['2'], neither=values['N'], pattern_variance=variance))


def residual_standardize(E):
    result = np.full_like(E, np.nan)
    for index, row in enumerate(E):
        observed = np.isfinite(row)
        if observed.any():
            sd = float(np.std(row[observed]))
            result[index, observed] = row[observed] / sd if sd > ZERO else 0.
    return result


def descriptive_shares(prepared):
    M = prepared['matrix'][prepared['productive_mask']]
    E = ANATOMY.pattern(M)[0] if len(M) else M
    a, b = prepared['a'], prepared['b']
    return dict(raw_ratio_residuals=moment_shares(E, a, b),
        standardized_raw_residuals=moment_shares(residual_standardize(E), a, b),
        standardized_new_residuals=moment_shares(residual_standardize(prepared['residuals']), a, b))


def ch_estimates(prepared, noise_scale):
    point, records = consistency_point(prepared)
    M = prepared['matrix'][prepared['productive_mask']]
    a, b = prepared['a'], prepared['b']
    if not len(M):
        point.update({key: None for key in ('halide_share', 'partner_share', 'neither_covariance_share',
                                            'noise_share', 'unassigned_share')})
        return point, records
    E, H, fit = ANATOMY.pattern(M)
    shares = ANATOMY.Shares(E, a, b)
    variance = float(shares.var.mean())
    if variance <= prepared['numerical_zero'] ** 2:
        point.update({key: None for key in ('halide_share', 'partner_share', 'neither_covariance_share',
                                            'noise_share', 'unassigned_share')})
        return point, records
    values = shares.compute(np.ones(len(M)))
    noise = ANATOMY.noise_variance(M, fit, H, noise_scale) / variance
    point.update(halide_share=values['1'], partner_share=values['2'], neither_covariance_share=values['N'],
        noise_share=noise, unassigned_share=1 - values['1'] - values['2'] - noise,
        noise_interpretation='transported P2 Hiyama-calibrated model; not externally identified')
    return point, records


def share_bootstrap(prepared, *, seed):
    M = prepared['matrix'][prepared['productive_mask']]
    if not len(M):
        raise ValueError('No productive pairs for bootstrap')
    a, b = prepared['a'], prepared['b']
    shares = ANATOMY.Shares(prepared['residuals'], a, b)
    numerical_zero = ZERO * max(1., float(np.nanmax(np.abs(M))))
    if shares.var.mean() <= numerical_zero ** 2:
        raise ValueError('Zero numerical pattern variance for bootstrap')
    rng, rows, rejected = np.random.default_rng(seed), [], Counter()
    for attempt in range(MAX_ATTEMPTS):
        weights = ANATOMY.molecule_weights(a, b, rng)
        with np.errstate(all='ignore'):
            values = shares.compute(weights)
        if not (np.isfinite(values['1']) and np.isfinite(values['2'])):
            rejected['nonfinite_role_share'] += 1
            continue
        rows.append(dict(draw=len(rows), attempt=attempt, halide_share=values['1'], partner_share=values['2'],
                         difference=values['1'] - values['2'], weighted_pairs=float(weights.sum())))
        if len(rows) == NBOOT:
            break
    if len(rows) != NBOOT:
        raise ValueError('Finite share draw cap exhausted')
    frame = pd.DataFrame(rows)
    return frame, dict(draws=NBOOT, attempts=attempt + 1, rejected=dict(rejected), undefined_accepted=0,
        difference_ci95=np.quantile(frame.difference, [.025, .975]).tolist(),
        fraction_positive=float((frame.difference > 0).mean()), seed=int(seed), residuals_refitted_each_draw=False)


def public_noise_scale(package):
    columns = ['screen', 'subpair', 'condition_id', 'yield']
    C = pd.concat([pd.read_csv(package / 'data' / filename, usecols=columns) for filename in
                  ('measurements_A1_analysis_arrays.csv.gz', 'measurements_A3_hiyama_duplicate.csv.gz')], ignore_index=True)
    C = C[C.screen.eq('Hiyama|1')]
    P = C[C.condition_id != 'C46'].pivot_table(index='subpair', columns='condition_id', values='yield')
    duplicate = C[C.condition_id == 'C46'].set_index('subpair')['yield'].reindex(P.index).to_numpy()
    M = P.to_numpy(float)
    E, H, fit = ANATOMY.pattern(M)
    j = list(P.columns).index('C43')
    ok = np.isfinite(duplicate) & np.isfinite(M[:, j])
    if ok.sum() < 50:
        raise ValueError('Released Hiyama duplicate not found')
    e43, e46 = E[ok, j], duplicate[ok] - fit[ok, j]
    empirical = np.mean(e43 * e46) / np.mean((e43 ** 2 + e46 ** 2) / 2)
    model = np.mean(np.maximum(.07 * np.clip(fit[ok, j], 0, 1), .012) ** 2 * (1 - H[ok, j]))
    return float(empirical), float((1 - empirical) * np.mean((e43 ** 2 + e46 ** 2) / 2) / model)


def p2_suzuki_benchmark(package):
    columns = ['screen', 'condition_id', 'subpair', 'sub_1_smiles', 'sub_2_smiles', 'yield']
    data = pd.read_csv(package / 'derived/DATASET_analysis.csv.gz', usecols=columns)
    data = data[data.screen.eq('Suzuki|1')]
    if not (data.subpair == data.sub_1_smiles + ' >> ' + data.sub_2_smiles).all():
        raise ValueError('Released P2 pair identities differ')
    matrix = data.pivot(index='subpair', columns='condition_id', values='yield')
    bits = np.asarray([pair.split(' >> ') for pair in matrix.index])
    er = int(ANATOMY.electrophile_role('Suzuki|1')) - 1
    prepared = prepare(matrix.to_numpy(float), bits[:, er], bits[:, 1 - er], ratio=True)
    point, records = consistency_point(prepared)
    return dict(screen='Suzuki|1', point=point, pairs=len(matrix),
        halides=len(prepared['halides']), partners=len(prepared['partners']), conditions=matrix.shape[1],
        observed_cells=int(np.isfinite(prepared['matrix']).sum()), descriptive_shares=descriptive_shares(prepared)), records


def calibration_training(package):
    data = pd.read_csv(package / 'derived/DATASET_analysis.csv.gz',
                       usecols=['screen', 'subpair', 'condition_id', 'yield'])
    screen, condition = sys.modules['common'].DUPLICATE
    return data[~(data.screen.eq(screen) & data.condition_id.eq(condition))]


def calibration_donors(package):
    data = calibration_training(package)
    donors = []
    for key in sorted(ANATOMY.ROLE):
        role = ANATOMY.electrophile_role(key)
        if role is None:
            continue
        matrix = data[data.screen == key].pivot(index='subpair', columns='condition_id', values='yield')
        bits = np.asarray([pair.split(' >> ') for pair in matrix.index])
        a, b = bits[:, int(role) - 1], bits[:, 2 - int(role)]
        M = matrix.to_numpy(float)
        E, _, fit = ANATOMY.pattern(M)
        ua, ia = np.unique(a, return_inverse=True)
        ub, ib = np.unique(b, return_inverse=True)
        U, W = ANATOMY.joint_profiles(E, ia, ib, len(ua), len(ub), np.arange(len(M)), rounds=10)
        donors.append(dict(key=key, U=U, W=W, fit=fit, Z=E - U[ia] - W[ib]))
    if len(donors) != 10:
        raise ValueError('Frozen calibration requires exactly ten halide training screens')
    return donors


def generate_calibration_sample(donor, replicate, *, ratio):
    seed = 20261028 if ratio else 20261027
    sequence = [seed, 1, replicate] if ratio else [seed, 1, 0, replicate]
    rng = np.random.default_rng(np.random.SeedSequence(sequence))
    nh, npart = (12, 10) if ratio else (8, 8)
    conditions = (np.sort(rng.choice(donor['fit'].shape[1], 12, replace=False)) if ratio
                  else np.arange(donor['fit'].shape[1]))
    h = rng.integers(0, len(donor['U']), nh)
    p = rng.integers(0, len(donor['W']), npart)
    source = rng.integers(0, len(donor['fit']), nh * npart)
    a, b = np.repeat(np.arange(nh), npart), np.tile(np.arange(npart), nh)
    if ratio:
        M = np.clip(donor['fit'][source][:, conditions] + donor['Z'][source][:, conditions]
                    + donor['U'][h[a]][:, conditions] + donor['W'][p[b]][:, conditions], 0, 1)
    else:
        M = np.clip(donor['fit'][source] + donor['Z'][source] + donor['U'][h][a] + donor['W'][p][b], 0, 1)
    sequence = [seed, 2, replicate] if ratio else [seed, 2, 0, replicate]
    bootstrap_seed = int(np.random.SeedSequence(sequence).generate_state(1)[0])
    return dict(M=M, a=a if ratio else a.astype(str), b=b if ratio else b.astype(str),
                conditions=conditions.tolist(), halide_profiles=h.tolist(), partner_profiles=p.tolist(),
                source_pair_blocks=source.tolist(), bootstrap_seed=bootstrap_seed)


def calibration_summary(rows):
    successes, total = sum(row['success'] for row in rows), len(rows)
    rate, z = successes / total, 1.959963984540054
    center = (rate + z*z/(2*total))/(1+z*z/total)
    half = z*np.sqrt(rate*(1-rate)/total + z*z/(4*total*total))/(1+z*z/total)
    return dict(simulations=total, successes=successes, conditional_detection_frequency=rate,
        monte_carlo_wilson95=[center-half, center+half],
        monte_carlo_standard_error=float(np.sqrt(rate*(1-rate)/total)), estimate_only=bool(rate < .8),
        reasons=dict(Counter(row['reason'] for row in rows)))


def calibrate(package, output):
    output = validate_output(package, output)
    output.mkdir(parents=True, exist_ok=True)
    load_core(package, output)
    _, noise_scale = public_noise_scale(package)
    donors = calibration_donors(package)
    result = dict(status='COMPLETE', classification='reproduction of frozen training-only calibration',
                  external_outcomes_read=False, simulations_per_calibration=500, bootstrap_draws=2000,
                  criteria_are_distinct=True, datasets={})
    started = time.monotonic()
    for key, ratio in [('CH_arylation', False), ('Suzuki', True)]:
        rows = []
        for replicate in range(500):
            donor = next(d for d in donors if d['key'] == 'Suzuki|1') if ratio else donors[replicate % len(donors)]
            sample = generate_calibration_sample(donor, replicate, ratio=ratio)
            prepared = prepare(sample['M'], sample['a'], sample['b'], ratio=ratio)
            if ratio:
                point, _ = consistency_point(prepared)
                interval, reason = None, 'undefined_point'
                if point['difference'] is not None:
                    _, interval = consistency_bootstrap(prepared, seed=sample['bootstrap_seed'])
                    reason = 'estimable'
                success = bool(point['difference'] is not None and point['difference'] > 0
                               and interval is not None and interval['difference_ci95'][0] > 0)
                row = dict(replicate=replicate, success=success, reason=reason, point=point, bootstrap=interval)
            else:
                point, _ = ch_estimates(prepared, noise_scale)
                try:
                    _, interval = share_bootstrap(prepared, seed=sample['bootstrap_seed'])
                    fields = ['halide_share', 'partner_share', 'halide_consistency', 'partner_consistency']
                    estimable = all(point[field] is not None for field in fields)
                    directions = (estimable and point['halide_share'] > point['partner_share']
                                  and point['halide_consistency'] > point['partner_consistency'])
                    success = bool(directions and interval['difference_ci95'][0] > 0)
                    reason = 'estimable' if estimable else 'undefined_point_estimate_or_consistency'
                except ValueError as error:
                    if not str(error).startswith(('No productive pairs', 'Finite share draw cap', 'Zero numerical')):
                        raise
                    interval, success = {}, False
                    reason = ('Could not obtain exactly the prespecified finite crossed draws'
                              if str(error).startswith('Finite share draw cap') else str(error))
                row = dict(study=key, simulation=replicate, donor=donor['key'], success=success,
                           reason=reason, productive_pairs=point['productive_pairs'], **interval)
            row.update({field: value for field, value in sample.items() if field not in ('M', 'a', 'b')})
            rows.append(row)
            if replicate % 25 == 24:
                print(f'{key}: training calibration {replicate + 1}/500; elapsed={time.monotonic() - started:.1f}s', flush=True)
        summary = calibration_summary(rows)
        summary.update(seed=20261028 if ratio else 20261027,
            nonestimable=sum(row['reason'] != 'estimable' for row in rows),
            criterion=('positive consistency difference and consistency-difference interval lower bound > 0' if ratio
                       else 'halide share > partner share; halide consistency > partner consistency; share-difference interval lower bound > 0'),
            actual_external_power=False,
            model=('Suzuki-only empirical raw-yield fit/residual/profile generator' if ratio
                   else 'ten-screen empirical shrunken-profile generator; native donor conditions and masks'))
        result['datasets'][key] = summary
        save_json(output / f'{key}_calibration_simulations.json', rows)
    inputs = ['derived/DATASET_analysis.csv.gz', 'data/measurements_A1_analysis_arrays.csv.gz',
              'data/measurements_A3_hiyama_duplicate.csv.gz', 'code/Core/00_common/ctb/anatomy.py',
              'code/Core/00_common/ctb/history.py', 'code/Core/00_common/ctb/common.py']
    result['public_input_sha256'] = {name: hashlib.sha256((package / name).read_bytes()).hexdigest() for name in inputs}
    save_json(output / 'CALIBRATION_RESULT.json', result)
    return result


def validate_output(package, output):
    package, output = Path(package).resolve(), Path(output).resolve()
    if output.is_relative_to(package):
        raise ValueError('--output must be outside the public package')
    return output


def locate_package(value=None):
    if value:
        package = Path(value).resolve()
    else:
        base = Path(__file__).resolve().parents[2]
        package = base if (base / 'data').is_dir() else base / '10_PUBLIC_RELEASE_CANDIDATE'
    if not (package / 'code/Core/00_common/ctb/anatomy.py').is_file():
        raise FileNotFoundError('Supply --package pointing to the released P2 package')
    return package


def load_core(package, output):
    global ANATOMY
    os.environ['NATURE_OUT'] = str(output / 'runtime')
    os.environ['MPLCONFIGDIR'] = str(output / 'runtime/mpl')
    sys.path.insert(0, str(package / 'code/Core/00_common/ctb'))
    spec = importlib.util.spec_from_file_location('anatomy', package / 'code/Core/00_common/ctb/anatomy.py')
    ANATOMY = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ANATOMY)


def write_draws(output, filename, draws):
    columns = {key: value for key, value in draws.items() if value.ndim == 1}
    for role in ('halide', 'partner'):
        counts = draws[role + '_counts']
        columns.update({f'{role}_count_{i}': counts[:, i] for i in range(counts.shape[1])})
    pd.DataFrame(columns).to_csv(output / filename, index=False)


def reproduce(package, output, input_dir=None):
    output = validate_output(package, output)
    output.mkdir(parents=True, exist_ok=True)
    load_core(package, output)
    empirical, noise = public_noise_scale(package)
    results = dict(status='COMPLETE', utc=datetime.now(timezone.utc).isoformat(),
        noise_calibration=dict(empirical_reliability=empirical, noise_scale=noise), datasets={})
    manifest, molecule_rows = {}, []
    for key in DATASETS:
        path, manifest[key] = download_source(key, output, input_dir)
        frame, structure = parse_source(path, key)
        print(f'{key}: checksum verified; {len(frame)} source records parsed', flush=True)
        frame.to_csv(output / f'{key}_parsed.csv', index=False)
        if key == 'CH_arylation':
            result = dict(structure=structure)
            for variant in ('main', 'sensitivity'):
                current = frame.copy()
                excluded = current.measurements_below0.gt(0) | current.measurements_above105.gt(0)
                if variant == 'sensitivity':
                    current.loc[excluded, 'value'] = np.nan
                M, a, b = matrix_from_frame(current)
                prepared = prepare(M, a, b, ratio=False)
                point, molecules = ch_estimates(prepared, noise)
                share_draws, share_interval = share_bootstrap(prepared,
                    seed=int(np.random.SeedSequence([20261027, 3, 0]).generate_state(1)[0]))
                draws, interval = consistency_bootstrap(prepared, seed=20261029)
                result[variant] = dict(estimates=point, interval=share_interval,
                    consistency_interval=interval,
                    consistency_interval_status='post-exposure supplemental interval; not the original decision criterion',
                    source_records=len(current), numeric_source_records=int(current.value.notna().sum()),
                    sensitivity_excluded_records=int(excluded.sum()) if variant == 'sensitivity' else 0)
                molecule_rows.extend(dict(dataset=key, variant=variant, **row) for row in molecules)
                share_draws.to_csv(output / f'CH_{variant}_share_draws.csv', index=False)
                write_draws(output, f'CH_{variant}_consistency_draws.csv', draws)
            numeric = frame.value.dropna()
            result['yield_range'] = dict(selected_quantitative_yield_measurements=int(frame.yield_measurements.sum()),
                measurements_below0=int(frame.measurements_below0.sum()),
                measurements_above105=int(frame.measurements_above105.sum()),
                measurements_100_to105=int(frame.measurements_100_to105.sum()),
                minimum_measurement_percentage=float(frame.minimum_measurement.min() * 100),
                maximum_measurement_percentage=float(frame.maximum_measurement.max() * 100),
                minimum_source_record_mean_percentage=float(numeric.min() * 100),
                maximum_source_record_mean_percentage=float(numeric.max() * 100))
        else:
            M, a, b = matrix_from_frame(frame)
            prepared = prepare(M, a, b, ratio=True)
            point, molecules = consistency_point(prepared)
            draws, interval = consistency_bootstrap(prepared, seed=20261028)
            result = dict(structure=structure, endpoint='Pdt-to-IS Ratio',
                estimator='pair-standardized-input residual consistency; new definition',
                point=point, bootstrap=interval, descriptive_shares=descriptive_shares(prepared))
            result['training_benchmark'], benchmark_molecules = p2_suzuki_benchmark(package)
            pd.DataFrame(benchmark_molecules).to_csv(output / 'p2_suzuki_molecules.csv', index=False)
            molecule_rows.extend(dict(dataset=key, variant='ratio', **row) for row in molecules)
            write_draws(output, 'Suzuki_consistency_draws.csv', draws)
        results['datasets'][key] = result
        print(f'{key}: frozen estimators and 2000-draw intervals complete', flush=True)
    pd.DataFrame(molecule_rows).to_csv(output / 'molecules.csv', index=False)
    versions = {}
    for name in ('numpy', 'pandas', 'pyarrow', 'protobuf', 'ord-schema', 'rdkit'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = 'not available as installed distribution'
    save_json(output / 'source_manifest.json', dict(sources=manifest, environment=versions,
        ord_citation='Open Reaction Database Project. The Open Reaction Database. https://doi.org/10.5281/zenodo.4321713',
        ord_preferred_paper='Kearnes et al. J. Am. Chem. Soc. 143, 18820-18826 (2021). https://doi.org/10.1021/jacs.1c09820',
        license='CC-BY-SA-4.0', license_url='https://github.com/open-reaction-database/ord-data/blob/main/LICENSE',
        citation_url='https://github.com/open-reaction-database/ord-data/blob/main/CITATION.cff',
        modifications='Original ORD records parsed, repeated source wells averaged within observed cells; derived outputs generated locally.'))
    save_json(output / 'RESULT.json', results)
    return results


class SelfTests(unittest.TestCase):
    def test_public_contract(self):
        for name in ('verify_bytes', 'extract_ratio', 'extract_yield', 'matrix_from_frame',
                     'prepare', 'consistency_bootstrap'):
            self.assertIn(name, globals(), 'Missing public reproduction operation: ' + name)

    def test_checksum_failure(self):
        payload = b'ORD source bytes'
        digest = hashlib.sha256(payload).hexdigest()
        self.assertEqual(verify_bytes(payload, digest), digest)
        with self.assertRaises(ValueError):
            verify_bytes(payload + b'changed', digest)

    def test_calibration_design_and_frozen_seeds(self):
        donor = dict(U=np.zeros((2, 20)), W=np.zeros((3, 20)),
                     fit=np.full((4, 20), .5), Z=np.zeros((4, 20)))
        ch = generate_calibration_sample(donor, 0, ratio=False)
        ratio = generate_calibration_sample(donor, 0, ratio=True)
        self.assertEqual(ch['M'].shape, (64, 20))
        self.assertEqual(ratio['M'].shape, (120, 12))
        self.assertEqual(ch['bootstrap_seed'], 4073359316)
        self.assertEqual(ratio['bootstrap_seed'], 1525748996)
        self.assertEqual(len(set(zip(ch['a'], ch['b']))), 64)
        self.assertEqual(len(set(ratio['conditions'])), 12)

    def test_calibration_excludes_only_registered_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary)
            (package / 'derived').mkdir()
            frame = pd.DataFrame([('Hiyama|1', 'h >> p', 'C43', .5),
                                  ('Hiyama|1', 'h >> p', 'C46', .4),
                                  ('Suzuki|1', 'h >> p', 'C46', .3)],
                                 columns=['screen', 'subpair', 'condition_id', 'yield'])
            frame.to_csv(package / 'derived/DATASET_analysis.csv.gz', index=False)
            actual = calibration_training(package)
            self.assertEqual(actual.index.tolist(), [0, 2])

    def test_strict_ratio_and_zero(self):
        reaction = pb.Reaction()
        product = reaction.outcomes.add().products.add(is_desired_product=True)
        measurement = product.measurements.add(type=pb.ProductMeasurement.AREA, details=' Pdt-to-IS Ratio ')
        measurement.float_value.value = 0
        self.assertEqual(extract_ratio(reaction)[0], 0.)
        measurement.type = pb.ProductMeasurement.YIELD
        with self.assertRaises(ValueError):
            extract_ratio(reaction)
        measurement.type = pb.ProductMeasurement.AREA
        product.measurements.add().CopyFrom(measurement)
        with self.assertRaises(ValueError):
            extract_ratio(reaction)

    def test_yield_range_retained(self):
        reaction = pb.Reaction()
        product = reaction.outcomes.add().products.add(is_desired_product=True)
        product.measurements.add(type=pb.ProductMeasurement.YIELD).percentage.value = 110
        value, info = extract_yield(reaction)
        self.assertEqual(value, 1.1)
        self.assertEqual(info['measurements_above105'], 1)

    def test_unspecified_role_preserves_frozen_condition_identity(self):
        reaction = pb.Reaction()
        for name, text, role in [('halide', 'Brc1ccccc1', pb.ReactionRole.REACTANT),
                                 ('partner', 'OB(O)c1ccccc1', pb.ReactionRole.REACTANT),
                                 ('base', '[K+].[OH-]', pb.ReactionRole.UNSPECIFIED)]:
            compound = reaction.inputs[name].components.add(reaction_role=role)
            compound.identifiers.add(type=pb.CompoundIdentifier.SMILES, value=text)
        condition = json.loads(structural_identity(reaction)[2])
        self.assertEqual(condition, [[['None', '[K+].[OH-]']], None])

    def test_duplicates_and_missing_masks(self):
        frame = pd.DataFrame([
            ('h1', 'p1', 'c1', 0.), ('h1', 'p1', 'c1', 2.),
            ('h1', 'p2', 'c1', 0.), ('h1', 'p2', 'c2', 4.),
        ], columns=['halide', 'partner', 'condition_signature', 'value'])
        M, a, b = matrix_from_frame(frame)
        np.testing.assert_allclose(M, [[1., np.nan], [0., 4.]], equal_nan=True)
        np.testing.assert_array_equal(a, ['h1', 'h1'])
        np.testing.assert_array_equal(b, ['p1', 'p2'])

    def test_weighted_median_expanded_copies(self):
        scores = np.array([[.8, -.3, .2]])
        counts = np.array([[2, 1, 3]])
        actual = weighted_median(scores, counts, np.ones_like(counts, bool))[0]
        self.assertEqual(actual, np.median(np.repeat(scores[0], counts[0])))

    def test_ratio_scale_and_chunk_invariance(self):
        rng = np.random.default_rng(441)
        a, b = np.repeat(np.arange(6), 6), np.tile(np.arange(6), 6)
        M = rng.uniform(.01, 2., (36, 8))
        M[0, 0] = np.nan
        M[1, 1] = 0.
        original = prepare(M, a, b, ratio=True)
        rescaled = prepare(M * np.geomspace(.001, 1000, len(M))[:, None], a, b, ratio=True)
        np.testing.assert_allclose(original['residuals'], rescaled['residuals'], atol=1e-12)
        first, summary1 = consistency_bootstrap(original, seed=1234, draws=40, chunk_size=7)
        second, summary2 = consistency_bootstrap(original, seed=1234, draws=40, chunk_size=53)
        self.assertEqual(summary1, summary2)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])

    def test_unique_original_partner_eligibility(self):
        rng = np.random.default_rng(441)
        a, b = np.repeat(np.arange(5), 5), np.tile(np.arange(5), 5)
        prepared = prepare(rng.uniform(.1, 2., (25, 7)), a, b, ratio=True)
        counts = np.array([[2, 2, 1, 0, 0]])
        _, _, eligible = _role_summary(prepared['halide_graph'], np.ones((1, 5), int), counts)
        self.assertFalse(eligible.any())

    def test_output_outside_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / 'package'
            with self.assertRaises(ValueError):
                validate_output(package, package / 'derived')


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--calibrate', action='store_true', help='Reproduce both frozen 500-simulation training-only calibrations; reads no ORD outcomes')
    parser.add_argument('--package', type=Path, help='Released P2 package root')
    parser.add_argument('--output', type=Path, help='User output directory outside the package')
    parser.add_argument('--input-dir', type=Path, help='Optional directory of the two pinned ORD parquets, for offline use')
    args = parser.parse_args()
    if args.self_test:
        with tempfile.TemporaryDirectory(prefix='p2_external_selftest_') as temporary:
            load_core(locate_package(args.package), Path(temporary))
            result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SelfTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
    if args.output is None:
        parser.error('--output is required')
    if args.calibrate:
        calibrate(locate_package(args.package), args.output.resolve())
    else:
        reproduce(locate_package(args.package), args.output.resolve(), args.input_dir)


if __name__ == '__main__':
    main()
