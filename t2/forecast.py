"""## Executive summary (read this first)
Produce CPU-only joint forecasts from the supplied, cutoff-limited public inputs.
The candidate uses a stationary block bootstrap for daily series and a filtered,
shrunk autoregressive bootstrap for monthly growth. No weights or network are used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import tomllib
from typing import Any

import numpy as np
import pandas as pd

from qfbench2_track_forecasting.horizons import monthly_horizon_steps
from qfbench2_track_forecasting.targets import log_return_steps
from qfbench2_track_forecasting.limits import ParseLimits
from qfbench2_common.taskcard import load_schema
import jsonschema


def read_histories(panels_dir: Path, assets: list[str], asof: str) -> dict[str, pd.Series]:
    """Read only panel files and refuse absent or ambiguous target histories."""
    paths = sorted(panels_dir.glob('*.parquet'))
    if not paths:
        paths = sorted((panels_dir / 'panels').glob('*.parquet'))
    if not paths:
        paths = sorted(panels_dir.parent.glob('*.parquet'))
    histories = {}
    for path in paths:
        frame = pd.read_parquet(path)
        asset_col = next((c for c in ('asset', 'asset_id') if c in frame), None)
        if asset_col is None or not {'date', 'value'} <= set(frame):
            continue
        for asset in assets:
            if asset in histories:
                continue
            part = frame.loc[frame[asset_col].astype(str) == asset, ['date', 'value']].copy()
            part['date'] = pd.to_datetime(part['date'], errors='raise').dt.tz_localize(None)
            part = part[part['date'] <= pd.Timestamp(asof)].sort_values('date')
            part = part.dropna(subset=['value'])
            if part.empty:
                continue
            if part['date'].duplicated().any() or not np.isfinite(part['value']).all():
                raise ValueError(f'Invalid or duplicate panel observations for {asset}')
            histories[asset] = part.set_index('date')['value'].astype(float)
    missing = set(assets) - set(histories)
    if missing:
        raise ValueError(f'Target assets absent before cutoff: {sorted(missing)}')
    return {asset: histories[asset] for asset in assets}


def step_frame(histories: dict[str, pd.Series], target: str, monthly: bool) -> pd.DataFrame:
    """Align changes on dates, removing changes across missing observation windows."""
    steps = {}
    for asset, original in histories.items():
        s = original.copy()
        if monthly:
            s.index = s.index.to_period('M')
            if s.index.has_duplicates:
                raise ValueError(f'Duplicate monthly observations for {asset}')
            gaps = np.r_[np.nan, np.diff(s.index.asi8)]
            valid = gaps == 1
        else:
            gaps = np.r_[np.nan, np.diff(s.index.to_numpy()).astype('timedelta64[D]').astype(float)]
            valid = gaps <= max(5.0, 10.0 * np.nanmedian(gaps))
        values = pd.Series(log_return_steps(s), index=s.index) if target == 'log_return' else s.diff()
        steps[asset] = values.where(valid)
    frame = pd.DataFrame(steps).dropna()
    if len(frame) < 20:
        raise ValueError(f'Only {len(frame)} date-aligned valid historical steps')
    return frame


def monthly_grid(histories: dict[str, pd.Series], card: dict, spec: dict, asof: str) -> np.ndarray | None:
    targets = card['targets']
    declared = targets.get('target_frequency', card.get('metadata', {}).get('target_frequency'))
    if declared != 'monthly':
        return None
    cadences = [np.median(np.diff(s.index.to_numpy()).astype('timedelta64[D]').astype(float)) for s in histories.values()]
    # Legacy exemplar describes a monthly horizon over genuinely daily observations.
    explicit = any('observation_periods' in src.get('targets', {}) or
                   any('observation_period' in q for q in src.get('questions', []))
                   for src in (card, spec))
    if all(0 < cadence <= 3 for cadence in cadences) and not explicit:
        return None
    if not all(20 <= cadence <= 35 for cadence in cadences):
        raise ValueError('Declared monthly targets do not have unambiguous monthly cadence')
    return monthly_horizon_steps(list(histories), targets['horizons'],
        {a: s.index[-1].strftime('%Y-%m-%d') for a, s in histories.items()},
        asof=asof, card=card, forecast_spec=spec)


def text_risk(text_dir: Path, asof: str) -> dict[str, Any]:
    """Treat dated files as data; bounded phrase matches can widen but never tighten."""
    result = {'width_multiplier': 1.0, 'documents_read': 0, 'evidence': [], 'excluded_after_cutoff': 0}
    index_path = text_dir / 'corpus_index.json'
    if not index_path.is_file():
        result['note'] = 'No dated corpus index; unindexed documents were not used.'
        return result
    index = json.loads(index_path.read_text(encoding='utf-8'))
    patterns = (r'financial (?:market )?(?:stress|instability|turmoil)', r'banking (?:stress|crisis)',
                r'heightened uncertainty', r'exceptionally uncertain', r'severe (?:downturn|disruption)',
                r'uncertainty (?:is|remains) (?:high|elevated)', r'pandemic')
    risks = []
    for doc in index.get('documents', []):
        date = str(doc.get('timestamp', ''))[:10]
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
            continue
        age = (pd.Timestamp(asof) - pd.Timestamp(date)).days
        if age < 0:
            result['excluded_after_cutoff'] += 1
            continue
        if age > 120:
            continue
        filename = doc.get('file', '')
        path = (text_dir / filename).resolve()
        if not path.is_relative_to(text_dir.resolve()) or not path.is_file():
            continue
        text = path.read_text(encoding='utf-8', errors='replace')[:250_000].lower()
        result['documents_read'] += 1
        matches = sorted({match.group(0) for pattern in patterns for match in re.finditer(pattern, text)})
        if matches:
            risks.append(float(np.exp(-age / 60.0)))
            result['evidence'].append({'doc_id': doc.get('doc_id', filename), 'date': date,
                'matched_phrases': matches, 'status': 'inferred uncertainty; not an outcome or directional claim'})
    # A single document has bounded influence; repeated copies do not accumulate it.
    if risks:
        result['width_multiplier'] = 1.0 + 0.05 * max(risks)
    return result


def _drift(values: np.ndarray, monthly: bool, target: str) -> np.ndarray:
    """Monthly local growth; daily mean heavily shrunk to avoid noisy trends."""
    if monthly:
        recent = values[-24:]
        long = values[-120:]
        return 0.65 * recent.mean(axis=0) + 0.35 * long.mean(axis=0)
    if target == 'log_return':
        return 0.5 * values[-1260:].mean(axis=0)
    return np.zeros(values.shape[1])


def _simulate_monthly(histories: dict[str, pd.Series], horizons: list[int], target: str,
                      n_draws: int, seed: int, panel_steps: np.ndarray,
                      width_multiplier: float) -> tuple[np.ndarray, dict]:
    """Joint filtered shocks and regularized monthly growth persistence.

    Filtered means that a historical shock is divided by its prior volatility.
    Autoregression uses the latest growth deviation to predict the next one; a
    24-observation zero-persistence prior and bounded coefficient reduce overfit.
    """
    assets = list(histories)
    frame = step_frame(histories, target, True)
    values = frame.to_numpy()[-240:]
    n_assets = len(assets)
    anchors = np.zeros(n_assets) if target == 'log_return' else np.array([s.iloc[-1] for s in histories.values()])
    steps = np.asarray(panel_steps, dtype=int)
    origins = np.array([s.index[-1].to_period('M').ordinal for s in histories.values()])
    offset = origins - origins.min()
    endpoints = steps + offset[:, None]
    maximum = int(endpoints.max())
    if maximum < 1 or maximum > 5000:
        raise ValueError('Forecast exceeds supported positive sampling horizon')
    drift = _drift(values, True, target)
    recent = values[-60:]
    x, y = recent[:-1] - drift, recent[1:] - drift
    denominator = np.sum(x * x, axis=0) + 24 * np.var(recent, axis=0)
    phi = np.divide(np.sum(x * y, axis=0), denominator,
                    out=np.zeros(n_assets), where=denominator > 1e-24)
    phi = np.clip(phi, -0.35, 0.70)
    residuals = values[1:] - drift - phi * (values[:-1] - drift)
    residuals -= residuals.mean(axis=0)
    long_var = residuals[-60:].var(axis=0, ddof=1)
    burn = min(24, len(residuals) // 3)
    initial = residuals[:burn].var(axis=0, ddof=1)
    floor = np.maximum(initial * 0.05, 1e-24)
    variance = np.maximum(initial, floor)
    decay = np.exp2(-1.0 / 12)
    standardized = []
    for i, residual in enumerate(residuals):
        if i >= burn:
            standardized.append(residual / np.sqrt(np.maximum(variance, floor)))
        variance = decay * variance + (1 - decay) * residual ** 2
    standardized = np.asarray(standardized)
    standardized -= standardized.mean(axis=0)
    empirical_sd = standardized.std(axis=0, ddof=1)
    standardized = np.divide(standardized, empirical_sd, out=np.zeros_like(standardized), where=empirical_sd > 1e-12)
    current_var = 0.75 * variance + 0.25 * long_var
    rng = np.random.default_rng(seed)
    # Keep the frozen experiment's exact random stream, including initialization.
    indices = rng.integers(0, len(standardized), size=n_draws)
    previous = np.tile(values[-1] - drift, (n_draws, 1))
    result = np.empty((n_draws, n_assets, len(horizons)))
    path = np.zeros((n_draws, n_assets))
    for step in range(1, maximum + 1):
        restart = rng.random(n_draws) < 1.0
        indices = np.where(restart, rng.integers(0, len(standardized), size=n_draws), (indices + 1) % len(standardized))
        persistence = np.exp2(-(step - 1) / 12)
        scale = np.sqrt(persistence * current_var + (1 - persistence) * long_var)
        innovation = standardized[indices] * scale * width_multiplier
        change = drift + phi * previous + innovation
        previous = np.where(step > offset, change - drift, previous)
        path += change * (step > offset)
        for ai, hi in np.argwhere(endpoints == step):
            result[:, ai, hi] = anchors[ai] + path[:, ai]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite forecast draws')
    return result, {'method': 'filtered_ar', 'step_unit': 'month',
        'n_aligned_steps': len(frame), 'n_estimation_steps': len(values),
        'anchor': dict(zip(assets, anchors.tolist())),
        'drift_per_step': dict(zip(assets, drift.tolist())),
        'growth_autoregression': dict(zip(assets, phi.tolist())),
        'innovation_scale': dict(zip(assets, np.sqrt(current_var).tolist())),
        'sampling_steps': {a: steps[i].tolist() for i, a in enumerate(assets)},
        'width_multiplier': width_multiplier}


def simulate(histories: dict[str, pd.Series], horizons: list[int], target: str,
             n_draws: int = 2000, seed: int = 773, panel_steps: np.ndarray | None = None,
             method: str = 'adaptive', width_multiplier: float = 1.0) -> tuple[np.ndarray, dict]:
    """Simulate one path per joint draw, reusing its earlier innovations at later horizons."""
    monthly = panel_steps is not None
    if monthly and method == 'adaptive':
        return _simulate_monthly(histories, horizons, target, n_draws, seed,
                                 panel_steps, width_multiplier)
    assets = list(histories)
    frame = step_frame(histories, target, monthly)
    values = frame.to_numpy()
    n_assets = len(assets)
    anchors = np.zeros(n_assets) if target == 'log_return' else np.array([s.iloc[-1] for s in histories.values()])
    steps = np.tile(horizons, (n_assets, 1)) if panel_steps is None else np.asarray(panel_steps, dtype=int)
    if monthly:
        origins = np.array([s.index[-1].to_period('M').ordinal for s in histories.values()])
        offset = origins - origins.min()
    else:
        offset = np.zeros(n_assets, dtype=int)
    endpoints = steps + offset[:, None]
    maximum = int(endpoints.max())
    if maximum < 1 or maximum > 5000:
        raise ValueError('Forecast exceeds supported positive sampling horizon')
    rng = np.random.default_rng(seed)
    result = np.empty((n_draws, n_assets, len(horizons)))
    path = np.zeros((n_draws, n_assets))
    if method == 'baseline':
        sample = values[-299:]
        drift = sample.mean(axis=0)
        cov = np.atleast_2d(np.cov(sample, rowvar=False))
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        root = eigenvectors @ np.diag(np.sqrt(np.clip(eigenvalues, 0, None)))
        scale = np.sqrt(np.diag(cov))
    else:
        limit = 240 if monthly else 2520
        sample = values[-limit:]
        drift = _drift(sample, monthly, target)
        # Remove the sample mean so the bootstrap does not add an unintended second drift.
        residuals = sample - sample.mean(axis=0)
        half_life = 12 if monthly else 63
        weights = np.exp2(-np.arange(len(sample) - 1, -1, -1) / half_life)
        weights /= weights.sum()
        recent_var = np.sum(weights[:, None] * residuals ** 2, axis=0)
        long_var = residuals[-(60 if monthly else 504):].var(axis=0, ddof=1)
        scale = np.sqrt(0.65 * recent_var + 0.35 * long_var)
        historical_scale = residuals.std(axis=0, ddof=1)
        multiplier = np.divide(scale, historical_scale, out=np.ones(n_assets), where=historical_scale > 0)
        residuals *= np.clip(multiplier, 0.45, 2.5) * width_multiplier
        # Shared block indices preserve simultaneous cross-asset shocks and serial dependence.
        indices = rng.integers(0, len(sample), size=n_draws)
        block_length = 3.0 if monthly else 5.0
    for step in range(1, maximum + 1):
        if method == 'baseline':
            innovation = rng.standard_normal((n_draws, n_assets)) @ root.T
        else:
            restart = rng.random(n_draws) < (1.0 / block_length)
            indices = np.where(restart, rng.integers(0, len(sample), size=n_draws), (indices + 1) % len(sample))
            innovation = residuals[indices]
        path += (innovation + drift) * (step > offset)
        for ai, hi in np.argwhere(endpoints == step):
            result[:, ai, hi] = anchors[ai] + path[:, ai]
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite forecast draws')
    stats = {'method': method, 'step_unit': 'month' if monthly else 'business observation',
             'n_aligned_steps': len(frame), 'n_estimation_steps': len(sample),
             'anchor': dict(zip(assets, anchors.tolist())), 'drift_per_step': dict(zip(assets, drift.tolist())),
             'innovation_scale': dict(zip(assets, scale.tolist())),
             'sampling_steps': {a: steps[i].tolist() for i, a in enumerate(assets)},
             'width_multiplier': width_multiplier}
    return result, stats


def run(panels: Path, text: Path, asof: str, out: Path, card_path: Path | None = None,
        n_draws: int = 2000, seed: int = 773, method: str = 'adaptive', use_text: bool = False) -> dict:
    if card_path is None:
        card_path = next((p for p in (panels / 'card.toml', panels.parent / 'card.toml') if p.is_file()), None)
    if card_path is None:
        raise ValueError('card.toml is required beside panels or in their parent')
    card = tomllib.loads(card_path.read_text(encoding='utf-8'))
    if asof != card['provenance']['data_cutoff']:
        raise ValueError('--asof must equal the card data_cutoff')
    targets = card['targets']
    assets, horizons = list(targets['asset_ids']), list(targets['horizons'])
    target = targets['target_type']
    if target not in ('level', 'log_return'):
        raise ValueError(f'Unsupported target type: {target}')
    limits = ParseLimits()
    n_draws = max(n_draws, limits.min_draws, int(card.get('scoring', {}).get('params', {}).get('n_draws_min', 0)))
    if n_draws > limits.max_draws:
        raise ValueError('Too many draws for official contract')
    histories = read_histories(panels, assets, asof)
    spec_path = card_path.parent / 'forecast_spec.json'
    spec = json.loads(spec_path.read_text(encoding='utf-8')) if spec_path.exists() else {}
    grid = monthly_grid(histories, card, spec, asof)
    evidence = text_risk(text, asof) if use_text else {'width_multiplier': 1.0, 'documents_read': 0, 'evidence': []}
    # Unit-specific stable seed is reproducible and has no knowledge of outcomes.
    salt = int.from_bytes(hashlib.sha256(card['task']['id'].encode()).digest()[:4], 'big')
    samples, stats = simulate(histories, horizons, target, n_draws, (seed + salt) % 2**32,
                              grid, method, evidence['width_multiplier'])
    method_name = ('CPU joint Gaussian walk on 299 aligned historical steps' if method == 'baseline'
                   else 'CPU filtered autoregressive monthly bootstrap' if grid is not None
                   else 'CPU stationary block bootstrap with adaptive volatility')
    method_name += '; experimental dated text risk enabled' if use_text else '; text-blind'
    metadata = {'unit_id': card['task']['id'], 'asof': asof, 'representation': 'samples',
                'asset_ids': assets, 'horizons': horizons, 'n_draws': n_draws, 'target': target,
                'rationale': {'file': 'forecast_rationale.md',
                              'method': method_name}}
    jsonschema.validate(metadata, load_schema('forecast.schema.json'))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'draw': np.repeat(np.arange(n_draws), len(assets) * len(horizons)),
                  'asset': np.tile(np.repeat(assets, len(horizons)), n_draws),
                  'horizon': np.tile(horizons, n_draws * len(assets)),
                  'value': samples.reshape(-1)}).to_parquet(out, index=False)
    (out.parent / 'forecast_meta.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    method_prose = (
        'A Gaussian walk uses the sample mean and covariance of the last 299 date-aligned '
        'historical steps. This participant comparison baseline is not the official M0 generator. '
        'Each draw accumulates shared cross-asset shocks along one path.\n\n'
        if method == 'baseline' else
        'A stationary block bootstrap resamples consecutive historical innovations. Each draw shares '
        'the same dated shock vector across assets, and later horizons accumulate the earlier path. '
        'The sample mean is removed before drift is added. Volatility combines recent exponentially '
        'weighted changes and a longer window. Daily level drift is zero. Daily return drift is a '
        'shrunk historical log-return mean. Monthly drift combines two historical growth windows. '
        'Monthly forecasts fit a bounded, ridge-shrunk autoregression to growth deviations, '
        'resample joint volatility-standardized residuals, and let current variance decay '
        'toward a longer-window estimate. Monthly residuals are sampled independently in '
        'time because growth persistence is modeled explicitly.\n\n')
    rationale = ('## Executive summary (read this first)\n\n'
        f"Forecast for {card['task']['id']} using only observations at or before {asof}. "
        'CPU only; no pretrained model, external API, network, retrieved outcomes, or stored answers.\n\n'
        + method_prose +
        'Monthly horizons use the official observation-period '
        'resolver and include publication lag; the original horizon keys are preserved.\n\n'
        'Drift and volatility are statistical inferences, '
        'not established facts about the future.\n\n'
        'Dated text matches can increase innovation width by at most five percent. They cannot '
        'shift the centre or tighten uncertainty. Phrase matches are weak inferred uncertainty '
        'signals; no text-performance advantage has been established. All listed documents are '
        'at or before the cutoff. Instructions inside documents are ignored.\n\n'
        '### Statistical ledger\n\n```json\n' + json.dumps(stats, indent=2) + '\n```\n\n'
        '### Text evidence\n\n```json\n' + json.dumps(evidence, indent=2) + '\n```\n')
    (out.parent / 'forecast_rationale.md').write_text(rationale, encoding='utf-8')
    return {'unit_id': card['task']['id'], 'rows': int(samples.size), 'draws': n_draws, 'stats': stats}


def main() -> int:
    parser = argparse.ArgumentParser(description='Eastern Eagles CPU forecasting candidate')
    parser.add_argument('--panels', type=Path, required=True)
    parser.add_argument('--text', type=Path, required=True)
    parser.add_argument('--asof', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--card', type=Path)
    parser.add_argument('--n-draws', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=773)
    parser.add_argument('--method', choices=['adaptive', 'baseline'], default='adaptive')
    parser.add_argument('--text-risk', action='store_true', help='Experimental dated text overlay; disabled until independently validated')
    args = parser.parse_args()
    result = run(args.panels, args.text, args.asof, args.out, args.card, args.n_draws,
                 args.seed, args.method, args.text_risk)
    print(json.dumps(result))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
