"""## Executive summary (read this first)
Check every public output with official gates and compare algorithms on honest
rolling cutoffs inside historical panels. These are not official hidden outcomes,
reference normalizers, leaderboard results, or evidence of world ranking.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import tomllib

import numpy as np
import pandas as pd

from forecast import read_histories, monthly_grid, run, simulate
from qfbench2_common.manifest import verify_manifest
from qfbench2_common.taskcard import load_and_validate
from qfbench2_track_forecasting.scoring import GATES, hydrate_ctx, _composite, _cmd_score
from qfbench2_track_forecasting.targets import log_return_steps
from qfbench2_track_forecasting.grid import grid_from_card, check_declared_grid, build_sample_matrix
from qfbench2_track_forecasting.limits import ParseLimits, inspect_parquet, rationale_has_content
from qfbench2_track_forecasting.cutoff import bind_metadata
from qfbench2_common.taskcard import load_schema
import jsonschema


def scorer_gates(units: Path, output: Path) -> dict:
    """Run the unmodified official gates-only CLI; staging validation is separate."""
    results = []
    for unit in sorted(units.iterdir()):
        if not (unit / 'card.toml').is_file():
            continue
        captured = io.StringIO()
        try:
            with redirect_stdout(captured):
                code = _cmd_score(argparse.Namespace(card=unit / 'card.toml',
                    forecast=output / unit.name / 'forecast.parquet', realized=None))
            payload = json.loads(captured.getvalue())
            payload['exit_code'] = code
            results.append(payload)
        except Exception as exc:
            results.append({'card_id': unit.name, 'admissible': False,
                            'error': f'{type(exc).__name__}: {exc}'})
    return {'mode': 'official_gates_only_cli', 'rankable': False, 'scored': False,
            'note': 'Unmodified official score CLI without --realized. Separate input manifest/staging traversal remains blocked on Windows.',
            'count': len(results), 'passed': sum(r.get('admissible', False) for r in results), 'results': results}


def check_outputs(units: Path, output: Path) -> dict:
    """Use official output parsers without pretending this replaces input security gates."""
    results = []
    for unit in sorted(units.iterdir()):
        if not (unit / 'card.toml').is_file():
            continue
        try:
            card = tomllib.loads((unit / 'card.toml').read_text(encoding='utf-8'))
            directory = output / unit.name
            meta = json.loads((directory / 'forecast_meta.json').read_text(encoding='utf-8'))
            jsonschema.validate(meta, load_schema('forecast.schema.json'))
            bind_metadata(meta, card, unit_handle=unit.name)
            spec = grid_from_card(card)
            check_declared_grid(meta, spec)
            limits = ParseLimits()
            path = directory / 'forecast.parquet'
            facts = inspect_parquet(path, what='forecast.parquet', limits=limits)
            matrix = build_sample_matrix(path, facts, spec, meta['n_draws'], limits=limits)
            if not rationale_has_content(directory / 'forecast_rationale.md', limits=limits):
                raise ValueError('Missing rationale')
            results.append({'unit_id': unit.name, 'passed': True, 'draws': len(matrix), 'cells': matrix.shape[1]})
        except Exception as exc:
            results.append({'unit_id': unit.name, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'})
    return {'mode': 'direct_official_output_parser_checks', 'rankable': False,
            'official_full_gate_pass': False,
            'note': 'Output schema/grid/cutoff binding/parser only. Does not replace manifest/input filesystem integrity gates.',
            'count': len(results), 'passed': sum(r['passed'] for r in results), 'results': results}


def gates(unit: Path, output: Path) -> dict:
    card, errors = load_and_validate(unit)
    errors += verify_manifest(unit)
    if errors:
        return {'passed': False, 'input_errors': errors}
    ctx = {'card': card, 'unit_dir': unit, 'output_dir': output, 'split': card['task']['split']}
    hydrate_ctx(ctx)
    results = {}
    for name, gate in GATES:
        result = gate(ctx)
        results[name] = {'passed': result.passed, 'detail': dict(result.detail)}
        if not result.passed:
            break
    return {'passed': len(results) == len(GATES) and all(r['passed'] for r in results.values()), 'gates': results}


def check_all(units: Path, output: Path, draws: int) -> dict:
    results = []
    for unit in sorted(units.iterdir()):
        if not (unit / 'card.toml').is_file():
            continue
        start = time.perf_counter()
        try:
            card = tomllib.loads((unit / 'card.toml').read_text(encoding='utf-8'))
            report = run(unit, unit / 'text', card['provenance']['data_cutoff'],
                         output / unit.name / 'forecast.parquet', n_draws=draws)
            report.update(gates(unit, output / unit.name))
        except Exception as exc:
            report = {'unit_id': unit.name, 'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
        report['seconds'] = time.perf_counter() - start
        results.append(report)
        print(json.dumps({'unit_id': unit.name, 'passed': report['passed'], 'seconds': round(report['seconds'], 2),
                          'error': report.get('error')}), flush=True)
    return {'mode': 'public_staging_plus_output_checks', 'rankable': False, 'scored': False,
            'note': 'Input loading and manifest traversal are attempted before scorer gates. On Windows those guards refuse; use scorer-gates for the separate official output-admissibility result.',
            'count': len(results), 'passed': sum(r['passed'] for r in results), 'results': results}


def score_draws(samples: np.ndarray, outcome: np.ndarray) -> dict:
    flat = samples.reshape(len(samples), -1)
    truth = outcome.reshape(-1)
    weights = (0.5, 0.3, 0.2) if len(truth) > 1 else (5/7, 0.0, 2/7)
    score = _composite(flat, truth, weights=weights, tail_levels=(0.01, 0.05, 0.95, 0.99),
                       joint='variogram', tail_metric='pinball', ref_scale=None)
    low, high = np.quantile(flat, [0.05, 0.95], axis=0)
    score['coverage_90'] = float(np.mean((truth >= low) & (truth <= high)))
    return score


def rolling(units: Path, draws: int, max_units: int = 0) -> dict:
    results, excluded = [], []
    # Disjoint final windows at each unit's own authored horizon. Evaluation labels
    # are existing panel observations strictly before that unit's actual cutoff.
    for unit in sorted(units.iterdir()):
        if not (unit / 'card.toml').is_file():
            continue
        if max_units and len({r['unit_id'] for r in results}) >= max_units:
            break
        try:
            card = tomllib.loads((unit / 'card.toml').read_text(encoding='utf-8'))
            assets, horizons = card['targets']['asset_ids'], card['targets']['horizons']
            target = card['targets']['target_type']
            asof = card['provenance']['data_cutoff']
            histories = read_histories(unit, assets, asof)
            spec_path = unit / 'forecast_spec.json'
            spec = json.loads(spec_path.read_text(encoding='utf-8')) if spec_path.exists() else {}
            monthly = monthly_grid(histories, card, spec, asof)
            if monthly is not None:
                h = [1, 3, 6]
            else:
                h = list(horizons)
            max_h = max(h)
            full = pd.DataFrame(histories).dropna()
            min_train = 60 if monthly is not None else 300
            for fold in range(1, 4):
                stop = len(full) - fold * max_h
                if stop < min_train:
                    continue
                future = full.iloc[stop:stop + max_h]
                # Reject synthetic evaluation segments spanning withheld transfer gaps.
                dates = full.index[stop-1:stop + max_h]
                gaps = np.diff(dates.to_numpy()).astype('timedelta64[D]').astype(float)
                if np.max(gaps) > (45 if monthly is not None else 10):
                    continue
                cutoff = full.index[stop-1]
                train = {a: s[s.index <= cutoff] for a, s in histories.items()}
                panel_steps = np.tile(h, (len(assets), 1)) if monthly is not None else None
                truth = np.empty((len(assets), len(h)))
                for ai, asset in enumerate(assets):
                    for hi, horizon in enumerate(h):
                        truth[ai, hi] = (float(log_return_steps(future[asset].iloc[:horizon]).sum())
                                         if target == 'log_return' else float(future[asset].iloc[horizon-1]))
                # Same source observations can occur in several cards; identify duplicate
                # evaluation problems so they do not silently multiply the evidence.
                key = hashlib.sha256(json.dumps({'assets': assets, 'horizons': h, 'cutoff': str(cutoff),
                     'target': target}, sort_keys=True).encode()).hexdigest()
                if any(r['evaluation_key'] == key for r in results):
                    continue
                seed = int(key[:8], 16)
                row = {'unit_id': unit.name, 'family': card['metadata']['category'],
                       'monthly': monthly is not None, 'target': target, 'fold': fold,
                       'cutoff': str(cutoff.date()), 'validation_end': str(future.index[max_h-1].date()),
                       'assets': assets, 'horizons': h, 'evaluation_key': key, 'methods': {}}
                for method in ('baseline', 'adaptive'):
                    samples, _ = simulate(train, h, target, draws, seed, panel_steps, method)
                    row['methods'][method] = score_draws(samples, truth)
                results.append(row)
            print(json.dumps({'unit_id': unit.name, 'rolling_rows_total': len(results)}), flush=True)
        except Exception as exc:
            excluded.append({'unit_id': unit.name, 'error': f'{type(exc).__name__}: {exc}'})
    summaries = {}
    groups = {'all': results, 'monthly': [r for r in results if r['monthly']],
              'daily_level': [r for r in results if not r['monthly'] and r['target'] == 'level'],
              'daily_return': [r for r in results if r['target'] == 'log_return']}
    for group, rows in groups.items():
        if not rows:
            continue
        # Paired relative raw components are descriptive only; they do not estimate
        # official M0-normalized scores. Avoid combining physical units across cards.
        summary = {'n_windows': len(rows), 'n_units': len({r['unit_id'] for r in rows})}
        for metric in ('marginal', 'tail', 'joint', 'composite'):
            ratios = [r['methods']['adaptive'][metric] / r['methods']['baseline'][metric]
                      for r in rows if r['methods']['baseline'][metric] > 1e-12]
            summary[metric + '_median_ratio'] = float(np.median(ratios)) if ratios else None
            summary[metric + '_win_fraction'] = float(np.mean(np.array(ratios) < 1)) if ratios else None
        for method in ('baseline', 'adaptive'):
            summary[method + '_coverage_90'] = float(np.mean([r['methods'][method]['coverage_90'] for r in rows]))
        summaries[group] = summary
    return {'mode': 'historical_rolling_diagnostic', 'rankable': False, 'hidden_outcomes_used': False,
            'baseline': 'Participant Gaussian joint walk: last299 aligned steps, sample mean/covariance; not official M0 generator',
            'metric': 'Imported official T2 raw composite (pinball tail); paired ratios, no sealed normalizers',
            'text_overlay': 'Disabled: present-day card corpora are not reused at earlier cutoffs',
            'limitations': ['Overlapping source histories across cards remain statistically dependent even after exact problem deduplication',
                            'Latest historical windows are development evidence, not a held-out competition estimate',
                            'Monthly diagnostics use 1/3/6 monthly steps, not authored horizons',
                            'Historical observations may be revised vintages; no first-release archive reconstructed'],
            'summaries': summaries, 'results': results, 'excluded': excluded}


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parent.parent
    parser.add_argument('--units', type=Path, default=root / 'upstream-t2' / 'units')
    parser.add_argument('--output', type=Path, default=root / 't2' / 'output')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--mode', choices=['gates', 'scorer-gates', 'rolling', 'outputs'], required=True)
    parser.add_argument('--n-draws', type=int, default=1000)
    parser.add_argument('--max-units', type=int, default=0)
    args = parser.parse_args()
    start = time.perf_counter()
    result = (check_all(args.units, args.output, args.n_draws) if args.mode == 'gates'
              else check_outputs(args.units, args.output) if args.mode == 'outputs'
              else scorer_gates(args.units, args.output) if args.mode == 'scorer-gates'
              else rolling(args.units, args.n_draws, args.max_units))
    result['timestamp_utc'] = datetime.now(timezone.utc).isoformat()
    result['elapsed_seconds'] = time.perf_counter() - start
    result['candidate_sha256'] = hashlib.sha256((Path(__file__).parent / 'forecast.py').read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('results', 'excluded')}), flush=True)


if __name__ == '__main__':
    main()
