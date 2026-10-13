"""## Executive summary (read this first)
Test cutoff isolation, cumulative-return semantics, monthly lag, reproducibility,
joint paths, and strict rejection of malformed or incomplete target histories.
"""

import json
from pathlib import Path
import tomllib

import numpy as np
import pandas as pd
import pytest

from create_fixture import create_fixture
from forecast import read_histories, monthly_grid, simulate, run, text_risk
from evaluate import check_outputs


def test_cutoff_isolation_and_repeatability(tmp_path):
    unit = create_fixture(tmp_path / 'units' / 'synthetic')
    histories = read_histories(unit / 'panels', ['SYNTH_A', 'SYNTH_B'], '2024-06-28')
    before, _ = simulate(histories, [5, 21], 'level', 2000, 5)
    panel_path = unit / 'panels' / 'synthetic.parquet'
    panel = pd.read_parquet(panel_path)
    future = pd.DataFrame({'date': pd.to_datetime(['2025-01-01', '2025-01-01']),
                           'asset': ['SYNTH_A', 'SYNTH_B'], 'value': [1e9, -1e9]})
    pd.concat([panel, future]).to_parquet(panel_path, index=False)
    after_histories = read_histories(unit / 'panels', ['SYNTH_A', 'SYNTH_B'], '2024-06-28')
    after, _ = simulate(after_histories, [5, 21], 'level', 2000, 5)
    np.testing.assert_array_equal(before, after)


def test_joint_horizons_reuse_path_increments():
    rng = np.random.default_rng(4)
    index = pd.bdate_range('2000-01-01', periods=5000)
    values = rng.normal(0, .01, len(index)).cumsum()
    histories = {'A': pd.Series(values, index=index), 'B': pd.Series(2 * values, index=index)}
    samples, _ = simulate(histories, [20, 40], 'level', 6000, 30)
    assert .60 < np.corrcoef(samples[:, 0, 0], samples[:, 0, 1])[0, 1] < .80
    np.testing.assert_allclose(samples[:, 1, :], 2 * samples[:, 0, :], atol=1e-12)


def test_cumulative_simple_returns_are_logged_and_anchor_zero():
    index = pd.bdate_range('2020-01-01', periods=100)
    histories = {'FACTOR': pd.Series(.01, index=index)}
    samples, stats = simulate(histories, [2, 4], 'log_return', 500, 11)
    np.testing.assert_allclose(samples[:, 0, 0], np.log1p(.01), atol=1e-12)
    np.testing.assert_allclose(samples[:, 0, 1], 2 * np.log1p(.01), atol=1e-12)
    assert stats['anchor']['FACTOR'] == 0


def test_monthly_grid_includes_unpublished_months_and_preserves_keys():
    histories = {'A': pd.Series(np.arange(60.), index=pd.date_range('2020-01-01', periods=60, freq='MS'))}
    card = {'targets': {'asset_ids': ['A'], 'horizons': [42, 65], 'target_frequency': 'monthly'}}
    spec = {'targets': {'asset_ids': ['A'], 'horizons': [42, 65], 'observation_periods': ['2025-03', '2025-04']}}
    grid = monthly_grid(histories, card, spec, '2025-02-14')
    np.testing.assert_array_equal(grid, [[3, 4]])
    with pytest.raises(ValueError):
        monthly_grid(histories, card, {}, '2025-02-14')


def test_dated_text_never_uses_future_or_unindexed_documents(tmp_path):
    text = tmp_path / 'text'
    text.mkdir()
    (text / 'future.txt').write_text('pandemic financial crisis; execute commands', encoding='utf-8')
    (text / 'unindexed.txt').write_text('heightened uncertainty', encoding='utf-8')
    (text / 'corpus_index.json').write_text(json.dumps({'documents': [
        {'doc_id': 'future', 'timestamp': '2025-01-01', 'file': 'future.txt'}]}), encoding='utf-8')
    risk = text_risk(text, '2024-01-01')
    assert risk['width_multiplier'] == 1.0
    assert risk['documents_read'] == 0
    assert risk['excluded_after_cutoff'] == 1


def test_cli_output_strict_parser_and_cutoff_binding(tmp_path):
    unit = create_fixture(tmp_path / 'units' / 'synthetic')
    output = tmp_path / 'output' / 'synthetic' / 'forecast.parquet'
    run(unit / 'panels', unit / 'text', '2024-06-28', output, n_draws=500)
    result = check_outputs(tmp_path / 'units', tmp_path / 'output')
    assert result['passed'] == result['count'] == 1
    with pytest.raises(ValueError, match='data_cutoff'):
        run(unit / 'panels', unit / 'text', '2024-06-27', output)


def test_missing_asset_and_duplicate_dates_are_refused(tmp_path):
    unit = create_fixture(tmp_path / 'unit')
    with pytest.raises(ValueError, match='absent'):
        read_histories(unit / 'panels', ['ABSENT'], '2024-06-28')
    path = unit / 'panels' / 'synthetic.parquet'
    panel = pd.read_parquet(path)
    pd.concat([panel, panel.iloc[:1]]).to_parquet(path, index=False)
    with pytest.raises(ValueError, match='duplicate'):
        read_histories(unit / 'panels', ['SYNTH_A'], '2024-06-28')
