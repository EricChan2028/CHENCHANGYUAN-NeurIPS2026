"""## Executive summary (read this first)
Create an entirely synthetic CPU smoke fixture; this is not a scored official unit.
It tests runtime outputs without redistributing any competition corpus or outcomes.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def create_fixture(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'panels').mkdir(exist_ok=True)
    (directory / 'text').mkdir(exist_ok=True)
    dates = pd.bdate_range(end='2024-06-28', periods=400)
    rng = np.random.default_rng(104)
    shared = rng.normal(0, .025, len(dates))
    frame = pd.concat([pd.DataFrame({'date': dates, 'asset': asset,
        'value': anchor + np.cumsum(shared + rng.normal(0, .01, len(dates)))})
        for asset, anchor in [('SYNTH_A', 4.0), ('SYNTH_B', 3.0)]], ignore_index=True)
    frame.to_parquet(directory / 'panels' / 'synthetic.parquet', index=False)
    card = '''# Executive summary: synthetic runtime fixture, not an official task.
schema_version = "2.0"
[task]
id = "t2-synthetic-cpu-smoke"
track = "forecasting"
split = "validation"
[metadata]
category = "T2-F1"
[provenance]
data_cutoff = "2024-06-28"
[targets]
asset_ids = ["SYNTH_A", "SYNTH_B"]
horizons = [5, 21]
target_type = "level"
target_frequency = "daily"
[scoring.params]
n_draws_min = 200
'''
    (directory / 'card.toml').write_text(card, encoding='utf-8')
    (directory / 'text' / 'corpus_index.json').write_text(json.dumps({'documents': []}), encoding='utf-8')
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    print(create_fixture(parser.parse_args().directory))
