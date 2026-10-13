## Executive summary (read this first)

This is Eastern Eagles' first CPU-only T2 candidate. It produces probabilistic
forecasts for all 104 public cards with no model weights or API calls. It uses
only supplied observations available by each card's cutoff.

The unmodified official T2 gates-only CLI passes g0, g1, g2 and g3 for all 104
artifacts. The official output schema and strict output parser also accept all
104 artifacts, and seven local tests pass. The separate common input-manifest
staging validation is blocked on Windows because its no-follow directory traversal
requires Linux primitives.
The Docker image has not yet been built or tested. Nothing here establishes a
competition score or rank.

### Run locally

From the campaign root in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' t2/forecast.py --panels upstream-t2/units/t2-EXAMPLE-ust-curve-1m --text upstream-t2/units/t2-EXAMPLE-ust-curve-1m/text --asof 2024-06-28 --out t2/output/t2-EXAMPLE-ust-curve-1m/forecast.parquet
& '.\.venv\Scripts\python.exe' -m pytest t2/test_forecast.py -q
& '.\.venv\Scripts\python.exe' t2/evaluate.py --mode gates --n-draws 2000 --report reports/t2_gates.json
& '.\.venv\Scripts\python.exe' t2/evaluate.py --mode outputs --report reports/t2_outputs.json
& '.\.venv\Scripts\python.exe' t2/evaluate.py --mode scorer-gates --report reports/t2_official_gates.json
& '.\.venv\Scripts\python.exe' t2/evaluate.py --mode rolling --n-draws 1000 --report reports/t2_rolling.json
```

The `forecast` executable in the image implements the required interface. It
writes `forecast.parquet`, `forecast_meta.json`, and `forecast_rationale.md` beside
the requested output file. Default draw count is 2,000; card and contract floors
are enforced. All output cells share coherent cross-asset and cross-horizon paths.

Run the official output gates without requesting nonexistent public outcomes:

```sh
python -m qfbench2_track_forecasting.scoring score --card /input/card.toml --forecast /output/forecast.parquet
```

`qfbench2-smoke` additionally requests a score; the public cards do not contain
the future outcomes needed to produce one. Do not fabricate an outcome to turn
that command green.

### Method

Daily changes are aligned by date; gaps are removed before estimating innovations.
For cumulative returns, the official helper transforms each simple return with
`log(1+r)` before accumulation. Monthly grids use the official observation-period
resolver, including the gap from the latest published observation to the target
month. Authored horizon keys remain unchanged.

The stationary bootstrap shares historical shock vectors across assets and
resamples short consecutive blocks. Recent and longer-window variances determine
its scale. Daily level drift is zero; daily return drift is a shrunk historical
mean. Monthly drift blends recent and longer growth windows. Missing assets,
duplicate dates, invalid monthly mappings and nonfinite output refuse explicitly.

An optional `--text-risk` experiment reads only indexed, cutoff-valid documents.
It can add at most 5% to width after a small set of uncertainty phrase matches.
It is **off by default** because its forecasting benefit has not been established.
The model-free default is honestly described as text-blind in its rationale.

### First historical diagnostic

304 rolling forecast windows from 103 public cards were compared to a participant
Gaussian random-walk baseline. Within each card, the three target windows do not
overlap. Identical evaluation problems are deduplicated, but overlapping data
across different cards still makes the observations statistically dependent.

- Median paired marginal CRPS ratio: approximately **0.939**.
- Median paired raw composite ratio: approximately **0.939**.
- Candidate has lower raw composite in **56.25%** of windows.
- Mean empirical 90% interval coverage: **81.66%** candidate vs **82.19%** baseline.

Lower ratios are better. These are development diagnostics against a locally
implemented baseline, not the official M0 generator. The imported T2 scorer uses
the current pinball tail metric. Sealed normalization scales are unavailable;
raw scores cannot be averaged across physical units as a leaderboard score.
Monthly diagnostics use 1/3/6 observation steps, separate from authored horizon
keys. Historical series may contain revisions rather than first-release vintages.
The historical text overlay is disabled to prevent moving current corpora backward
across a forecast cutoff.

### Next experiment, before making a new performance claim

The observed interval undercoverage motivates checking width calibration. The
304 inspected windows are development evidence and must not be relabelled as an
untouched test set. First build an exposure registry across assets and source
histories using each report row's `cutoff`, `validation_end`, and `assets`.
Reserve previously uninspected nonoverlapping time intervals before changing
the method. Select among predeclared width multipliers 1.00, 1.15 and 1.30 using
only earlier calibration windows, freeze the selection, and evaluate once on a
later reserved window. Respect vintage availability and source overlap. If a
sufficient untouched partition does not exist, obtain new permitted data or
continue to label the exercise as exploratory.

Before any upload: complete the Linux gates, offline/read-only container execution,
reproducibility and immutable image-digest recording. Future standings must come
from the official platform rather than these local diagnostics.
