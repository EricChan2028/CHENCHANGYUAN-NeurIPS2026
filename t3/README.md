# Eastern Eagles CPU market simulator

This candidate runs the genuine pinned ABIDES market simulator entirely on CPU. It
preserves market and agent behavior while replacing scalar latency clipping and
projecting logs directly into the required trace columns. Version 002 also caches
pure second-resolution timestamp formatting, uses the same heap comparisons without
thread locks in each single-thread kernel, and copies flat immutable log dictionaries
without recursively redispatching every scalar. Nested/mutable log values retain the
original deep-copy path. Version 003 computes the same end-of-market liquidity
dropout metrics using tuple traversal instead of allocating a pandas Series for
every book snapshot. It retains the original DataFrame coercion, arithmetic order,
all records, and treatment of an unclosed final dropout. Version 004 builds typed
message-ledger columns directly
while retaining delivered-message filtering, stable sequence order, broadcast keys,
exact nullable nanosecond integers, and canonical Parquet schema/bytes. Unsupported
input domains use the unchanged reference conversion. Local parity, all public
markets, fresh seeds and a separate Linux comparison passed before this build.
Independent batch markets
run in at most four separate processes. No reference trace, pretrained model, API,
GPU library or network request is used during simulation.

The implementation is an initial measured candidate. Local tests do not establish an
official rank, final admissibility or comparable performance on the organizer fleet.

## Local setup

The separate `.venv` uses Python 3.11 and the exact official numerical stack. The
campaign's parent `.venv` uses Python 3.13 for the official scoring toolkit; they must
remain separate because the ABIDES baseline requires pandas 1.5.3.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
git clone https://github.com/jpmorganchase/abides-jpmc-public.git vendor/abides
git -C vendor/abides checkout f9cbe51342b7dedd9587e4e069040d68a5c6477f
git -C vendor/abides apply ../track3_runtime/patches/order_size_model.pomegranate-free.patch
git -C vendor/abides apply ../track3_runtime/patches/kernel_message_ledger.patch
git -C vendor/abides apply ../track3_runtime/patches/exchange_protocol_stp.patch
git -C vendor/abides apply ../track3_runtime/patches/oracle_scheduled_jump.patch
.\.venv\Scripts\python.exe -m pip install --no-deps vendor/abides/abides-core vendor/abides/abides-markets
.\.venv\Scripts\python.exe smoke_test.py
```

On Windows, the runner makes the upstream duration conversion explicitly int64.
The upstream NumPy 1.x `astype(int)` uses Windows' 32-bit C integer and otherwise
overflows durations above 2.1 seconds. This local compatibility adjustment matches
Linux's int64 result. Linux submissions do not apply this adjustment.

## Running and testing

```powershell
.\.venv\Scripts\python.exe cpu_agent.py simulate --config <scenario.json> --out <output/trace.parquet>
.\.venv\Scripts\python.exe cpu_agent.py simulate-batch --batch-dir <scenarios> --out-dir <output> --workers 4
.\.venv\Scripts\python.exe verify_public.py --upstream 'D:\NeurIPS Challenge\upstream-t3' --output runs/public --report ../reports/t3-public-exact.json --workers 2
.\.venv\Scripts\python.exe verify_batch.py --upstream 'D:\NeurIPS Challenge\upstream-t3' --output runs/public-batch --report ../reports/t3-public-batch.json --workers 2
.\.venv\Scripts\python.exe benchmark_local.py --config <scenario.json> --output runs/benchmark --report ../reports/t3-local-benchmark.json --repeats 3
.\.venv\Scripts\python.exe verify_v002.py --report ../reports/t3-v002-synthetic.json
.\.venv\Scripts\python.exe benchmark_v002.py --upstream 'D:\NeurIPS Challenge\upstream-t3' --output runs/v002-benchmark --report ../reports/t3-v002-benchmark.json --repeats 3
.\.venv\Scripts\python.exe verify_v003.py --report ../reports/t3-v003-synthetic.json
.\.venv\Scripts\python.exe benchmark_v003.py --upstream 'D:\NeurIPS Challenge\upstream-t3' --output runs/v003-benchmark --report ../reports/t3-v003-benchmark.json --repeats 3
```

`smoke_test.py` compares 40,000 latency draws against the original implementation,
then executes three new synthetic markets with all four agent types, stochastic
latencies, self-trade prevention and a scheduled shock. It compares candidate and
baseline traces and ledgers, checks repeat byte stability, and checks two-process
batch isolation. The baseline computes outputs at test time.

`verify_public.py` and `verify_batch.py` compare every table cell, column dtype and row count against
frozen **public** traces and ledgers. This is stricter than approximate table
comparison, but does not replace the official Linux container/scoring tests.
`benchmark_local.py` alternates baseline/candidate order, includes Python startup in
process timing, and requires identical stable output bytes across all timed runs.
`verify_v002.py` independently checks timestamp boundaries, queue tie ordering and
mutable/aliased/cyclic log values, followed by eight fresh market configurations with
all four latency models, exchange delays, shocks, agent populations and depths. It
checks one- and two-worker batch equivalence against separate simulation calls.
`benchmark_v002.py` compares the frozen prior submitted source (`cpu_agent_v001.py`,
SHA-256 `2a58e80cb2a502817e38420995fe700da4737e05c81ce510ba495514226ad309`)
against v002 in alternating fresh processes across small and large markets. The
snapshot and all development checks stay outside the container build context.

`verify_v003.py` adds 260 book-history cases checking exact liquidity metric values,
types and numerical warnings, including zero duration and trailing empty books. It
also compares the metrics on eight newly seeded markets while retaining complete
trace, ledger and batch-isolation checks. The implementation-specific NumPy
deprecation warning triggered only by pandas `iterrows` is excluded from the warning
comparison because the deprecated API is no longer used. `benchmark_v003.py`
compares whole fresh processes against `cpu_agent_v002.py`, whose SHA-256 is
`d1b832f3e9fc3697da0c4b7d5620f8cfbef85fee7d624d85675730f8beb546fe`.
Its optional `--pin-cpu <allowed-logical-cpu>` restricts only the benchmark child
processes to the same permitted logical CPU. Preserve unpinned and pinned results
separately; neither replaces timing on the official Linux resource profile.

The official `qfbench2 smoke` manifest verifier refuses Windows because the required
`O_NOFOLLOW` and `O_DIRECTORY` facilities are unavailable. Its checks are retained
unchanged. Run official gates on Linux (the campaign's GitHub Actions workflow is
prepared for that), before treating the image as submission-ready.

## Container

```text
docker build --platform linux/amd64 -t eastern-eagles-t3:local t3
docker run --rm --network none --cpus 4 --memory 16g --read-only --tmpfs /tmp:rw,size=64m -v <unit>:/input:ro -v <output>:/output eastern-eagles-t3:local simulate --config /input/scenario.json --out /output/trace.parquet
```

Build context is `t3/`. The `.dockerignore` whitelist permits only the runtime,
dependency list and official adapter/patch sources. It excludes local runs, reference
data, the Python environment, development tools and the ABIDES checkout. The build
fetches the pinned public ABIDES commit and applies the four official overlays.
The image has no Docker `VOLUME`. It writes Snappy Parquet and honest timing sidecars;
the organizer's externally measured time, not these sidecars, determines ranking.

## Provenance

- Participant code: `cpu_agent.py`, `smoke_test.py`, `verify_public.py`, `verify_batch.py`,
  `benchmark_local.py`, Docker packaging and this README; Eastern Eagles, Team 773.
- Official Track 3 adapter/patch source: public repository commit
  `fe32378d92b9767a0452c8fbb9911dcf722a8c87` from
  https://github.com/Agenthon-2026/track3-simulation-public, MIT license retained in
  `vendor/track3_runtime/LICENSE`.
- ABIDES upstream: https://github.com/jpmorganchase/abides-jpmc-public, commit
  `f9cbe51342b7dedd9587e4e069040d68a5c6477f`, BSD-3-Clause. Its license is retained
  in the image as `vendor/track3_runtime/ABIDES-LICENSE`.
- The four organizer overlays are applied in the Dockerfile's documented order.
- No learned model, external training data, private evaluation data or paid service
  is included. Public reference data is used only by external local verification.
- The base Python image tag remains floating until the campaign build workflow
  resolves and records the final immutable image digest. No image digest is invented.

Future optimizations must preserve exact RNG draws, event ordering and mutable log
snapshots. Compare against the prior submitted source and fresh seeds before adoption;
public reference outputs are validation data and never part of the runtime.
