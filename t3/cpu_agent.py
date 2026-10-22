"""Eastern Eagles' CPU simulator: pinned ABIDES semantics and cheaper trace extraction.

No reference traces, scenario identifiers or expected outputs are used at runtime.
The baseline switch is a local verification aid using the same pinned upstream engine.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from types import SimpleNamespace

# Limit numerical-library threads before importing NumPy; independent batch workers
# share the competition's four CPUs rather than starting four BLAS pools each.
for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "vendor" / "track3_runtime"))

import numpy as np
import pandas as pd
from abides_core import abides
from abides_core.message import Message
if sys.platform == "win32":
    # NumPy 1.x uses a 32-bit C long for astype(int) on Windows; upstream
    # str_to_ns overflows at 2.1 seconds. Match the Linux int64 calculation
    # solely for local verification. The Linux submission needs no patch.
    import abides_core.utils as _abides_utils
    _abides_utils.str_to_ns = lambda value: np.int64(pd.to_timedelta(value).value)
from abides_markets.orders import Order
from abides_fork.config import ScenarioLatencyModel, build_config
from abides_fork.scenario_io import read_scenario
from abides_fork.trace import (
    TRACE_COLUMNS, _TRACE_DTYPES, _side_to_str,
    extract_trace as reference_extract_trace,
    extract_message_trace as reference_extract_message_trace, MESSAGE_TRACE_COLUMNS,
)

_ORIGINAL_LATENCY = ScenarioLatencyModel.get_latency
import abides_core.utils as _core_utils
import abides_core.kernel as _kernel_module
import abides_core.agent as _agent_module
from abides_markets.orders import Side
from abides_markets.agents.exchange_agent import ExchangeAgent

_ORIGINAL_FMT_TS = _core_utils.fmt_ts
_ORIGINAL_KERNEL_QUEUE = _kernel_module.queue
_ORIGINAL_LOG_DEEPCOPY = _agent_module.deepcopy
_ORIGINAL_TIME_DROPOUT = ExchangeAgent.get_time_dropout
_COPY_ATOMIC_TYPES = frozenset((type(None), int, float, bool, str, bytes, complex, Side))


def snapshot_copy(value):
    """Fast exact deepcopy for scalar-only log dictionaries; fallback for all else.

    These exact builtin types and Side enum values are unchanged by deepcopy.
    Nested containers, NumPy scalars, custom objects and mutable order tags retain
    Python's complete memoized deepcopy behavior.
    """
    value_type = type(value)
    if value_type in _COPY_ATOMIC_TYPES:
        return value
    if value_type is dict and all(type(k) in _COPY_ATOMIC_TYPES and type(v) in _COPY_ATOMIC_TYPES
                                  for k, v in value.items()):
        return value.copy()
    return _ORIGINAL_LOG_DEEPCOPY(value)


@lru_cache(maxsize=1024)
def _format_whole_second(second: int) -> str:
    return _ORIGINAL_FMT_TS(second * 1_000_000_000)


def cached_fmt_ts(timestamp) -> str:
    """Reuse identical second-resolution strings, never simulation state or outputs."""
    # The pinned formatter discards fractional seconds. Handle integer nanoseconds
    # only, with pandas' supported endpoints retained by the original function.
    if (type(timestamp) is int or isinstance(timestamp, np.integer)) and -9223372036000000000 <= timestamp <= 9223372036854775807:
        return _format_whole_second(int(timestamp) // 1_000_000_000)
    return _ORIGINAL_FMT_TS(timestamp)


class KernelHeapQueue:
    """Unbounded heap for the kernel's one-thread event loop.

    The exact event tuples and heapq comparisons match queue.PriorityQueue. Batch
    workers each own their kernel and heap; no producer or consumer crosses threads.
    """
    def __init__(self):
        self.queue = []

    def put(self, item):
        heapq.heappush(self.queue, item)

    def get(self):
        if not self.queue:
            raise _ORIGINAL_KERNEL_QUEUE.Empty
        return heapq.heappop(self.queue)

    def empty(self):
        return not self.queue

    def qsize(self):
        return len(self.queue)


def fast_time_dropout(self, book, symbol):
    """Preserve liquidity metrics while avoiding a pandas Series per book row.

    The original DataFrame coercion, arithmetic order, and treatment of a final
    unclosed dropout are retained. Only the row traversal changes; no records,
    exchange metrics, or simulation events are omitted.
    """
    if len(book) == 0:
        return
    df = pd.DataFrame(book)
    total_time = df["QuoteTime"].iloc[-1] - df["QuoteTime"].iloc[0]
    is_null_bids = False
    t_null_bids_first = 0
    T_null_bids = 0
    is_null_asks = False
    t_null_asks_first = 0
    T_null_asks = 0
    for quote_time, bids, asks in df[["QuoteTime", "bids", "asks"]].itertuples(index=False, name=None):
        if (len(bids) == 0) & (is_null_bids == False):
            t_null_bids_first = quote_time
            is_null_bids = True
        elif (len(bids) != 0) & (is_null_bids == True):
            T_null_bids += quote_time - t_null_bids_first
            is_null_bids = False
        if (len(asks) == 0) & (is_null_asks == False):
            t_null_asks_first = quote_time
            is_null_asks = True
        elif (len(asks) != 0) & (is_null_asks == True):
            T_null_asks += quote_time - t_null_asks_first
            is_null_asks = False
    self.metric_trackers[symbol] = self.MetricTracker(
        total_time_no_liquidity_asks=T_null_asks / 1e9,
        total_time_no_liquidity_bids=T_null_bids / 1e9,
        pct_time_no_liquidity_asks=100 * T_null_asks / total_time,
        pct_time_no_liquidity_bids=100 * T_null_bids / total_time,
    )


def configure_hotpaths(baseline: bool) -> None:
    """Install or restore pure-formatting and single-thread queue substitutions."""
    formatter = _ORIGINAL_FMT_TS if baseline else cached_fmt_ts
    # ABIDES imports fmt_ts by value. Update only those known aliases; do not
    # overwrite similarly named functions belonging to other libraries.
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("abides_core", "abides_markets")):
            value = getattr(module, "fmt_ts", None)
            if value is _ORIGINAL_FMT_TS or value is cached_fmt_ts:
                module.fmt_ts = formatter
    _kernel_module.queue = _ORIGINAL_KERNEL_QUEUE if baseline else SimpleNamespace(PriorityQueue=KernelHeapQueue)
    _agent_module.deepcopy = _ORIGINAL_LOG_DEEPCOPY if baseline else snapshot_copy
    ExchangeAgent.get_time_dropout = _ORIGINAL_TIME_DROPOUT if baseline else fast_time_dropout


def scalar_latency(self, sender_id: int, recipient_id: int) -> int:
    """Identical random draws and rounding, without scalar NumPy dispatch per message."""
    if sender_id == recipient_id:
        return 0
    if self._model == "log_normal":
        value = self.random_state.lognormal(mean=self._mu, sigma=self._sigma)
    elif self._model == "uniform":
        value = self.random_state.uniform(self._min_ns, self._max_ns)
    elif self._model == "pareto":
        base = self._min_ns if self._min_ns > 0 else 1.0
        value = base * (1.0 + self.random_state.pareto(self._alpha))
    else:
        value = self._mean_ns
    # The public NumPy clip contract also returns max when min > max. Keep that
    # behavior, and preserve its non-finite behavior instead of silently fixing input.
    if not (math.isfinite(value) and math.isfinite(self._min_ns) and math.isfinite(self._max_ns)):
        return int(round(float(np.clip(value, self._min_ns, self._max_ns))))
    if value < self._min_ns:
        value = self._min_ns
    if value > self._max_ns:
        value = self._max_ns
    return int(round(float(value)))


def fast_extract_trace(end_state: dict) -> pd.DataFrame:
    """Project logs directly onto seven trace columns, retaining upstream tie ordering.

The engine and all log emissions are untouched. This avoids materializing unused
debugging columns. Orders sort by timestamp before final-fill classification; quotes
keep each (timestamp, side)'s final observation in first-appearance order, exactly
as the organizer adapter does. The final (timestamp, order_id) sort is stable.
"""
    orders = []
    quotes = {}
    event_types = {"ORDER_SUBMITTED", "ORDER_ACCEPTED", "ORDER_CANCELLED", "ORDER_REPLACED", "ORDER_EXECUTED"}
    sides = {}
    for agent in end_state["agents"]:
        for ts, event_type, event in agent.log:
            if event_type not in event_types and event_type not in ("BEST_BID", "BEST_ASK"):
                continue
            ts = ts if isinstance(ts, (int, np.int64)) else 0
            if event_type in event_types:
                if not isinstance(event, dict) or event.get("order_id") is None:
                    continue
                oid = event["order_id"]
                if pd.isna(oid):
                    continue
                side = event.get("side")
                if side not in sides:
                    sides[side] = _side_to_str(side)
                price = event.get("fill_price" if event_type == "ORDER_EXECUTED" else "limit_price")
                size = event.get("quantity")
                aid = event.get("agent_id")
                orders.append([
                    ts, agent.id if aid is None else aid, event_type, sides[side],
                    0 if price is None or pd.isna(price) else price,
                    0 if size is None or pd.isna(size) else size, oid,
                ])
            else:
                # Upstream BEST_BID/BEST_ASK values are SYMBOL,price,volume.
                if isinstance(event, dict):
                    event = event.get("ScalarEventValue")
                parts = str(event).split(",")
                if len(parts) != 3:
                    continue
                try:
                    price, size = int(parts[1]), int(parts[2])
                except ValueError:
                    # Preserve the upstream parser on unusual numeric strings.
                    return reference_extract_trace(end_state)
                side = "BID" if event_type == "BEST_BID" else "ASK"
                quotes[(ts, side)] = [ts, agent.id, "QUOTE_UPDATE", side, price, size, -1]
    orders.sort(key=lambda r: r[0])
    final_execution = {}
    for pos, row in enumerate(orders):
        if row[2] == "ORDER_EXECUTED":
            final_execution[row[6]] = pos
    for pos, row in enumerate(orders):
        if row[2] == "ORDER_EXECUTED":
            row[2] = "ORDER_FILLED" if final_execution[row[6]] == pos else "PARTIAL_FILL"
    rows = orders + list(quotes.values())
    rows.sort(key=lambda r: (r[0], r[6]))
    return pd.DataFrame(rows, columns=TRACE_COLUMNS).astype(_TRACE_DTYPES)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def peak_rss_bytes() -> int:
    if sys.platform == "win32":
        import psutil
        info = psutil.Process().memory_info()
        return int(getattr(info, "peak_wset", info.rss))
    import resource
    result = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(result if sys.platform == "darwin" else result * 1024)


_LEDGER_INT_TYPES = frozenset((int, np.int8, np.int16, np.int32, np.int64))


def fast_extract_message_trace(end_state: dict) -> pd.DataFrame:
    """Construct exact typed ledger columns without DataFrame inference/recasting.

    The fast path is restricted to replayable built-in containers and scalar
    types emitted by the pinned kernel. Other domains use the unchanged
    reference, retaining its coercion, missing-value and exception behavior.
    """
    if type(end_state) is not dict:
        return reference_extract_message_trace(end_state)
    ledger = end_state.get("message_ledger")
    seqmap = end_state.get("deliver_seq_by_key")
    if type(ledger) is not list or not ledger or type(seqmap) is not dict:
        return reference_extract_message_trace(end_state)
    # Validate lookup operands before conversion or lookup. In particular, never
    # narrow destination IDs before finding the broadcast-specific delivery key.
    if any(type(row) is not dict or type(row.get("message_id")) not in _LEDGER_INT_TYPES
           or type(row.get("dst_id")) not in _LEDGER_INT_TYPES for row in ledger):
        return reference_extract_message_trace(end_state)
    rows = [(seq, row) for row in ledger
            if (seq := seqmap.get((int(row["message_id"]), int(row["dst_id"])))) is not None]
    if not rows or any(type(seq) not in _LEDGER_INT_TYPES for seq, _ in rows):
        return reference_extract_message_trace(end_state)
    rows.sort(key=lambda item: item[0])  # Python's stable ordering retains tied deliveries.
    led = [row for _, row in rows]
    columns = {}
    try:
        for name in MESSAGE_TRACE_COLUMNS:
            values = [seq for seq, _ in rows] if name == "seq" else [row[name] for row in led]
            if name == "msg_type":
                if any(value is not None and type(value) is not str for value in values):
                    raise TypeError("noncanonical ledger string domain")
                columns[name] = pd.array(values, dtype="string")
            elif name in ("t_send_ns", "order_id", "causal_parent"):
                if any(value is not None and type(value) not in _LEDGER_INT_TYPES for value in values):
                    raise TypeError("noncanonical nullable ledger integer domain")
                data = np.fromiter((0 if value is None else value for value in values),
                                   dtype=np.int64, count=len(values))
                mask = np.fromiter((value is None for value in values),
                                   dtype=np.bool_, count=len(values))
                columns[name] = pd.arrays.IntegerArray(data, mask)
            else:
                if any(type(value) not in _LEDGER_INT_TYPES for value in values):
                    raise TypeError("noncanonical ledger integer domain")
                data = np.fromiter(values, dtype=np.int64, count=len(values))
                # Match pandas' int64 -> int32 wrapping, including cast edges.
                columns[name] = data.astype(np.int32) if name in ("src_id", "dst_id") else data
    except (KeyError, OverflowError, ValueError, TypeError):
        return reference_extract_message_trace(end_state)
    return pd.DataFrame(columns, columns=MESSAGE_TRACE_COLUMNS)


def simulate(config_path: str | Path, out_path: str | Path, baseline: bool = False) -> dict:
    started = time.perf_counter()
    scenario = json.loads(read_scenario(config_path))
    Order._order_id_counter = 0
    setattr(Message, "_Message__message_id_counter", 1)
    ScenarioLatencyModel.get_latency = _ORIGINAL_LATENCY if baseline else scalar_latency
    configure_hotpaths(baseline)
    config = build_config(scenario)
    built = time.perf_counter()
    # Windows and Linux both get an absolute writable temporary log path. Each
    # concurrent market has a separate directory; nothing enters /output except deliverables.
    with tempfile.TemporaryDirectory(prefix="eastern-eagles-t3-") as log_dir:
        end_state = abides.run(config, log_dir=log_dir)
    simulated = time.perf_counter()
    trace = (reference_extract_trace if baseline else fast_extract_trace)(end_state)
    message_trace = (reference_extract_message_trace if baseline else fast_extract_message_trace)(end_state)
    extracted = time.perf_counter()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace.to_parquet(out_path, compression="snappy", index=False)
    message_path = out_path.parent / "message_trace.parquet"
    message_trace.to_parquet(message_path, compression="snappy", index=False)
    trace_hash, message_hash = sha256(out_path), sha256(message_path)
    elapsed = time.perf_counter() - started
    events = {
        "scenario_id": str(scenario["scenario_id"]), "seed": int(scenario["seed"]),
        "n_events": int(len(trace)), "wall_clock_sec": elapsed,
        "events_per_sec": len(trace) / elapsed if elapsed > 0 else 0.0,
        "trace_sha256": trace_hash, "n_messages": int(len(message_trace)),
        "message_trace_sha256": message_hash, "peak_memory_bytes": peak_rss_bytes(),
        "gpu_seconds": 0.0,
    }
    (out_path.parent / "events.json").write_text(json.dumps(events, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scenario_id": scenario["scenario_id"], "baseline": baseline,
                      "configure_seconds": built - started, "simulate_seconds": simulated - built,
                      "extract_seconds": extracted - simulated, "serialize_seconds": elapsed - (extracted - started),
                      "total_seconds": elapsed, "events": len(trace)}), file=sys.stderr)
    return events


def _batch_job(args):
    path, out, baseline = args
    return Path(path).stem, simulate(path, out, baseline)


def simulate_batch(batch_dir: str | Path, out_dir: str | Path, workers: int = 4, baseline: bool = False) -> dict:
    batch_dir, out_dir = Path(batch_dir), Path(out_dir)
    paths = sorted(batch_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No subscenario JSON files in {batch_dir}")
    jobs = [(str(p), str(out_dir / p.stem / "trace.parquet"), baseline) for p in paths]
    started = time.perf_counter()
    workers = min(4, max(1, workers), len(jobs))
    if workers == 1:
        results = [_batch_job(job) for job in jobs]
    else:
        # No mutable simulation state is shared across processes. Every call resets
        # the order/message counters and seeds all streams from its own config.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_batch_job, jobs))
    elapsed = time.perf_counter() - started
    total = sum(ev["n_events"] for _, ev in results)
    aggregate = {"n_scenarios": len(results), "total_events": total,
                 "wall_clock_sec": elapsed, "events_per_sec": total / elapsed,
                 "gpu_seconds": 0.0,
                 "per_scenario": [{"sub": sub, "n_events": ev["n_events"], "trace_sha256": ev["trace_sha256"]}
                                  for sub, ev in results]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "batch_events.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("verb", choices=["simulate", "simulate-batch"])
    parser.add_argument("--config")
    parser.add_argument("--out")
    parser.add_argument("--batch-dir")
    parser.add_argument("--out-dir")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--baseline", action="store_true", help="Local differential-check baseline")
    args = parser.parse_args(argv)
    if args.verb == "simulate":
        if not args.config or not args.out:
            parser.error("simulate requires --config and --out")
        result = simulate(args.config, args.out, args.baseline)
    else:
        if not args.batch_dir or not args.out_dir:
            parser.error("simulate-batch requires --batch-dir and --out-dir")
        result = simulate_batch(args.batch_dir, args.out_dir, args.workers, args.baseline)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
