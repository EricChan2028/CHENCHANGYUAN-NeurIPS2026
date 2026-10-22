"""Bounded ledger-only parity and microtiming screen; no markets or network calls."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from decimal import Decimal
from fractions import Fraction
import gc
import hashlib
import io
import json
from pathlib import Path
import pickle
import platform
import random
import statistics
import time
import warnings

import cpu_agent_v004_candidate as candidate
import numpy as np
import pandas as pd
import psutil
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SELECTED_HASH = "b858c806e7a9209366a7b1a43718662305f619c227086b83ad33521500a61a6c"
REFERENCE = candidate.reference_extract_message_trace
FALLBACK_CALLS = 0


def tracked_reference(state):
    global FALLBACK_CALLS
    FALLBACK_CALLS += 1
    return REFERENCE(state)


def state_with_rows(size=7, mixed_numpy=False):
    rows = []
    seqmap = {}
    for i in range(size):
        message = i // 3
        destination = i % 3
        sent = 1_600_000_000_000_000_001 + i * 173
        row = {"t_recv_ns": sent + 19, "t_send_ns": None if i % 7 == 0 else sent,
               "latency_ns": 0 if i % 7 == 0 else 19, "src_id": -1 if i % 7 == 0 else i % 19,
               "dst_id": destination, "message_id": message,
               "msg_type": ("WAKEUP", "ORDER_ACCEPTED", "ORDER_EXECUTED", "QUERY_SPREAD")[i % 4],
               "order_id": None if i % 5 == 0 else i * 11,
               "causal_parent": None if i % 7 == 0 else max(0, message - 3)}
        if mixed_numpy:
            # Pinned str_to_ns returns np.int64 on Linux; kernel arithmetic can
            # retain NumPy signed scalars. Mix these with plain integer events.
            for field in ("t_recv_ns", "t_send_ns", "latency_ns"):
                if row[field] is not None and i % 3 != 0:
                    row[field] = np.int64(row[field])
        rows.append(row)
        if i % 23 != 22:
            # Nontrivial stable order; adjacent broadcasts can share a seq.
            seqmap[(message, destination)] = (i * 17 % max(1, size)) // 2
    return {"message_ledger": rows, "deliver_seq_by_key": seqmap}


def parquet_bytes(frame):
    target = io.BytesIO()
    frame.to_parquet(target, compression="snappy", index=False)
    return target.getvalue()


def call(function, state):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", message="np.find_common_type is deprecated.*", category=DeprecationWarning)
        try:
            result = ("returned", function(state))
        except Exception as error:
            result = ("raised", type(error).__name__, str(error))
    return result, [(type(item.message).__name__, str(item.message)) for item in caught]


def compare(name, state):
    before = pickle.dumps(state, protocol=5)
    reference_state, actual_state = copy.deepcopy(state), copy.deepcopy(state)
    expected, expected_warnings = call(REFERENCE, reference_state)
    fallbacks_before = FALLBACK_CALLS
    actual, actual_warnings = call(candidate.fast_extract_message_trace, actual_state)
    assert pickle.dumps(reference_state, protocol=5) == before, (name, "reference input mutation")
    assert pickle.dumps(actual_state, protocol=5) == before, (name, "candidate input mutation")
    assert actual[0] == expected[0], (name, expected[0], actual[0])
    assert actual_warnings == expected_warnings, (name, expected_warnings, actual_warnings)
    result = {"name": name, "result": actual[0], "fallback": FALLBACK_CALLS != fallbacks_before}
    if expected[0] == "raised":
        assert actual == expected, (name, expected, actual)
        result["exception_type"] = actual[1]
        return result
    pd.testing.assert_frame_equal(actual[1], expected[1], check_exact=True)
    expected_bytes, actual_bytes = parquet_bytes(expected[1]), parquet_bytes(actual[1])
    assert actual_bytes == expected_bytes, (name, "Parquet bytes differ")
    actual_schema = pq.read_schema(io.BytesIO(actual_bytes))
    assert actual_schema.equals(pq.read_schema(io.BytesIO(expected_bytes)), check_metadata=True), name
    result.update(rows=len(actual[1]), parquet_sha256=hashlib.sha256(actual_bytes).hexdigest())
    # An output edit must not mutate its source ledger or another extraction.
    if len(actual[1]):
        actual[1].iloc[0, actual[1].columns.get_loc("t_recv_ns")] = 123
        assert pickle.dumps(actual_state, protocol=5) == before, (name, "output aliases ledger")
        pd.testing.assert_frame_equal(REFERENCE(actual_state), expected[1], check_exact=True)
    return result


def parity_cases():
    cases = [("absent", {}), ("empty", state_with_rows(0)), ("broadcasts-stable-ties", state_with_rows(30)),
             ("mixed-native-and-numpy-timestamps", state_with_rows(100, mixed_numpy=True))]
    state = state_with_rows()
    state["deliver_seq_by_key"] = {}
    cases.append(("no-deliveries", state))
    state = state_with_rows()
    state["message_ledger"].insert(2, copy.deepcopy(state["message_ledger"][0]))
    cases.append(("duplicate-row-and-seq-zero", state))
    state = state_with_rows()
    state["message_ledger"] = tuple(state["message_ledger"])
    cases.append(("tuple-ledger-fallback", state))
    fields = ("t_recv_ns", "t_send_ns", "latency_ns", "src_id", "dst_id", "message_id", "order_id", "causal_parent")
    values = [-(2**63), -(2**63) + 1, -2**31 - 1, -2**31, -1, 0,
              2**31 - 1, 2**31, 2**32 - 1, 2**53 + 1, 2**63 - 1,
              2**63, 2**64 - 1, 2**64, -(2**63) - 1,
              None, pd.NA, float("nan"), 2.0, 2.5, float("inf"), True,
              np.int64(2**53 + 1), np.uint64(2**63), np.int32(-7),
              "2", "bad", Decimal("2"), Fraction(3, 2)]
    for field in fields:
        for index, value in enumerate(values):
            state = state_with_rows()
            row = state["message_ledger"][1]
            old_key = (row["message_id"], row["dst_id"])
            row[field] = value
            if field in ("message_id", "dst_id"):
                try:
                    state["deliver_seq_by_key"][(int(row["message_id"]), int(row["dst_id"]))] = state["deliver_seq_by_key"].pop(old_key)
                except (ValueError, TypeError, OverflowError):
                    pass
            cases.append((f"{field}-edge-{index}", state))
    for value in (None, pd.NA, float("nan"), True, 1, "", "A" * 1024, "送信🦅"):
        state = state_with_rows()
        state["message_ledger"][1]["msg_type"] = value
        cases.append((f"msg-type-{len(cases)}", state))
    for value in (0, -1, 2**63 - 1, 2**63, 2.5, True, np.int64(7), "7", None):
        state = state_with_rows()
        state["deliver_seq_by_key"][(0, 1)] = value
        cases.append((f"seq-edge-{len(cases)}", state))
    for field in fields + ("msg_type",):
        state = state_with_rows()
        del state["message_ledger"][1][field]
        cases.append(("missing-" + field, state))
    rng = random.Random(773004)
    for index in range(24):
        state = state_with_rows(31 + index)
        rng.shuffle(state["message_ledger"])
        for row in state["message_ledger"]:
            for field in ("t_send_ns", "order_id", "causal_parent"):
                row[field] = None if rng.randrange(3) == 0 else rng.choice([-(2**63), 2**53 + 1, 2**63 - 1, rng.randrange(-1000, 1000)])
        cases.append((f"random-nullable-extremes-{index}", state))
    return cases


def non_replayable_parity():
    consumed = []
    def fresh():
        state = state_with_rows()
        def rows():
            for row in state["message_ledger"]:
                consumed.append(row["message_id"])
                yield row
        return {"message_ledger": rows(), "deliver_seq_by_key": state["deliver_seq_by_key"]}
    expected = REFERENCE(fresh())
    expected_consumed = list(consumed)
    consumed.clear()
    actual = candidate.fast_extract_message_trace(fresh())
    assert consumed == expected_consumed
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert parquet_bytes(actual) == parquet_bytes(expected)
    return {"name": "generator-not-consumed-before-fallback", "rows": len(actual), "consumed_once": True}


def fallback_invoked_once():
    calls = []
    def rejecting_reference(state):
        calls.append(state)
        raise ValueError("reference error with a visible side effect")
    state = state_with_rows()
    state["message_ledger"][0]["t_send_ns"] = 1.25
    original = candidate.reference_extract_message_trace
    candidate.reference_extract_message_trace = rejecting_reference
    try:
        result, _ = call(candidate.fast_extract_message_trace, state)
        assert result == ("raised", "ValueError", "reference error with a visible side effect")
        assert len(calls) == 1
    finally:
        candidate.reference_extract_message_trace = original
    return {"name": "unsupported-value-reference-exception-invoked-once", "reference_calls": len(calls)}


def benchmark():
    records = []
    for size_index, size in enumerate((50_000, 200_000)):
        state = state_with_rows(size, mixed_numpy=True)
        # Exact comparison and warm-up precede timing; generation/serialization are untimed.
        before = FALLBACK_CALLS
        expected = REFERENCE(state)
        actual = candidate.fast_extract_message_trace(state)
        assert FALLBACK_CALLS == before, "representative data unexpectedly used fallback"
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert parquet_bytes(actual) == parquet_bytes(expected)
        count = len(expected)
        del expected, actual
        raw = []
        for pair in range(3):
            order = ("v003", "v004") if (pair + size_index) % 2 == 0 else ("v004", "v003")
            times = {}
            for mode in order:
                gc.collect()
                function = REFERENCE if mode == "v003" else candidate.fast_extract_message_trace
                start = time.perf_counter()
                result = function(state)
                elapsed = time.perf_counter() - start
                assert len(result) == count
                del result
                times[mode] = elapsed
            raw.append({"pair": pair, "order": list(order), "seconds": times,
                        "conversion_rate_ratio_v004_over_v003": times["v003"] / times["v004"]})
        median = {mode: statistics.median(row["seconds"][mode] for row in raw) for mode in ("v003", "v004")}
        ratio = median["v003"] / median["v004"]
        records.append({"ledger_rows": size, "delivered_rows": count, "scalar_domain": "mixed built-in/np.int64 timestamps",
                        "fast_path_used": True, "pairs": raw,
                        "median_seconds": median, "conversion_rate_ratio_v004_over_v003": ratio,
                        "screen_passed": ratio >= 1.10})
    return {"cases": records, "screen_passed": all(row["screen_passed"] for row in records),
            "predeclared_gate": "median ledger-conversion rate >=1.10x v003 on BOTH 50000 and 200000 rows",
            "whole_process_performance_claim": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-only", action="store_true")
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/t3-v004-local-audit.json")
    args = parser.parse_args()
    assert candidate.sha256(ROOT / "t3/cpu_agent.py") == SELECTED_HASH
    assert candidate.sha256(ROOT / "t3/cpu_agent_v003.py") == SELECTED_HASH
    process = psutil.Process()
    process.cpu_affinity([args.cpu])
    candidate.reference_extract_message_trace = tracked_reference
    report = {"kind": "bounded_synthetic_ledger_screen", "rankable": False, "cpu_only": True,
              "worker_processes": 1, "affinity": process.cpu_affinity(),
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {"selected_v003": SELECTED_HASH,
                                "v004_candidate": candidate.sha256(ROOT / "t3/cpu_agent_v004_candidate.py"),
                                "verification": candidate.sha256(Path(__file__))},
              "environment": {"python": platform.python_version(), "system": platform.platform(),
                              "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
              "selection_changed": False, "limitations": "Synthetic local microtiming only; no whole-process, Linux, official score or rank claim"}
    if args.report.exists():
        previous = json.loads(args.report.read_text(encoding="utf-8"))
        if "microbenchmark" in previous:
            report["superseded_initial_native_int_screen"] = {
                "source_sha256": previous["source_sha256"], "microbenchmark": previous["microbenchmark"],
                "parity_cases": previous["parity"]["cases"],
                "reason": "Native-int-only prototype was too narrow for NumPy kernel timestamps; fixed fallback double-call control flow before further evaluation"}
    try:
        results = [compare(name, state) for name, state in parity_cases()]
        results.append(non_replayable_parity())
        results.append(fallback_invoked_once())
        report["parity"] = {"passed": True, "cases": len(results), "results": results,
                            "exact_dataframe_dtypes_schema_parquet": True, "input_and_output_isolation": True,
                            "fallback_calls": FALLBACK_CALLS}
        print(json.dumps({"parity_passed": len(results), "fallback_calls": FALLBACK_CALLS}), flush=True)
        if not args.parity_only:
            report["microbenchmark"] = benchmark()
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["selected_source_unchanged"] = candidate.sha256(ROOT / "t3/cpu_agent.py") == SELECTED_HASH
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"report": str(args.report), "status": report["status"],
                          "screen_passed": report.get("microbenchmark", {}).get("screen_passed")}), flush=True)


if __name__ == "__main__":
    main()
