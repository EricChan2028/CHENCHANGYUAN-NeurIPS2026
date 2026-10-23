"""Bounded full trace-extraction parity and microtiming; no markets or network."""
from __future__ import annotations

import argparse
import ast
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
from types import SimpleNamespace
import warnings

import cpu_agent_v004 as reference
import cpu_agent_v005_candidate as candidate
import numpy as np
import pandas as pd
import psutil
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SELECTED_HASH = "f04b88e6c25c15c029857d3071ec2ea8817a9a18ab3e4c942f746f61084b93ad"
FALLBACK = candidate._trace_frame_reference
FALLBACK_CALLS = 0
GATE = "Median full-extraction rate >=1.10x v004 on BOTH 50000 and 200000 input logs, three balanced pairs per size"


def tracked_fallback(rows):
    global FALLBACK_CALLS
    FALLBACK_CALLS += 1
    return FALLBACK(rows)


def state_with_logs(size=31):
    agents = [SimpleNamespace(id=i, log=[]) for i in range(8)]
    types = ("ORDER_SUBMITTED", "ORDER_ACCEPTED", "ORDER_EXECUTED", "ORDER_EXECUTED",
             "ORDER_CANCELLED", "ORDER_REPLACED", "BEST_BID", "BEST_ASK", "IGNORED", "BEST_BID")
    for i in range(size):
        timestamp = 1_600_000_000_000_000_001 + (i // 7) * 173
        if i % 3:
            timestamp = np.int64(timestamp)
        kind = types[i % len(types)]
        agent = agents[i % len(agents)]
        if kind in ("BEST_BID", "BEST_ASK"):
            value = f"ABM,{10000 + i % 41},{1 + i % 73}"
            if i % 2:
                value = {"ScalarEventValue": value}
        else:
            value = {"order_id": i // 10, "side": (candidate.Side.BID, candidate.Side.ASK, None)[i % 3],
                     "agent_id": agent.id if i % 2 else np.int32(agent.id),
                     "limit_price": 10000 + i % 41, "fill_price": 10003 + i % 31,
                     "quantity": 1 + i % 73 if i % 2 else np.int64(1 + i % 73)}
        agent.log.append((timestamp, kind, value))
    return {"agents": agents}


def simple_state():
    def order(oid, size, price=100, side="BID"):
        return {"order_id": oid, "agent_id": 7, "quantity": size,
                "limit_price": price, "fill_price": price + 1, "side": side}
    return {"agents": [SimpleNamespace(id=7, log=[
        (40, "ORDER_EXECUTED", order(11, 2)),
        (10, "ORDER_SUBMITTED", order(11, 6)),
        (20, "BEST_BID", "ABM,99,4"),
        (20, "BEST_ASK", "ABM,101,5"),
        (30, "ORDER_EXECUTED", order(11, 4)),
        (20, "BEST_BID", "ABM,100,9"),
        (40, "ORDER_SUBMITTED", order(10, 3, side=None)),
        (40, "ORDER_EXECUTED", order(10, 1)),
        (40, "ORDER_EXECUTED", order(10, 2)),
    ])]}


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
    expected, expected_warnings = call(reference.fast_extract_trace, reference_state)
    start_fallbacks = FALLBACK_CALLS
    actual, actual_warnings = call(candidate.fast_extract_trace, actual_state)
    assert pickle.dumps(reference_state, protocol=5) == before, (name, "reference input mutation")
    assert pickle.dumps(actual_state, protocol=5) == before, (name, "candidate input mutation")
    assert actual[0] == expected[0], (name, expected[0], actual[0])
    assert actual_warnings == expected_warnings, (name, expected_warnings, actual_warnings)
    result = {"name": name, "result": actual[0], "constructor_fallbacks": FALLBACK_CALLS - start_fallbacks}
    assert result["constructor_fallbacks"] <= 1, (name, "fallback invoked more than once")
    if expected[0] == "raised":
        assert actual == expected, (name, expected, actual)
        result["exception_type"] = actual[1]
        return result
    pd.testing.assert_frame_equal(actual[1], expected[1], check_exact=True)
    expected_bytes, actual_bytes = parquet_bytes(expected[1]), parquet_bytes(actual[1])
    assert actual_bytes == expected_bytes, (name, "Parquet bytes differ")
    assert pq.read_schema(io.BytesIO(actual_bytes)).equals(
        pq.read_schema(io.BytesIO(expected_bytes)), check_metadata=True), name
    result.update(rows=len(actual[1]), parquet_sha256=hashlib.sha256(actual_bytes).hexdigest())
    # Source logs and separately returned frames must not alias writable arrays.
    if len(actual[1]):
        actual[1].iloc[0, actual[1].columns.get_loc("t_ns")] = 123
        assert pickle.dumps(actual_state, protocol=5) == before, (name, "output aliases source")
        pd.testing.assert_frame_equal(reference.fast_extract_trace(actual_state), expected[1], check_exact=True)
        pd.testing.assert_frame_equal(candidate.fast_extract_trace(actual_state), expected[1], check_exact=True)
    return result


def parity_cases():
    cases = [("empty-agents", {"agents": []}), ("empty-logs", state_with_logs(0)),
             ("mixed-signed-numpy-and-native", state_with_logs(101)),
             ("timestamp-ties-final-fills-and-last-quotes", simple_state())]
    values = [-(2**63), -(2**63) + 1, -2**31 - 1, -2**31, -1, 0, 2**31 - 1, 2**31,
              2**32 - 1, 2**53 + 1, 2**63 - 1, 2**63, 2**64 - 1, 2**64, -(2**63) - 1,
              None, pd.NA, float("nan"), 2.0, 2.5, float("inf"), True,
              np.int8(-5), np.int16(-17), np.int32(-7), np.int64(2**53 + 1), np.uint64(2**63),
              "2", "bad", Decimal("2"), Fraction(3, 2)]
    fields = ("agent_id", "limit_price", "fill_price", "quantity", "order_id")
    for field in fields:
        for index, value in enumerate(values):
            state = simple_state()
            state["agents"][0].log[0 if field == "fill_price" else 1][2][field] = value
            cases.append((f"{field}-edge-{index}", state))
    for index, value in enumerate(values):
        state = simple_state()
        _, kind, event = state["agents"][0].log[1]
        state["agents"][0].log[1] = (value, kind, event)
        cases.append((f"timestamp-edge-{index}", state))
    for value in (None, pd.NA, float("nan"), True, 1, "", "bid", "AsK", "unknown", candidate.Side.ASK):
        state = simple_state()
        state["agents"][0].log[1][2]["side"] = value
        cases.append((f"nullable-side-{len(cases)}", state))
    for field in fields + ("side",):
        state = simple_state()
        del state["agents"][0].log[0 if field == "fill_price" else 1][2][field]
        cases.append(("missing-" + field, state))
    for value in ("ABM,2.0,3", "ABM,nan,3", "ABM,2,3,4", "ABM,2,3", "ABM,-2,-3",
                  "ABM,9223372036854775808,2", {"ScalarEventValue": "ABM,100,5"}, None):
        state = simple_state()
        state["agents"][0].log[2] = (20, "BEST_BID", value)
        cases.append((f"quote-value-{len(cases)}", state))
    # Integer extremes in otherwise float columns exercise pandas' original
    # lossy float inference instead of incorrectly taking the typed path.
    for field in ("agent_id", "quantity", "order_id", "limit_price"):
        state = simple_state()
        state["agents"][0].log[1][2][field] = 2**53 + 1
        state["agents"][0].log[6][2][field] = 2.5
        cases.append(("mixed-large-integer-and-float-" + field, state))
    rng = random.Random(773005)
    for index in range(24):
        state = state_with_logs(71 + index)
        for agent in state["agents"]:
            rng.shuffle(agent.log)
            for _, kind, row in agent.log:
                if kind.startswith("ORDER_"):
                    row["agent_id"] = rng.choice([-2**31 - 1, 2**31, np.int32(-1), 7])
                    row["quantity"] = rng.choice([-(2**63), 2**53 + 1, 2**63 - 1, np.int64(2)])
                    row["side"] = rng.choice([None, "BID", "ASK"])
        cases.append((f"shuffled-tied-extremes-{index}", state))
    return cases


def projected_row_fallback_once():
    calls = []
    conversions = []
    class ObservableInt:
        def __int__(self):
            conversions.append("int")
            return 9
    rows = [[1, 2, "ORDER_SUBMITTED", None, ObservableInt(), 4, 5]]
    before = list(rows[0])
    expected = FALLBACK(rows)
    reference_conversions = list(conversions)
    conversions.clear()
    actual = candidate._typed_trace_frame(rows)
    assert conversions == reference_conversions, "unsupported scalar conversion repeated"
    assert rows[0] == before, "projected rows mutated"
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert parquet_bytes(actual) == parquet_bytes(expected)
    original = candidate._trace_frame_reference
    def rejecting_reference(value):
        calls.append(value)
        raise ValueError("reference constructor with a visible side effect")
    candidate._trace_frame_reference = rejecting_reference
    try:
        result, _ = call(candidate._typed_trace_frame, [[1, 2, "ORDER_SUBMITTED", None, 2.5, 4, 5]])
        assert result == ("raised", "ValueError", "reference constructor with a visible side effect")
        assert len(calls) == 1
    finally:
        candidate._trace_frame_reference = original
    return {"name": "unsupported-scalar-and-raising-constructor-each-invoked-once",
            "reference_calls": len(calls), "scalar_conversions": len(conversions)}


def verify_source_scope():
    base = ast.parse((ROOT / "t3/cpu_agent_v004.py").read_text(encoding="utf-8"))
    changed = ast.parse((ROOT / "t3/cpu_agent_v005_candidate.py").read_text(encoding="utf-8"))
    added = {"_TRACE_INT_TYPES", "_trace_frame_reference", "_typed_trace_frame"}
    def name(node):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            return node.name
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
    changed.body = [node for node in changed.body if name(node) not in added]
    base_trace = next(node for node in base.body if name(node) == "fast_extract_trace")
    changed_trace = next(node for node in changed.body if name(node) == "fast_extract_trace")
    assert ast.dump(changed_trace.body[-1]) == ast.dump(ast.parse("return _typed_trace_frame(rows)").body[0])
    changed_trace.body[-1] = base_trace.body[-1]
    assert ast.dump(changed) == ast.dump(base), "changes extend beyond final trace construction"
    return {"only_final_trace_construction_changed": True, "engine_and_message_ledger_unchanged": True}


def benchmark(report, save):
    records = []
    for size_index, size in enumerate((50_000, 200_000)):
        state = state_with_logs(size)
        before = FALLBACK_CALLS
        expected = reference.fast_extract_trace(state)
        actual = candidate.fast_extract_trace(state)
        assert FALLBACK_CALLS == before, "representative data unexpectedly used fallback"
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        expected_bytes = parquet_bytes(expected)
        assert parquet_bytes(actual) == expected_bytes
        count = len(expected)
        digest = hashlib.sha256(expected_bytes).hexdigest()
        del expected, actual, expected_bytes
        record = {"input_logs": size, "trace_rows": count, "fast_path_used": True,
                  "parquet_sha256": digest, "pairs": []}
        records.append(record)
        report["microbenchmark"] = {"cases": records, "predeclared_gate": GATE,
                                    "whole_process_performance_claim": False, "screen_passed": None}
        save()
        for pair in range(3):
            order = ("v004", "v005") if (pair + size_index) % 2 == 0 else ("v005", "v004")
            times = {}
            for mode in order:
                gc.collect()
                function = reference.fast_extract_trace if mode == "v004" else candidate.fast_extract_trace
                start = time.perf_counter()
                result = function(state)
                elapsed = time.perf_counter() - start
                assert len(result) == count
                del result
                times[mode] = elapsed
            record["pairs"].append({"pair": pair, "order": list(order), "seconds": times,
                                    "extraction_rate_ratio_v005_over_v004": times["v004"] / times["v005"]})
            save()
        median = {mode: statistics.median(row["seconds"][mode] for row in record["pairs"])
                  for mode in ("v004", "v005")}
        ratio = median["v004"] / median["v005"]
        record.update(median_seconds=median, extraction_rate_ratio_v005_over_v004=ratio,
                      screen_passed=ratio >= 1.10)
        save()
    report["microbenchmark"]["screen_passed"] = all(row["screen_passed"] for row in records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--report", type=Path, default=ROOT / "reports/t3-v005-local-audit.json")
    args = parser.parse_args()
    assert candidate.sha256(ROOT / "t3/cpu_agent.py") == SELECTED_HASH
    assert candidate.sha256(ROOT / "t3/cpu_agent_v004.py") == SELECTED_HASH
    if args.report.exists():
        raise FileExistsError("Preserve the existing report; use a separately named report for any justified repair")
    process = psutil.Process()
    process.cpu_affinity([args.cpu])
    candidate._trace_frame_reference = tracked_fallback
    report = {"kind": "bounded_synthetic_full_trace_extraction_screen", "rankable": False,
              "status": "started", "cpu_only": True, "worker_processes": 1,
              "affinity": process.cpu_affinity(), "predeclared_gate": GATE,
              "numeric_threads": {name: candidate.os.environ[name] for name in
                                  ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "source_sha256": {"selected_v004": SELECTED_HASH,
                                "v005_candidate": candidate.sha256(ROOT / "t3/cpu_agent_v005_candidate.py"),
                                "verification": candidate.sha256(Path(__file__))},
              "environment": {"python": platform.python_version(), "system": platform.platform(),
                              "numpy": np.__version__, "pandas": pd.__version__, "pyarrow": pyarrow.__version__},
              "selection_changed": False,
              "limitations": "Synthetic local microtiming only; no markets, Linux throughput, official score or rank claim"}
    def save():
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save()
    try:
        report["source_scope"] = verify_source_scope()
        results = []
        report["parity"] = {"passed": False, "results": results}
        for name, state in parity_cases():
            results.append(compare(name, state))
        results.append(projected_row_fallback_once())
        report["parity"].update(passed=True, cases=len(results), exact_frames_dtypes_schema_parquet=True,
                               input_and_output_isolation=True, fallback_calls=FALLBACK_CALLS)
        save()
        print(json.dumps({"parity_passed": len(results), "fallback_calls": FALLBACK_CALLS}), flush=True)
        benchmark(report, save)
        report["status"] = "complete"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report["selected_source_unchanged"] = candidate.sha256(ROOT / "t3/cpu_agent.py") == SELECTED_HASH
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(json.dumps({"report": str(args.report), "status": report["status"],
                          "screen_passed": report.get("microbenchmark", {}).get("screen_passed")}), flush=True)


if __name__ == "__main__":
    main()
