"""Fresh-input semantic tests for the v002 hotpaths; excluded from the image.

Checks optimized primitives independently against the pinned engine, then runs new
seed/config combinations against that engine and through isolated-equivalent batch.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import queue
import random
import tempfile

import numpy as np
import pandas as pd

from cpu_agent import (KernelHeapQueue, cached_fmt_ts, _ORIGINAL_FMT_TS,
                       snapshot_copy, sha256, simulate, simulate_batch, Side)
from smoke_test import scenario
from abides_core.message import Message, WakeupMsg, MessageBatch


def verify_primitives():
    rng = random.Random(77320261013)
    timestamps = [rng.randrange(-8_000_000_000_000_000_000, 8_000_000_000_000_000_000)
                  for _ in range(3000)]
    timestamps += [pd.Timestamp.min.value, pd.Timestamp.min.value + 1,
                   pd.Timestamp.max.value, -1, 0, 1,
                   -1_000_000_001, -1_000_000_000, -999_999_999,
                   999_999_999, 1_000_000_000, 1_000_000_001]
    for timestamp in timestamps:
        for value in (timestamp, np.int64(timestamp)):
            assert cached_fmt_ts(value) == _ORIGINAL_FMT_TS(value)
    for value in (True, False, pd.NaT):
        results = []
        for formatter in (_ORIGINAL_FMT_TS, cached_fmt_ts):
            try:
                results.append(("returned", formatter(value)))
            except Exception as exc:
                results.append(("raised", type(exc).__name__, str(exc)))
        assert results[0] == results[1]
    normal, optimized = queue.PriorityQueue(), KernelHeapQueue()
    operations = 0
    for index in range(20_000):
        if normal.empty() or rng.random() < 0.6:
            # Many exact timestamp/agent ties ensure no tie-breaking substitution.
            event = (rng.randrange(20), (rng.randrange(4), rng.randrange(4), index))
            normal.put(event)
            optimized.put(event)
        else:
            assert normal.get() == optimized.get()
        assert normal.empty() == optimized.empty()
        assert normal.qsize() == optimized.qsize()
        operations += 1
    while not normal.empty():
        assert normal.get() == optimized.get()
    assert optimized.empty()
    message_queue, message_heap = queue.PriorityQueue(), KernelHeapQueue()
    entries = []
    for index in range(400):
        message = (Message, WakeupMsg, lambda: MessageBatch([Message(), WakeupMsg()]))[index % 3]()
        entries.append((rng.randrange(5), (rng.randrange(3), rng.randrange(3), message)))
    entries += entries[:30]
    rng.shuffle(entries)
    for entry in entries:
        message_queue.put(entry)
        message_heap.put(entry)
    while not message_queue.empty():
        assert message_queue.get() is message_heap.get()
    assert message_heap.empty()
    flat = {"order_id": 2, "quantity": 12, "price": 99543, "side": Side.BID,
            "fill_price": None, "time_placed": "2021-02-05 09:30:04"}
    copied = snapshot_copy(flat)
    assert copied == copy.deepcopy(flat) and copied is not flat
    for key in flat:
        assert copied[key] is flat[key]
    flat["quantity"] = 97
    assert copied["quantity"] == 12
    shared = [3, {"state": 4}]
    nested = {"a": shared, "b": shared}
    nested_copy = snapshot_copy(nested)
    assert nested_copy == copy.deepcopy(nested)
    assert nested_copy["a"] is nested_copy["b"]
    assert nested_copy["a"] is not shared
    shared[1]["state"] = 999
    assert nested_copy["a"][1]["state"] == 4
    cycle = {}
    cycle["self"] = cycle
    cycle_copy = snapshot_copy(cycle)
    assert cycle_copy is cycle_copy["self"] and cycle_copy is not cycle
    class MutableInt(int):
        pass
    custom = MutableInt(7)
    custom.state = [8]
    custom_copy = snapshot_copy({"tag": custom})
    assert custom_copy["tag"] is not custom
    assert custom_copy["tag"].state is not custom.state
    return {"timestamps_compared": len(timestamps) * 2,
            "invalid_timestamp_exception_parity": True, "actual_message_ties": len(entries),
            "queue_operations_compared": operations,
            "snapshot_alias_mutability_cycle_and_subclass_checks": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    primitives = verify_primitives()
    results = []
    with tempfile.TemporaryDirectory(prefix="t3-v002-fresh-") as directory:
        root = Path(directory)
        configs = root / "configs"
        configs.mkdir()
        # These seeds are distinct from the initial smoke and all timing scenarios.
        for index, seed in enumerate((837721, 402709, 971003, 251701, 677011, 312647, 732421, 992681)):
            cfg = scenario(seed)
            cfg["scenario_id"] = f"77300200-0000-4000-8000-{index + 1:012d}"
            cfg["horizon_ns"] = (4 + index % 3) * 1_000_000_000
            cfg["exchange_config"].update(stp_policy=("cancel_oldest" if index % 2 else "cancel_newest"),
                                           ack_delay_ns=311 + index * 197, compute_delay_ns=43 + index * 29)
            cfg["latency_config"]["model"] = ("deterministic", "uniform", "log_normal", "pareto")[index % 4]
            cfg["latency_config"]["params"].update(alpha=1.2 + index / 10, sigma=0.2 + index / 5)
            cfg["oracle_config"]["params"]["sigma"] = 0.0001 + index * 0.0001
            if index % 3 == 0:
                cfg["oracle_config"]["params"].pop("scheduled_jump")
            else:
                cfg["oracle_config"]["params"]["scheduled_jump"] = {
                    "time_ns": 1_230_000_000 + index * 99_000_000, "magnitude": 120 * (-1) ** index}
            for agent_index, agent_cfg in enumerate(cfg["agent_configs"]):
                agent_cfg["count"] += (index + agent_index) % 3
            cfg["agent_configs"][1]["params"].update(depth_levels=2 + index % 5,
                                                      rebalance_interval_ns=71_000_000 + index * 11_000_000)
            path = configs / f"fresh-{index}.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            baseline = root / "baseline" / path.stem
            candidate = root / "candidate" / path.stem
            baseline_events = simulate(path, baseline / "trace.parquet", baseline=True)
            candidate_events = simulate(path, candidate / "trace.parquet")
            assert candidate_events["n_events"] == baseline_events["n_events"]
            assert candidate_events["n_messages"] == baseline_events["n_messages"]
            for name in ("trace.parquet", "message_trace.parquet"):
                pd.testing.assert_frame_equal(pd.read_parquet(baseline / name),
                                              pd.read_parquet(candidate / name), check_exact=True)
                assert sha256(baseline / name) == sha256(candidate / name)
            results.append({"seed": seed, "config_sha256": sha256(path),
                            "events": candidate_events["n_events"], "exact_and_byte_equal": True})
        for workers in (1, 2):
            result = simulate_batch(configs, root / f"batch-{workers}", workers=workers)
            assert result["n_scenarios"] == len(results)
            for path in configs.glob("*.json"):
                for name in ("trace.parquet", "message_trace.parquet"):
                    assert sha256(root / "candidate" / path.stem / name) == sha256(root / f"batch-{workers}" / path.stem / name)
    report = {"ok": True, "cpu_only": True, "rankable": False,
              "source_sha256": sha256(Path(__file__).with_name("cpu_agent.py")),
              "primitives": primitives, "fresh_synthetic_cases": results,
              "baseline": "Unmodified pinned ABIDES and official adapter, computed at test time",
              "batch_workers": [1, 2], "batch_isolation_and_repeat_byte_equal": True}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
