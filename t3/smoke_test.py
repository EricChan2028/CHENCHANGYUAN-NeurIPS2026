"""Synthetic regression for real simulation, batch isolation and byte reproducibility.

These locally authored configs are independent of organizer references. The baseline
engine computes the comparison at test time; no precomputed answers are shipped.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import pandas as pd
from cpu_agent import simulate, simulate_batch, ScenarioLatencyModel, _ORIGINAL_LATENCY, scalar_latency, sha256
import numpy as np


def scenario(seed=7081):
    return {
        "scenario_id": "77300000-0000-4000-8000-000000000001", "seed": seed,
        "horizon_ns": 3_000_000_000,
        "exchange_config": {"symbol": "TEST", "protocol_enforcement": True,
                            "stp_policy": "cancel_oldest", "ack_delay_ns": 500, "compute_delay_ns": 80},
        "oracle_config": {"type": "mean_reverting", "params": {
            "initial_price": 100000, "kappa": 0.07, "sigma": 0.0004,
            "scheduled_jump": {"time_ns": 1_300_000_000, "magnitude": -150}}},
        "latency_config": {"model": "log_normal", "params": {
            "mean_ns": 1400, "sigma": 0.8, "min_ns": 80, "max_ns": 10000}},
        "agent_configs": [
            {"agent_type": "NoiseTrader", "count": 3, "params": {"arrival_rate_hz": 8.0}},
            {"agent_type": "MarketMaker", "count": 2, "params": {"rebalance_interval_ns": 250000000, "depth_levels": 4}},
            {"agent_type": "ValueTrader", "count": 2, "params": {"arrival_rate_hz": 4.0}},
            {"agent_type": "MomentumTrader", "count": 2, "params": {"arrival_rate_hz": 4.0, "lookback": 2}},
        ],
    }


def main():
    # Exercise the clipping/RNG optimization on all supported distributions, clipping
    # both tails and following the stream for thousands of draws.
    for model in ("uniform", "log_normal", "pareto", "deterministic"):
        cfg = {"model": model, "params": {"mean_ns": 2000, "sigma": 3.0, "min_ns": 37, "max_ns": 9000, "alpha": 1.2}}
        a = ScenarioLatencyModel(2, cfg, np.random.RandomState(17867))
        b = ScenarioLatencyModel(2, cfg, np.random.RandomState(17867))
        for _ in range(10000):
            assert _ORIGINAL_LATENCY(a, 0, 1) == scalar_latency(b, 0, 1)
        assert _ORIGINAL_LATENCY(a, 1, 1) == scalar_latency(b, 1, 1) == 0
    with tempfile.TemporaryDirectory(prefix="t3-synthetic-test-") as directory:
        root = Path(directory)
        configs = root / "scenarios"
        configs.mkdir()
        for index, seed in enumerate((7081, 9157, 12091)):
            cfg = scenario(seed)
            if index == 1:
                cfg["exchange_config"]["stp_policy"] = "cancel_newest"
                cfg["latency_config"]["model"] = "pareto"
            if index == 2:
                cfg["latency_config"]["model"] = "uniform"
                cfg["oracle_config"]["params"].pop("scheduled_jump")
            path = configs / f"market{index}.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            baseline = root / "baseline" / path.stem
            candidate = root / "candidate" / path.stem
            simulate(path, baseline / "trace.parquet", baseline=True)
            simulate(path, candidate / "trace.parquet")
            for name in ("trace.parquet", "message_trace.parquet"):
                pd.testing.assert_frame_equal(pd.read_parquet(baseline / name), pd.read_parquet(candidate / name), check_exact=True)
                assert sha256(baseline / name) == sha256(candidate / name)
        result = simulate_batch(configs, root / "batch", workers=2)
        assert result["n_scenarios"] == 3
        for path in configs.glob("*.json"):
            for name in ("trace.parquet", "message_trace.parquet"):
                assert sha256(root / "candidate" / path.stem / name) == sha256(root / "batch" / path.stem / name)
    print(json.dumps({"ok": True, "latency_draws_compared": 40000, "synthetic_markets": 3,
                      "baseline_exact_match": True, "repeat_byte_stability": True,
                      "batch_isolation_byte_match": True, "gpu_used": False}))


if __name__ == "__main__":
    main()
