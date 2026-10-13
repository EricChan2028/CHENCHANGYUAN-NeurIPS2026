"""Alternate frozen submitted v001 and v002 in fresh whole Python processes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from smoke_test import scenario


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sources = {"v001": root / "cpu_agent_v001.py", "v002": root / "cpu_agent.py"}
    assert digest(sources["v001"]) == "2a58e80cb2a502817e38420995fe700da4737e05c81ce510ba495514226ad309"
    args.output.mkdir(parents=True, exist_ok=True)
    small_config = args.output / "fresh-small.json"
    small_config.write_text(json.dumps(scenario(seed=941083)), encoding="utf-8")
    cases = ["fresh-small", "t3-as06-throughput-fast", "t3-mp05-cancel-churn-newest", "t3-gb-horizon-240s"]
    records = []
    for unit in cases:
        config = small_config if unit == "fresh-small" else args.upstream / "units" / unit / "scenario.json"
        rows, hashes = [], set()
        for repeat in range(args.repeats):
            for mode in (("v001", "v002") if repeat % 2 == 0 else ("v002", "v001")):
                output = args.output / unit / f"{mode}-{repeat}"
                output.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(sources[mode]), "simulate", "--config", str(config),
                           "--out", str(output / "trace.parquet")]
                started = time.perf_counter()
                with (output / "process.log").open("w", encoding="utf-8") as log:
                    subprocess.run(command, check=True, timeout=300, stdout=log, stderr=log)
                elapsed = time.perf_counter() - started
                events = json.loads((output / "events.json").read_text())
                hashes.add((events["trace_sha256"], events["message_trace_sha256"]))
                row = {"mode": mode, "repeat": repeat, "process_seconds": elapsed,
                       "agent_seconds": events["wall_clock_sec"], "events": events["n_events"],
                       "peak_memory_bytes": events["peak_memory_bytes"]}
                rows.append(row)
                print(json.dumps({"case": unit, **row}), flush=True)
        medians = {mode: statistics.median(row["process_seconds"] for row in rows if row["mode"] == mode)
                   for mode in sources}
        assert len(hashes) == 1, f"Nondeterministic or unequal output: {unit}"
        records.append({"case": unit, "config_sha256": digest(config), "stable_output_bytes": True,
                        "median_process_seconds": medians, "speedup": medians["v001"] / medians["v002"],
                        "runs": rows})
        report = {"rankable": False, "cpu_only": True,
                  "environment": "Local shared Windows CPU; sequential paired fresh processes including startup",
                  "reference": "Frozen prior submitted v001 source, including its initial optimizations",
                  "source_sha256": {mode: digest(path) for mode, path in sources.items()},
                  "repeats_per_case": args.repeats, "cases": records}
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
