"""Alternate frozen submitted v002 and v003 in fresh whole Python processes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import psutil
from smoke_test import scenario


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pin-cpu", type=int, help="Use the same permitted logical CPU for both child processes")
    args = parser.parse_args()
    if args.pin_cpu is not None and args.pin_cpu not in psutil.Process().cpu_affinity():
        parser.error("Requested CPU is outside the current process affinity")
    root = Path(__file__).resolve().parent
    sources = {"v002": root / "cpu_agent_v002.py", "v003": root / "cpu_agent.py"}
    assert digest(sources["v002"]) == "d1b832f3e9fc3697da0c4b7d5620f8cfbef85fee7d624d85675730f8beb546fe"
    args.output.mkdir(parents=True, exist_ok=True)
    small_config = args.output / "fresh-small.json"
    small_config.write_text(json.dumps(scenario(seed=276499)), encoding="utf-8")
    cases = ["fresh-small", "t3-as06-throughput-fast", "t3-mp05-cancel-churn-newest", "t3-gb-horizon-240s"]
    records = []
    for unit in cases:
        config = small_config if unit == "fresh-small" else args.upstream / "units" / unit / "scenario.json"
        rows, hashes = [], set()
        for repeat in range(args.repeats):
            for mode in (("v002", "v003") if repeat % 2 == 0 else ("v003", "v002")):
                output = args.output / unit / f"{mode}-{repeat}"
                output.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(sources[mode]), "simulate", "--config", str(config),
                           "--out", str(output / "trace.parquet")]
                if args.pin_cpu is not None:
                    bootstrap = ("import sys,psutil,runpy; "
                                 f"psutil.Process().cpu_affinity([{args.pin_cpu}]); "
                                 "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0],run_name='__main__')")
                    command = [sys.executable, "-c", bootstrap] + command[1:]
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
                        "median_process_seconds": medians, "speedup": medians["v002"] / medians["v003"],
                        "runs": rows})
        report = {"rankable": False, "cpu_only": True,
                  "environment": "Local shared Windows CPU; sequential paired fresh processes including startup",
                  "reference": "Frozen prior submitted v002 source, including its initial optimizations",
                  "cpu_affinity": args.pin_cpu,
                  "source_sha256": {mode: digest(path) for mode, path in sources.items()},
                  "repeats_per_case": args.repeats, "cases": records}
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
