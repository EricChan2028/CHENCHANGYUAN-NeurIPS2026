"""Paired CPU timings with exact-output verification; never a leaderboard predictor."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    agent = Path(__file__).with_name("cpu_agent.py")
    rows = []
    hashes = set()
    for repeat in range(args.repeats):
        for baseline in ([True, False] if repeat % 2 == 0 else [False, True]):
            name = "baseline" if baseline else "candidate"
            output = args.output / f"{name}-{repeat}"
            output.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(agent), "simulate", "--config", str(args.config),
                       "--out", str(output / "trace.parquet")]
            if baseline:
                command.append("--baseline")
            started = time.perf_counter()
            with (args.output / f"{name}-{repeat}.log").open("w", encoding="utf-8") as log:
                subprocess.run(command, check=True, timeout=300, stdout=log, stderr=log)
            elapsed = time.perf_counter() - started
            events = json.loads((output / "events.json").read_text())
            hashes.add((events["trace_sha256"], events["message_trace_sha256"]))
            row = {"mode": name, "repeat": repeat, "process_seconds": elapsed,
                   "agent_seconds": events["wall_clock_sec"], "events": events["n_events"]}
            rows.append(row)
            print(json.dumps(row), flush=True)
    medians = {mode: statistics.median(row["process_seconds"] for row in rows if row["mode"] == mode)
               for mode in ("baseline", "candidate")}
    report = {"rankable": False, "environment": "local Windows CPU, includes Python process startup; shared desktop load",
              "reference": "Pinned ABIDES engine and original organizer latency/trace extraction",
              "candidate_source_sha256": hashlib.sha256(agent.read_bytes()).hexdigest(),
              "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
              "repeats": args.repeats, "stable_output_bytes": len(hashes) == 1,
              "median_process_seconds": medians, "local_speedup": medians["baseline"] / medians["candidate"],
              "runs": rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0 if len(hashes) == 1 else 1


if __name__ == "__main__":
    raise SystemExit(main())
