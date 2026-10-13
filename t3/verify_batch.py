"""Verify all public batch scenarios against published isolated traces and ledgers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for batch_path in sorted((args.upstream / "units").glob("*/batch.json")):
        unit = batch_path.parent
        dest = args.output / unit.name
        row = {"unit": unit.name, "ok": False}
        started = time.perf_counter()
        try:
            cmd = [sys.executable, str(Path(__file__).with_name("cpu_agent.py")), "simulate-batch",
                   "--batch-dir", str(unit / "scenarios"), "--out-dir", str(dest), "--workers", str(args.workers)]
            with (args.output / (unit.name + ".log")).open("w", encoding="utf-8") as log:
                subprocess.run(cmd, check=True, timeout=300, stdout=log, stderr=log)
            subs = json.loads(batch_path.read_text())["subs"]
            count = 0
            for sub in subs:
                for name in ("trace.parquet", "message_trace.parquet"):
                    expected = pd.read_parquet(unit / "checks" / "reference_data" / sub["sub"] / name)
                    candidate = pd.read_parquet(dest / sub["sub"] / name)
                    pd.testing.assert_frame_equal(candidate, expected, check_exact=True)
                    if name == "trace.parquet":
                        count += len(candidate)
            aggregate = json.loads((dest / "batch_events.json").read_text())
            assert aggregate["total_events"] == count == sum(s["n_events"] for s in subs)
            assert aggregate["n_scenarios"] == len(subs)
            assert abs(aggregate["events_per_sec"] * aggregate["wall_clock_sec"] - count) < 1e-6
            row.update(ok=True, subscenarios=len(subs), events=count)
        except Exception as exc:
            row["error"] = str(exc)[:4000]
        row["process_and_check_seconds"] = time.perf_counter() - started
        results.append(row)
        report = {"verification": "Exact public isolated reference tables, counts and aggregate timing consistency",
                  "rankable": False, "passed": sum(r["ok"] for r in results),
                  "failed": sum(not r["ok"] for r in results), "results": results}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(row), flush=True)
    return 0 if results and all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
