"""Run a bounded public-only regression and compare every trace value to public references.

Exact table equality is a strong local semantic check, not official container acceptance
or leaderboard timing. This script is excluded from the submitted image.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time
import pandas as pd


def check_unit(unit: Path, output: Path, timeout: float) -> dict:
    started = time.perf_counter()
    dest = output / unit.name
    dest.mkdir(parents=True, exist_ok=True)
    record = {"unit": unit.name, "ok": False, "verification": "exact_public_trace_and_ledger_tables"}
    command = [sys.executable, str(Path(__file__).with_name("cpu_agent.py")), "simulate",
               "--config", str(unit / "scenario.json"), "--out", str(dest / "trace.parquet")]
    try:
        with (dest / "process.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=log, check=True, timeout=timeout)
        for name in ("trace.parquet", "message_trace.parquet"):
            reference = unit / name
            if not reference.exists():
                raise FileNotFoundError(f"Public reference unavailable: {reference}")
            candidate = pd.read_parquet(dest / name)
            expected = pd.read_parquet(reference)
            pd.testing.assert_frame_equal(candidate, expected, check_exact=True)
            record[name + "_rows"] = len(candidate)
        events = json.loads((dest / "events.json").read_text())
        assert events["n_events"] == record["trace.parquet_rows"]
        reference_events = json.loads((unit / "events.json").read_text())
        assert events["n_events"] == reference_events["n_events"]
        assert abs(events["events_per_sec"] * events["wall_clock_sec"] - events["n_events"]) < 1e-6
        record["agent_seconds"] = events["wall_clock_sec"]
        record["ok"] = True
    except Exception as exc:
        record["error"] = str(exc)[:4000]
    record["process_and_check_seconds"] = time.perf_counter() - started
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--units", nargs="*", help="Exact public unit directory names; default all singles")
    args = parser.parse_args()
    units = sorted(p.parent for p in (args.upstream / "units").glob("*/scenario.json"))
    if args.units:
        wanted = set(args.units)
        units = [u for u in units if u.name in wanted]
        if {u.name for u in units} != wanted:
            parser.error("Some named public units do not exist")
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    def save():
        report = {"verification": "Exact public reference table comparison; not official container acceptance",
                  "rankable": False, "cpu_only": True, "selected_units": len(units),
                  "completed": len(results), "passed": sum(r["ok"] for r in results),
                  "failed": sum(not r["ok"] for r in results), "elapsed_seconds": time.perf_counter() - started,
                  "results": sorted(results, key=lambda r: r["unit"])}
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=min(4, max(1, args.workers))) as executor:
        futures = {executor.submit(check_unit, unit, args.output, args.timeout): unit for unit in units}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result), flush=True)
            save()
    save()
    return 0 if results and all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
