"""Paired Linux CPU confirmation against frozen v003; this is not Final timing.

The same inspected image supplies both runs' dependencies and vendor tree. Only
the baseline's /app/cpu_agent.py is replaced by a read-only frozen source mount.
Every pair uses fresh offline containers and the same staged public scenario.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

from container_smoke import inspect_image, stage_inputs, verify_checkout, verify_output_sizes

CASES = (
    "t3-stp-cancel-newest",
    "t3-as06-throughput-fast",
    "t3-mp05-cancel-churn-newest",
    "t3-gb-horizon-240s",
)
MODES = ("v003", "v004")
PAIRS = 3
BASELINE_SHA256 = "b858c806e7a9209366a7b1a43718662305f619c227086b83ad33521500a61a6c"
CANDIDATE_SHA256 = "f04b88e6c25c15c029857d3071ec2ea8817a9a18ab3e4c942f746f61084b93ad"
GATES = {"equal_unit_mean_eps_ratio_min": 1.03,
         "larger_case_eps_ratio_min": 0.95,
         "small_case_eps_ratio_min": 0.90}
ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=void", "OMP_NUM_THREADS=1",
    "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1",
    "PYTHONDONTWRITEBYTECODE=1", "TMPDIR=/tmp",
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def schedule() -> list[dict]:
    """Six baseline-first and six candidate-first pairs across four cases."""
    rows = []
    for case_index, case in enumerate(CASES):
        for pair in range(PAIRS):
            order = MODES if (case_index + pair) % 2 == 0 else MODES[::-1]
            for mode in order:
                rows.append({"case": case, "pair": pair, "mode": mode})
    return rows


def docker_command(image: str, inputs: Path, outputs: Path, name: str,
                   baseline_source: Path | None = None) -> list[str]:
    argv = ["docker", "run", "--rm", "--name", name, "--pull", "never",
            "--platform", "linux/amd64", "--network", "none", "--read-only",
            "--cpus", "4", "--memory", "8g", "--memory-swap", "8g",
            "--pids-limit", "128", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,nosuid,size=64m",
            "--mount", f"type=bind,src={inputs.resolve()},dst=/input,readonly",
            "--mount", f"type=bind,src={outputs.resolve()},dst=/output"]
    if baseline_source is not None:
        argv += ["--mount", f"type=bind,src={baseline_source.resolve()},dst=/app/cpu_agent.py,readonly"]
    for setting in ENVIRONMENT:
        argv += ["--env", setting]
    return argv + [image, "simulate", "--config", "/input/scenario.json",
                   "--out", "/output/trace.parquet"]


def run_container(image: str, inputs: Path, outputs: Path, log: Path,
                  baseline_source: Path | None = None) -> float:
    outputs.mkdir(parents=True, exist_ok=False)
    outputs.chmod(0o777)
    log.parent.mkdir(parents=True, exist_ok=True)
    name = "agenthon-t3-pair-" + uuid.uuid4().hex
    argv = docker_command(image, inputs, outputs, name, baseline_source)
    try:
        with log.open("x", encoding="utf-8") as handle:
            started = time.perf_counter()
            subprocess.run(argv, check=True, stdout=handle, stderr=subprocess.STDOUT, timeout=300)
            elapsed = time.perf_counter() - started
    finally:
        # A timed-out Docker CLI need not stop its container. This cleanup is
        # outside the measured process window, which already includes --rm.
        subprocess.run(["docker", "rm", "--force", name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    if not math.isfinite(elapsed) or not 0 < elapsed <= 300:
        raise ValueError("run exceeded the 300-second process limit")
    return elapsed


def output_evidence(outputs: Path, scenario: dict) -> dict:
    import pyarrow.parquet as pq

    verify_output_sizes(outputs, "t3")
    events = json.loads((outputs / "events.json").read_text(encoding="utf-8"))
    trace = outputs / "trace.parquet"
    ledger = outputs / "message_trace.parquet"
    trace_rows = pq.ParquetFile(trace).metadata.num_rows
    ledger_rows = pq.ParquetFile(ledger).metadata.num_rows
    if (type(events.get("n_events")) is not int or events["n_events"] != trace_rows
            or trace_rows <= 0 or type(events.get("n_messages")) is not int
            or events["n_messages"] != ledger_rows):
        raise ValueError("reported counts disagree with actual parquet row counts")
    if (events.get("scenario_id") != str(scenario["scenario_id"])
            or events.get("seed") != scenario["seed"]):
        raise ValueError("output identifies a different scenario or seed")
    hashes = {"trace_sha256": sha256(trace), "message_trace_sha256": sha256(ledger)}
    if any(events.get(key) != value for key, value in hashes.items()):
        raise ValueError("reported hashes disagree with actual output bytes")
    wall = events.get("wall_clock_sec")
    rate = events.get("events_per_sec")
    if (isinstance(wall, bool) or not isinstance(wall, (int, float))
            or not math.isfinite(wall) or wall <= 0
            or isinstance(rate, bool) or not isinstance(rate, (int, float))
            or not math.isfinite(rate) or abs(rate - trace_rows / wall) > 0.05 * trace_rows / wall):
        raise ValueError("invalid or inconsistent self-reported timing")
    return {**hashes, "events": trace_rows, "messages": ledger_rows,
            "agent_seconds": wall}


def summarize(records: list[dict]) -> dict:
    expected = schedule()
    if len(records) != len(expected) or any(
            any(row.get(key) != want[key] for key in ("case", "pair", "mode"))
            for row, want in zip(records, expected)):
        raise ValueError("paired record roster/order is incomplete or changed")
    cases = []
    for case in CASES:
        rows = [row for row in records if row["case"] == case]
        bindings = {(r["trace_sha256"], r["message_trace_sha256"], r["events"], r["messages"])
                    for r in rows}
        if len(bindings) != 1:
            raise ValueError("trace, ledger or event count differs across versions/repeats")
        if any(not math.isfinite(r["process_seconds"]) or not 0 < r["process_seconds"] <= 300
               or type(r["events"]) is not int or r["events"] <= 0 for r in rows):
            raise ValueError("invalid benchmark rate inputs")
        medians = {mode: statistics.median(r["process_seconds"] for r in rows if r["mode"] == mode)
                   for mode in MODES}
        rates = {mode: rows[0]["events"] / medians[mode] for mode in MODES}
        ratio = rates["v004"] / rates["v003"]
        threshold = GATES["small_case_eps_ratio_min"] if case == CASES[0] else GATES["larger_case_eps_ratio_min"]
        cases.append({"case": case, "events": rows[0]["events"], "messages": rows[0]["messages"],
                      "stable_output_bytes": True, "median_process_seconds": medians,
                      "median_process_eps": rates, "eps_ratio": ratio,
                      "minimum_eps_ratio": threshold, "case_gate_passed": ratio >= threshold})
    means = {mode: statistics.mean(case["median_process_eps"][mode] for case in cases) for mode in MODES}
    ratio = means["v004"] / means["v003"]
    passed = ratio >= GATES["equal_unit_mean_eps_ratio_min"] and all(case["case_gate_passed"] for case in cases)
    return {"cases": cases, "equal_unit_mean_eps": means, "equal_unit_mean_eps_ratio": ratio,
            "promotion_passed": passed}


def write_report(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def benchmark(image: str, work: Path, report_path: Path, baseline_source: Path,
              candidate_source: Path) -> dict:
    if sys.platform != "linux":
        raise RuntimeError("this confirmation requires Linux and a working Docker daemon")
    work, report_path = work.resolve(), report_path.resolve()
    if not report_path.is_relative_to(work) or report_path.exists():
        raise ValueError("report must be a new file inside the work directory")
    checkout = work / "t3"
    verify_checkout("t3", checkout)
    detail = inspect_image(image)
    image_id = detail["Id"]
    if detail["Config"].get("Entrypoint") != ["python", "/app/cpu_agent.py"]:
        raise ValueError("expected the unmodified T3 Python entrypoint")
    for source in (baseline_source, candidate_source):
        if not source.is_file() or source.is_symlink() or source.is_junction():
            raise ValueError("source must be an ordinary source file")
    if sha256(baseline_source) != BASELINE_SHA256:
        raise ValueError("baseline source is not the frozen verified v003")
    candidate_hash = sha256(candidate_source)
    if candidate_hash != CANDIDATE_SHA256:
        raise ValueError("candidate source is not the frozen v004 under confirmation")
    # Verify the actual image contains the host candidate used to identify this
    # experiment. This read-only inspection is outside all timed paired runs.
    inspection_name = "agenthon-t3-source-" + uuid.uuid4().hex
    try:
        result = subprocess.run(
            ["docker", "run", "--rm", "--name", inspection_name, "--pull", "never",
             "--network", "none", "--read-only", "--entrypoint", "python", image_id, "-c",
             "import hashlib; print(hashlib.sha256(open('/app/cpu_agent.py','rb').read()).hexdigest())"],
            check=True, capture_output=True, text=True, timeout=30)
    finally:
        subprocess.run(["docker", "rm", "--force", inspection_name], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    if result.stdout.strip() != candidate_hash:
        raise ValueError("image candidate source does not match the reviewed host source")
    output_root = work / "t3-paired-benchmark"
    output_root.mkdir(parents=True, exist_ok=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    inputs, scenarios, config_hashes = {}, {}, {}
    for case in CASES:
        inputs[case] = stage_inputs(checkout / "units" / case, output_root / "inputs" / case,
                                   ("scenario.json",))
        config = inputs[case] / "scenario.json"
        scenarios[case] = json.loads(config.read_text(encoding="utf-8"))
        config_hashes[case] = sha256(config)
    report = {"schema_version": 1, "kind": "paired_linux_cpu_confirmation",
              "rankable": False, "official_final_timing": False, "cpu_only": True,
              "status": "running", "promotion_passed": False,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "image_id": image_id,
              "source_sha256": {"v003": BASELINE_SHA256, "v004": candidate_hash},
              "source_revision": os.environ.get("GITHUB_SHA"), "config_sha256": config_hashes,
              "protocol": {"pairs_per_case": PAIRS, "schedule": schedule(), "gates": GATES,
                           "timing": "Fresh Docker CLI whole-process elapsed time, including startup and --rm; no warm-up discarded",
                           "baseline": "Same image/entrypoint/vendor; frozen v003 file mounted read-only over /app/cpu_agent.py",
                           "case_metric": "events divided by median of three process times",
                           "aggregate": "equal arithmetic mean of four per-case EPS values",
                           "resources": "4 CPU quota, 8 GiB RAM/swap cap, 64 MiB tmpfs, no network/GPU, numeric threads=1, 300 seconds/run"},
              "limitations": "Four public cases on a shared CI host; diagnostic proxy, not the organizer Final clock or ranking.",
              "runs": []}
    write_report(report_path, report)  # Protocol is recorded before the first timed run.
    try:
        for row in schedule():
            case, pair, mode = row["case"], row["pair"], row["mode"]
            if mode == "v003" and sha256(baseline_source) != BASELINE_SHA256:
                raise ValueError("frozen baseline changed during the benchmark")
            output = output_root / "outputs" / case / f"{pair}-{mode}"
            log = output_root / "logs" / case / f"{pair}-{mode}.log"
            seconds = run_container(image_id, inputs[case], output, log,
                                    baseline_source if mode == "v003" else None)
            evidence = output_evidence(output, scenarios[case])
            # Fail immediately on a semantic mismatch rather than finish timing
            # an invalid candidate and possibly select a favourable subset.
            previous = [x for x in report["runs"] if x["case"] == case]
            binding_keys = ("trace_sha256", "message_trace_sha256", "events", "messages")
            if previous and any(previous[0][key] != evidence[key] for key in binding_keys):
                raise ValueError("trace, ledger or counts differ within a paired case")
            measured = {**row, **evidence, "process_seconds": seconds}
            report["runs"].append(measured)
            write_report(report_path, report)
            print(json.dumps({"paired_run": measured}, allow_nan=False), flush=True)
        report.update(summarize(report["runs"]))
        report["status"] = "complete"
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(report_path, report)
        return report
    except Exception as error:
        report.update(status="failed", promotion_passed=False, failure_type=type(error).__name__,
                      failure_message=str(error), finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_report(report_path, report)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        raise


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--baseline-source", type=Path, default=root / "t3/cpu_agent_v003.py")
    parser.add_argument("--candidate-source", type=Path, default=root / "t3/cpu_agent.py")
    args = parser.parse_args()
    path = args.report or args.work / "t3-paired-report.json"
    report = benchmark(args.image, args.work, path, args.baseline_source, args.candidate_source)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if report["promotion_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
