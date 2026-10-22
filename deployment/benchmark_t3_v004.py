"""Fixed 24-run Linux confirmation of v004 over the verified v003 CPU image.

This never builds/publishes an image or changes the selected runtime. Expected
output bindings stay on the host; containers receive public scenario JSON only.
Whole Docker process times are diagnostics, not organizer Final scores.
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
from urllib.request import urlopen
import uuid

IMAGE = "ghcr.io/ericchan2028/eastern-eagles-t3@sha256:43540c8bcf3ccef55d635d46a2caa107d034ee4d8ebb89e1a84021c08a6d9fe9"
IMAGE_ID = "sha256:d1493d0f5bfefe897ef70013683d0b0b76272c250a4698bca475ee22d825aeba"
RUNTIME_REVISION = "6f8891d54a7a26df0137d07e043a3ac7f32b84dd"
BASELINE_SHA256 = "b858c806e7a9209366a7b1a43718662305f619c227086b83ad33521500a61a6c"
CANDIDATE_SHA256 = "f04b88e6c25c15c029857d3071ec2ea8817a9a18ab3e4c942f746f61084b93ad"
PUBLIC_COMMIT = "2fa71ea2468db98bb1b1d06339a87d7e3a9d651c"
MODES = ("v003", "v004")
PAIRS = 3
OUTPUT_CAP = 256 * 1024 * 1024
BINDING_KEYS = ("trace_sha256", "message_trace_sha256", "events", "messages")
GATES = {"equal_unit_mean_eps_ratio_min": 1.03,
         "larger_case_eps_ratio_min": 0.95, "small_case_eps_ratio_min": 0.90}
CASES = {
    "t3-stp-cancel-newest": {
        "scenario_sha256": "42d8f6f9e6413d4bc02949d5dfec2a2ef0f181a10eb9984108859a43a6d80c1b",
        "trace_sha256": "7c5cb6651c1d02facd446c2cae21d4001b54e5f8ef3cb9c11c8e0132fb657c8b",
        "message_trace_sha256": "07fcbbb977151cd8cd6ce3782be5033812b9dff1f22318123417acc04f2bd442",
        "events": 72048, "messages": 63590},
    "t3-as06-throughput-fast": {
        "scenario_sha256": "15a2bc778742e778ef895cc5a905483bc967d3857f9c9219b1b87ad34e82ad5e",
        "trace_sha256": "1e7761ee74ade7e23bb4f40410f8525705601f10cc9e6e855d5df798c3e94350",
        "message_trace_sha256": "16f64ee89dd18988792dadc8c6ac3b02881e3d29fc5c80d4df19ae14fa738f77",
        "events": 74502, "messages": 83337},
    "t3-mp05-cancel-churn-newest": {
        "scenario_sha256": "0af9f298f191c4a33982852cf344b5dbe81475846beb274ca56a92f8c95a9183",
        "trace_sha256": "f83b6143c66629a0e2887ca5944b3360cd3bcb7c0b46e57de763f192452f6c14",
        "message_trace_sha256": "8342cfb62afbbccf6ae765d9c36493096bfaa265e2c7dfd873cc8b0c40426afb",
        "events": 240829, "messages": 205166},
    "t3-gb-horizon-240s": {
        "scenario_sha256": "650e0d29b40f3709f224a6af278b9e0cc9899294bcdfa31deb4f95bb5e84dc8b",
        "trace_sha256": "8423a8c5da94afce610421456ed56488c42cf4d398e1fa656c9d427880eb9df6",
        "message_trace_sha256": "6ed82dcbf48f278e3d6971fe1f50be011d02163bfc754f0aecfb639c711b2059",
        "events": 555032, "messages": 692641},
}
ENVIRONMENT = ("CUDA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=void", "OMP_NUM_THREADS=1",
               "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1",
               "PYTHONDONTWRITEBYTECODE=1", "TMPDIR=/tmp")


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def schedule():
    rows = []
    for case_index, case in enumerate(CASES):
        for pair in range(PAIRS):
            for mode in (MODES if (case_index + pair) % 2 == 0 else MODES[::-1]):
                rows.append({"case": case, "pair": pair, "mode": mode})
    return rows


def container_prefix(mode, candidate_source, name):
    if mode not in MODES:
        raise ValueError("unknown benchmark mode")
    args = ["docker", "run", "--rm", "--name", name, "--pull", "never", "--platform", "linux/amd64",
            "--network", "none", "--read-only", "--cpus", "4", "--memory", "8g", "--memory-swap", "8g",
            "--pids-limit", "128", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,nosuid,size=64m"]
    if mode == "v004":
        args += ["--mount", f"type=bind,src={candidate_source.resolve()},dst=/app/cpu_agent.py,readonly"]
    for setting in ENVIRONMENT:
        args += ["--env", setting]
    return args


def docker_command(mode, candidate_source, inputs, outputs, name):
    return container_prefix(mode, candidate_source, name) + [
        "--mount", f"type=bind,src={inputs.resolve()},dst=/input,readonly",
        "--mount", f"type=bind,src={outputs.resolve()},dst=/output",
        IMAGE_ID, "simulate", "--config", "/input/scenario.json", "--out", "/output/trace.parquet"]


def cleanup(name):
    subprocess.run(["docker", "rm", "--force", name], check=False, timeout=15,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def check_image(detail):
    config = detail.get("Config") or {}
    if (detail.get("Id") != IMAGE_ID or detail.get("Os") != "linux" or detail.get("Architecture") != "amd64"
            or IMAGE not in detail.get("RepoDigests", []) or config.get("Volumes")
            or config.get("Entrypoint") != ["python", "/app/cpu_agent.py"]
            or (config.get("Labels") or {}).get("org.opencontainers.image.revision") != RUNTIME_REVISION):
        raise ValueError("image identity/config differs from the verified v003 runtime")


def inspect_sources(candidate_source):
    for mode, expected in (("v003", BASELINE_SHA256), ("v004", CANDIDATE_SHA256)):
        name = "agenthon-v004-inspect-" + uuid.uuid4().hex
        args = container_prefix(mode, candidate_source, name) + ["--entrypoint", "python", IMAGE_ID, "-c",
                "import hashlib; print(hashlib.sha256(open('/app/cpu_agent.py','rb').read()).hexdigest())"]
        try:
            result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=30)
        finally:
            cleanup(name)
        if result.stdout.strip() != expected:
            raise ValueError("actual container source differs from frozen " + mode)


def run_container(mode, candidate_source, inputs, outputs, log):
    outputs.mkdir(parents=True, exist_ok=False)
    outputs.chmod(0o777)
    log.parent.mkdir(parents=True, exist_ok=True)
    name = "agenthon-v004-pair-" + uuid.uuid4().hex
    command = docker_command(mode, candidate_source, inputs, outputs, name)
    try:
        with log.open("x", encoding="utf-8") as handle:
            start = time.perf_counter()
            subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, timeout=300)
            elapsed = time.perf_counter() - start
    finally:
        cleanup(name)  # Timeout cleanup is outside the measured Docker --rm interval.
    if not math.isfinite(elapsed) or not 0 < elapsed <= 300:
        raise ValueError("invalid or excessive whole-process time")
    if log.stat().st_size > 1024 * 1024:
        raise ValueError("candidate console output exceeded 1 MiB")
    return elapsed


def output_evidence(outputs, scenario):
    import pyarrow.parquet as pq
    files = list(outputs.iterdir())
    if ({p.name for p in files} != {"trace.parquet", "message_trace.parquet", "events.json"}
            or any(p.is_symlink() or not p.is_file() for p in files)):
        raise ValueError("unexpected output members")
    size = sum(p.stat().st_size for p in files)
    if size > OUTPUT_CAP:
        raise ValueError("output tree exceeds 256 MiB")
    events = json.loads((outputs / "events.json").read_text(encoding="utf-8"))
    counts = {"events": pq.ParquetFile(outputs / "trace.parquet").metadata.num_rows,
              "messages": pq.ParquetFile(outputs / "message_trace.parquet").metadata.num_rows}
    hashes = {"trace_sha256": sha256(outputs / "trace.parquet"),
              "message_trace_sha256": sha256(outputs / "message_trace.parquet")}
    if (type(events.get("n_events")) is not int or events["n_events"] != counts["events"]
            or type(events.get("n_messages")) is not int or events["n_messages"] != counts["messages"]
            or counts["events"] <= 0 or any(events.get(k) != v for k, v in hashes.items())
            or events.get("scenario_id") != str(scenario["scenario_id"])
            or events.get("seed") != scenario["seed"] or events.get("gpu_seconds") != 0):
        raise ValueError("sidecar differs from scenario or actual output bytes/counts")
    wall, rate = events.get("wall_clock_sec"), events.get("events_per_sec")
    if (type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0
            or type(rate) not in (int, float) or not math.isfinite(rate)
            or abs(rate - counts["events"] / wall) > 0.05 * counts["events"] / wall):
        raise ValueError("invalid self-reported timing")
    return {**hashes, **counts, "agent_seconds": wall, "output_bytes": size}


def validate_bindings(case, evidence):
    if any(evidence.get(key) != CASES[case][key] for key in BINDING_KEYS):
        raise ValueError("output differs from verified v003 Linux bindings")
    if type(evidence.get("events")) is not int or type(evidence.get("messages")) is not int:
        raise ValueError("noninteger output counts")


def summarize(records):
    expected = schedule()
    if len(records) != len(expected) or any(
            any(row.get(key) != want[key] for key in ("case", "pair", "mode"))
            for row, want in zip(records, expected)):
        raise ValueError("paired run roster/order is incomplete or changed")
    cases = []
    for case in CASES:
        rows = [row for row in records if row["case"] == case]
        for row in rows:
            validate_bindings(case, row)
            seconds = row.get("process_seconds")
            if type(seconds) not in (int, float) or not math.isfinite(seconds) or not 0 < seconds <= 300:
                raise ValueError("invalid benchmark time")
        medians = {mode: statistics.median(row["process_seconds"] for row in rows if row["mode"] == mode)
                   for mode in MODES}
        rates = {mode: rows[0]["events"] / medians[mode] for mode in MODES}
        ratio = rates["v004"] / rates["v003"]
        threshold = GATES["small_case_eps_ratio_min"] if case == next(iter(CASES)) else GATES["larger_case_eps_ratio_min"]
        cases.append({"case": case, "events": rows[0]["events"], "messages": rows[0]["messages"],
                      "stable_output_bytes": True, "median_process_seconds": medians,
                      "median_process_eps": rates, "eps_ratio": ratio,
                      "minimum_eps_ratio": threshold, "case_gate_passed": ratio >= threshold})
    means = {mode: statistics.mean(case["median_process_eps"][mode] for case in cases) for mode in MODES}
    ratio = means["v004"] / means["v003"]
    return {"cases": cases, "equal_unit_mean_eps": means, "equal_unit_mean_eps_ratio": ratio,
            "promotion_passed": ratio >= GATES["equal_unit_mean_eps_ratio_min"] and all(c["case_gate_passed"] for c in cases)}


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def benchmark(work, candidate_source):
    if sys.platform != "linux":
        raise RuntimeError("Linux and Docker are required")
    if not candidate_source.is_file() or candidate_source.is_symlink():
        raise ValueError("candidate must be an ordinary source file")
    candidate_source = candidate_source.resolve()
    if sha256(candidate_source) != CANDIDATE_SHA256:
        raise ValueError("candidate file is not the frozen reviewed v004")
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    report_path = work / "t3-v004-paired-report.json"
    report = {"schema_version": 1, "kind": "paired_linux_v004_confirmation", "rankable": False,
              "official_final_timing": False, "cpu_only": True, "status": "preparing", "promotion_passed": False,
              "image": IMAGE, "image_id": IMAGE_ID, "runtime_revision": RUNTIME_REVISION,
              "experiment_revision": os.environ.get("GITHUB_SHA"), "official_public_commit": PUBLIC_COMMIT,
              "source_sha256": {"v003": BASELINE_SHA256, "v004": CANDIDATE_SHA256},
              "started_at_utc": datetime.now(timezone.utc).isoformat(),
              "protocol": {"pairs_per_case": PAIRS, "schedule": schedule(), "gates": GATES,
                           "timing": "Fresh Docker CLI whole-process time including startup and --rm; no discarded warm-up",
                           "baseline": "Unmodified verified v003 image /app/cpu_agent.py",
                           "candidate": "Same image/entrypoint/dependencies; frozen v004 source mounted read-only over /app/cpu_agent.py",
                           "case_metric": "events divided by median of three process times",
                           "aggregate": "ratio of equal arithmetic means of four per-case EPS values (same prior promotion gate)",
                           "resources": "4 CPU,8 GiB memory/swap,64 MiB temp,300s/run,threads1,offline/read-only,no GPU",
                           "output_limit": "256 MiB tree checked immediately after each run",
                           "binding_provenance": "Verified v003 Linux confirmation39745316032 attempt2; host-only expected values"},
              "expected_bindings": CASES, "runs": [],
              "limitations": "Four public cases on one shared Linux CI host; diagnostic proxy, not official Final timing/rank"}
    write_report(report_path, report)
    print(json.dumps({"v004_protocol": report}, separators=(",", ":")), flush=True)
    try:
        subprocess.run(["docker", "pull", "--platform", "linux/amd64", IMAGE], check=True, timeout=120, stdout=subprocess.DEVNULL)
        detail = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE], text=True, timeout=30))[0]
        check_image(detail)
        inspect_sources(candidate_source)
        inputs, scenarios = {}, {}
        for case, binding in CASES.items():
            url = f"https://raw.githubusercontent.com/Agenthon-2026/track3-simulation-public/{PUBLIC_COMMIT}/units/{case}/scenario.json"
            with urlopen(url, timeout=15) as response:
                raw = response.read(65537)
            if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != binding["scenario_sha256"]:
                raise ValueError("public scenario bytes differ from fixed official binding")
            inputs[case] = work / "inputs" / case
            inputs[case].mkdir(parents=True)
            (inputs[case] / "scenario.json").write_bytes(raw)
            scenarios[case] = json.loads(raw)
        report["status"] = "running"
        write_report(report_path, report)
        for row in schedule():
            case, pair, mode = row["case"], row["pair"], row["mode"]
            if sha256(candidate_source) != CANDIDATE_SHA256:
                raise ValueError("frozen candidate changed during experiment")
            if sha256(inputs[case] / "scenario.json") != CASES[case]["scenario_sha256"]:
                raise ValueError("public scenario changed during experiment")
            output = work / "outputs" / case / f"{pair}-{mode}"
            seconds = run_container(mode, candidate_source, inputs[case], output, work / "logs" / case / f"{pair}-{mode}.log")
            evidence = output_evidence(output, scenarios[case])
            validate_bindings(case, evidence)  # Fail immediately; never select a successful subset.
            measured = {**row, **evidence, "process_seconds": seconds}
            report["runs"].append(measured)
            write_report(report_path, report)
            print(json.dumps({"v004_paired_run": measured}, separators=(",", ":"), allow_nan=False), flush=True)
        report.update(summarize(report["runs"]))
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", promotion_passed=False, failure_type=type(error).__name__, failure_message=str(error))
        raise
    finally:
        report["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(report_path, report)
        print("T3_V004_REPORT " + json.dumps(report, separators=(",", ":"), allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path(".v004-confirmation"))
    parser.add_argument("--candidate-source", type=Path,
                        default=Path(__file__).resolve().parents[1] / "t3/cpu_agent_v004_candidate.py")
    args = parser.parse_args()
    report = benchmark(args.work, args.candidate_source)
    return 0 if report["promotion_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
