"""Bounded diagnostic profiling of the already verified T3 v003 CPU image.

No candidate code is built or edited. Only public scenario JSON and a generic
profiling worker are mounted; expected output bindings remain on the host.
Profiled times are instrumented diagnostics, never throughput or ranking claims.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
from urllib.request import urlopen

IMAGE = "ghcr.io/ericchan2028/eastern-eagles-t3@sha256:43540c8bcf3ccef55d635d46a2caa107d034ee4d8ebb89e1a84021c08a6d9fe9"
IMAGE_ID = "sha256:d1493d0f5bfefe897ef70013683d0b0b76272c250a4698bca475ee22d825aeba"
SOURCE_SHA256 = "b858c806e7a9209366a7b1a43718662305f619c227086b83ad33521500a61a6c"
RUNTIME_REVISION = "6f8891d54a7a26df0137d07e043a3ac7f32b84dd"
PUBLIC_COMMIT = "2fa71ea2468db98bb1b1d06339a87d7e3a9d651c"
OUTPUT_CAP = 256 * 1024 * 1024
TOP_ROWS = 12
SENTINEL = "T3_PROFILE_JSON "
CASES = {
    "t3-as06-throughput-fast": {
        "scenario_sha256": "15a2bc778742e778ef895cc5a905483bc967d3857f9c9219b1b87ad34e82ad5e",
        "trace_sha256": "1e7761ee74ade7e23bb4f40410f8525705601f10cc9e6e855d5df798c3e94350",
        "message_trace_sha256": "16f64ee89dd18988792dadc8c6ac3b02881e3d29fc5c80d4df19ae14fa738f77",
        "events": 74502, "messages": 83337,
    },
    "t3-mp05-cancel-churn-newest": {
        "scenario_sha256": "0af9f298f191c4a33982852cf344b5dbe81475846beb274ca56a92f8c95a9183",
        "trace_sha256": "f83b6143c66629a0e2887ca5944b3360cd3bcb7c0b46e57de763f192452f6c14",
        "message_trace_sha256": "8342cfb62afbbccf6ae765d9c36493096bfaa265e2c7dfd873cc8b0c40426afb",
        "events": 240829, "messages": 205166,
    },
}


def frame_file(filename):
    """Expose relevant module names rather than arbitrary absolute profile paths."""
    value = filename.replace("\\", "/")
    if value == "/app/cpu_agent.py":
        return "cpu_agent.py"
    for module in ("abides_fork", "abides_core", "abides_markets", "numpy", "pandas", "pyarrow"):
        marker = "/" + module + "/"
        if marker in value:
            return module + "/" + value.split(marker, 1)[1]
    if value == "~":
        return "builtin"
    return None


def profile_summary(stats):
    rows = []
    for (filename, line, function), (primitive, calls, own, cumulative, _callers) in stats.stats.items():
        module = frame_file(filename)
        if module is None:
            continue
        rows.append({"file": module, "line": line, "function": function[:160],
                     "primitive_calls": primitive, "calls": calls,
                     "own_seconds": own, "cumulative_seconds": cumulative})
    simulate = [r for r in rows if r["file"] == "cpu_agent.py" and r["function"] == "simulate"]
    ledger = [r for r in rows if r["file"] == "abides_fork/trace.py" and r["function"] == "extract_message_trace"]
    if len(simulate) != 1 or len(ledger) != 1 or simulate[0]["calls"] != 1 or ledger[0]["calls"] != 1:
        raise ValueError("profile does not contain exactly one candidate simulate and ledger conversion")
    whole, candidate, target = stats.total_tt, simulate[0]["cumulative_seconds"], ledger[0]["cumulative_seconds"]
    if any(not math.isfinite(x) or x <= 0 for x in (whole, candidate, target)):
        raise ValueError("invalid profile totals")
    top = sorted(rows, key=lambda r: r["cumulative_seconds"], reverse=True)[:TOP_ROWS]
    return {"whole_profile_seconds": whole, "candidate_simulate_seconds": candidate,
            "ledger_cumulative_seconds": target, "ledger_fraction_of_simulate": target / candidate,
            "candidate_simulate_row": simulate[0], "target_function_row": ledger[0],
            "top_cumulative": top,
            "top_own": sorted(rows, key=lambda r: r["own_seconds"], reverse=True)[:TOP_ROWS],
            "pstats_text": "ncalls  tottime  cumtime  file:line(function)\n" + "\n".join(
                f"{r['calls']:7d} {r['own_seconds']:8.4f} {r['cumulative_seconds']:8.4f} "
                f"{r['file']}:{r['line']}({r['function']})" for r in top)}


class CaptureTail:
    """Bound candidate console capture; profiling logs need not include market messages."""
    def __init__(self):
        self.characters = 0
        self.tail = ""

    def write(self, value):
        self.characters += len(value)
        self.tail = (self.tail + value)[-8192:]
        return len(value)

    def flush(self):
        pass


def collect_output(directory, scenario):
    import pyarrow.parquet as pq
    allowed = {"trace.parquet", "message_trace.parquet", "events.json", "profile.pstats"}
    entries = list(directory.iterdir())
    if {p.name for p in entries} != allowed or any(p.is_symlink() or not p.is_file() for p in entries):
        raise ValueError("unexpected profiling output members")
    size = sum(p.stat().st_size for p in entries)
    if size > OUTPUT_CAP:
        raise ValueError("profiling output exceeds its 256 MiB cap")
    events = json.loads((directory / "events.json").read_text())
    hashes = {key: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
              for key, filename in (("trace_sha256", "trace.parquet"),
                                    ("message_trace_sha256", "message_trace.parquet"))}
    counts = {"events": pq.ParquetFile(directory / "trace.parquet").metadata.num_rows,
              "messages": pq.ParquetFile(directory / "message_trace.parquet").metadata.num_rows}
    if (type(events.get("n_events")) is not int or events["n_events"] != counts["events"]
            or type(events.get("n_messages")) is not int or events["n_messages"] != counts["messages"]
            or any(events.get(k) != v for k, v in hashes.items())
            or events.get("scenario_id") != scenario["scenario_id"]
            or events.get("seed") != scenario["seed"] or events.get("gpu_seconds") != 0):
        raise ValueError("profiled outputs disagree with scenario or their actual bytes/counts")
    return {**hashes, **counts, "output_bytes_including_profile": size,
            "profile_bytes": (directory / "profile.pstats").stat().st_size}


def worker_main():
    import contextlib
    import cProfile
    import pstats
    import runpy
    scenario_path = Path("/input/scenario.json")
    scenario = json.loads(scenario_path.read_text())
    directory = Path("/output")
    source = Path("/app/cpu_agent.py")
    capture = CaptureTail()
    profiler = cProfile.Profile()
    sys.argv = [str(source), "simulate", "--config", str(scenario_path),
                "--out", str(directory / "trace.parquet")]
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            profiler.enable()
            try:
                runpy.run_path(str(source), run_name="__main__")
            finally:
                profiler.disable()
        profiler.dump_stats(str(directory / "profile.pstats"))
        result = {"kind": "instrumented_cpu_profile", "rankable": False,
                  "runtime_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                  "scenario_sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
                  "suppressed_console_characters": capture.characters,
                  "outputs": collect_output(directory, scenario),
                  "profile": profile_summary(pstats.Stats(profiler))}
        print(SENTINEL + json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
    except Exception as error:
        print(json.dumps({"profile_failed": type(error).__name__, "message": str(error)[:1000]}), flush=True)
        raise


def worker_source():
    # These functions contain no reference output hashes/counts or platform credentials.
    imports = "import hashlib,json,math,sys\nfrom pathlib import Path\n"
    constants = f"OUTPUT_CAP={OUTPUT_CAP}\nTOP_ROWS={TOP_ROWS}\nSENTINEL={SENTINEL!r}\n"
    functions = (frame_file, profile_summary, CaptureTail, collect_output, worker_main)
    return imports + constants + "\n".join(inspect.getsource(f) for f in functions) + "\nworker_main()\n"


def docker_command(inputs, worker, name):
    args = ["docker", "run", "--rm", "--name", name, "--pull", "never", "--platform", "linux/amd64",
            "--network", "none", "--read-only", "--cpus", "4", "--memory", "8g", "--memory-swap", "8g",
            "--pids-limit", "128", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,nosuid,size=64m", "--tmpfs", "/output:rw,nosuid,size=256m",
            "--mount", f"type=bind,src={inputs.resolve()},dst=/input,readonly",
            "--mount", f"type=bind,src={worker.resolve()},dst=/audit/profile_worker.py,readonly"]
    for value in ("CUDA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=void", "OMP_NUM_THREADS=1",
                  "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1",
                  "PYTHONDONTWRITEBYTECODE=1", "TMPDIR=/tmp"):
        args += ["--env", value]
    return args + ["--entrypoint", "python", IMAGE, "/audit/profile_worker.py"]


def validate_bindings(case, result):
    expected = CASES[case]
    if result.get("runtime_source_sha256") != SOURCE_SHA256 or result.get("rankable") is not False:
        raise ValueError("profile used an unverified runtime source")
    if result.get("scenario_sha256") != expected["scenario_sha256"]:
        raise ValueError("profile used another scenario")
    actual = result.get("outputs", {})
    for key in ("trace_sha256", "message_trace_sha256", "events", "messages"):
        if actual.get(key) != expected[key] or (key in ("events", "messages") and type(actual.get(key)) is not int):
            raise ValueError("profiled output differs from verified v003 Linux confirmation")
    if not 0 < actual.get("output_bytes_including_profile", 0) <= OUTPUT_CAP:
        raise ValueError("invalid profiled output size")


def interpret(results):
    if [r["case"] for r in results] != list(CASES):
        raise ValueError("both fixed public cases are required")
    ratios = [r["profile"]["ledger_fraction_of_simulate"] for r in results]
    if any(not math.isfinite(v) or not 0 < v <= 1 for v in ratios):
        raise ValueError("invalid ledger profile fraction")
    worthwhile = min(ratios) >= 0.03 and max(ratios) >= 0.05
    return {"typed_ledger_experiment_justified": worthwhile,
            "decision": "Profile threshold met; a separately validated experiment may be justified"
                        if worthwhile else "Retain frozen v003 and defer typed-ledger work",
            "threshold": "ledger cumulative >=5% of candidate simulate in at least one case and >=3% in the other",
            "speedup_claim": False}


def run_profile(work):
    if sys.platform != "linux":
        raise RuntimeError("run this reviewed harness on Linux with Docker")
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    protocol = {"image": IMAGE, "runtime_revision": RUNTIME_REVISION, "runtime_source_sha256": SOURCE_SHA256,
                "official_public_commit": PUBLIC_COMMIT, "cases": list(CASES), "rankable": False,
                "interpretation": "ledger cumulative >=5% of candidate simulate in one case and >=3% in the other",
                "resources": "4 CPU,8 GiB RAM/swap,64 MiB tmp,256 MiB output tmpfs,300s/case,no network/GPU",
                "binding_provenance": "verified v003 Linux paired confirmation39745316032 attempt2",
                "profiling_revision": __import__("os").environ.get("GITHUB_SHA"),
                "started_at_utc": datetime.now(timezone.utc).isoformat()}
    (work / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"profile_protocol": protocol}, separators=(",", ":")), flush=True)
    subprocess.run(["docker", "pull", "--platform", "linux/amd64", IMAGE], check=True, timeout=120,
                   stdout=subprocess.DEVNULL)
    detail = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE], text=True, timeout=30))[0]
    if (detail.get("Id") != IMAGE_ID or detail.get("Os") != "linux" or detail.get("Architecture") != "amd64"
            or detail["Config"].get("Volumes") or IMAGE not in detail.get("RepoDigests", [])
            or (detail["Config"].get("Labels") or {}).get("org.opencontainers.image.revision") != RUNTIME_REVISION):
        raise ValueError("Docker image differs from the immutable verified runtime")
    worker = work / "profile_worker.py"
    worker.write_text(worker_source(), encoding="utf-8")
    results = []
    for case, binding in CASES.items():
        inputs = work / "inputs" / case
        inputs.mkdir(parents=True)
        url = f"https://raw.githubusercontent.com/Agenthon-2026/track3-simulation-public/{PUBLIC_COMMIT}/units/{case}/scenario.json"
        with urlopen(url, timeout=15) as response:
            raw = response.read(65537)
        if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != binding["scenario_sha256"]:
            raise ValueError("public scenario bytes differ from their committed binding")
        (inputs / "scenario.json").write_bytes(raw)
        name = "agenthon-profile-" + case
        try:
            completed = subprocess.run(docker_command(inputs, worker, name), capture_output=True, text=True,
                                       check=True, timeout=300)
        finally:
            subprocess.run(["docker", "rm", "--force", name], check=False, timeout=15,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if len(completed.stdout.encode()) > 65536:
            raise ValueError("profiling report exceeded its log bound")
        records = [line[len(SENTINEL):] for line in completed.stdout.splitlines() if line.startswith(SENTINEL)]
        if len(records) != 1:
            raise ValueError("missing or duplicate profiling result")
        result = json.loads(records[0])
        validate_bindings(case, result)
        result["case"] = case
        results.append(result)
        print(json.dumps({"verified_profile": result}, separators=(",", ":"), allow_nan=False), flush=True)
    report = {"protocol": protocol, "results": results, "interpretation": interpret(results),
              "status": "complete", "rankable": False,
              "limitations": "cProfile instrumentation distorts time; diagnostic only, no throughput or Final score claim"}
    (work / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"profile_summary": {"status": "complete", **report["interpretation"]}}, separators=(",", ":")), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path(".profile-t3"))
    args = parser.parse_args()
    run_profile(args.work)


if __name__ == "__main__":
    main()
