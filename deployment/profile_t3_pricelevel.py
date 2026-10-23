"""Two-pass PriceLevel diagnostic in the frozen v004 CPU image; no optimization.

Each public case runs first with cProfile and an unwrapped getter, then in a
separate container with a delegating length counter and no cProfile. Expected
outputs and source hashes stay on the host. Neither pass is a speed benchmark.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from urllib.request import urlopen
import uuid

from profile_t3_v004 import (CASES, IMAGE, IMAGE_ID, SOURCE_SHA256, RUNTIME_REVISION,
                             PUBLIC_COMMIT, CaptureTail, frame_file, validate_image)

DEPENDENCY_COMMIT = "f9cbe51342b7dedd9587e4e069040d68a5c6477f"
DEPENDENCY_SHA256 = "d448ccf12f49c8078994c6d1d2ca150bfed169dbfa393fcfd4d8bc7e4eb9c1b1"
GETTER_SHA256 = "c2b003502cd4dba89d7be401b57bd34c1e6cf0457f0c48aae41df460b76af5f7"
VERSIONS = {"numpy": "1.26.4", "pandas": "1.5.3", "pyarrow": "15.0.2"}
MODES = ("profile", "histogram")
BUCKETS = ("0", "1", "2", "3-7", "8-31", "32+")
OUTPUT_CAP = 256 * 1024 * 1024
SENTINEL = "T3_PRICELEVEL_RESULT "
GATES = {"getter_fraction_of_simulate_min_each": 0.05,
         "length_0_1_2_fraction_min_each": 0.75,
         "later_component_rate_ratio_min_each": 2.50,
         "later_whole_container_mean_eps_ratio_min": 1.03,
         "later_larger_case_eps_ratio_min": 0.95,
         "later_small_case_eps_ratio_min": 0.90}


def profile_summary(stats, target_file, target_line):
    simulate = [(key, row) for key, row in stats.stats.items()
                if key[0] == "/app/cpu_agent.py" and key[2] == "simulate"]
    target = [(key, row) for key, row in stats.stats.items()
              if key == (target_file, target_line, "total_quantity")]
    if (len(simulate) != 1 or len(target) != 1 or simulate[0][1][:2] != (1, 1)
            or target[0][1][0] != target[0][1][1] or target[0][1][1] <= 0):
        raise ValueError("profile lacks one simulate and the exact unwrapped getter")
    whole, candidate, cumulative = stats.total_tt, simulate[0][1][3], target[0][1][3]
    if (any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in (whole, candidate, cumulative)) or cumulative > candidate):
        raise ValueError("invalid target profile durations")
    def row_data(pair):
        key, row = pair
        return {"file": frame_file(key[0]), "line": key[1], "function": key[2],
                "primitive_calls": row[0], "calls": row[1],
                "own_seconds": row[2], "cumulative_seconds": row[3]}
    return {"whole_profile_seconds": whole, "candidate_simulate_seconds": candidate,
            "getter_cumulative_seconds": cumulative,
            "getter_fraction_of_simulate": cumulative / candidate,
            "candidate_simulate_row": row_data(simulate[0]),
            "target_function_row": row_data(target[0])}


def histogram_property(original):
    """Observe only list length; always delegate once to the original getter."""
    counts = {key: 0 for key in BUCKETS}
    state = {"calls": 0, "delegations": 0, "maximum_length": 0, "buckets": counts}
    def counted(instance):
        length = len(instance.visible_orders)
        key = str(length) if length <= 2 else "3-7" if length <= 7 else "8-31" if length <= 31 else "32+"
        state["calls"] += 1
        counts[key] += 1
        state["maximum_length"] = max(state["maximum_length"], length)
        state["delegations"] += 1
        return original.fget(instance)
    return property(counted, original.fset, original.fdel, original.__doc__), state


def collect_output(directory, scenario, mode):
    import pyarrow.parquet as pq
    allowed = {"trace.parquet", "message_trace.parquet", "events.json"}
    if mode == "profile":
        allowed.add("profile.pstats")
    entries = list(directory.iterdir())
    if {path.name for path in entries} != allowed or any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("unexpected diagnostic output members")
    size = sum(path.stat().st_size for path in entries)
    if size > OUTPUT_CAP:
        raise ValueError("diagnostic output exceeds 256 MiB")
    events = json.loads((directory / "events.json").read_text())
    hashes = {key: hashlib.sha256((directory / filename).read_bytes()).hexdigest()
              for key, filename in (("trace_sha256", "trace.parquet"),
                                    ("message_trace_sha256", "message_trace.parquet"))}
    counts = {"events": pq.ParquetFile(directory / "trace.parquet").metadata.num_rows,
              "messages": pq.ParquetFile(directory / "message_trace.parquet").metadata.num_rows}
    if (type(events.get("n_events")) is not int or events["n_events"] != counts["events"]
            or type(events.get("n_messages")) is not int or events["n_messages"] != counts["messages"]
            or any(events.get(key) != value for key, value in hashes.items())
            or events.get("scenario_id") != scenario["scenario_id"]
            or events.get("seed") != scenario["seed"] or events.get("gpu_seconds") != 0):
        raise ValueError("sidecar disagrees with actual bytes, footer counts or scenario")
    return {**hashes, **counts, "output_bytes": size,
            "profile_bytes": (directory / "profile.pstats").stat().st_size if mode == "profile" else 0}


def worker_main():
    import contextlib
    import cProfile
    import pstats
    import runpy
    import numpy
    import pandas
    import pyarrow
    from abides_markets.price_level import PriceLevel
    mode = sys.argv[1]
    if mode not in ("profile", "histogram"):
        raise ValueError("unknown diagnostic pass")
    scenario_path = Path("/input/scenario.json")
    scenario = json.loads(scenario_path.read_text())
    directory, source = Path("/output"), Path("/app/cpu_agent.py")
    original = PriceLevel.total_quantity
    if (type(original) is not property or original.fget is None
            or original.fget.__name__ != "total_quantity"
            or original.fget.__qualname__ != "PriceLevel.total_quantity"):
        raise ValueError("target is not the expected PriceLevel property getter")
    original_code = original.fget.__code__
    dependency = Path(inspect.getsourcefile(original.fget))
    dependency_bytes = dependency.read_bytes()
    bindings = {"runtime_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "dependency_source_sha256": hashlib.sha256(dependency_bytes).hexdigest(),
                "dependency_lf_source_sha256": hashlib.sha256(dependency_bytes.replace(b"\r\n", b"\n")).hexdigest(),
                "getter_source_sha256": hashlib.sha256(inspect.getsource(original.fget).encode()).hexdigest(),
                "scenario_sha256": hashlib.sha256(scenario_path.read_bytes()).hexdigest(),
                "dependency_versions": {"numpy": numpy.__version__, "pandas": pandas.__version__, "pyarrow": pyarrow.__version__}}
    capture = CaptureTail()
    profiler, histogram = None, None
    sys.argv = [str(source), "simulate", "--config", str(scenario_path), "--out", str(directory / "trace.parquet")]
    try:
        with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
            if mode == "profile":
                # Deliberately no wrapper or histogram instrumentation in this pass.
                profiler = cProfile.Profile()
                profiler.enable()
                try:
                    runpy.run_path(str(source), run_name="__main__")
                finally:
                    profiler.disable()
            else:
                wrapped, histogram = histogram_property(original)
                PriceLevel.total_quantity = wrapped
                try:
                    runpy.run_path(str(source), run_name="__main__")
                finally:
                    PriceLevel.total_quantity = original
        if PriceLevel.total_quantity is not original or original.fget.__code__ is not original_code:
            raise ValueError("original target property/function was not preserved")
        result = {"kind": "pricelevel_diagnostic", "mode": mode, "rankable": False, **bindings,
                  "getter_wrapper_installed": mode == "histogram", "cprofile_enabled": mode == "profile",
                  "getter_code_unchanged": True, "original_property_restored": True,
                  "suppressed_console_characters": capture.characters}
        if profiler is not None:
            profiler.dump_stats(str(directory / "profile.pstats"))
            result["profile"] = profile_summary(pstats.Stats(profiler), original_code.co_filename, original_code.co_firstlineno)
        else:
            result["histogram"] = histogram
        result["outputs"] = collect_output(directory, scenario, mode)
        print(SENTINEL + json.dumps(result, separators=(",", ":"), allow_nan=False), flush=True)
    except Exception as error:
        print(json.dumps({"diagnostic_failed": type(error).__name__, "message": str(error)[:1000]}), flush=True)
        raise


def worker_source():
    imports = "import hashlib,inspect,json,math,sys\nfrom pathlib import Path\n"
    constants = f"OUTPUT_CAP={OUTPUT_CAP}\nBUCKETS={BUCKETS!r}\nSENTINEL={SENTINEL!r}\n"
    functions = (frame_file, profile_summary, CaptureTail, histogram_property, collect_output, worker_main)
    return imports + constants + "\n".join(inspect.getsource(function) for function in functions) + "\nworker_main()\n"


def schedule():
    return [{"case": case, "mode": mode} for case in CASES for mode in MODES]


def docker_command(inputs, worker, name, mode):
    if mode not in MODES:
        raise ValueError("unknown diagnostic pass")
    args = ["docker", "run", "--rm", "--name", name, "--pull", "never", "--platform", "linux/amd64",
            "--network", "none", "--read-only", "--cpus", "4", "--memory", "8g", "--memory-swap", "8g",
            "--pids-limit", "128", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--tmpfs", "/tmp:rw,nosuid,size=64m", "--tmpfs", "/output:rw,nosuid,size=256m",
            "--mount", f"type=bind,src={inputs.resolve()},dst=/input,readonly",
            "--mount", f"type=bind,src={worker.resolve()},dst=/audit/profile_worker.py,readonly"]
    for value in ("CUDA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=void", "OMP_NUM_THREADS=1",
                  "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1", "TMPDIR=/tmp"):
        args += ["--env", value]
    return args + ["--entrypoint", "python", IMAGE_ID, "/audit/profile_worker.py", mode]


def validate_result(case, mode, result):
    if (result.get("kind") != "pricelevel_diagnostic" or result.get("mode") != mode
            or result.get("rankable") is not False
            or result.get("runtime_source_sha256") != SOURCE_SHA256
            or result.get("dependency_source_sha256") != DEPENDENCY_SHA256
            or result.get("dependency_lf_source_sha256") != DEPENDENCY_SHA256
            or result.get("getter_source_sha256") != GETTER_SHA256
            or result.get("dependency_versions") != VERSIONS
            or result.get("scenario_sha256") != CASES[case]["scenario_sha256"]):
        raise ValueError("diagnostic source, dependency, mode or scenario binding differs")
    if (result.get("getter_wrapper_installed") is not (mode == "histogram")
            or result.get("cprofile_enabled") is not (mode == "profile")
            or result.get("getter_code_unchanged") is not True or result.get("original_property_restored") is not True):
        raise ValueError("diagnostic passes were mixed or the original getter changed")
    output = result.get("outputs", {})
    for key in ("trace_sha256", "message_trace_sha256", "events", "messages"):
        if output.get(key) != CASES[case][key] or key in ("events", "messages") and type(output.get(key)) is not int:
            raise ValueError("diagnostic outputs differ from known exact bindings")
    size, profiled = output.get("output_bytes"), output.get("profile_bytes")
    if (type(size) is not int or not 0 < size <= OUTPUT_CAP or type(profiled) is not int
            or (not 0 < profiled <= size if mode == "profile" else profiled != 0)):
        raise ValueError("diagnostic output cap/profile membership differs")
    if mode == "profile":
        if "histogram" in result:
            raise ValueError("timed pass unexpectedly includes histogram instrumentation")
        profile = result["profile"]
        row = profile["target_function_row"]
        fraction = profile["getter_fraction_of_simulate"]
        if (row["file"] != "abides_markets/price_level.py" or row["function"] != "total_quantity"
                or type(row["calls"]) is not int or row["calls"] <= 0 or row["primitive_calls"] != row["calls"]
                or type(fraction) not in (int, float) or not math.isfinite(fraction) or not 0 < fraction <= 1):
            raise ValueError("invalid unwrapped target profile")
    else:
        if "profile" in result:
            raise ValueError("histogram pass unexpectedly includes cProfile")
        histogram = result["histogram"]
        counts = histogram["buckets"]
        if (set(counts) != set(BUCKETS) or any(type(value) is not int or value < 0 for value in counts.values())
                or type(histogram["calls"]) is not int or histogram["calls"] <= 0
                or type(histogram["delegations"]) is not int or histogram["delegations"] != histogram["calls"]
                or sum(counts.values()) != histogram["calls"]
                or type(histogram["maximum_length"]) is not int or histogram["maximum_length"] < 0):
            raise ValueError("invalid getter length/delegation histogram")
        maximum = histogram["maximum_length"]
        highest = str(maximum) if maximum <= 2 else "3-7" if maximum <= 7 else "8-31" if maximum <= 31 else "32+"
        if counts[highest] == 0 or any(counts[key] for key in BUCKETS[BUCKETS.index(highest) + 1:]):
            raise ValueError("histogram maximum disagrees with its occupied buckets")


def interpret(results):
    if [{"case": row["case"], "mode": row["mode"]} for row in results] != schedule():
        raise ValueError("both exact two-pass public cases are required")
    cases = []
    for offset, case in enumerate(CASES):
        measured, counted = results[2 * offset:2 * offset + 2]
        validate_result(case, "profile", measured)
        validate_result(case, "histogram", counted)
        calls = measured["profile"]["target_function_row"]["calls"]
        if calls != counted["histogram"]["calls"]:
            raise ValueError("getter call counts differ across isolated diagnostic passes")
        fraction = measured["profile"]["getter_fraction_of_simulate"]
        counts = counted["histogram"]["buckets"]
        small_share = sum(counts[key] for key in ("0", "1", "2")) / calls
        cases.append({"case": case, "getter_calls": calls, "getter_fraction_of_simulate": fraction,
                      "length_0_1_2_fraction": small_share,
                      "profile_gate_passed": fraction >= GATES["getter_fraction_of_simulate_min_each"],
                      "length_gate_passed": small_share >= GATES["length_0_1_2_fraction_min_each"]})
    justified = all(case["profile_gate_passed"] and case["length_gate_passed"] for case in cases)
    return {"cases": cases, "component_experiment_justified": justified,
            "decision": "A separately reviewed component experiment may proceed" if justified else "Retain v004; do not implement this optimization",
            "speedup_claim": False, "gates": GATES}


def run_diagnostics(work):
    if sys.platform != "linux":
        raise RuntimeError("run this reviewed diagnostic on Linux with Docker")
    work = work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    report = {"kind": "two_pass_pricelevel_diagnostic", "status": "preparing", "rankable": False,
              "protocol": {"image": IMAGE, "image_id": IMAGE_ID, "runtime_revision": RUNTIME_REVISION,
                           "runtime_source_sha256": SOURCE_SHA256, "dependency_commit": DEPENDENCY_COMMIT,
                           "dependency_source_sha256": DEPENDENCY_SHA256, "getter_source_sha256": GETTER_SHA256,
                           "dependency_provenance": "Exact official ABIDES Git blob at pinned commit; raw Linux bytes must match, LF-normalized hash also recorded",
                           "official_public_commit": PUBLIC_COMMIT, "dependency_versions": VERSIONS,
                           "schedule": schedule(), "gates": GATES,
                           "pass_separation": "Fresh containers: first unwrapped cProfile fraction, second delegating length histogram without cProfile",
                           "resources": "4CPU,8GiB memory/swap,64MiB tmp,256MiB output tmpfs,300s/pass,threads1,offline/read-only,no GPU",
                           "expected_bindings_host_only": True, "profiling_revision": os.environ.get("GITHUB_SHA")},
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "results": [],
              "limitations": "Instrumented public diagnostics only; no runtime candidate, performance or Final rank claim"}
    def save():
        (work / "t3-pricelevel-profile.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    save()
    print(json.dumps({"pricelevel_protocol": report}, separators=(",", ":")), flush=True)
    try:
        subprocess.run(["docker", "pull", "--platform", "linux/amd64", IMAGE], check=True, timeout=120, stdout=subprocess.DEVNULL)
        detail = json.loads(subprocess.check_output(["docker", "image", "inspect", IMAGE], text=True, timeout=30))[0]
        validate_image(detail)
        worker = work / "profile_worker.py"
        worker.write_text(worker_source(), encoding="utf-8")
        for case, binding in CASES.items():
            inputs = work / "inputs" / case
            inputs.mkdir(parents=True)
            url = f"https://raw.githubusercontent.com/Agenthon-2026/track3-simulation-public/{PUBLIC_COMMIT}/units/{case}/scenario.json"
            with urlopen(url, timeout=15) as response:
                raw = response.read(65537)
            if len(raw) > 65536 or hashlib.sha256(raw).hexdigest() != binding["scenario_sha256"]:
                raise ValueError("public scenario differs from fixed official binding")
            (inputs / "scenario.json").write_bytes(raw)
            for mode in MODES:
                report["status"] = "running"
                save()
                name = "agenthon-pricelevel-" + uuid.uuid4().hex
                try:
                    done = subprocess.run(docker_command(inputs, worker, name, mode), capture_output=True, text=True,
                                          check=True, timeout=300)
                finally:
                    subprocess.run(["docker", "rm", "--force", name], check=False, timeout=15,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if len(done.stdout.encode()) > 65536:
                    raise ValueError("diagnostic report exceeds its log bound")
                payloads = [line[len(SENTINEL):] for line in done.stdout.splitlines() if line.startswith(SENTINEL)]
                if len(payloads) != 1:
                    raise ValueError("missing or duplicate diagnostic result")
                result = json.loads(payloads[0])
                validate_result(case, mode, result)
                result["case"] = case
                report["results"].append(result)
                save()
                print(json.dumps({"verified_pricelevel_pass": result}, separators=(",", ":"), allow_nan=False), flush=True)
        report["interpretation"] = interpret(report["results"])
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", failure_type=type(error).__name__, failure_message=str(error))
        raise
    finally:
        report["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print("T3_PRICELEVEL_REPORT " + json.dumps(report, separators=(",", ":"), allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path(".profile-pricelevel"))
    args = parser.parse_args()
    run_diagnostics(args.work)


if __name__ == "__main__":
    main()
