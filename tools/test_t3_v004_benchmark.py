"""Offline tests for v004 source selection, exact bindings, resource bounds and gates."""
from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "deployment/benchmark_t3_v004.py"
spec = importlib.util.spec_from_file_location("benchmark_t3_v004", PATH)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


def records(ratios=(1.1, 1.1, 1.1, 1.1), baseline_eps=(10000, 10000, 10000, 10000)):
    rows = []
    for row in benchmark.schedule():
        index = list(benchmark.CASES).index(row["case"])
        binding = benchmark.CASES[row["case"]]
        speed = baseline_eps[index] * (ratios[index] if row["mode"] == "v004" else 1)
        rows.append({**row, **{key: binding[key] for key in benchmark.BINDING_KEYS},
                     "process_seconds": binding["events"] / speed})
    return rows


def image_detail():
    return {"Id": benchmark.IMAGE_ID, "Os": "linux", "Architecture": "amd64",
            "RepoDigests": [benchmark.IMAGE], "Config": {"Entrypoint": ["python", "/app/cpu_agent.py"],
            "Labels": {"org.opencontainers.image.revision": benchmark.RUNTIME_REVISION}}}


class ConfirmationTests(unittest.TestCase):
    def test_baseline_original_and_only_candidate_source_overlay(self):
        source = Path("cpu_agent_v004_candidate.py")
        for mode in benchmark.MODES:
            command = benchmark.docker_command(mode, source, Path("scenario-only"), Path("output"), "unit-test")
            mounts = [command[index + 1] for index, value in enumerate(command) if value == "--mount"]
            overlays = [value for value in mounts if "dst=/app/cpu_agent.py" in value]
            self.assertEqual(len(overlays), int(mode == "v004"))
            self.assertEqual(len(mounts), 3 if mode == "v004" else 2)
            if overlays:
                self.assertIn(str(source.resolve()), overlays[0])
                self.assertTrue(overlays[0].endswith(",readonly"))
            self.assertNotIn("--entrypoint", command)
            self.assertIn(benchmark.IMAGE_ID, command)
            self.assertFalse(any("reference" in value or "docker.sock" in value for value in mounts))
        with self.assertRaises(ValueError):
            benchmark.container_prefix("v002", source, "unit-test")

    def test_cpu_offline_bounds(self):
        command = benchmark.container_prefix("v003", Path("unused"), "unit-test")
        for flag, value in (("--cpus", "4"), ("--memory", "8g"), ("--memory-swap", "8g"),
                            ("--network", "none"), ("--pids-limit", "128"), ("--pull", "never")):
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--read-only", command)
        self.assertIn("--rm", command)
        self.assertIn("/tmp:rw,nosuid,size=64m", command)
        self.assertIn("OPENBLAS_NUM_THREADS=1", command)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("--privileged", command)

    def test_image_identity_architecture_entrypoint_volume_and_revision(self):
        benchmark.check_image(image_detail())
        for key, value in (("Id", "sha256:wrong"), ("Architecture", "arm64"), ("RepoDigests", [])):
            changed = image_detail()
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                benchmark.check_image(changed)
        for key, value in (("Volumes", {"/output": {}}), ("Entrypoint", ["python", "/different.py"]), ("Labels", {})):
            changed = image_detail()
            changed["Config"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                benchmark.check_image(changed)

    def test_schedule_has_four_cases_three_pairs_and_balanced_order(self):
        schedule = benchmark.schedule()
        self.assertEqual(len(schedule), 24)
        first_modes = [schedule[index]["mode"] for index in range(0, 24, 2)]
        self.assertEqual(first_modes.count("v003"), 6)
        self.assertEqual(first_modes.count("v004"), 6)
        for index in range(0, 24, 2):
            a, b = schedule[index:index + 2]
            self.assertEqual((a["case"], a["pair"]), (b["case"], b["pair"]))
            self.assertEqual({a["mode"], b["mode"]}, set(benchmark.MODES))

    def test_same_promotion_gates_and_equal_mean_eps_formula(self):
        self.assertTrue(benchmark.summarize(records())["promotion_passed"])
        self.assertFalse(benchmark.summarize(records((1.029,) * 4))["promotion_passed"])
        self.assertTrue(benchmark.summarize(records((1.031,) * 4))["promotion_passed"])
        self.assertFalse(benchmark.summarize(records((0.899, 1.2, 1.2, 1.2)))["promotion_passed"])
        self.assertFalse(benchmark.summarize(records((1.2, 0.949, 1.2, 1.2)))["promotion_passed"])
        speeds = (10000, 5000, 9000, 12000)
        ratios = (0.91, 1.2, 1.1, 1.2)
        summary = benchmark.summarize(records(ratios, speeds))
        expected = sum(speed * ratio for speed, ratio in zip(speeds, ratios)) / sum(speeds)
        self.assertAlmostEqual(summary["equal_unit_mean_eps_ratio"], expected)
        self.assertTrue(summary["promotion_passed"])

    def test_all_repeats_must_match_fixed_external_bindings(self):
        for key, value in (("trace_sha256", "0" * 64), ("message_trace_sha256", "0" * 64),
                           ("events", 1), ("messages", 1)):
            changed = records()
            for row in changed:
                if row["case"] == next(iter(benchmark.CASES)):
                    row[key] = value  # Even stable-but-wrong output across both modes must fail.
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "verified v003"):
                benchmark.summarize(changed)
        changed = records()
        changed[-1]["events"] = float(changed[-1]["events"])
        with self.assertRaisesRegex(ValueError, "noninteger"):
            benchmark.summarize(changed)

    def test_partial_reordered_and_invalid_time_runs_cannot_pass(self):
        with self.assertRaisesRegex(ValueError, "roster"):
            benchmark.summarize(records()[:-1])
        with self.assertRaisesRegex(ValueError, "roster"):
            benchmark.summarize(records()[::-1])
        for value in (0, -1, 301, math.nan, math.inf, True):
            changed = records()
            changed[-1]["process_seconds"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "benchmark time"):
                benchmark.summarize(changed)

    def test_container_timeout_and_cleanup_are_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch.object(benchmark.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
                with patch.object(benchmark.time, "perf_counter", side_effect=(100.0, 102.0)):
                    elapsed = benchmark.run_container("v003", root / "candidate.py", root / "input", root / "out", root / "log")
            self.assertEqual(elapsed, 2.0)
            self.assertEqual(run.call_args_list[0].kwargs["timeout"], 300)
            self.assertEqual(run.call_args_list[1].kwargs["timeout"], 15)
            self.assertEqual(run.call_args_list[1].args[0][:3], ["docker", "rm", "--force"])
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with patch.object(benchmark.subprocess, "run", side_effect=(subprocess.TimeoutExpired("docker", 300), None)) as run:
                with self.assertRaises(subprocess.TimeoutExpired):
                    benchmark.run_container("v004", root / "candidate.py", root / "input", root / "out", root / "log")
            self.assertEqual(run.call_count, 2)

    def test_actual_parquet_metadata_hashes_and_sidecar(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as root:
            output = Path(root)
            pq.write_table(pa.table({"x": [1, 2]}), output / "trace.parquet")
            pq.write_table(pa.table({"x": [1, 2, 3]}), output / "message_trace.parquet")
            sidecar = {"n_events": 2, "n_messages": 3, "scenario_id": "test", "seed": 7, "gpu_seconds": 0,
                       "wall_clock_sec": 1.0, "events_per_sec": 2.0,
                       "trace_sha256": benchmark.sha256(output / "trace.parquet"),
                       "message_trace_sha256": benchmark.sha256(output / "message_trace.parquet")}
            (output / "events.json").write_text(json.dumps(sidecar))
            evidence = benchmark.output_evidence(output, {"scenario_id": "test", "seed": 7})
            self.assertEqual((evidence["events"], evidence["messages"]), (2, 3))
            for key, value in (("n_events", 99), ("message_trace_sha256", "0" * 64), ("seed", 8), ("gpu_seconds", 1)):
                (output / "events.json").write_text(json.dumps(sidecar | {key: value}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    benchmark.output_evidence(output, {"scenario_id": "test", "seed": 7})

    def test_protocol_and_failure_report_exist_before_any_timing(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            candidate = root / "candidate.py"
            candidate.write_text("pass\n")
            work = root / "work"
            def stop_at_pull(*args, **kwargs):
                protocol = json.loads((work / "t3-v004-paired-report.json").read_text())
                self.assertEqual(protocol["status"], "preparing")
                self.assertEqual(protocol["protocol"]["gates"], benchmark.GATES)
                self.assertEqual(protocol["runs"], [])
                raise RuntimeError("offline unit test stops before Docker")
            with patch.object(benchmark.sys, "platform", "linux"), patch.object(benchmark, "sha256", return_value=benchmark.CANDIDATE_SHA256):
                with patch.object(benchmark.subprocess, "run", side_effect=stop_at_pull), patch("builtins.print"):
                    with self.assertRaisesRegex(RuntimeError, "offline unit"):
                        benchmark.benchmark(work, candidate)
            result = json.loads((work / "t3-v004-paired-report.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["promotion_passed"])

    def test_workflow_is_dedicated_free_cpu_without_publish_storage(self):
        text = (PATH.parent / "cpu-v004.yml").read_text()
        for required in ("branches: [t3-experiment]", "github.event.repository.private == false",
                         "runs-on: ubuntu-24.04", "timeout-minutes: 20", "--no-cache-dir"):
            self.assertIn(required, text)
        for forbidden in ("deployment/**", "cpu-build.yml", "docker build", "docker push", "upload-artifact", "actions/cache", "packages: write"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
