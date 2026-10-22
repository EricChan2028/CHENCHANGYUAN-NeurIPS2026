"""Synthetic command/evidence tests; they establish no Linux performance result."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deployment"))
import benchmark_t3_pair as bench


def records(ratio=1.1):
    return [{**row, "events": 1000, "messages": 1100,
             "trace_sha256": "a" * 64, "message_trace_sha256": "b" * 64,
             "process_seconds": 10.0 if row["mode"] == "v003" else 10.0 / ratio}
            for row in bench.schedule()]


class PairedBenchmarkTests(unittest.TestCase):
    def test_exact_balanced_fresh_container_roster(self):
        rows = bench.schedule()
        self.assertEqual(len(rows), 24)
        self.assertEqual(sum(rows[i]["mode"] == "v003" for i in range(0, 24, 2)), 6)
        for i in range(0, 24, 2):
            self.assertEqual(rows[i]["case"], rows[i + 1]["case"])
            self.assertEqual(rows[i]["pair"], rows[i + 1]["pair"])
            self.assertEqual({rows[i]["mode"], rows[i + 1]["mode"]}, set(bench.MODES))

    def test_baseline_changes_only_readonly_source_mount(self):
        inputs, outputs, baseline = Path("inputs"), Path("outputs"), Path("cpu_agent_v003.py")
        candidate = bench.docker_command("sha256:synthetic", inputs, outputs, "one")
        old = bench.docker_command("sha256:synthetic", inputs, outputs, "one", baseline)
        mount = f"type=bind,src={baseline.resolve()},dst=/app/cpu_agent.py,readonly"
        index = old.index(mount)
        self.assertEqual(old[:index-1] + old[index+1:], candidate)
        self.assertNotIn("--entrypoint", old)
        self.assertNotIn("--gpus", old)
        for flag, value in (("--network", "none"), ("--cpus", "4"), ("--memory", "8g"),
                            ("--memory-swap", "8g"), ("--tmpfs", "/tmp:rw,nosuid,size=64m")):
            self.assertEqual(old[old.index(flag) + 1], value)
        self.assertIn("--read-only", old)
        self.assertEqual(old[-6:], ["sha256:synthetic", "simulate", "--config", "/input/scenario.json", "--out", "/output/trace.parquet"])

    def test_timeout_still_forces_container_cleanup(self):
        calls = []
        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            if argv[1] == "run":
                self.assertEqual(kwargs["timeout"], 300)
                raise subprocess.TimeoutExpired(argv, 300)
            return subprocess.CompletedProcess(argv, 0)
        with tempfile.TemporaryDirectory() as directory, patch.object(bench.subprocess, "run", side_effect=fake_run):
            root = Path(directory)
            with self.assertRaises(subprocess.TimeoutExpired):
                bench.run_container("sha256:test", root / "inputs", root / "out", root / "run.log")
        self.assertEqual(calls[1][0][:3], ["docker", "rm", "--force"])
        self.assertEqual(calls[0][0][calls[0][0].index("--name") + 1], calls[1][0][-1])

    def test_incomplete_or_semantically_changed_results_fail_closed(self):
        rows = records()
        with self.assertRaisesRegex(ValueError, "incomplete or changed"):
            bench.summarize(rows[:-1])
        for key, value in (("trace_sha256", "c" * 64), ("message_trace_sha256", "d" * 64),
                           ("events", 999), ("messages", 1099)):
            changed = copy.deepcopy(rows)
            changed[1][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "differs"):
                bench.summarize(changed)

    def test_mean_eps_ratio_is_arithmetic_rate_proxy_not_mean_speedup(self):
        rows = records()
        for row in rows:
            row["events"] = 1000 if row["case"] == bench.CASES[0] else 10
            row["process_seconds"] = 1000 / 104 if row["case"] == bench.CASES[0] and row["mode"] == "v004" else 10
        result = bench.summarize(rows)
        self.assertAlmostEqual(result["equal_unit_mean_eps_ratio"], 107 / 103)
        self.assertTrue(result["promotion_passed"])

    def test_promotion_requires_all_predeclared_gates(self):
        self.assertTrue(bench.summarize(records(1.1))["promotion_passed"])
        self.assertFalse(bench.summarize(records(1.02))["promotion_passed"])
        for case, ratio in ((bench.CASES[0], 0.89), (bench.CASES[2], 0.94)):
            rows = records(2.0)
            for row in rows:
                if row["case"] == case and row["mode"] == "v004":
                    row["process_seconds"] = 10 / ratio
            result = bench.summarize(rows)
            self.assertGreater(result["equal_unit_mean_eps_ratio"], 1.03)
            self.assertFalse(result["promotion_passed"])

    def test_bad_times_cannot_produce_a_passing_report(self):
        for value in (0, -1, float("nan"), float("inf"), 300.1):
            rows = records()
            rows[0]["process_seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.summarize(rows)

    def test_output_evidence_reads_real_footer_and_hashes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(pa.table({"value": [1, 2]}), root / "trace.parquet")
            pq.write_table(pa.table({"value": [3, 4, 5]}), root / "message_trace.parquet")
            events = {"scenario_id": "synthetic", "seed": 17, "n_events": 2, "n_messages": 3,
                      "wall_clock_sec": 1.0, "events_per_sec": 2.0,
                      "trace_sha256": bench.sha256(root / "trace.parquet"),
                      "message_trace_sha256": bench.sha256(root / "message_trace.parquet")}
            (root / "events.json").write_text(json.dumps(events))
            self.assertEqual(bench.output_evidence(root, {"scenario_id": "synthetic", "seed": 17})["events"], 2)
            events["n_events"] = 3
            (root / "events.json").write_text(json.dumps(events))
            with self.assertRaisesRegex(ValueError, "row counts"):
                bench.output_evidence(root, {"scenario_id": "synthetic", "seed": 17})

    def test_nonlinux_refuses_before_docker_or_files(self):
        with patch.object(bench.sys, "platform", "win32"), patch.object(bench, "inspect_image") as inspect:
            with self.assertRaisesRegex(RuntimeError, "requires Linux"):
                bench.benchmark("test", Path("unused"), Path("unused/out.json"), Path("old.py"), Path("new.py"))
            inspect.assert_not_called()

    def test_cli_returns_two_only_for_completed_performance_failure(self):
        with patch.object(sys, "argv", ["benchmark", "--image", "synthetic", "--work", "unused"]), \
                patch.object(bench, "benchmark", return_value={"promotion_passed": False}), \
                patch("builtins.print"):
            self.assertEqual(bench.main(), 2)


if __name__ == "__main__":
    unittest.main()
