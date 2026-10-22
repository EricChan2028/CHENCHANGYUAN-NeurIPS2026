"""Pure smoke/boundary tests; never launch Docker, fetch data or profile a worker."""
from __future__ import annotations
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

PATH = Path(__file__).resolve().parents[1] / "deployment/profile_t3_linux.py"
spec = importlib.util.spec_from_file_location("profile_t3_linux", PATH)
profile = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = profile
spec.loader.exec_module(profile)


def fake_stats():
    return SimpleNamespace(total_tt=10.0, stats={
        ("/app/cpu_agent.py", 275, "simulate"): (1, 1, 0.1, 5.0, {}),
        ("/app/vendor/track3_runtime/abides_fork/trace.py", 233, "extract_message_trace"): (1, 1, 0.05, 0.25, {}),
        ("/audit/profile_worker.py", 1, "worker_main"): (1, 1, 0.1, 9.0, {}),
        ("/private/credential_store.py", 99, "sensitive_function"): (1, 1, 1.0, 8.0, {}),
        **{(f"/usr/local/lib/python3.11/site-packages/pandas/core/frame{i}.py", 2, "construct"): (2, 3, 0.02, 0.2, {}) for i in range(30)},
    })


def bound_result(case):
    binding = profile.CASES[case]
    return {"rankable": False, "runtime_source_sha256": profile.SOURCE_SHA256,
            "scenario_sha256": binding["scenario_sha256"],
            "outputs": {k: binding[k] for k in ("trace_sha256", "message_trace_sha256", "events", "messages")}
                       | {"output_bytes_including_profile": 12345}}


class ProfilingTests(unittest.TestCase):
    def test_container_is_cpu_offline_read_only_and_hard_capped(self):
        command = profile.docker_command(Path("inputs"), Path("worker.py"), "test-profile")
        for key, value in (("--cpus", "4"), ("--memory", "8g"), ("--memory-swap", "8g"),
                           ("--network", "none"), ("--pids-limit", "128"), ("--pull", "never")):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn("--read-only", command)
        self.assertIn("/output:rw,nosuid,size=256m", command)
        self.assertIn("/tmp:rw,nosuid,size=64m", command)
        self.assertIn(profile.IMAGE, command)
        self.assertNotIn("--gpus", command)
        self.assertNotIn("--privileged", command)
        mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
        self.assertEqual(len(mounts), 2)
        self.assertTrue(all(value.endswith(",readonly") for value in mounts))
        self.assertFalse(any("reference" in value or "docker.sock" in value for value in mounts))

    def test_report_filters_wrapper_and_unrelated_paths_and_bounds_rows(self):
        result = profile.profile_summary(fake_stats())
        self.assertEqual(result["whole_profile_seconds"], 10.0)
        self.assertEqual(result["candidate_simulate_seconds"], 5.0)
        self.assertEqual(result["ledger_fraction_of_simulate"], 0.05)
        self.assertEqual(result["target_function_row"]["function"], "extract_message_trace")
        self.assertLessEqual(len(result["top_cumulative"]), 12)
        self.assertLessEqual(len(result["top_own"]), 12)
        rendered = json.dumps(result)
        self.assertNotIn("credential_store", rendered)
        self.assertNotIn("worker_main", rendered)
        self.assertNotIn("/usr/local", rendered)

    def test_missing_target_is_an_error_not_a_zero_claim(self):
        data = fake_stats()
        data.stats = {k: v for k, v in data.stats.items() if k[2] != "extract_message_trace"}
        with self.assertRaisesRegex(ValueError, "exactly one"):
            profile.profile_summary(data)

    def test_predeclared_threshold_requires_both_cases(self):
        def rows(ratios):
            return [{"case": name, "profile": {"ledger_fraction_of_simulate": ratio}}
                    for name, ratio in zip(profile.CASES, ratios)]
        self.assertTrue(profile.interpret(rows((0.05, 0.03)))["typed_ledger_experiment_justified"])
        self.assertFalse(profile.interpret(rows((0.049, 0.049)))["typed_ledger_experiment_justified"])
        self.assertFalse(profile.interpret(rows((0.06, 0.029)))["typed_ledger_experiment_justified"])
        with self.assertRaisesRegex(ValueError, "both fixed"):
            profile.interpret(rows((0.06,)))

    def test_bindings_reject_changed_trace_counts_source_and_scenario(self):
        case = next(iter(profile.CASES))
        expected = bound_result(case)
        profile.validate_bindings(case, expected)
        for location, key, value in (("outputs", "trace_sha256", "0" * 64),
                                     ("outputs", "messages", 1),
                                     ("outputs", "events", float(expected["outputs"]["events"])),
                                     (None, "runtime_source_sha256", "0" * 64),
                                     (None, "scenario_sha256", "0" * 64),
                                     ("outputs", "output_bytes_including_profile", profile.OUTPUT_CAP + 1)):
            with self.subTest(key=key):
                changed = copy.deepcopy(expected)
                (changed[location] if location else changed)[key] = value
                with self.assertRaises(ValueError):
                    profile.validate_bindings(case, changed)

    def test_generated_worker_has_no_reference_bindings_and_compiles(self):
        source = profile.worker_source()
        compile(source, "profile_worker.py", "exec")
        self.assertIn("cProfile.Profile()", source)
        self.assertNotIn("raw.githubusercontent", source)
        self.assertNotIn(profile.IMAGE, source)
        for value in profile.CASES.values():
            for key in ("scenario_sha256", "trace_sha256", "message_trace_sha256"):
                self.assertNotIn(value[key], source)

    def test_console_capture_is_bounded(self):
        capture = profile.CaptureTail()
        self.assertEqual(capture.write("x" * 20000), 20000)
        self.assertEqual(len(capture.tail), 8192)
        self.assertEqual(capture.characters, 20000)

    def test_workflow_is_only_the_public_profile_branch_with_no_billed_storage(self):
        text = (PATH.parent / "cpu-profile.yml").read_text()
        self.assertIn("branches: [t3-profile]", text)
        self.assertIn("timeout-minutes: 15", text)
        self.assertIn("github.event.repository.private == false", text)
        self.assertIn("runs-on: ubuntu-24.04", text)
        self.assertNotIn("deployment/**", text)
        self.assertNotIn("cpu-build.yml", text)
        self.assertNotIn("upload-artifact", text)
        self.assertNotIn("actions/cache", text)
        self.assertNotIn("packages: write", text)


if __name__ == "__main__":
    unittest.main()
