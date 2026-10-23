"""Offline safeguards for v004 profiling: no Docker, downloads or market runs."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / "deployment/profile_t3_v004.py"
spec = importlib.util.spec_from_file_location("profile_t3_v004", PATH)
profile = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = profile
spec.loader.exec_module(profile)


def fake_stats():
    return SimpleNamespace(total_tt=10.0, stats={
        ("/app/cpu_agent.py", 323, "simulate"): (1, 1, 0.1, 5.0, {}),
        ("/app/cpu_agent.py", 193, "fast_extract_trace"): (1, 1, 0.05, 0.25, {}),
        ("/app/cpu_agent.py", 272, "fast_extract_message_trace"): (1, 1, 0.04, 0.15, {}),
        ("/app/vendor/track3_runtime/abides_fork/trace.py", 233, "extract_message_trace"): (1, 1, 0.01, 0.01, {}),
        ("/app/vendor/track3_runtime/abides_fork/trace.py", 200, "fast_extract_trace"): (1, 1, 0.1, 3.0, {}),
        ("/audit/profile_worker.py", 1, "worker_main"): (1, 1, 0.1, 9.0, {}),
        ("/private/credential_store.py", 99, "sensitive_function"): (1, 1, 1.0, 8.0, {}),
        **{(f"/usr/local/lib/python3.11/site-packages/pandas/core/frame{i}.py", 2, "construct"):
           (2, 3, 0.02, 0.2, {}) for i in range(30)},
    })


def bound_result(case):
    binding = profile.CASES[case]
    return {"kind": "instrumented_cpu_profile", "rankable": False,
            "runtime_source_sha256": profile.SOURCE_SHA256,
            "scenario_sha256": binding["scenario_sha256"],
            "outputs": {k: binding[k] for k in ("trace_sha256", "message_trace_sha256", "events", "messages")}
                       | {"output_bytes_including_profile": 12345, "profile_bytes": 111}}


def image_detail():
    return {"Id": profile.IMAGE_ID, "Os": "linux", "Architecture": "amd64",
            "RepoDigests": [profile.IMAGE], "Config": {"Volumes": None, "Labels": {
                "org.opencontainers.image.revision": profile.RUNTIME_REVISION,
                "qfbench2.interface_version": "2.0"}}}


class ProfilingTests(unittest.TestCase):
    def test_container_cpu_offline_read_only_hard_caps_and_mount_scope(self):
        command = profile.docker_command(Path("inputs"), Path("worker.py"), "test-profile")
        for key, value in (("--cpus", "4"), ("--memory", "8g"), ("--memory-swap", "8g"),
                           ("--network", "none"), ("--pids-limit", "128"), ("--pull", "never"),
                           ("--platform", "linux/amd64"), ("--cap-drop", "ALL"),
                           ("--security-opt", "no-new-privileges")):
            self.assertEqual(command[command.index(key) + 1], value)
        self.assertIn("--read-only", command)
        self.assertIn("/output:rw,nosuid,size=256m", command)
        self.assertIn("/tmp:rw,nosuid,size=64m", command)
        self.assertEqual(command[-3:], ["python", profile.IMAGE, "/audit/profile_worker.py"])
        self.assertNotIn("--gpus", command)
        self.assertNotIn("--privileged", command)
        envs = [command[i + 1] for i, value in enumerate(command) if value == "--env"]
        for value in ("CUDA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=void", "OMP_NUM_THREADS=1",
                      "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1"):
            self.assertIn(value, envs)
        mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
        self.assertEqual(len(mounts), 2)
        self.assertTrue(all(value.endswith(",readonly") for value in mounts))
        self.assertTrue(any("dst=/input," in value for value in mounts))
        self.assertTrue(any("dst=/audit/profile_worker.py," in value for value in mounts))
        self.assertFalse(any("reference" in value or "docker.sock" in value or "dst=/app" in value
                             for value in mounts))

    def test_actual_v004_trace_target_and_ledger_context_are_selected(self):
        result = profile.profile_summary(fake_stats())
        self.assertEqual(result["whole_profile_seconds"], 10.0)
        self.assertEqual(result["candidate_simulate_seconds"], 5.0)
        self.assertEqual(result["trace_fraction_of_simulate"], 0.05)
        self.assertEqual(result["ledger_fraction_of_simulate"], 0.03)
        self.assertEqual(result["target_function_row"]["function"], "fast_extract_trace")
        self.assertEqual(result["target_function_row"]["file"], "cpu_agent.py")
        self.assertEqual(result["ledger_context_row"]["function"], "fast_extract_message_trace")
        self.assertLessEqual(len(result["top_cumulative"]), 12)
        self.assertLessEqual(len(result["top_own"]), 12)
        rendered = json.dumps(result)
        for excluded in ("credential_store", "worker_main", "/usr/local"):
            self.assertNotIn(excluded, rendered)

    def test_vendor_or_similar_function_cannot_replace_actual_target(self):
        data = fake_stats()
        del data.stats[("/app/cpu_agent.py", 193, "fast_extract_trace")]
        data.stats[("/app/cpu_agent.py", 199, "fast_extract_trace_wrapper")] = (1, 1, 0.1, 4.0, {})
        with self.assertRaisesRegex(ValueError, "exactly one"):
            profile.profile_summary(data)

    def test_missing_duplicate_or_repeated_target_calls_fail(self):
        for name in ("simulate", "fast_extract_trace", "fast_extract_message_trace"):
            for mutation in ("missing", "duplicate", "repeated", "recursive"):
                with self.subTest(target=name, mutation=mutation):
                    data = fake_stats()
                    key = next(k for k in data.stats if k[0] == "/app/cpu_agent.py" and k[2] == name)
                    row = data.stats[key]
                    if mutation == "missing":
                        del data.stats[key]
                    elif mutation == "duplicate":
                        data.stats[(key[0], key[1] + 1, key[2])] = row
                    elif mutation == "repeated":
                        data.stats[key] = (2, 2, *row[2:])
                    else:
                        data.stats[key] = (1, 2, *row[2:])
                    with self.assertRaisesRegex(ValueError, "exactly one"):
                        profile.profile_summary(data)

    def test_threshold_uses_trace_only_and_requires_both_fixed_cases(self):
        def rows(ratios):
            return [{"case": name, "profile": {"trace_fraction_of_simulate": ratio,
                                               "ledger_fraction_of_simulate": 0.99}}
                    for name, ratio in zip(profile.CASES, ratios)]
        for ratios, expected in (((0.05, 0.03), True), ((0.03, 0.05), True),
                                 ((0.049, 0.049), False), ((0.06, 0.029), False)):
            with self.subTest(ratios=ratios):
                decision = profile.interpret(rows(ratios))
                self.assertEqual(decision["trace_experiment_justified"], expected)
                self.assertIs(decision["speedup_claim"], False)
        for incomplete in (rows((0.06,)), rows((0.05, 0.03))[::-1]):
            with self.assertRaisesRegex(ValueError, "both fixed"):
                profile.interpret(incomplete)
        for invalid in (float("nan"), float("inf"), 0, -0.1, 1.1, True, "0.05"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                profile.interpret(rows((invalid, 0.03)))

    def test_invalid_profile_totals_fail(self):
        for value in (0, -1, float("nan"), float("inf"), True, 6.0):
            with self.subTest(value=value):
                data = fake_stats()
                data.stats[("/app/cpu_agent.py", 193, "fast_extract_trace")] = (1, 1, 0.1, value, {})
                with self.assertRaisesRegex(ValueError, "invalid profile"):
                    profile.profile_summary(data)

    def test_bindings_reject_changed_bytes_counts_source_scenario_and_caps(self):
        for case in profile.CASES:
            expected = bound_result(case)
            profile.validate_bindings(case, expected)
            for location, key, value in (("outputs", "trace_sha256", "0" * 64),
                                         ("outputs", "message_trace_sha256", "0" * 64),
                                         ("outputs", "messages", 1),
                                         ("outputs", "events", float(expected["outputs"]["events"])),
                                         (None, "runtime_source_sha256", "0" * 64),
                                         (None, "scenario_sha256", "0" * 64), (None, "rankable", True),
                                         (None, "kind", "benchmark"),
                                         ("outputs", "output_bytes_including_profile", profile.OUTPUT_CAP + 1),
                                         ("outputs", "output_bytes_including_profile", float("nan")),
                                         ("outputs", "profile_bytes", True),
                                         ("outputs", "profile_bytes", 0),
                                         ("outputs", "profile_bytes", 12346)):
                with self.subTest(case=case, key=key, value=value):
                    changed = copy.deepcopy(expected)
                    (changed[location] if location else changed)[key] = value
                    with self.assertRaises(ValueError):
                        profile.validate_bindings(case, changed)

    def test_preserved_cases_and_bindings_equal_prior_v003_profiler(self):
        prior_spec = importlib.util.spec_from_file_location("prior_t3_profile", PATH.parent / "profile_t3_linux.py")
        prior = importlib.util.module_from_spec(prior_spec)
        prior_spec.loader.exec_module(prior)
        self.assertEqual(profile.CASES, prior.CASES)
        self.assertNotEqual(profile.SOURCE_SHA256, prior.SOURCE_SHA256)
        self.assertNotEqual(profile.IMAGE, prior.IMAGE)

    def test_image_id_digest_labels_architecture_and_no_volume_are_bound(self):
        expected = image_detail()
        profile.validate_image(expected)
        for location, key, value in ((None, "Id", "sha256:" + "0" * 64),
                                     (None, "RepoDigests", []), (None, "Os", "windows"),
                                     (None, "Architecture", "arm64"),
                                     ("Config", "Volumes", {"/data": {}}),
                                     ("Labels", "org.opencontainers.image.revision", "0" * 40),
                                     ("Labels", "qfbench2.interface_version", "1.0")):
            with self.subTest(key=key):
                changed = copy.deepcopy(expected)
                target = changed if location is None else changed["Config"] if location == "Config" else changed["Config"]["Labels"]
                target[key] = value
                with self.assertRaisesRegex(ValueError, "immutable verified"):
                    profile.validate_image(changed)

    def test_generated_worker_has_no_expected_bindings_and_compiles(self):
        source = profile.worker_source()
        compile(source, "profile_worker.py", "exec")
        self.assertIn("cProfile.Profile()", source)
        self.assertIn("fast_extract_trace", source)
        self.assertIn("fast_extract_message_trace", source)
        for excluded in ("raw.githubusercontent", profile.IMAGE, profile.IMAGE_ID, profile.SOURCE_SHA256,
                         profile.RUNTIME_REVISION, "CASES", "validate_bindings"):
            self.assertNotIn(excluded, source)
        for value in profile.CASES.values():
            for key in ("scenario_sha256", "trace_sha256", "message_trace_sha256"):
                self.assertNotIn(value[key], source)

    def test_console_capture_is_bounded(self):
        capture = profile.CaptureTail()
        self.assertEqual(capture.write("x" * 20000), 20000)
        self.assertEqual(len(capture.tail), 8192)
        self.assertEqual(capture.characters, 20000)

    def test_timeout_stops_case_and_forces_container_cleanup_without_retry(self):
        scenario_bytes = b'{"scenario_id":"offline-test","seed":7}'
        response = mock.MagicMock()
        response.__enter__.return_value.read.return_value = scenario_bytes
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            if command[:2] == ["docker", "run"]:
                raise subprocess.TimeoutExpired(command, kwargs["timeout"])
            return subprocess.CompletedProcess(command, 0, "", "")

        fake_cases = {"offline-test": {"scenario_sha256": hashlib.sha256(scenario_bytes).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(profile.sys, "platform", "linux"), \
                mock.patch.object(profile, "CASES", fake_cases), \
                mock.patch.object(profile, "urlopen", return_value=response), \
                mock.patch.object(profile.subprocess, "check_output", return_value=json.dumps([image_detail()])), \
                mock.patch.object(profile.subprocess, "run", side_effect=fake_run), \
                mock.patch("builtins.print"):
            with self.assertRaises(subprocess.TimeoutExpired):
                profile.run_profile(Path(directory) / "new-work")
        runs = [(c, k) for c, k in calls if c[:2] == ["docker", "run"]]
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0][1]["timeout"], 300)
        self.assertEqual(calls[-1][0], ["docker", "rm", "--force", "agenthon-profile-v004-offline-test"])
        self.assertEqual(calls[-1][1]["timeout"], 15)

    def test_workflow_is_only_public_dedicated_branch_and_has_no_billed_storage(self):
        text = (PATH.parent / "cpu-profile-v004.yml").read_text()
        for required in ("branches: [t3-profile-v004]", "timeout-minutes: 15",
                         "github.event.repository.private == false", "runs-on: ubuntu-24.04",
                         "deployment/profile_t3_v004.py", "test_t3_v004_profile.py"):
            self.assertIn(required, text)
        for forbidden in ("deployment/**", "cpu-build.yml", "upload-artifact", "actions/cache",
                          "packages: write", "self-hosted"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
