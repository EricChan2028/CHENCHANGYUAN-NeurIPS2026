"""Offline tests for exact getter targeting, two-pass separation and fixed gates."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

PATH = Path(__file__).resolve().parents[1] / "deployment/profile_t3_pricelevel.py"
sys.path.insert(0, str(PATH.parent))
spec = importlib.util.spec_from_file_location("profile_t3_pricelevel", PATH)
profile = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = profile
spec.loader.exec_module(profile)


class PriceLevel:
    def __init__(self, quantities):
        self.visible_orders = quantities
        self.getter_calls = 0

    @property
    def total_quantity(self):
        self.getter_calls += 1
        if self.visible_orders == ["raise"]:
            raise ValueError("original exception")
        return sum(self.visible_orders)


def result(case, mode, fraction=0.05, short_calls=75):
    binding = profile.CASES[case]
    value = {"kind": "pricelevel_diagnostic", "case": case, "mode": mode, "rankable": False,
             "runtime_source_sha256": profile.SOURCE_SHA256,
             "dependency_source_sha256": profile.DEPENDENCY_SHA256,
             "dependency_lf_source_sha256": profile.DEPENDENCY_SHA256,
             "getter_source_sha256": profile.GETTER_SHA256,
             "dependency_versions": profile.VERSIONS.copy(), "scenario_sha256": binding["scenario_sha256"],
             "getter_wrapper_installed": mode == "histogram", "cprofile_enabled": mode == "profile",
             "getter_code_unchanged": True, "original_property_restored": True,
             "outputs": {key: binding[key] for key in ("trace_sha256", "message_trace_sha256", "events", "messages")}
                        | {"output_bytes": 1000, "profile_bytes": 100 if mode == "profile" else 0}}
    if mode == "profile":
        value["profile"] = {"getter_fraction_of_simulate": fraction,
                            "target_function_row": {"file": "abides_markets/price_level.py", "function": "total_quantity",
                                                    "calls": 100, "primitive_calls": 100}}
    else:
        value["histogram"] = {"calls": 100, "delegations": 100, "maximum_length": 3,
                              "buckets": {"0": 0, "1": short_calls, "2": 0, "3-7": 100 - short_calls,
                                          "8-31": 0, "32+": 0}}
    return value


def all_results(fractions=(0.05, 0.05), shares=(75, 75)):
    rows = []
    for index, case in enumerate(profile.CASES):
        for mode in profile.MODES:
            rows.append(result(case, mode, fractions[index], shares[index]))
    return rows


class DiagnosticTests(unittest.TestCase):
    def test_target_is_exact_original_module_line_and_name(self):
        filename = "/usr/local/lib/python3.11/site-packages/abides_markets/price_level.py"
        stats = SimpleNamespace(total_tt=11.0, stats={
            ("/app/cpu_agent.py", 323, "simulate"): (1, 1, 0.2, 10.0, {}),
            (filename, 260, "total_quantity"): (100, 100, 0.3, 0.5, {}),
            ("/another/price_level.py", 260, "total_quantity"): (100, 100, 8.0, 9.0, {}),
            ("/audit/profile_worker.py", 75, "counted"): (100, 100, 0.1, 0.2, {}),
        })
        summary = profile.profile_summary(stats, filename, 260)
        self.assertEqual(summary["getter_fraction_of_simulate"], 0.05)
        self.assertEqual(summary["target_function_row"]["file"], "abides_markets/price_level.py")
        with self.assertRaisesRegex(ValueError, "exact unwrapped"):
            profile.profile_summary(stats, filename, 261)
        del stats.stats[(filename, 260, "total_quantity")]
        with self.assertRaisesRegex(ValueError, "exact unwrapped"):
            profile.profile_summary(stats, filename, 260)

    def test_histogram_delegates_exactly_once_without_changing_values_or_inputs(self):
        original = PriceLevel.total_quantity
        wrapped, histogram = profile.histogram_property(original)
        for length in (0, 1, 2, 3, 7, 8, 31, 32, 100):
            instance = PriceLevel(list(range(length)))
            before = list(instance.visible_orders)
            self.assertEqual(wrapped.fget(instance), sum(before))
            self.assertEqual(instance.getter_calls, 1)
            self.assertEqual(instance.visible_orders, before)
        self.assertEqual(histogram, {"calls": 9, "delegations": 9, "maximum_length": 100,
                                    "buckets": {"0": 1, "1": 1, "2": 1, "3-7": 2, "8-31": 2, "32+": 2}})
        instance = PriceLevel(["raise"])
        with self.assertRaisesRegex(ValueError, "original exception"):
            wrapped.fget(instance)
        self.assertEqual(instance.getter_calls, 1)
        self.assertIs(PriceLevel.total_quantity, original)

    def run_worker(self, mode, fail=False):
        original = PriceLevel.total_quantity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "input").mkdir()
            (root / "output").mkdir()
            (root / "app").mkdir()
            (root / "input/scenario.json").write_text('{"scenario_id":"test","seed":7}')
            (root / "app/cpu_agent.py").write_text("pass\n")
            def mapped_path(value):
                return root / str(value).lstrip("/") if str(value).startswith(("/app", "/input", "/output")) else Path(value)
            def run_path(*args, **kwargs):
                self.assertEqual(PriceLevel.total_quantity is original, mode == "profile")
                instance = PriceLevel([2, 3])
                self.assertEqual(instance.total_quantity, 5)
                self.assertEqual(instance.getter_calls, 1)
                if fail:
                    raise ValueError("simulation fixture failure")
            fake_profiler = mock.MagicMock()
            fake_modules = {name: SimpleNamespace(__version__=version) for name, version in profile.VERSIONS.items()}
            fake_modules["abides_markets.price_level"] = SimpleNamespace(PriceLevel=PriceLevel)
            with mock.patch.dict(sys.modules, fake_modules), mock.patch.object(profile, "Path", side_effect=mapped_path), \
                    mock.patch.object(sys, "argv", ["worker.py", mode]), \
                    mock.patch("cProfile.Profile", return_value=fake_profiler) as profiler_constructor, \
                    mock.patch("pstats.Stats"), mock.patch.object(profile, "profile_summary", return_value={"fake": True}), \
                    mock.patch.object(profile, "collect_output", return_value={"fake": True}), \
                    mock.patch("runpy.run_path", side_effect=run_path), mock.patch("builtins.print") as printed:
                if fail:
                    with self.assertRaisesRegex(ValueError, "fixture failure"):
                        profile.worker_main()
                    payload = None
                else:
                    profile.worker_main()
                    line = printed.call_args.args[0]
                    self.assertTrue(line.startswith(profile.SENTINEL))
                    payload = json.loads(line[len(profile.SENTINEL):])
                self.assertIs(PriceLevel.total_quantity, original)
                self.assertEqual(profiler_constructor.call_count, int(mode == "profile"))
                self.assertEqual(fake_profiler.enable.call_count, int(mode == "profile"))
                self.assertEqual(fake_profiler.disable.call_count, int(mode == "profile"))
            return payload

    def test_worker_profile_pass_never_wraps_getter(self):
        payload = self.run_worker("profile")
        self.assertFalse(payload["getter_wrapper_installed"])
        self.assertTrue(payload["cprofile_enabled"])
        self.assertNotIn("histogram", payload)

    def test_worker_histogram_pass_has_no_profiler_and_restores_property(self):
        payload = self.run_worker("histogram")
        self.assertTrue(payload["getter_wrapper_installed"])
        self.assertFalse(payload["cprofile_enabled"])
        self.assertNotIn("profile", payload)
        self.assertEqual(payload["histogram"]["calls"], 1)
        self.assertEqual(payload["histogram"]["buckets"]["2"], 1)

    def test_property_restored_and_profiler_disabled_after_exception(self):
        self.run_worker("histogram", fail=True)
        self.run_worker("profile", fail=True)

    def test_fixed_both_case_profile_and_length_gates(self):
        self.assertTrue(profile.interpret(all_results())["component_experiment_justified"])
        for fractions, shares in (((0.04999, 0.10), (75, 99)), ((0.10, 0.04999), (99, 75)),
                                  ((0.05, 0.05), (74, 99)), ((0.05, 0.05), (99, 74))):
            with self.subTest(fractions=fractions, shares=shares):
                verdict = profile.interpret(all_results(fractions, shares))
                self.assertFalse(verdict["component_experiment_justified"])
                self.assertFalse(verdict["speedup_claim"])
        self.assertEqual(profile.GATES["later_component_rate_ratio_min_each"], 2.5)
        self.assertEqual(profile.GATES["later_whole_container_mean_eps_ratio_min"], 1.03)

    def test_partial_reordered_mismatched_calls_and_bad_histograms_fail(self):
        for rows in (all_results()[:-1], all_results()[::-1]):
            with self.assertRaisesRegex(ValueError, "both exact"):
                profile.interpret(rows)
        rows = all_results()
        rows[0]["profile"]["target_function_row"].update(calls=99, primitive_calls=99)
        with self.assertRaisesRegex(ValueError, "call counts differ"):
            profile.interpret(rows)
        for key, value in (("calls", 99), ("delegations", 99), ("delegations", True), ("maximum_length", 2)):
            rows = all_results()
            rows[1]["histogram"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                profile.interpret(rows)
        rows = all_results()
        rows[1]["histogram"]["buckets"]["1"] = -1
        with self.assertRaises(ValueError):
            profile.interpret(rows)

    def test_exact_runtime_dependency_getter_and_outputs_bindings(self):
        case = next(iter(profile.CASES))
        for mode in profile.MODES:
            expected = result(case, mode)
            profile.validate_result(case, mode, expected)
            for key in ("runtime_source_sha256", "dependency_source_sha256", "dependency_lf_source_sha256",
                        "getter_source_sha256", "scenario_sha256"):
                changed = copy.deepcopy(expected)
                changed[key] = "0" * 64
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    profile.validate_result(case, mode, changed)
            for key, value in (("trace_sha256", "0" * 64), ("message_trace_sha256", "0" * 64),
                               ("events", float(expected["outputs"]["events"])), ("messages", 1),
                               ("output_bytes", profile.OUTPUT_CAP + 1), ("output_bytes", True)):
                changed = copy.deepcopy(expected)
                changed["outputs"][key] = value
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    profile.validate_result(case, mode, changed)

    def test_mixed_pass_flags_or_payloads_fail(self):
        case = next(iter(profile.CASES))
        for mode in profile.MODES:
            for key in ("getter_wrapper_installed", "cprofile_enabled", "getter_code_unchanged", "original_property_restored"):
                changed = result(case, mode)
                changed[key] = not changed[key]
                with self.subTest(mode=mode, key=key), self.assertRaises(ValueError):
                    profile.validate_result(case, mode, changed)
            changed = result(case, mode)
            changed["histogram" if mode == "profile" else "profile"] = {}
            with self.assertRaises(ValueError):
                profile.validate_result(case, mode, changed)

    def test_worker_has_no_expected_bindings_or_registry_credentials(self):
        source = profile.worker_source()
        compile(source, "profile_worker.py", "exec")
        for value in (profile.IMAGE, profile.SOURCE_SHA256, profile.DEPENDENCY_SHA256, profile.GETTER_SHA256,
                      "raw.githubusercontent", "GHCR_TOKEN", "CASES"):
            self.assertNotIn(value, source)
        for expected in profile.CASES.values():
            for key in ("trace_sha256", "message_trace_sha256", "scenario_sha256"):
                self.assertNotIn(expected[key], source)

    def test_cpu_resources_and_only_input_worker_mounts(self):
        for mode in profile.MODES:
            command = profile.docker_command(Path("input"), Path("worker.py"), "test", mode)
            for flag, value in (("--cpus", "4"), ("--memory", "8g"), ("--memory-swap", "8g"),
                                ("--network", "none"), ("--pull", "never"), ("--pids-limit", "128")):
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--read-only", command)
            self.assertIn("/tmp:rw,nosuid,size=64m", command)
            self.assertIn("/output:rw,nosuid,size=256m", command)
            mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
            self.assertEqual(len(mounts), 2)
            self.assertTrue(all(value.endswith(",readonly") for value in mounts))
            self.assertFalse(any("dst=/app" in value for value in mounts))
            self.assertNotIn("--gpus", command)
            self.assertEqual(command[-3:], [profile.IMAGE_ID, "/audit/profile_worker.py", mode])

    def test_protocol_saved_before_any_diagnostic_or_network_fetch(self):
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory) / "work"
            def stop(*args, **kwargs):
                report = json.loads((work / "t3-pricelevel-profile.json").read_text())
                self.assertEqual(report["status"], "preparing")
                self.assertEqual(report["protocol"]["gates"], profile.GATES)
                self.assertEqual(report["protocol"]["schedule"], profile.schedule())
                raise RuntimeError("offline test stops before Docker")
            with mock.patch.object(profile.sys, "platform", "linux"), \
                    mock.patch.object(profile.subprocess, "run", side_effect=stop), mock.patch("builtins.print"):
                with self.assertRaisesRegex(RuntimeError, "offline test"):
                    profile.run_diagnostics(work)
            report = json.loads((work / "t3-pricelevel-profile.json").read_text())
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["results"], [])

    def test_workflow_is_dedicated_readonly_public_cpu_no_build_publish_storage(self):
        text = (PATH.parent / "cpu-pricelevel-profile.yml").read_text()
        for value in ("branches: [t3-pricelevel-profile]", "refs/heads/t3-pricelevel-profile",
                      "github.event.repository.private == false", "runs-on: ubuntu-24.04", "timeout-minutes: 20"):
            self.assertIn(value, text)
        for value in ("docker build", "docker push", "packages: write", "upload-artifact", "actions/cache", "self-hosted"):
            self.assertNotIn(value, text)


if __name__ == "__main__":
    unittest.main()
