"""Offline mocks only: never invoke Docker, a registry or any submission API."""
from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

PATH = Path(__file__).resolve().parents[1] / "deployment/push_public_image.py"
spec = importlib.util.spec_from_file_location("push_public_image", PATH)
push = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = push
spec.loader.exec_module(push)
IMAGE = "ghcr.io/ericchan2028/eastern-eagles-t3:sha-" + "a" * 40


class FakeProcess:
    def __init__(self, output, returncode=0):
        self.stdout = io.BytesIO(output)
        self.returncode = returncode
        self.killed = False

    def poll(self):
        return self.returncode

    def wait(self, timeout):
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class PushTests(unittest.TestCase):
    def test_success_stops_even_if_message_contains_unknown_blob(self):
        with patch.object(push, "run_push", return_value=(0, b"unknown blob recovered")) as run, patch.object(push.time, "sleep") as sleep:
            self.assertEqual(push.push_image(IMAGE), 0)
        run.assert_called_once_with(IMAGE)
        sleep.assert_not_called()

    def test_transient_unknown_blob_recovers_same_exact_image(self):
        with patch.object(push, "run_push", side_effect=[(1, b"unknown blob"), (0, b"pushed")]) as run:
            with patch.object(push.time, "sleep") as sleep, patch("builtins.print"):
                self.assertEqual(push.push_image(IMAGE), 0)
        self.assertEqual([call.args for call in run.call_args_list], [(IMAGE,), (IMAGE,)])
        sleep.assert_called_once_with(10)

    def test_retry_cap_is_three_attempts_and_two_sleeps(self):
        with patch.object(push, "run_push", return_value=(7, b"unknown blob")) as run:
            with patch.object(push.time, "sleep") as sleep, patch("builtins.print"):
                self.assertEqual(push.push_image(IMAGE), 7)
        self.assertEqual(run.call_count, 3)
        self.assertEqual([call.args for call in run.call_args_list], [(IMAGE,)] * 3)
        self.assertEqual([call.args for call in sleep.call_args_list], [(10,), (10,)])

    def test_no_retry_authentication_other_error_or_wrong_literal(self):
        for output in (b"unauthorized: authentication required", b"denied: permission denied", b"TLS timeout", b"Unknown Blob"):
            with self.subTest(output=output):
                with patch.object(push, "run_push", return_value=(1, output)) as run, patch.object(push.time, "sleep") as sleep:
                    self.assertEqual(push.push_image(IMAGE), 1)
                run.assert_called_once_with(IMAGE)
                sleep.assert_not_called()

    def test_timeout_and_capture_overflow_never_retry(self):
        for error in (subprocess.TimeoutExpired(["docker", "push", IMAGE], 300), push.PushOutputLimitError("cap")):
            with self.subTest(error=type(error).__name__):
                with patch.object(push, "run_push", side_effect=error) as run, patch.object(push.time, "sleep") as sleep:
                    with self.assertRaises(type(error)):
                        push.push_image(IMAGE)
                run.assert_called_once_with(IMAGE)
                sleep.assert_not_called()

    def test_auth_error_after_unknown_blob_stops_without_third_attempt(self):
        with patch.object(push, "run_push", side_effect=[(1, b"unknown blob"), (1, b"unauthorized")]) as run:
            with patch.object(push.time, "sleep") as sleep, patch("builtins.print"):
                self.assertEqual(push.push_image(IMAGE), 1)
        self.assertEqual(run.call_count, 2)
        sleep.assert_called_once_with(10)

    def test_unknown_cli_arguments_are_not_echoed(self):
        with patch.object(push.sys, "argv", ["push_public_image.py", IMAGE, "--password", "test-sensitive-value"]):
            with patch.object(push.sys, "stderr", io.StringIO()) as error:
                with self.assertRaises(SystemExit):
                    push.main()
                self.assertNotIn("test-sensitive-value", error.getvalue())

    def test_registry_tag_validation_rejects_credentials_and_shell_text(self):
        invalid = ["docker.io/owner/image:sha-" + "a" * 40, "ghcr.io/image:sha-" + "a" * 40,
                   "ghcr.io/owner/image:latest", "ghcr.io/user:password@owner/image:sha-" + "a" * 40,
                   IMAGE + "; echo bad", IMAGE + "\n", IMAGE[:-1], IMAGE + " --password secret"]
        with patch.object(push, "run_push") as run:
            for image in invalid:
                with self.subTest(image=image), self.assertRaises(ValueError):
                    push.push_image(image)
        run.assert_not_called()

    def test_stream_uses_argv_tees_progress_and_sets_300_second_timer(self):
        process = FakeProcess(b"layer: pushed\nunknown blob\n", returncode=1)
        timer = Mock()
        with patch.object(push.subprocess, "Popen", return_value=process) as popen:
            with patch.object(push.threading, "Timer", return_value=timer) as make_timer, patch.object(push.sys, "stdout", io.StringIO()) as output:
                result = push.run_push(IMAGE)
                progress = output.getvalue()
        self.assertEqual(result, (1, b"layer: pushed\nunknown blob\n"))
        self.assertEqual(progress, result[1].decode())
        self.assertEqual(popen.call_args.args[0], ["docker", "push", IMAGE])
        self.assertNotIn("shell", popen.call_args.kwargs)
        self.assertEqual(make_timer.call_args.args[0], 300)
        timer.start.assert_called_once()
        timer.cancel.assert_called_once()

    def test_stream_capture_never_exceeds_one_mib(self):
        process = FakeProcess(b"x" * (push.MAX_OUTPUT + 8192), returncode=None)
        with patch.object(push.subprocess, "Popen", return_value=process), patch.object(push.threading, "Timer", return_value=Mock()):
            with patch.object(push.sys, "stdout", io.StringIO()) as output:
                with self.assertRaises(push.PushOutputLimitError):
                    push.run_push(IMAGE)
                self.assertEqual(len(output.getvalue()), push.MAX_OUTPUT)
        self.assertTrue(process.killed)

    def test_expired_timer_kills_process_and_raises_timeout(self):
        process = FakeProcess(b"", returncode=None)
        def timer_factory(seconds, callback):
            timer = Mock()
            timer.start.side_effect = callback
            return timer
        with patch.object(push.subprocess, "Popen", return_value=process), patch.object(push.threading, "Timer", side_effect=timer_factory):
            with self.assertRaises(subprocess.TimeoutExpired) as caught:
                push.run_push(IMAGE)
        self.assertTrue(process.killed)
        self.assertEqual(caught.exception.timeout, 300)


if __name__ == "__main__":
    unittest.main()
