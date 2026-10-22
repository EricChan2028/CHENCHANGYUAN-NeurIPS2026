"""Push one immutable-source GHCR tag; retry only Docker's 'unknown blob' failure."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time

MAX_OUTPUT = 1024 * 1024
PUSH_TIMEOUT = 300
MAX_ATTEMPTS = 3
RETRY_DELAY = 10
IMAGE_PATTERN = re.compile(r"ghcr\.io/[a-z0-9][a-z0-9._-]*/(?:[a-z0-9][a-z0-9._-]*/)*[a-z0-9][a-z0-9._-]*:sha-[0-9a-f]{40}\Z")


class PushOutputLimitError(RuntimeError):
    pass


def run_push(image):
    """Tee ordinary push progress while capturing at most one MiB for classification."""
    command = ["docker", "push", image]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
    expired = threading.Event()
    output = bytearray()

    def timeout():
        expired.set()
        if process.poll() is None:
            process.kill()

    timer = threading.Timer(PUSH_TIMEOUT, timeout)
    timer.daemon = True
    timer.start()
    try:
        while chunk := process.stdout.read(8192):
            remaining = MAX_OUTPUT - len(output)
            kept = chunk[:remaining]
            output.extend(kept)
            target = getattr(sys.stdout, "buffer", None)
            if target is not None:
                target.write(kept)
                target.flush()
            else:
                sys.stdout.write(kept.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            if len(chunk) > remaining:
                raise PushOutputLimitError("Docker push output exceeded 1 MiB; no retry")
        returncode = process.wait(timeout=10)
        if expired.is_set():
            raise subprocess.TimeoutExpired(command, PUSH_TIMEOUT)
        return returncode, bytes(output)
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        process.stdout.close()


def push_image(image):
    if not IMAGE_PATTERN.fullmatch(image):
        raise ValueError("Expected ghcr.io/owner/image:sha- followed by 40 lowercase hexadecimal characters")
    for attempt in range(1, MAX_ATTEMPTS + 1):
        returncode, output = run_push(image)
        if returncode == 0:
            return 0
        if b"unknown blob" not in output or attempt == MAX_ATTEMPTS:
            return returncode
        print(f"Retrying the same image after 'unknown blob' ({attempt}/{MAX_ATTEMPTS}); waiting {RETRY_DELAY}s.", flush=True)
        time.sleep(RETRY_DELAY)
    raise AssertionError("unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="Public GHCR image tag tied to a 40-character source commit")
    args, extra = parser.parse_known_args()
    if extra:
        parser.error("Only the image argument is accepted")
    try:
        return push_image(args.image)
    except (ValueError, PushOutputLimitError, subprocess.TimeoutExpired, OSError) as error:
        print(f"Image push failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
