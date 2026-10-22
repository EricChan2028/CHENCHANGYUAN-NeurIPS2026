"""Real Linux container checks; public scoring inputs stay outside submitted images."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid


UPSTREAM = {
    "t2": ("https://github.com/Agenthon-2026/track2-forecasting-public.git",
           "3b51e16b3f7c0fa0e56bcba323c2bb5e81b8bd56"),
    "t3": ("https://github.com/Agenthon-2026/track3-simulation-public.git",
           "2fa71ea2468db98bb1b1d06339a87d7e3a9d651c"),
}
T3_UNITS = ("t3-stp-cancel-newest", "t3-gbatch-dense-3")
OUTPUT_LIMITS = {"t2": 64 * 1024 * 1024, "t3": 256 * 1024 * 1024}


def run(argv: list[str], *, capture: bool = False, timeout: int = 600) -> str:
    result = subprocess.run(argv, check=True, text=True, timeout=timeout,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout or ""


def fetch(track: str, work: Path) -> None:
    target = work / track
    if target.exists():
        raise ValueError("validator checkout already exists; use a fresh work directory")
    target.mkdir(parents=True)
    if track == "t3":
        # The standard Ubuntu runner currently provides Git LFS; fail if it does not.
        run(["git", "lfs", "version"])
    url, commit = UPSTREAM[track]
    run(["git", "init", str(target)])
    run(["git", "-C", str(target), "remote", "add", "origin", url])
    run(["git", "-C", str(target), "fetch", "--depth", "1", "origin", commit])
    # Avoid downloading the entire public T3 reference corpus. Pull only our two
    # selected public smoke units after checkout, and reject unresolved pointers.
    run(["git", "-C", str(target), "-c", "filter.lfs.smudge=",
         "-c", "filter.lfs.process=", "-c", "filter.lfs.required=false",
         "checkout", "--detach", "FETCH_HEAD"])
    if track == "t3":
        run(["git", "-C", str(target), "lfs", "install", "--local"])
        run(["git", "-C", str(target), "lfs", "pull", "--include",
             ",".join(f"units/{unit}/**" for unit in T3_UNITS), "--exclude", ""])
        for name in T3_UNITS:
            parquets = list((target / "units" / name).rglob("*.parquet"))
            if not parquets:
                raise ValueError("required public T3 reference parquet is absent")
            for path in parquets:
                with path.open("rb") as handle:
                    if handle.read(4) != b"PAR1":
                        raise ValueError("public fixture has an unresolved LFS pointer or invalid parquet")
    verify_checkout(track, target)


def verify_checkout(track: str, target: Path) -> None:
    actual = run(["git", "-C", str(target), "rev-parse", "HEAD"], capture=True).strip()
    if actual != UPSTREAM[track][1]:
        raise ValueError("unexpected official validator revision")
    if run(["git", "-C", str(target), "status", "--porcelain", "--untracked-files=no"], capture=True).strip():
        raise ValueError("official validator checkout has modifications")


def copy_input(source: Path, destination: Path) -> None:
    if source.is_symlink() or source.is_junction():
        raise ValueError("public fixture contains an unexpected link")
    if source.is_dir():
        destination.mkdir()
        for item in sorted(source.iterdir()):
            copy_input(item, destination / item.name)
    elif source.is_file():
        shutil.copyfile(source, destination)
    else:
        raise ValueError("public fixture contains an unexpected special file")


def stage_inputs(unit: Path, destination: Path, names: tuple[str, ...]) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    for name in names:
        source = unit / name
        if source.exists():
            copy_input(source, destination / name)
    return destination


def verify_public_inputs(unit: Path) -> dict:
    """Run the unchanged common card and manifest checks on the ORIGINAL unit.

    The reduced container mount intentionally omits reference files, so it is not
    the unit covered by the official manifest and must not be substituted here.
    """
    from qfbench2_common import manifest, taskcard

    card, card_errors = taskcard.load_and_validate(unit)
    if card_errors:
        raise ValueError("official input card validation failed:\n" + "\n".join(card_errors))
    manifest_errors = manifest.verify_manifest(unit)
    if manifest_errors:
        raise ValueError("official input manifest verification failed:\n" + "\n".join(manifest_errors))
    return card


def inspect_image(image: str) -> dict:
    detail = json.loads(run(["docker", "image", "inspect", image], capture=True))[0]
    if detail.get("Os") != "linux" or detail.get("Architecture") != "amd64":
        raise ValueError("submission must be a Linux/amd64 image")
    config = detail["Config"]
    if (config.get("Labels") or {}).get("qfbench2.interface_version") != "2.0":
        raise ValueError("missing qfbench2.interface_version=2.0 image label")
    if config.get("Volumes"):
        raise ValueError("Final rejects image VOLUME declarations, including inherited ones")
    return detail


def verify_output_sizes(outputs: Path, track: str) -> None:
    limit = OUTPUT_LIMITS[track]
    total = 0
    for path in outputs.rglob("*"):
        if path.is_symlink() or path.is_junction():
            raise ValueError("output links are not permitted")
        if path.is_file():
            size = path.stat().st_size
            if size > limit:
                raise ValueError("output file exceeds the published Final size cap")
            total += size
    if total > limit:
        raise ValueError("total output exceeds the published Final size cap")


def container_run(image: str, inputs: Path, outputs: Path, args: list[str], *, track: str) -> None:
    outputs.mkdir(parents=True, exist_ok=False)
    outputs.chmod(0o777)  # Permit the image's declared non-root user to write its bind mount.
    name = "agenthon-cpu-smoke-" + uuid.uuid4().hex
    argv = [
        "docker", "run", "--rm", "--name", name, "--platform", "linux/amd64",
        "--network", "none", "--read-only", "--cpus", "4", "--memory", "8g",
        "--memory-swap", "8g", "--pids-limit", "128", "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges", "--tmpfs", "/tmp:rw,nosuid,size=64m",
        "--mount", f"type=bind,src={inputs.resolve()},dst=/input,readonly",
        "--mount", f"type=bind,src={outputs.resolve()},dst=/output",
    ]
    for setting in (
        "CUDA_VISIBLE_DEVICES=", "NVIDIA_VISIBLE_DEVICES=void", "OMP_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1", "NUMEXPR_NUM_THREADS=1",
        "PYTHONDONTWRITEBYTECODE=1", "TMPDIR=/tmp",
    ):
        argv.extend(["--env", setting])
    try:
        run(argv + [image] + args, timeout=300)
    finally:
        # Also stop a still-running container after a subprocess timeout.
        subprocess.run(["docker", "rm", "--force", name], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=False, timeout=30)
    if not any(p.is_file() for p in outputs.rglob("*")):
        raise ValueError("container produced no files")
    verify_output_sizes(outputs, track)


def check(track: str, image: str, work: Path) -> None:
    if sys.platform != "linux":
        raise RuntimeError("container checks require Linux with a working Docker daemon")
    checkout = work / track
    verify_checkout(track, checkout)
    detail = inspect_image(image)
    checks: list[str] = []
    if track == "t2":
        # Exercise both branches: the v002 change is monthly, while the original
        # Treasury exemplar exercises the unchanged daily engine.
        for name, panel, label in (
            ("t2-EXAMPLE-ust-curve-1m", "rates_daily.parquet", "exemplar"),
            ("t2-F1-cpi-glidepath-2023", "macro_monthly.parquet", "monthly"),
            ("t2-F3-curve-ust3m-ust6m-ust1y-ust2y-005-ps1", "rates_daily.parquet", "short-ust"),
            ("t2-F3-fx-krw-inr-zar-017-ps1", "em_fx_ecb_daily.parquet", "short-fx"),
        ):
            unit = checkout / "units" / name
            card = verify_public_inputs(unit)
            inputs = stage_inputs(unit, work / f"runtime-inputs/t2-{label}", (
                "card.toml", "forecast_spec.json", "panels", "text",
            ))
            if not (inputs / "panels").exists():
                (inputs / "panels").mkdir()
                copy_input(unit / panel, inputs / "panels" / panel)
            outputs = work / f"runtime-outputs/t2-{label}"
            container_run(image, inputs, outputs, [
                "forecast", "--panels", "/input/panels", "--text", "/input/text",
                "--asof", card["provenance"]["data_cutoff"], "--out", "/output/forecast.parquet",
            ], track="t2")
            run([sys.executable, "-m", "qfbench2_track_forecasting.scoring", "score",
                 "--card", str(unit / "card.toml"), "--forecast", str(outputs / "forecast.parquet")])
            checks.append({"exemplar": "T2 official gates-only exemplar",
                           "monthly": "T2 official monthly gates-only unit",
                           "short-ust": "T2 Final short-horizon UST rehearsal",
                           "short-fx": "T2 Final short-horizon FX rehearsal"}[label])
        checks.append("T2 official input card and manifest validation on original unit")
        checks.append("T2 output file and total size caps checked")
    else:
        unit = checkout / "units/t3-stp-cancel-newest"
        inputs = stage_inputs(unit, work / "runtime-inputs/t3-single", ("scenario.json", "card.toml"))
        hashes: list[dict[str, str]] = []
        # One warm-up plus four measured-repeat equivalents. This checks output
        # stability, not the organizers' controlled Final throughput score.
        for repeat in range(5):
            outputs = work / f"runtime-outputs/t3-single-{repeat}"
            container_run(image, inputs, outputs, [
                "simulate", "--config", "/input/scenario.json", "--out", "/output/trace.parquet",
            ], track="t3")
            run(["qfbench2-smoke", str(unit), str(outputs), "--track", "simulation"])
            hashes.append({name: hashlib.sha256((outputs / name).read_bytes()).hexdigest()
                           for name in ("trace.parquet", "message_trace.parquet")})
        if any(value != hashes[0] for value in hashes[1:]):
            raise ValueError("T3 stable trace or ledger differs between identical repeats")
        checks.extend(["T3 single public semantic gates twice", "T3 byte-stable trace and ledger"])
        checks.append("T3 Final five-repeat byte-stability rehearsal")
        unit = checkout / "units/t3-gbatch-dense-3"
        inputs = stage_inputs(unit, work / "runtime-inputs/t3-batch", (
            "card.toml", "batch.json", "scenarios",
        ))
        outputs = work / "runtime-outputs/t3-batch"
        container_run(image, inputs, outputs, [
            "simulate-batch", "--batch-dir", "/input/scenarios", "--out-dir", "/output",
        ], track="t3")
        run(["qfbench2-smoke", str(unit), str(outputs), "--track", "simulation"])
        checks.append("T3 batch public isolation and semantic gates")
        checks.append("T3 output file and total size caps checked")
    report = {"track": track, "image_id": detail["Id"], "passed_checks": checks,
              "official_private_evaluation": False,
              "limitations": "Public-example checks only; no official score or rank."}
    (work / f"checked-{track}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


def published(track: str, image: str, work: Path) -> None:
    detail = inspect_image(image)
    checked = json.loads((work / f"checked-{track}.json").read_text(encoding="utf-8"))
    if checked["image_id"] != detail["Id"]:
        raise ValueError("published image differs from the container that passed checks")
    repository = image.rsplit(":", 1)[0]
    matching = [value for value in detail.get("RepoDigests", []) if value.startswith(repository + "@sha256:")]
    if len(matching) != 1:
        raise ValueError("cannot establish an unambiguous pushed image digest")
    reference = matching[0]
    record = {"track": track, "image_reference": reference,
              "source_revision": os.environ.get("GITHUB_SHA"),
              "anonymous_access": "pending separate check",
              "submission_uploaded": False}
    (work / f"published-{track}.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "check", "published"))
    parser.add_argument("--track", choices=tuple(UPSTREAM), required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--image")
    args = parser.parse_args()
    if args.action == "fetch":
        fetch(args.track, args.work.resolve())
    elif not args.image:
        parser.error("--image is required for this action")
    elif args.action == "check":
        check(args.track, args.image, args.work.resolve())
    else:
        published(args.track, args.image, args.work.resolve())


if __name__ == "__main__":
    main()
