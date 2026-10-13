"""Export only reviewed source paths into a deterministic public-source tarball.

This does not build, publish, authenticate, or package a competition submission.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile


DENIED_PARTS = {
    ".secrets", "private", ".git", ".venv", "venv", "__pycache__",
    "runs", "output", "outputs", "reference_data", "units", "reports",
}
SECRET_PATTERNS = (
    re.compile(rb"(?<![A-Za-z0-9])[A-Za-z0-9]{4}(?:-[A-Za-z0-9]{4}){4}(?![A-Za-z0-9])"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9_]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{24,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb'"(?:team_key|descriptor_sha256|proof)"\s*:'),
)


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (not value or value != path.as_posix() or "\\" in value or path.is_absolute()
            or any(p in {"", ".", ".."} or ":" in p for p in path.parts)
            or any(p.lower() in DENIED_PARTS for p in path.parts)
            or any(p.lower().startswith(".env") for p in path.parts)):
        raise ValueError("unsafe or excluded export path")
    return path


def read_source(root: Path, relative: str) -> bytes:
    path = safe_relative(relative)
    current = root
    for part in path.parts:
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or current.is_junction():
            raise ValueError(f"links are not exported: {relative}")
    if not current.resolve().is_relative_to(root):
        raise ValueError(f"source escaped root: {relative}")
    info = current.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError(f"source must be an ordinary, unlinked file: {relative}")
    if info.st_size > 2 * 1024 * 1024:
        raise ValueError(f"source exceeds 2 MiB: {relative}")
    payload = current.read_bytes()
    payload.decode("utf-8")  # Source-only: reject opaque binary payloads.
    if any(pattern.search(payload) for pattern in SECRET_PATTERNS):
        raise ValueError(f"possible credential or submission proof in: {relative}")
    return payload


def collect(root: Path, manifest: Path) -> dict[str, bytes]:
    config = json.loads(manifest.read_text(encoding="utf-8"))
    if set(config) != {"format_version", "files"} or config["format_version"] != 1:
        raise ValueError("unexpected export whitelist format")
    entries: dict[str, bytes] = {}
    folded: set[str] = set()
    for row in config["files"]:
        if set(row) != {"source", "destination"}:
            raise ValueError("each whitelist entry requires source and destination")
        destination = safe_relative(row["destination"]).as_posix()
        if destination.casefold() in folded:
            raise ValueError(f"duplicate export path: {destination}")
        folded.add(destination.casefold())
        entries[destination] = read_source(root, row["source"])
    metadata = {
        "format_version": 1,
        "purpose": "Reviewed public source only; no submission ZIP or credentials",
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in sorted(entries.items())
        },
    }
    entries["PUBLIC_SOURCE_MANIFEST.json"] = (
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return entries


def archive_bytes(entries: dict[str, bytes]) -> bytes:
    result = io.BytesIO()
    with gzip.GzipFile(fileobj=result, mode="wb", filename="", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for name, payload in sorted(entries.items()):
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                info.mode = 0o644
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(payload))
    return result.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    entries = collect(root, root / "deployment/public-source-files.json")
    payload = archive_bytes(entries)
    out = args.out.absolute()
    out.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite a previous reviewed artifact, follow a link, or replace a directory.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(out, flags, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
    print(json.dumps({"artifact": str(out), "sha256": hashlib.sha256(payload).hexdigest(),
                      "files": len(entries), "bytes": len(payload)}, indent=2))


if __name__ == "__main__":
    main()
