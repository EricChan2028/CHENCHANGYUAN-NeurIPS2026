"""Check anonymous GHCR manifest/config/layer access without registry credentials."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen


MEDIA = ", ".join(("application/vnd.oci.image.manifest.v1+json",
                   "application/vnd.docker.distribution.manifest.v2+json"))


def request(url: str, token: str | None = None, *, head: bool = False) -> bytes:
    headers = {"Accept": MEDIA}
    if token:
        headers["Authorization"] = "Bearer " + token
    with urlopen(Request(url, headers=headers, method="HEAD" if head else "GET"), timeout=60) as response:
        if response.status != 200:
            raise ValueError("anonymous registry access did not return HTTP 200")
        return b"" if head else response.read()


def verify_layer(url: str, token: str, expected: str) -> None:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", expected) is None:
        raise ValueError("invalid layer digest")
    digest = hashlib.sha256()
    with urlopen(Request(url, headers={"Authorization": "Bearer " + token}), timeout=60) as response:
        if response.status != 200:
            raise ValueError("anonymous layer download did not return HTTP 200")
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
    if "sha256:" + digest.hexdigest() != expected:
        raise ValueError("anonymous layer download digest mismatch")


def verify(reference: str) -> None:
    match = re.fullmatch(r"ghcr\.io/([a-z0-9._/-]+)@(sha256:[0-9a-f]{64})", reference)
    if not match:
        raise ValueError("expected a GHCR image reference with an immutable sha256 digest")
    repository, digest = match.groups()
    query = urlencode({"scope": f"repository:{repository}:pull", "service": "ghcr.io"})
    token_reply = json.loads(request("https://ghcr.io/token?" + query))
    token = token_reply.get("token") or token_reply.get("access_token")
    if not token:
        raise ValueError("registry did not grant an anonymous pull token")
    base = "https://ghcr.io/v2/" + repository
    raw = request(base + "/manifests/" + digest, token)
    if "sha256:" + hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("registry manifest bytes do not match the submitted digest")
    manifest = json.loads(raw)
    if "manifests" in manifest:
        raise ValueError("expected the single-platform image built and tested by this workflow")
    config_digest = manifest["config"]["digest"]
    raw_config = request(base + "/blobs/" + config_digest, token)
    if "sha256:" + hashlib.sha256(raw_config).hexdigest() != config_digest:
        raise ValueError("registry config digest mismatch")
    config = json.loads(raw_config)
    if (config.get("os"), config.get("architecture")) != ("linux", "amd64"):
        raise ValueError("published image is not Linux/amd64")
    if (config.get("config", {}).get("Labels") or {}).get("qfbench2.interface_version") != "2.0":
        raise ValueError("published image lacks the interface label")
    if config.get("config", {}).get("Volumes"):
        raise ValueError("published image has a prohibited VOLUME")
    for layer in manifest["layers"]:
        verify_layer(base + "/blobs/" + layer["digest"], token, layer["digest"])
    # Docker resolves the same reference with a clean config. Above, every layer was
    # downloaded anonymously and hashed, regardless of this daemon's existing cache.
    import os
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory(prefix="anonymous-registry-") as directory:
        environment = dict(os.environ, DOCKER_CONFIG=directory)
        environment.pop("DOCKER_AUTH_CONFIG", None)
        subprocess.run(["docker", "pull", "--platform", "linux/amd64", reference],
                       check=True, env=environment, timeout=600)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path)
    args = parser.parse_args()
    record = json.loads(args.record.read_text(encoding="utf-8"))
    verify(record["image_reference"])
    record["anonymous_access"] = "verified manifest, config, layers and credential-free Docker pull"
    args.record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
