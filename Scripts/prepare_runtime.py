"""Prepare a relocatable, offline Python runtime payload from the pinned standalone CPython."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

from audit_runtime import audit_runtime

# The app's installer accepts identifiers of at most 160 characters.
IDENTIFIER_LIMIT = 160


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked(command, **kwargs):
    subprocess.run(command, check=True, stdout=sys.stderr, **kwargs)


def pinned_packages(requirements_text):
    """(name, version) of each exact pin in the lock, in order."""
    pins = []
    for line in requirements_text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([0-9][A-Za-z0-9._-]*)", line)
        if not match:
            raise RuntimeError(f"The runtime lock must contain only exact version pins: {line}")
        pins.append((match.group(1).lower(), match.group(2)))
    if not pins:
        raise RuntimeError("The runtime lock pins no packages.")
    return pins


def runtime_identifier(python_version, requirements_text, architecture, payload_hash):
    """Readable, content-addressed install identifier, for example
    cpython-3.14.10-numpy-2.4.6-arm64-<first 20 payload digest characters>."""
    packages = "-".join(f"{name}-{version}" for name, version in pinned_packages(requirements_text))
    identifier = f"cpython-{python_version}-{packages}-{architecture}-{payload_hash[:20]}"
    if len(identifier) > IDENTIFIER_LIMIT or not re.fullmatch(r"[A-Za-z0-9._-]+", identifier):
        raise RuntimeError("The runtime identifier is invalid or exceeds the installer's size limit.")
    return identifier


def payload_fingerprint(root, archive_name, architecture):
    """Identify the exact preparation recipes and pinned artifact manifests."""
    root = Path(root)
    requirements = root / "Engine/requirements-lock.txt"
    wheel_lock = root / f"Engine/runtime-wheels-{architecture}.json"
    return hashlib.sha256(
        Path(archive_name).name.encode() + requirements.read_bytes() + wheel_lock.read_bytes()
        + (root / "Scripts/prepare_runtime.py").read_bytes()
        + (root / "Scripts/audit_runtime.py").read_bytes() + (root / "Engine/runtime-source.json").read_bytes()
        + (root / "Scripts/check_runtime.py").read_bytes()
    ).hexdigest()[:16]


def prepare(root, archive, architecture):
    source = json.loads((root / "Engine/runtime-source.json").read_text())
    requirements = root / "Engine/requirements-lock.txt"
    pinned_packages(requirements.read_text())
    wheel_lock = root / f"Engine/runtime-wheels-{architecture}.json"
    fingerprint = payload_fingerprint(root, archive.name, architecture)
    destination = root / ".build/cad-runtime" / architecture
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("buildFingerprint") == fingerprint and digest(destination / "payload.tar.gz") == previous.get("sha256"):
            return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{architecture}-", dir=destination.parent))
    try:
        expanded = staging / "expanded"
        expanded.mkdir()
        with tarfile.open(archive) as bundle:
            bundle.extractall(expanded, filter="data")
        python = expanded / "python/bin/python3"
        wheels = staging / "wheels"
        wheels.mkdir()
        checked([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--no-deps",
                 "--dest", str(wheels), "--requirement", str(requirements)])
        wheel_hashes = {path.name: digest(path) for path in sorted(wheels.iterdir())}
        if wheel_hashes != json.loads(wheel_lock.read_text()):
            raise RuntimeError("Downloaded runtime wheels differ from the committed artifact hashes.")
        checked([str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
                 "--find-links", str(wheels), "--requirement", str(requirements)])
        checked([str(python), "-m", "pip", "check"])
        native_audit = audit_runtime(expanded / "python", architecture, repair_build_rpaths=True)
        # The payload is prepared before the app and its native SDF kernel are
        # built, so this check never loads a kernel named by the environment.
        environment = {key: value for key, value in os.environ.items() if key != "RAD_SDF_LIBRARY"}
        checked([str(python), "-I", "-B", str(root / "Scripts/check_runtime.py")], env=environment)
        for cache in expanded.rglob("__pycache__"):
            shutil.rmtree(cache)
        # pip-generated launchers retain the build path. The app uses `python -m`
        # and creates a fresh venv; remove these unnecessary stale entry points.
        for entry in (expanded / "python/bin").iterdir():
            if entry.is_file() and not entry.is_symlink() and entry.read_bytes()[:2] == b"#!":
                entry.unlink()
        for entry in expanded.rglob("*"):
            if entry.is_symlink():
                resolved = entry.resolve()
                if not resolved.is_relative_to(expanded.resolve()):
                    raise RuntimeError(f"Payload contains an external symlink: {entry}")
        shutil.copy2(requirements, expanded / "requirements-lock.txt")
        (expanded / "wheel-hashes.json").write_text(json.dumps(wheel_hashes, indent=2) + "\n")
        payload = staging / "payload.tar.gz"
        checked(["/usr/bin/tar", "-czf", str(payload), "-C", str(expanded), "python", "requirements-lock.txt", "wheel-hashes.json"])
        payload_hash = digest(payload)
        # Different built bytes never share an installation path, including
        # rebuilds by a different builder or a later runtime-packaging change.
        identifier = runtime_identifier(source["pythonVersion"], requirements.read_text(), architecture, payload_hash)
        manifest = {"formatVersion": 1, "identifier": identifier, "architecture": architecture,
                    "pythonVersion": source["pythonVersion"], "archive": "payload.tar.gz",
                    "sha256": payload_hash, "buildFingerprint": fingerprint, "requirementsSHA256": digest(requirements),
                    "pythonSourceURL": source[architecture]["url"],
                    "pythonSourceSHA256": source[architecture]["sha256"], "nativeAudit": native_audit}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        shutil.rmtree(wheels)
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
        return destination
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--architecture", required=True, choices=["arm64"])
    args = parser.parse_args()
    print(prepare(args.root, args.archive, args.architecture))
