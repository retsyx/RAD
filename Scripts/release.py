"""Assemble and verify an Apple Silicon RAD distribution.

Production requires an installed Developer ID Application identity and a
notarytool keychain profile. --local exercises packaging with ad-hoc signatures
and deliberately produces an artifact that is not a distributable release.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import plistlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

from audit_runtime import audit_runtime, load_commands, validate_commands
from prepare_app_binaries import KERNEL_LIBRARY, verify_app_binaries, verify_app_symbols
from prepare_runtime import payload_fingerprint
from release_runtime import file_hash, native_images, pack_runtime, sign_image, sign_runtime, signing_team

# The installed-payload check builds a 20 x 15 x 6 mm plate with a 6 mm
# diameter through hole in the sandboxed worker.
VERIFICATION_PROGRAM = "result = difference(box((20, 15, 6)), cylinder(3, 10))"
VERIFICATION_VOLUME = 20 * 15 * 6 - math.pi * 3 ** 2 * 6


class ReleaseError(RuntimeError):
    pass


def run(command, *, capture=False, **kwargs):
    result = subprocess.run([str(value) for value in command], check=False,
                            stdout=subprocess.PIPE if capture else None,
                            stderr=subprocess.PIPE if capture else None, text=True, **kwargs)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip() if capture else "See the command output above."
        raise ReleaseError(f"{Path(command[0]).name} failed ({result.returncode}): {detail}")
    return result


def validate_inputs(version, build_number, local, identity, notary_profile):
    if not re.fullmatch(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)", version):
        raise ReleaseError("Version must contain three numeric components, for example 0.3.0.")
    if not re.fullmatch(r"[1-9][0-9]*", build_number):
        raise ReleaseError("Build number must be a positive integer.")
    if local:
        if identity or notary_profile:
            raise ReleaseError("--local cannot be combined with release signing credentials.")
    elif not identity or not notary_profile or identity == "-":
        raise ReleaseError("A release requires --identity and --notary-profile. Use --local for local packaging validation.")


def snapshot_source(root, revision, destination):
    """Build production binaries from committed bytes, never a changing checkout."""
    destination.mkdir(parents=True)
    archive = destination.parent / "source.tar"
    try:
        run(["/usr/bin/git", "-C", root, "archive", "--format=tar", "--output", archive, revision])
        with tarfile.open(archive) as source:
            source.extractall(destination, filter="data")
    finally:
        archive.unlink(missing_ok=True)
    return destination


def verify_checkout_unchanged(root, revision):
    current = run(["/usr/bin/git", "-C", root, "rev-parse", "HEAD"], capture=True).stdout.strip()
    dirty = run(["/usr/bin/git", "-C", root, "status", "--porcelain"], capture=True).stdout
    if current != revision or dirty:
        raise ReleaseError("The source checkout changed during packaging. No release will be published; retry from a clean checkout.")


def verify_payload_provenance(source_root, expanded, manifest):
    """Match the cached payload to the frozen release's preparation contract."""
    source = json.loads((source_root / "Engine/runtime-source.json").read_text())
    pinned_python = source["arm64"]
    requirements = source_root / "Engine/requirements-lock.txt"
    wheel_hashes = json.loads((source_root / "Engine/runtime-wheels-arm64.json").read_text())
    expected_fingerprint = payload_fingerprint(source_root, pinned_python["sha256"] + ".tar.gz", "arm64")
    if (manifest.get("buildFingerprint") != expected_fingerprint
            or manifest.get("pythonVersion") != source["pythonVersion"]
            or manifest.get("pythonSourceURL") != pinned_python["url"]
            or manifest.get("pythonSourceSHA256") != pinned_python["sha256"]
            or manifest.get("requirementsSHA256") != file_hash(requirements)
            or (expanded / "requirements-lock.txt").read_bytes() != requirements.read_bytes()
            or json.loads((expanded / "wheel-hashes.json").read_text()) != wheel_hashes):
        raise ReleaseError("The prepared runtime payload does not match the frozen release source. Prepare the runtime from that checkout and retry.")


def unpack_payload(payload, destination):
    manifest = json.loads((payload / "manifest.json").read_text())
    if manifest.get("architecture") != "arm64" or manifest.get("formatVersion") != 1:
        raise ReleaseError("The runtime payload must be an Apple Silicon format-1 runtime.")
    if manifest.get("archive") != "payload.tar.gz":
        raise ReleaseError("The runtime payload archive name is invalid.")
    archive = payload / "payload.tar.gz"
    if file_hash(archive) != manifest.get("sha256"):
        raise ReleaseError("The runtime payload checksum does not match its manifest.")
    destination.mkdir(parents=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    for path in destination.rglob("*"):
        if path.is_symlink() and not path.resolve().is_relative_to(destination.resolve()):
            raise ReleaseError("The runtime payload contains an external symlink.")
    return manifest


def notarize(path, label, profile, logs):
    """Retain Apple's complete log and fail closed on any incomplete submission."""
    print(f"Notarizing {label}…", flush=True)
    result = subprocess.run(["/usr/bin/xcrun", "notarytool", "submit", str(path),
                             "--keychain-profile", profile, "--wait", "--output-format", "json"],
                            capture_output=True, text=True)
    (logs / f"{label}-submission.json").write_text(result.stdout)
    (logs / f"{label}-submission.stderr.txt").write_text(result.stderr)
    try:
        response = json.loads(result.stdout)
    except ValueError as error:
        raise ReleaseError(f"Notarization returned no result for {label}; see {logs}.") from error
    submission_id = response.get("id")
    if submission_id:
        run(["/usr/bin/xcrun", "notarytool", "log", submission_id, str(logs / f"{label}-log.json"),
             "--keychain-profile", profile], capture=True)
    if result.returncode or response.get("status") != "Accepted" or not submission_id:
        raise ReleaseError(f"Notarization was not accepted for {label}; see {logs}.")
    log = json.loads((logs / f"{label}-log.json").read_text())
    if log.get("status") != "Accepted" or any(issue.get("severity") == "error" for issue in log.get("issues") or []):
        raise ReleaseError(f"The notarization log is not successful for {label}; see {logs}.")
    return {"id": submission_id, "status": "Accepted", "issues": log.get("issues") or []}


def zip_for_notarization(source, destination):
    run(["/usr/bin/ditto", "-c", "-k", "--keepParent", "--norsrc", "--noqtn", source, destination])


def staple(path):
    run(["/usr/bin/xcrun", "stapler", "staple", path])
    run(["/usr/bin/xcrun", "stapler", "validate", path])


def verify_app(app, version, build_number, production):
    # Audit all signed native images again, including the copy from the DMG.
    # Signing, notarization and the archive round trip must not bypass the
    # stripping and private-path checks performed before signing.
    verify_app_binaries(app)
    with (app / "Contents/Info.plist").open("rb") as stream:
        info = plistlib.load(stream)
    if (info.get("CFBundleShortVersionString"), info.get("CFBundleVersion")) != (version, build_number):
        raise ReleaseError("The assembled app has incorrect release version metadata.")
    if info.get("LSMinimumSystemVersion") != "14.0" or info.get("LSArchitecturePriority") != ["arm64"]:
        raise ReleaseError("The assembled app has incorrect platform metadata.")
    executable = app / "Contents/MacOS/RAD"
    if run(["/usr/bin/lipo", "-archs", executable], capture=True).stdout.split() != ["arm64"]:
        raise ReleaseError("The RAD executable must contain only arm64 code.")
    commands = run(["/usr/bin/otool", "-arch", "arm64", "-l", executable], capture=True).stdout
    validate_commands(load_commands(commands), "RAD")
    run(["/usr/bin/codesign", "--verify", "--deep", "--strict", app], capture=True)
    details = run(["/usr/bin/codesign", "--display", "--verbose=4", app], capture=True).stderr
    flags = re.search(r"\bflags=0x([0-9a-fA-F]+)", details)
    if not flags or not int(flags.group(1), 16) & 0x10000:
        raise ReleaseError("The RAD app must have hardened runtime enabled.")
    if production:
        if "Authority=Developer ID Application:" not in details or "Timestamp=" not in details:
            raise ReleaseError("The RAD app lacks a timestamped Developer ID Application signature.")
        run(["/usr/sbin/spctl", "--assess", "--type", "execute", "--verbose=2", app], capture=True)
        run(["/usr/bin/xcrun", "stapler", "validate", app], capture=True)
    thumbnail = app / "Contents/PlugIns/RADThumbnailExtension.appex"
    thumbnail_executable = thumbnail / "Contents/MacOS/RADThumbnailExtension"
    expected_images = {executable.resolve(), thumbnail_executable.resolve(), (app / KERNEL_LIBRARY).resolve()}
    if {path.resolve() for path in native_images(app)} != expected_images:
        raise ReleaseError("Unexpected unpacked native code in the app; add explicit inside-out signing before shipping it.")
    with (thumbnail / "Contents/Info.plist").open("rb") as stream:
        thumbnail_info = plistlib.load(stream)
    extension = thumbnail_info.get("NSExtension", {})
    if (thumbnail_info.get("CFBundleIdentifier") != "studio.rad.cad.thumbnail"
            or thumbnail_info.get("CFBundleVersion") != build_number
            or thumbnail_info.get("CFBundleShortVersionString") != version
            or extension.get("NSExtensionPointIdentifier") != "com.apple.quicklook.thumbnail"
            or extension.get("NSExtensionAttributes", {}).get("QLSupportedContentTypes") != ["studio.rad.project"]):
        raise ReleaseError("The Finder thumbnail extension has incorrect metadata.")
    if run(["/usr/bin/lipo", "-archs", thumbnail_executable], capture=True).stdout.split() != ["arm64"]:
        raise ReleaseError("The Finder thumbnail extension must contain only arm64 code.")
    run(["/usr/bin/codesign", "--verify", "--strict", thumbnail], capture=True)
    thumbnail_details = run(["/usr/bin/codesign", "--display", "--verbose=4", thumbnail], capture=True).stderr
    thumbnail_flags = re.search(r"\bflags=0x([0-9a-fA-F]+)", thumbnail_details)
    if not thumbnail_flags or not int(thumbnail_flags.group(1), 16) & 0x10000:
        raise ReleaseError("The Finder thumbnail extension must have hardened runtime enabled.")
    if production and ("Authority=Developer ID Application:" not in thumbnail_details or "Timestamp=" not in thumbnail_details):
        raise ReleaseError("The Finder thumbnail extension lacks a timestamped Developer ID Application signature.")
    verify_kernel(app, production)


def verify_kernel(app, production):
    """The SDF kernel must be a hardened arm64 library signed by the runtime's
    team: release Python keeps library validation and loads only same-team code."""
    kernel = app / KERNEL_LIBRARY
    if run(["/usr/bin/lipo", "-archs", kernel], capture=True).stdout.split() != ["arm64"]:
        raise ReleaseError("The SDF kernel library must contain only arm64 code.")
    commands = run(["/usr/bin/otool", "-arch", "arm64", "-l", kernel], capture=True).stdout
    validate_commands(load_commands(commands), KERNEL_LIBRARY.name)
    run(["/usr/bin/codesign", "--verify", "--strict", kernel], capture=True)
    details = run(["/usr/bin/codesign", "--display", "--verbose=4", kernel], capture=True).stderr
    flags = re.search(r"\bflags=0x([0-9a-fA-F]+)", details)
    if not flags or not int(flags.group(1), 16) & 0x10000:
        raise ReleaseError("The SDF kernel library must have hardened runtime enabled.")
    if production:
        if "Authority=Developer ID Application:" not in details or "Timestamp=" not in details:
            raise ReleaseError("The SDF kernel library lacks a timestamped Developer ID Application signature.")
        manifest = json.loads((app / "Contents/Resources/CADRuntime/manifest.json").read_text())
        team = signing_team(details)
        if team in {None, "not set"} or team != manifest.get("signing", {}).get("teamIdentifier"):
            raise ReleaseError("The SDF kernel library and the runtime must share one signing team.")


def verify_installed_payload(app, work, production):
    """Rebuild from the bytes copied out of the DMG, in an unrelated path.

    This complements the Swift provisioning tests: signatures must survive the
    archive round trip and venv --copies used by the real runtime installer,
    and the installed interpreter must load the app's SDF kernel both directly
    and inside the worker sandbox.
    """
    expanded = work / "offline runtime with spaces"
    manifest = unpack_payload(app / "Contents/Resources/CADRuntime", expanded)
    python = expanded / "python/bin/python3"
    # The sandbox profile matches real paths.
    kernel = (app / KERNEL_LIBRARY).resolve()
    environment = {"PATH": "/usr/bin:/bin", "HOME": str(work), "TMPDIR": str(work),
                   "LANG": "en_US.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1",
                   "XDG_CACHE_HOME": str(work / "cache"), "RAD_SDF_LIBRARY": str(kernel)}
    venv = expanded / "venv"
    run([python, "-I", "-B", "-m", "venv", "--copies", "--system-site-packages", "--without-pip", venv], env=environment)
    interpreter = venv / "bin/python"
    run(["/usr/bin/codesign", "--verify", "--strict", interpreter], capture=True)
    # With RAD_SDF_LIBRARY set, the runtime check also loads, evaluates and
    # meshes with the kernel; release Python keeps library validation.
    run([interpreter, "-I", "-B", Path(__file__).with_name("check_runtime.py")], env=environment, timeout=180)
    output = work / "worker-output"
    output.mkdir()
    private = work / "private.txt"
    private.write_text("private release verification fixture")
    program = f"""
from pathlib import Path
import socket, subprocess, errno
for action in [lambda: Path({str(private)!r}).read_text(),
               lambda: Path({str(work / 'forbidden.txt')!r}).write_text('forbidden'),
               lambda: subprocess.run(['/bin/echo','forbidden'], check=True)]:
    try: action()
    except PermissionError: pass
    else: raise AssertionError('The worker sandbox allowed forbidden access')
try:
    with socket.socket() as connection: connection.connect(('127.0.0.1',9))
except OSError as error: assert error.errno in (errno.EPERM, errno.EACCES), error
else: raise AssertionError('The worker sandbox allowed network access')
{VERIFICATION_PROGRAM}
"""
    request = output / "request.json"
    request.write_text(json.dumps({"title": "Release verification", "summary": "Bundled sandbox and SDF kernel verification.",
                                   "units": "mm", "script": program}))
    engine = app / "Contents/Resources/Engine"
    parameters = {"RUNTIME": venv, "PYTHON_BASE": expanded / "python", "WORKER": engine,
                  "OUTPUT": output, "PYTHON": interpreter, "ENTRY_PYTHON": interpreter, "KERNEL": kernel}
    command = ["/usr/bin/sandbox-exec"]
    for name, value in parameters.items():
        command += ["-D", f"{name}={value}"]
    command += ["-f", engine / "script.sb", interpreter, engine / "worker.py",
                "--request", request, "--output", output, "--threads", "2"]
    environment.update({"RAD_MODEL_OUTPUT": str(output), "XDG_CACHE_HOME": str(output / ".cache"), "TMPDIR": str(output)})
    run(command, env=environment, cwd=output, timeout=180)
    result = json.loads((output / "result.json").read_text())
    volume = result.get("volume")
    if (result.get("solids") != 1 or not isinstance(volume, (int, float))
            or abs(volume - VERIFICATION_VOLUME) > 0.01 * VERIFICATION_VOLUME):
        raise ReleaseError("The packaged worker did not build the expected solid with the SDF kernel.")
    artifact = output / "model.rsdf"
    if not artifact.is_file() or artifact.read_bytes()[:4] != b"RSDF":
        raise ReleaseError("The packaged worker did not write its native RSDF graph.")
    if (work / "forbidden.txt").exists() or private.read_text() != "private release verification fixture":
        raise ReleaseError("The packaged worker changed a file outside its scratch directory.")
    images = native_images(expanded)
    # The signing pass checks all images before packing; validate the copied
    # interpreter and native library signatures after the archive round trip.
    for image in images:
        run(["/usr/bin/codesign", "--verify", "--strict", image], capture=True)
    return {"runtimeIdentifier": manifest["identifier"], "nativeImagesVerified": len(images),
            "offlineVenv": True, "nativeKernel": True, "sandboxedBuild": True}


def verify_dmg(dmg, work, version, build_number, production):
    run(["/usr/bin/hdiutil", "verify", dmg], capture=True)
    mount = work / "mounted"
    mount.mkdir()
    attached = False
    try:
        run(["/usr/bin/hdiutil", "attach", dmg, "-readonly", "-nobrowse", "-noautoopen",
             "-mountpoint", mount], capture=True)
        attached = True
        if not (mount / "Applications").is_symlink() or os.readlink(mount / "Applications") != "/Applications":
            raise ReleaseError("The DMG is missing its Applications shortcut.")
        copied = work / "copied from DMG" / "RAD.app"
        copied.parent.mkdir()
        run(["/usr/bin/ditto", "--noqtn", mount / "RAD.app", copied])
    finally:
        if attached:
            run(["/usr/bin/hdiutil", "detach", mount], capture=True)
    verify_app(copied, version, build_number, production)
    return verify_installed_payload(copied, work, production)


def release(args):
    validate_inputs(args.version, args.build_number, args.local, args.identity, args.notary_profile)
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise ReleaseError("RAD releases require an Apple Silicon Mac.")
    root, payload = args.root.resolve(), args.payload.resolve()
    notes = args.notes.resolve()
    if not notes.is_file():
        raise ReleaseError("Provide a nonempty release notes file with --notes.")
    with notes.open("rb") as stream:
        notes_data = stream.read(1_048_577)
    if len(notes_data) > 1_048_576:
        raise ReleaseError("Release notes must be smaller than 1 MiB.")
    if not notes_data.decode("utf-8").strip():
        raise ReleaseError("Provide a nonempty release notes file with --notes.")
    output = args.output or root / "dist/releases" / f"{args.version}+{args.build_number}{'-local' if args.local else ''}"
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise ReleaseError(f"The output already exists: {output}. Choose a new build number or output directory.")
    revision = run(["/usr/bin/git", "-C", root, "rev-parse", "HEAD"], capture=True).stdout.strip()
    dirty = bool(run(["/usr/bin/git", "-C", root, "status", "--porcelain"], capture=True).stdout)
    if not args.local and dirty:
        raise ReleaseError("Release builds require a clean checkout. Commit the intended changes before signing.")
    production = not args.local
    identity = args.identity if production else "-"
    if production:
        identities = run(["/usr/bin/security", "find-identity", "-v", "-p", "codesigning"], capture=True).stdout
        matching = [line for line in identities.splitlines() if args.identity in line and '"Developer ID Application:' in line]
        if len(matching) != 1:
            raise ReleaseError("The requested Developer ID Application identity is not uniquely available in the keychain.")
        run(["/usr/bin/xcrun", "--find", "notarytool"], capture=True)
        run(["/usr/bin/xcrun", "notarytool", "history", "--keychain-profile", args.notary_profile,
             "--output-format", "json"], capture=True)
    work_root = root / ".build/releases"
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{args.version}-{args.build_number}-", dir=work_root))
    print(f"Release workspace: {work}", flush=True)
    print("Local validation only; this artifact is not notarized." if args.local else "Building a Developer ID release.", flush=True)
    logs = work / "logs"
    logs.mkdir()
    symbols = work / "symbols"
    try:
        frozen_notes = work / "release-notes.md"
        frozen_notes.write_bytes(notes_data)
        source_root = snapshot_source(root, revision, work / "source") if production else root
        expanded = work / "expanded-runtime"
        source_manifest = unpack_payload(payload, expanded)
        if production:
            verify_payload_provenance(source_root, expanded, source_manifest)
        signing = sign_runtime(expanded, identity, production)
        notarization = {}
        if production:
            runtime_zip = work / "RAD-CADRuntime.zip"
            zip_for_notarization(expanded, runtime_zip)
            notarization["runtime"] = notarize(runtime_zip, "runtime", args.notary_profile, logs)
            runtime_zip.unlink()
        final_payload = work / "payload"
        manifest = pack_runtime(expanded, source_manifest, final_payload, signing)
        app = work / "application/RAD.app"
        environment = dict(os.environ, RAD_BUILD_CONFIGURATION="release", RAD_APP_OUTPUT=str(app),
                           RAD_RUNTIME_PAYLOAD=str(final_payload), RAD_APP_SIGNING="none", RAD_REGISTER_APP="0",
                           RAD_VERSION=args.version, RAD_BUILD_NUMBER=args.build_number,
                           RAD_APP_SYMBOLS_OUTPUT=str(symbols), RAD_BUILD_PYTHON=sys.executable)
        run([source_root / "Scripts/build-app.sh"], env=environment)
        # Enforce preparation independently of build-app.sh before any release
        # signatures, retaining the complete matching symbols outside the app.
        verify_app_binaries(app, build_roots=(root, source_root))
        verify_app_symbols(app, symbols)
        # SwiftPM can retain an unused Xcode-toolchain rpath. Remove build-only
        # searches before signing, while rejecting actual external dependencies.
        app_audit = audit_runtime(app / "Contents", "arm64", repair_build_rpaths=True)
        # Sign inside out: the kernel library, the extension, then the app.
        # Release Python loads only libraries signed by its own team.
        if sign_image(app / KERNEL_LIBRARY, identity, production) != signing["teamIdentifier"]:
            raise ReleaseError("The SDF kernel library and the runtime must share one signing team.")
        thumbnail = app / "Contents/PlugIns/RADThumbnailExtension.appex"
        run(["/usr/bin/codesign", "--force", "--sign", identity, "--options", "runtime",
             "--timestamp" if production else "--timestamp=none", "--entitlements",
             source_root / "Scripts/ThumbnailExtension.entitlements", thumbnail])
        run(["/usr/bin/codesign", "--force", "--sign", identity, "--options", "runtime",
             "--timestamp" if production else "--timestamp=none", app])
        if production:
            app_zip = work / "RAD.zip"
            zip_for_notarization(app, app_zip)
            notarization["app"] = notarize(app_zip, "app", args.notary_profile, logs)
            staple(app)
            app_zip.unlink()
        verify_app(app, args.version, args.build_number, production)
        image_root = work / "image-root"
        image_root.mkdir()
        run(["/usr/bin/ditto", app, image_root / "RAD.app"])
        (image_root / "Applications").symlink_to("/Applications")
        artifacts = work / "artifacts"
        artifacts.mkdir()
        filename = f"RAD-{args.version}-{args.build_number}-arm64{'-local' if args.local else ''}.dmg"
        dmg = artifacts / filename
        run(["/usr/bin/hdiutil", "create", "-volname", f"RAD {args.version}", "-srcfolder", image_root,
             "-fs", "HFS+", "-format", "UDZO", dmg])
        if production:
            run(["/usr/bin/codesign", "--sign", identity, "--timestamp", "--identifier", "studio.rad.cad.dmg", dmg])
            notarization["dmg"] = notarize(dmg, "dmg", args.notary_profile, logs)
            staple(dmg)
            run(["/usr/sbin/spctl", "--assess", "--type", "open", "--context", "context:primary-signature", dmg], capture=True)
        verification = verify_dmg(dmg, work, args.version, args.build_number, production)
        shutil.copyfile(frozen_notes, artifacts / "release-notes.md")
        record = {"formatVersion": 1, "version": args.version, "buildNumber": args.build_number,
                  "architecture": "arm64", "minimumMacOS": "14.0", "distributable": production,
                  "createdAt": datetime.now(timezone.utc).isoformat(), "sourceRevision": revision,
                  "sourceDirty": dirty, "xcode": run(["/usr/bin/xcrun", "xcodebuild", "-version"], capture=True).stdout.strip(),
                  "runtime": manifest, "runtimeSigning": signing, "appNativeAudit": app_audit, "notarization": notarization,
                  "verification": verification, "files": {filename: {"sha256": file_hash(dmg), "bytes": dmg.stat().st_size}}}
        (artifacts / "release.json").write_text(json.dumps(record, indent=2) + "\n")
        checksums = "".join(f"{file_hash(artifacts / name)}  {name}\n" for name in [filename, "release-notes.md", "release.json"])
        (artifacts / "SHA256SUMS").write_text(checksums)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Artifacts become visible only after every required check has passed.
        if production:
            verify_checkout_unchanged(root, revision)
        if output.exists() or output.is_symlink():
            raise ReleaseError("The release output appeared during packaging; refusing to replace it.")
        # Stage on the destination filesystem as --output may be on another
        # volume. Only the final rename exposes the complete artifact set.
        publication = Path(tempfile.mkdtemp(prefix=".RAD-release-", dir=output.parent))
        try:
            for artifact in artifacts.iterdir():
                shutil.copy2(artifact, publication / artifact.name)
            if output.exists() or output.is_symlink():
                raise ReleaseError("The release output appeared during publication; refusing to replace it.")
            publication.rename(output)
        finally:
            if publication.exists():
                shutil.rmtree(publication)
        print(f"Verified {'release' if production else 'local validation'} artifacts: {output}", flush=True)
        print(f"Private debugging symbols retained separately: {symbols}", flush=True)
    except BaseException:
        print(f"Release failed. No release output was published. Diagnostic workspace: {work}", file=sys.stderr)
        raise
    # Keep notarization logs and UUID-matched symbols privately for diagnostics.
    # Symbols contain build paths and must never enter the published artifacts.
    for child in work.iterdir():
        if child not in {logs, symbols}:
            if child.is_dir(): shutil.rmtree(child)
            else: child.unlink()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--payload", required=True, type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--version", required=True)
    parser.add_argument("--build-number", required=True)
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--identity", default=os.environ.get("RAD_SIGNING_IDENTITY", ""))
    parser.add_argument("--notary-profile", default=os.environ.get("RAD_NOTARY_PROFILE", ""))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--local", action="store_true", help="Build an ad-hoc signed, non-distributable validation DMG")
    try:
        release(parser.parse_args())
    except (ReleaseError, ValueError, OSError, RuntimeError) as error:
        print(f"Release error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
