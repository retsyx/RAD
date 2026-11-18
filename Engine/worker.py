#!/usr/bin/env python3
"""RAD's SDF CAD worker.

Every invocation runs in its own process, keeping design programs, the native
SDF kernel and meshing away from the app's main thread. A build runs one
design program (sdf_build); the long-lived serving worker answers native
geometry operations on a build's staged model.rsdf (sdf_ops). The macOS app
sandboxes builds before launch.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

from sdf_build import HostError, ModelError, build, runtime_info

MAX_REQUEST_BYTES = 16 * 1024 * 1024
MAX_SPEC_BYTES = 2 * 1024 * 1024
# Operations the unsandboxed serving worker accepts; it never runs programs.
NATIVE_OPERATIONS = {"runtime", "measure", "clearance", "print_orient", "export", "validate", "probe"}


def failure_kind(error):
    """Report "design" when changing the program could fix the failure.

    Program exceptions arrive wrapped in ModelError, so an OSError or
    ImportError reaching here comes from the runtime or its file access.
    """
    if isinstance(error, (HostError, ImportError, OSError)):
        return "runtime"
    return "design"


def write_error_report(error, output):
    """Publish bounded failure details independently of diagnostic log tails."""
    import tempfile
    report = {"formatVersion": 1, "stage": "worker", **(getattr(error, "diagnostic", None) or {}),
              "failureKind": failure_kind(error),
              "message": str(error).encode("utf-8")[:32_000].decode("utf-8", errors="ignore")}
    data = json.dumps(report, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode("utf-8")
    if len(data) > 65_536:
        report.pop("failures", None)
        report["detailsOmitted"] = True
        data = json.dumps(report, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode("utf-8")
    while len(data) > 65_536 and report["message"]:
        # Control characters need escaping even without ASCII-only encoding.
        # Bound the serialized bytes, not just the source string's length.
        report["message"] = report["message"][:len(report["message"]) // 2]
        report["messageTruncated"] = True
        data = json.dumps(report, separators=(",", ":"), allow_nan=False, ensure_ascii=False).encode("utf-8")
    output.mkdir(parents=True, exist_ok=True)
    path = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".error-", dir=output, delete=False) as stream:
            path = Path(stream.name)
            stream.write(data)
        path.replace(output / "error.json")
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def native_operation(spec, directory, output, threads, serving=False):
    """Answer a native geometry operation; sdf_ops owns every one but runtime.
    A serving worker keeps recently used graphs loaded between requests."""
    if spec == {"operation": "runtime"}:
        return {"runtime": runtime_info()}
    if spec["operation"] == "runtime":
        raise ModelError("The runtime operation takes no other fields.")
    try:
        import sdf_ops
    except ModuleNotFoundError as error:
        if error.name != "sdf_ops":
            raise
        raise HostError(f"The native operation {spec['operation']} is not available.") from error
    return sdf_ops.execute(spec, directory, output, threads, serving=serving)


def run(request, output, threads, native_only=False):
    """Run one request file: a build (no `operation`) or a native operation.
    Results are published in `output`; returns the result dictionary."""
    started = time.perf_counter()
    if not 1 <= threads <= 256:
        raise HostError("Thread count must be between 1 and 256.")
    size = request.stat().st_size
    if size > MAX_REQUEST_BYTES:
        raise ModelError("CAD request exceeds the 16 MiB size limit.")
    spec = json.loads(request.read_text())
    operation = spec.get("operation") if isinstance(spec, dict) else None
    native = isinstance(operation, str) and operation in NATIVE_OPERATIONS
    if native_only and not native:
        # The serving worker runs outside the program sandbox.
        raise ModelError("This worker performs only native geometry operations.")
    output.mkdir(parents=True, exist_ok=True)
    if operation is None:
        if size > MAX_SPEC_BYTES:
            raise ModelError("Design specification exceeds the 2 MiB size limit.")
        return build(spec, request.parent.resolve(), output, threads, started=started)
    if not native:
        raise ModelError(f"Unknown CAD operation {operation!r}.")
    result = native_operation(spec, request.parent.resolve(), output, threads, serving=native_only)
    write_native_result(result, output)
    return result


def write_native_result(result, output):
    """Publish a native operation's result.json (plain JSON, no mesh.bin)."""
    staged = output / "result.json.tmp"
    try:
        staged.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
        staged.replace(output / "result.json")
    finally:
        staged.unlink(missing_ok=True)


def serve():
    """Answer native-geometry requests from the app, one JSON line each.

    Each request names a request file, an output folder and a thread count; the
    reply reports completion or the error, and results stay in that folder.
    Native output and stray prints go to stderr so they cannot corrupt replies.
    The loop ends when the app closes stdin, including when it exits.
    """
    replies = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    for line in sys.stdin:
        message, output = None, None
        try:
            message = json.loads(line)
            output = Path(message["output"])
            run(Path(message["request"]), output, int(message["threads"]), native_only=True)
            reply = {"id": message["id"], "status": "complete"}
        except Exception as error:
            if output is not None:
                try:
                    write_error_report(error, output)
                except Exception:
                    pass
            reply = {"id": message.get("id") if isinstance(message, dict) else None, "status": "error",
                     "message": f"RAD CAD engine: {error}"[:16_000]}
        replies.write(json.dumps(reply) + "\n")
    return 0


BUILD_OPTIONS = {"threads"}


def await_build(start, output):
    """Load the CAD runtime, then wait for one build's options on `start`.

    The app starts a sandboxed worker before its next build is requested, so
    Python, numpy, the modeling modules and the native kernel are already
    loaded when it is. The app stages the request in `output` and writes the
    options to this private FIFO; the worker still runs exactly one program.
    It exits once RAD does.
    """
    import threading
    parent = os.getppid()

    def watch():
        # An orphan is reparented to launchd, possibly before this started.
        while os.getppid() == parent != 1:
            time.sleep(1)
        os._exit(1)

    threading.Thread(target=watch, daemon=True).start()
    import numpy  # noqa: F401 -- loaded ahead of the build
    import radsdf
    import radsdf_assembly  # noqa: F401
    try:
        radsdf.kernel_version()
    except radsdf.SDFError:
        pass  # the build reports the kernel failure in error.json
    with open(start, "rb") as stream:
        options = json.loads(stream.read(4_096))
    if (not isinstance(options, dict) or set(options) - BUILD_OPTIONS
            or isinstance(options.get("threads"), bool) or not isinstance(options.get("threads"), int)):
        raise HostError("The CAD build options are invalid.")
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="Answer native-geometry requests on stdin.")
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=max(1, min(32, os.cpu_count() or 1)))
    parser.add_argument("--await-build", type=Path,
                        help="Load the CAD runtime, then read one build's options from this FIFO; the request is "
                             "in --output.")
    arguments = parser.parse_args()
    if arguments.serve:
        return serve()
    if arguments.await_build is not None and arguments.output is not None:
        try:
            options = await_build(arguments.await_build, arguments.output)
        except Exception as error:
            try:
                write_error_report(error, arguments.output)
            except Exception:
                pass
            print(f"RAD CAD engine: {error}", file=sys.stderr)
            return 1
        arguments.request = arguments.output / "request.json"
        arguments.threads = options["threads"]
    if arguments.request is None or arguments.output is None:
        parser.error("--request and --output are required")
    try:
        result = run(arguments.request, arguments.output, arguments.threads)
        print(json.dumps({"status": "complete", "solids": result.get("solids")}))
    except Exception as error:
        try:
            write_error_report(error, arguments.output)
        except Exception:
            # Report publication must not replace the original CAD failure.
            pass
        print(f"RAD CAD engine: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
