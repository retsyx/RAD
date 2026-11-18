#!/usr/bin/env python3
"""Profile one worker build of a retained build directory using RAD's bundled Python.

Usage: RAD_SDF_LIBRARY=<libRADSDF.dylib> <runtime python> Scripts/profile-worker.py BUILD_DIRECTORY
The directory needs the retained request.json and its inputs (for example from
ProjectOpeningProfileTests). The worker runs unsandboxed through its command
line; outputs stay in BUILD_DIRECTORY (profile-output/, worker.pstats,
worker-stages.json). Timings are nested, not additive.
"""
import argparse
import cProfile
import json
import os
from pathlib import Path
import pstats
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("build_directory", type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    folder = args.build_directory.resolve()
    if not (folder / "request.json").is_file():
        parser.error("build_directory must contain the retained request.json and its inputs")
    output = folder / "profile-output"
    if output.exists():
        parser.error("profile-output already exists; use a fresh copy of the build directory")
    if not os.environ.get("RAD_SDF_LIBRARY"):
        parser.error("set RAD_SDF_LIBRARY to the libRADSDF.dylib to profile with")
    engine = Path(__file__).resolve().parents[1] / "Engine"
    sys.path.insert(0, str(engine))
    times = {}
    start = time.perf_counter()
    import radsdf
    radsdf.kernel_version()  # loads the native kernel, which buildSeconds excludes
    times["kernelLoad"] = time.perf_counter() - start
    output.mkdir()
    worker = engine / "worker.py"
    sys.argv = [str(worker), "--request", str(folder / "request.json"), "--output", str(output),
                "--threads", str(args.threads)]
    profile = cProfile.Profile()
    status = 0
    start = time.perf_counter()
    profile.enable()
    try:
        runpy.run_path(str(worker), run_name="__main__")
    except SystemExit as exit:
        status = exit.code if isinstance(exit.code, int) else (0 if exit.code is None else 1)
    finally:
        profile.disable()
    times["workerTotal"] = time.perf_counter() - start
    result = output / "result.json"
    if result.is_file():
        times["workerReportedBuild"] = json.loads(result.read_text()).get("buildSeconds")
    profile.dump_stats(str(folder / "worker.pstats"))
    (folder / "worker-stages.json").write_text(json.dumps(times, indent=2))
    pstats.Stats(profile).strip_dirs().sort_stats("tottime").print_stats(40)
    print(json.dumps(times, indent=2))
    if status:
        print(f"The worker exited with status {status}; see {output}.", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
