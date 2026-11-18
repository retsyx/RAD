#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
RUNTIME_PAYLOAD="$("$ROOT/Scripts/prepare-runtime.sh")"
export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-/tmp/rad-clang-cache}"
export SWIFTPM_MODULECACHE_OVERRIDE="${SWIFTPM_MODULECACHE_OVERRIDE:-/tmp/rad-swift-cache}"
# The native SDF kernel is a dynamic library product, which `swift test` does
# not build. Python tests and workers load it from RAD_SDF_LIBRARY; the worker
# sandbox matches real paths, so export the resolved location.
swift build --disable-sandbox --arch arm64 --product RADSDF
KERNEL_DIRECTORY="$(swift build --disable-sandbox --arch arm64 --show-bin-path)"
[[ -f "$KERNEL_DIRECTORY/libRADSDF.dylib" ]] || { echo "The native SDF kernel library is missing." >&2; exit 1; }
export RAD_SDF_LIBRARY="$(cd "$KERNEL_DIRECTORY" && pwd -P)/libRADSDF.dylib"
swift test --disable-sandbox --arch arm64
"$RUNTIME_PAYLOAD/expanded/python/bin/python3" -m unittest discover -s Engine/tests -v
