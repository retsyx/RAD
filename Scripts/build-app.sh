#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
[[ "$(uname -m)" == arm64 ]] || { echo "RAD requires Apple Silicon (arm64)." >&2; exit 1; }
CONFIG="${RAD_BUILD_CONFIGURATION:-release}"
RUNTIME_PAYLOAD="${RAD_RUNTIME_PAYLOAD:-}"
if [[ -z "$RUNTIME_PAYLOAD" ]]; then RUNTIME_PAYLOAD="$("$ROOT/Scripts/prepare-runtime.sh")"; fi
[[ -f "$RUNTIME_PAYLOAD/manifest.json" && -f "$RUNTIME_PAYLOAD/payload.tar.gz" ]] || {
    echo "The prepared runtime payload is incomplete." >&2; exit 1;
}
APP="${RAD_APP_OUTPUT:-$ROOT/dist/RAD.app}"
[[ "$(basename "$APP")" == RAD.app ]] || { echo "The app output must be named RAD.app." >&2; exit 1; }
mkdir -p "$(dirname "$APP")"
STAGING="$(mktemp -d "$(dirname "$APP")/.RAD-build.XXXXXX")"
trap 'rm -rf "$STAGING"' EXIT
ASSEMBLED="$STAGING/RAD.app"
export CLANG_MODULE_CACHE_PATH="${CLANG_MODULE_CACHE_PATH:-/tmp/rad-clang-cache}"
export SWIFTPM_MODULECACHE_OVERRIDE="${SWIFTPM_MODULECACHE_OVERRIDE:-/tmp/rad-swift-cache}"
swift build --disable-sandbox --arch arm64 -c "$CONFIG"
# The native SDF kernel is a dynamic library product; build it explicitly.
swift build --disable-sandbox --arch arm64 -c "$CONFIG" --product RADSDF
BIN="$(swift build --disable-sandbox --arch arm64 -c "$CONFIG" --show-bin-path)"
[[ -f "$BIN/libRADSDF.dylib" ]] || { echo "The native SDF kernel library is missing." >&2; exit 1; }
mkdir -p "$ASSEMBLED/Contents/MacOS" "$ASSEMBLED/Contents/Resources" "$ASSEMBLED/Contents/Frameworks"
cp "$BIN/RAD" "$ASSEMBLED/Contents/MacOS/RAD"
# Python workers load the kernel through RAD_SDF_LIBRARY. Native code stays out
# of Resources so the app signature seals it as nested code.
KERNEL="$ASSEMBLED/Contents/Frameworks/libRADSDF.dylib"
cp "$BIN/libRADSDF.dylib" "$KERNEL"
THUMBNAIL_EXTENSION="$ASSEMBLED/Contents/PlugIns/RADThumbnailExtension.appex"
mkdir -p "$THUMBNAIL_EXTENSION/Contents/MacOS"
cp "$BIN/RADThumbnailExtension" "$THUMBNAIL_EXTENSION/Contents/MacOS/RADThumbnailExtension"
cp "$ROOT/Scripts/ThumbnailExtension-Info.plist" "$THUMBNAIL_EXTENSION/Contents/Info.plist"
if [[ "$CONFIG" == release ]]; then
    SYMBOLS="${RAD_APP_SYMBOLS_OUTPUT:-}"
    if [[ -z "$SYMBOLS" ]]; then
        mkdir -p "$ROOT/.build/app-symbols"
        SYMBOLS="$(mktemp -d "$ROOT/.build/app-symbols/release.XXXXXX")"
    fi
    BUILD_PYTHON="${RAD_BUILD_PYTHON:-$RUNTIME_PAYLOAD/expanded/python/bin/python3}"
    "$BUILD_PYTHON" "$ROOT/Scripts/prepare_app_binaries.py" \
        --app "$ASSEMBLED" --symbols "$SYMBOLS" --build-root "$ROOT"
fi
# Bundle.module searches Contents/Resources in a packaged macOS app.
[[ -d "$BIN/RAD_RAD.bundle" ]] || { echo "The Swift resource bundle is missing." >&2; exit 1; }
ditto "$BIN/RAD_RAD.bundle" "$ASSEMBLED/Contents/Resources/RAD_RAD.bundle"
PACKAGED_ENGINE="$ASSEMBLED/Contents/Resources/Engine"
mkdir -p "$PACKAGED_ENGINE"
# Every top-level Engine module ships (worker, radsdf modeling API and its
# helpers, motion checks, agent tool server); Engine/tests stays behind.
cp "$ROOT"/Engine/*.py "$ROOT/Engine/agent-tools.json" "$ROOT/Engine/script.sb" "$ROOT/Engine/requirements-lock.txt" \
   "$PACKAGED_ENGINE/"
[[ -f "$PACKAGED_ENGINE/worker.py" && -f "$PACKAGED_ENGINE/radsdf.py" ]] || { echo "The Engine modules are incomplete." >&2; exit 1; }
ditto "$ROOT/Examples" "$ASSEMBLED/Contents/Resources/Examples"
# Ship a verified offline payload. First use prepares a versioned private
# interpreter/venv in Application Support without changing the signed app.
mkdir -p "$ASSEMBLED/Contents/Resources/CADRuntime"
cp "$RUNTIME_PAYLOAD/manifest.json" "$RUNTIME_PAYLOAD/payload.tar.gz" "$ASSEMBLED/Contents/Resources/CADRuntime/"
cp "$ROOT/LICENSE" "$ASSEMBLED/Contents/Resources/LICENSE.txt"
cp "$ROOT/Scripts/Info.plist" "$ASSEMBLED/Contents/Info.plist"
if [[ -n "${RAD_VERSION:-}" ]]; then
    /usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString $RAD_VERSION" "$ASSEMBLED/Contents/Info.plist"
fi
if [[ -n "${RAD_BUILD_NUMBER:-}" ]]; then
    /usr/libexec/PlistBuddy -c "Set :CFBundleVersion $RAD_BUILD_NUMBER" "$ASSEMBLED/Contents/Info.plist"
fi
for KEY in CFBundleShortVersionString CFBundleVersion; do
    VALUE="$(/usr/libexec/PlistBuddy -c "Print :$KEY" "$ASSEMBLED/Contents/Info.plist")"
    /usr/libexec/PlistBuddy -c "Set :$KEY $VALUE" "$THUMBNAIL_EXTENSION/Contents/Info.plist"
done
mkdir -p "$STAGING/icons"
xcrun swift -O "$ROOT/Scripts/make-icon.swift" "$STAGING/icons"
for ICON in RAD RADDocument RADFolder; do
    cp "$STAGING/icons/$ICON.icns" "$ASSEMBLED/Contents/Resources/"
done
if [[ "${RAD_APP_SIGNING:-adhoc}" != none ]]; then
    # Inside out: nested code first, then the bundle that seals it.
    codesign --force --sign - "$KERNEL"
    codesign --force --sign - --entitlements "$ROOT/Scripts/ThumbnailExtension.entitlements" "$THUMBNAIL_EXTENSION"
    codesign --force --sign - "$ASSEMBLED"
fi
# A fresh assembly never retains deleted resources or old runtime payloads.
if [[ -e "$APP" ]]; then rm -rf "$APP"; fi
mv "$ASSEMBLED" "$APP"
if [[ "${RAD_REGISTER_APP:-1}" == 1 ]]; then
    /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP"
fi
echo "Built $APP"
echo "Launch: open \"$APP\""
