#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAYLOAD="$("$ROOT/Scripts/prepare-runtime.sh")"
echo "Python runtime payload ready: $PAYLOAD"
echo "Run Scripts/build-app.sh to build RAD and its native SDF kernel, then open dist/RAD.app. RAD prepares its private runtime automatically."
