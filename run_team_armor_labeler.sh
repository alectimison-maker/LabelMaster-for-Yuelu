#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins"
export QT_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins"
exec python3 "$BASE_DIR/app/main.py" --model "$BASE_DIR/assets/models/model-opt.onnx"
