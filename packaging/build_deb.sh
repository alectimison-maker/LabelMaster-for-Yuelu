#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="1.1.0"
ARCH="amd64"
PKG_NAME="labelmaster-for-yuelu"
OUT_DIR="$ROOT_DIR/dist"
WORK_DIR="$(mktemp -d)"
PKG_DIR="$WORK_DIR/${PKG_NAME}_${VERSION}_${ARCH}"

mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/opt/$PKG_NAME"
mkdir -p "$PKG_DIR/usr/bin"
mkdir -p "$PKG_DIR/usr/share/applications"

cat > "$PKG_DIR/DEBIAN/control" <<EOF
Package: $PKG_NAME
Version: $VERSION
Section: utils
Priority: optional
Architecture: $ARCH
Maintainer: Team Vision <team@example.com>
Depends: python3, python3-pyqt5, python3-pyqt5.qtsvg, python3-opencv, python3-numpy
Description: Team armor and energy rune annotation tool
EOF

cp -r "$ROOT_DIR/app" "$PKG_DIR/opt/$PKG_NAME/"
cp -r "$ROOT_DIR/assets" "$PKG_DIR/opt/$PKG_NAME/"

cat > "$PKG_DIR/usr/bin/$PKG_NAME" <<'EOF'
#!/usr/bin/env bash
export QT_QPA_PLATFORM_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins"
export QT_PLUGIN_PATH="/usr/lib/x86_64-linux-gnu/qt5/plugins"
exec python3 /opt/labelmaster-for-yuelu/app/main.py \
  --ov-xml /opt/labelmaster-for-yuelu/assets/models/yolov5.xml \
  --model /opt/labelmaster-for-yuelu/assets/models/model-opt.onnx \
  --energy-model /opt/labelmaster-for-yuelu/assets/models/energy_rune_5kp.onnx "$@"
EOF
chmod 755 "$PKG_DIR/usr/bin/$PKG_NAME"

cat > "$PKG_DIR/usr/share/applications/$PKG_NAME.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=LabelMaster for Yuelu
Exec=/usr/bin/$PKG_NAME
Terminal=false
Categories=Utility;Development;
EOF

mkdir -p "$OUT_DIR"
DEB_PATH="$OUT_DIR/${PKG_NAME}_${VERSION}_${ARCH}.deb"
if command -v dpkg-deb >/dev/null 2>&1; then
  dpkg-deb --build "$PKG_DIR" "$DEB_PATH" >/dev/null
else
  python3 "$ROOT_DIR/packaging/build_deb_portable.py" "$PKG_DIR" "$DEB_PATH"
fi

echo "DEB_BUILT=$DEB_PATH"
