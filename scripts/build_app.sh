#!/bin/zsh
#
# Builds (and re-publishes) LocalTranscriber.app: a thin wrapper bundle whose
# launcher runs `python3 -m app.main` from this checked-out project directory
# using its .venv. Re-run this after any code change to update the installed
# app - nothing is frozen/copied into the bundle, so most of the time you
# don't even need to rebuild, but re-running keeps the version stamp and
# bundle metadata current.
#
# The bundle is staged in a throwaway temp directory, not inside the project
# - a build artifact left sitting in the project tree is itself a real .app
# bundle, so Spotlight/Launch Services index it as a second, separate app
# alongside the one actually installed in /Applications. Building outside
# the project and cleaning up after avoids that entirely.
#
# Usage:
#   ./scripts/build_app.sh              # installs into /Applications
#   APP_INSTALL_DIR=~/Applications ./scripts/build_app.sh   # install elsewhere

set -eu

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
cd "$PROJECT_DIR"

APP_NAME="LocalTranscriber"
BUNDLE_ID="com.local-transcriber.app"
INSTALL_DIR="${APP_INSTALL_DIR:-/Applications}"
DIST_DIR="$(mktemp -d)"
trap 'rm -rf "$DIST_DIR"' EXIT
BUNDLE_PATH="$DIST_DIR/$APP_NAME.app"
LOGO_PATH="$PROJECT_DIR/logo.jpg"

if [ ! -f "$PROJECT_DIR/VERSION" ]; then
  echo "ERROR: VERSION file not found in $PROJECT_DIR" >&2
  exit 1
fi
VERSION="$(cat "$PROJECT_DIR/VERSION" | tr -d '[:space:]')"

echo "Local Transcriber build"
echo "Project: $PROJECT_DIR"
echo "Version: $VERSION"
echo

PYTHON_BIN="$PROJECT_DIR/.venv/bin/python3"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "No .venv found, falling back to system python3 for tests."
  PYTHON_BIN="python3"
fi

echo "Running tests..."
"$PYTHON_BIN" -m pytest -q
echo

echo "Building bundle at $BUNDLE_PATH"
rm -rf "$BUNDLE_PATH"
mkdir -p "$BUNDLE_PATH/Contents/MacOS"
mkdir -p "$BUNDLE_PATH/Contents/Resources"

ICON_PLIST_KEYS=""
if [ -f "$LOGO_PATH" ]; then
  echo "Generating app icon from $LOGO_PATH"
  ICONSET_DIR="$DIST_DIR/AppIcon.iconset"
  mkdir -p "$ICONSET_DIR"
  for size in 16 32 128 256 512; do
    sips -s format png -z "$size" "$size" "$LOGO_PATH" --out "$ICONSET_DIR/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -s format png -z "$double" "$double" "$LOGO_PATH" --out "$ICONSET_DIR/icon_${size}x${size}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET_DIR" -o "$BUNDLE_PATH/Contents/Resources/AppIcon.icns"
  ICON_PLIST_KEYS="    <key>CFBundleIconFile</key>
    <string>AppIcon</string>"
else
  echo "No logo.jpg found at $LOGO_PATH - building without a custom icon."
fi

cat > "$BUNDLE_PATH/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleDisplayName</key>
    <string>Local Transcriber</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleVersion</key>
    <string>$VERSION</string>
    <key>CFBundleShortVersionString</key>
    <string>$VERSION</string>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
$ICON_PLIST_KEYS
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
</dict>
</plist>
PLIST

cat > "$BUNDLE_PATH/Contents/MacOS/$APP_NAME" <<LAUNCHER
#!/bin/bash
set -eu
PROJECT_DIR="$PROJECT_DIR"
cd "\$PROJECT_DIR"
# Apps launched via Finder/Spotlight (launchd) get a minimal PATH that does
# not include Homebrew, so ffmpeg/ffprobe wouldn't be found even though a
# Terminal shell finds them fine. Add the common Homebrew bin dirs.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:\$PATH"
# Run the venv's interpreter by absolute path rather than \`source activate\`:
# a venv's activate script bakes in an absolute VIRTUAL_ENV, so if the project
# is ever moved/renamed activation silently prepends a dead path and \`python3\`
# falls through to a system Python without our deps (tkinter/tkinterdnd2).
# The venv interpreter finds its own site-packages via pyvenv.cfg regardless.
if [ -x ".venv/bin/python3" ]; then
  exec ".venv/bin/python3" -m app.main
else
  exec python3 -m app.main
fi
LAUNCHER
chmod +x "$BUNDLE_PATH/Contents/MacOS/$APP_NAME"

echo "Installing to $INSTALL_DIR/$APP_NAME.app"
if [ -w "$INSTALL_DIR" ] || [ -w "$INSTALL_DIR/$APP_NAME.app" ] 2>/dev/null; then
  rm -rf "$INSTALL_DIR/$APP_NAME.app"
  cp -R "$BUNDLE_PATH" "$INSTALL_DIR/$APP_NAME.app"
else
  echo "ERROR: no write permission for $INSTALL_DIR." >&2
  echo "Re-run with: APP_INSTALL_DIR=\$HOME/Applications ./scripts/build_app.sh" >&2
  echo "or: sudo ./scripts/build_app.sh" >&2
  exit 1
fi

echo
echo "Installed $APP_NAME.app version $VERSION to $INSTALL_DIR"
echo "Launch it from Finder/Spotlight - it always runs the live code in:"
echo "  $PROJECT_DIR"
