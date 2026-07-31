#!/bin/zsh
#
# Builds (and re-publishes) LocalTranscriber.app: a thin wrapper bundle whose
# launcher runs `python3 -m app.main` from this checked-out project directory
# using its .venv. Re-run this after any code change to update the installed
# app - nothing is frozen/copied into the bundle, so most of the time you
# don't even need to rebuild, but re-running keeps the version stamp and
# bundle metadata current.
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
DIST_DIR="$PROJECT_DIR/dist"
BUNDLE_PATH="$DIST_DIR/$APP_NAME.app"

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
if [ -d ".venv" ]; then
  source ".venv/bin/activate"
fi
exec python3 -m app.main
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
