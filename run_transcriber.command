#!/bin/zsh

set -u

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR" || exit 1

# Belt-and-suspenders: make sure Homebrew's ffmpeg/ffprobe are found even if
# this isn't a login shell with Homebrew's shellenv already sourced.
export PATH="/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:$PATH"

echo "Local Transcriber"
echo "Project folder: $SCRIPT_DIR"
echo

# Run the venv's interpreter by absolute path rather than `source activate`:
# activate bakes in an absolute VIRTUAL_ENV, so a moved/renamed project makes
# activation prepend a dead path and `python3` falls through to a system
# Python without our deps. The venv interpreter still finds its own
# site-packages via pyvenv.cfg.
if [ -x ".venv/bin/python3" ]; then
  PYTHON_BIN=".venv/bin/python3"
else
  echo "No .venv found. Using system python3."
  PYTHON_BIN="python3"
fi

echo "Starting Local Transcriber..."
echo
"$PYTHON_BIN" -m app.main
STATUS=$?

echo
echo "Transcriber exited with status $STATUS."
echo "Press Return to close this window."
read -r
exit "$STATUS"
