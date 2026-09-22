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

if [ -d ".venv" ]; then
  echo "Activating .venv..."
  source ".venv/bin/activate"
else
  echo "No .venv found. Using system python3."
fi

echo "Starting Local Transcriber..."
echo
python3 -m app.main
STATUS=$?

echo
echo "Transcriber exited with status $STATUS."
echo "Press Return to close this window."
read -r
exit "$STATUS"
