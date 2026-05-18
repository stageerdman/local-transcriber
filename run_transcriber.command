#!/bin/zsh

set -u

SCRIPT_DIR="${0:A:h}"
cd "$SCRIPT_DIR" || exit 1

echo "Local Transcriber"
echo "Project folder: $SCRIPT_DIR"
echo

if [ -d ".venv" ]; then
  echo "Activating .venv..."
  source ".venv/bin/activate"
else
  echo "No .venv found. Using system python3."
fi

echo "Starting transcriber..."
echo
python3 transcriber.py
STATUS=$?

echo
echo "Transcriber exited with status $STATUS."
echo "Press Return to close this window."
read -r
exit "$STATUS"
