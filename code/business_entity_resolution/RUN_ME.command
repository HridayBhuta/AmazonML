#!/bin/bash
# Double-click in Finder to run everything (if macOS blocks it, see README "If macOS blocks the file").
cd "$(dirname "$0")"
bash run.sh auto
STATUS=$?
echo
if [ $STATUS -eq 0 ]; then
  echo "Done. The files to upload are listed in final/SUBMIT_THESE.txt"
else
  echo "The run FAILED (exit $STATUS). Scroll up or open the logs/ folder to see why."
fi
echo "Press Enter to close."; read -r _
