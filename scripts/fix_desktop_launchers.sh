#!/usr/bin/env bash
set -e

DESKTOP_DIR="/home/wxm/桌面"

FILES=(
  "$DESKTOP_DIR/Start_Robot_Stack.desktop"
  "$DESKTOP_DIR/Run_Stack_Task.desktop"
  "$DESKTOP_DIR/Stop_Robot_Stack.desktop"
)

for f in "${FILES[@]}"; do
  if [ -f "$f" ]; then
    chmod +x "$f"
    gio set "$f" metadata::trusted true || true
  fi
done
