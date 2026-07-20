#!/usr/bin/env bash

PGID_FILE="/tmp/robot_stack_moveit.pgid"
TMUX_SESSION="robot_stack"

if [ -s "$PGID_FILE" ]; then
  MOVEIT_PGID="$(tr -d '[:space:]' < "$PGID_FILE")"

  if [[ "$MOVEIT_PGID" =~ ^[0-9]+$ ]]; then
    echo "[stop] stopping MoveIt process group: $MOVEIT_PGID"

    kill -TERM -- "-$MOVEIT_PGID" 2>/dev/null || true
    sleep 2

    kill -KILL -- "-$MOVEIT_PGID" 2>/dev/null || true
  fi

  rm -f "$PGID_FILE"
fi

tmux kill-session -t "$TMUX_SESSION" 2>/dev/null || true

echo "[stop] robot stack stopped"
