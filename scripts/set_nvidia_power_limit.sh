#!/usr/bin/env bash
set -euo pipefail

power_limit_w="${1:-250}"
attempt=0
while ! /usr/bin/nvidia-smi -L >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 30 ]; then
    echo "NVIDIA GPU did not become ready within 60 seconds" >&2
    exit 1
  fi
  sleep 2
done

/usr/bin/nvidia-smi -pm 1
/usr/bin/nvidia-smi -pl "$power_limit_w"
/usr/bin/nvidia-smi \
  --query-gpu=name,power.limit,power.default_limit \
  --format=csv,noheader
