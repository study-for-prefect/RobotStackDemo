#!/usr/bin/env python3
"""Single entry point for realtime YOLO and depth monitoring."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.monitoring.realtime_monitor import main

if __name__ == "__main__":
    raise SystemExit(main())
