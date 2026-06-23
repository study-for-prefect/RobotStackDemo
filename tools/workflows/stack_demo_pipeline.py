#!/usr/bin/env python3
"""Single entry point for the closed-loop stack demo workflow."""

import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.workflows.stack_demo import main


if __name__ == "__main__":
    raise SystemExit(main())
