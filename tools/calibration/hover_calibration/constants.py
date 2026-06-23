"""Shared paths and orientation defaults for hover calibration."""

import os
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(
    "runtime",
    "tool_offset_calibration",
    "run_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
)
DEFAULT_DOWNWARD_QUAT_XYZW = [1.0, 0.0, 0.0, 0.0]
