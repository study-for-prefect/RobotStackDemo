"""Shared paths for XY bias calibration."""

import os
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_OUTPUT_DIR = os.path.join(
    "runtime",
    "xy_bias_diagnosis",
    "run_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
)
