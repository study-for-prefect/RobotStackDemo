#!/usr/bin/env python3
"""Single entry point and compatibility exports for XY bias diagnosis."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.calibration.xy_bias import (
    analyze_dataset,
    build_stack_calibration,
    fit_workspace_model,
    main,
    trial_key,
)

__all__ = ["analyze_dataset", "build_stack_calibration", "fit_workspace_model", "main", "trial_key"]

if __name__ == "__main__":
    raise SystemExit(main())
