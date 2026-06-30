#!/usr/bin/env python3
"""Single entry point and compatibility exports for two-stage visual pick."""

import os
import sys


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.workflows.two_stage_pick import (
    build_corrected_plan,
    build_tcp_error_corrected_plan,
    camera_optical_vector_to_base,
    camera_vector_to_base,
    main,
)

__all__ = [
    "build_corrected_plan",
    "build_tcp_error_corrected_plan",
    "camera_optical_vector_to_base",
    "camera_vector_to_base",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
