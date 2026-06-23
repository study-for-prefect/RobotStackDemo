"""Default output paths for perception stability diagnostics."""

import os
from datetime import datetime

DEFAULT_OUTPUT_DIR = os.path.join(
    "runtime",
    "perception_stability",
    "run_{}".format(datetime.now().strftime("%Y%m%d_%H%M%S")),
)
