"""JSON and subprocess helpers for XY bias calibration."""

import json
import subprocess

from .constants import PROJECT_ROOT

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run(command):
    print("$ {}".format(" ".join(str(value) for value in command)), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
