"""Process and JSON I/O helpers."""

import json
import os
import subprocess

from .constants import PROJECT_ROOT

def run(command, execute=True):
    print("\n$ {}".format(" ".join(command)), flush=True)
    if execute:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def run_checked(command, execute=True):
    print("\n$ {}".format(" ".join(command)), flush=True)
    if not execute:
        return True
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode == 0


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
