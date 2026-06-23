"""Top-level XY bias diagnostic command."""

from typing import Optional

from .analysis import analyze
from .arguments import parse_args
from .collection import collect

def main() -> Optional[int]:
    args = parse_args()
    if args.command == "collect":
        return collect(args)
    return analyze(args)
