"""Command launcher for Geo-chi-Former checkpoint inference."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

SCRIPT_COMMANDS = {
    "predict": "scripts/predict.py",
    "evaluate": "scripts/evaluate.py",
    "validate": "scripts/validate_data.py",
}

HELP = """usage: python run.py COMMAND [arguments]

commands:
  predict     run the published checkpoint
  evaluate    evaluate a prediction table
  validate    validate prepared checkpoint inputs
  test        run the project test suite

examples:
  python run.py predict --help
  python run.py evaluate --help
  python run.py validate --help
  python run.py test
"""


def _script_command(relative_path: str, forwarded: list[str]) -> list[str]:
    return [sys.executable, "-B", str(ROOT / relative_path), *forwarded]


def build_command(arguments: list[str]) -> list[str] | None:
    if not arguments or arguments[0] in {"-h", "--help", "help"}:
        print(HELP)
        return None

    command, *forwarded = arguments
    if command in SCRIPT_COMMANDS:
        return _script_command(SCRIPT_COMMANDS[command], forwarded)
    if command == "test":
        return [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-v", *forwarded]
    raise SystemExit(f"unknown command: {command!r}; run 'python run.py --help'")


def main(arguments: list[str] | None = None) -> int:
    command = build_command(sys.argv[1:] if arguments is None else arguments)
    if command is None:
        return 0
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
