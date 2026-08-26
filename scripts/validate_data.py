"""Validate prepared inputs against their schemas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.runtime import CommandError, validate_prepared_inputs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--allow-unverified-inputs", action="store_true")
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    args = parser.parse_args(argv)
    config = {
        "data_path": "eop_data_xy_EAM.csv",
        "use_future_observed_eam": True,
        "future_eam_forecast_path": "eam14forecast_daily.csv",
    }
    try:
        report = validate_prepared_inputs(
            config,
            Path(args.data_dir).resolve(),
            verify_hashes=not args.allow_unverified_inputs,
        )
    except CommandError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        report_path = Path(args.report).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
