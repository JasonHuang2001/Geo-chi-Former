"""Independent evaluation of saved polar-motion prediction CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.runtime import (  # noqa: E402
    CommandError,
    parse_horizons,
    resolve_path,
)


REQUIRED_COLUMNS = {
    "sample_index",
    "lead",
    "date",
    "pred_xpole",
    "pred_ypole",
    "true_xpole",
    "true_ypole",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute PM RMSE/MAE from saved predictions and, when a matched Bulletin A "
            "prediction CSV is supplied, skill = 1 - RMSE_model / RMSE_BulletinA."
        )
    )
    parser.add_argument("--predictions", required=True, help="Geo-chi-Former prediction CSV.")
    parser.add_argument(
        "--bulletin-a-predictions",
        default=None,
        help="Optional matched Bulletin A CSV using the same column schema.",
    )
    parser.add_argument("--horizons", default="1,7,10,14,30", help="Comma-separated lead horizons.")
    parser.add_argument("--output", default=None, help="Output JSON; defaults beside the prediction CSV.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise CommandError(f"Prediction file is missing: {path}\nNo output was created.")
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise CommandError(
                f"Prediction CSV {path} is missing columns: {missing}\nNo output was created."
            )
        rows = list(reader)
    if not rows:
        raise CommandError(f"Prediction CSV is empty: {path}\nNo output was created.")
    try:
        for row in rows:
            row["sample_index"] = int(row["sample_index"])
            row["lead"] = int(row["lead"])
            for name in ["pred_xpole", "pred_ypole", "true_xpole", "true_ypole"]:
                row[name] = float(row[name])
    except (TypeError, ValueError) as exc:
        raise CommandError(f"Prediction CSV contains an invalid numeric value: {path}") from exc
    return rows


def _key(row: dict) -> tuple[int, int, str]:
    return row["sample_index"], row["lead"], row["date"]


def _metrics(rows: list[dict]) -> dict:
    squared = []
    absolute = []
    component = {"xpole": {"squared": [], "absolute": []}, "ypole": {"squared": [], "absolute": []}}
    for row in rows:
        for axis in ["xpole", "ypole"]:
            error = row[f"pred_{axis}"] - row[f"true_{axis}"]
            squared.append(error * error)
            absolute.append(abs(error))
            component[axis]["squared"].append(error * error)
            component[axis]["absolute"].append(abs(error))
    result = {
        "rmse_mas": math.sqrt(sum(squared) / len(squared)),
        "mae_mas": sum(absolute) / len(absolute),
        "row_count": len(rows),
    }
    for axis, values in component.items():
        result[axis] = {
            "rmse_mas": math.sqrt(sum(values["squared"]) / len(values["squared"])),
            "mae_mas": sum(values["absolute"]) / len(values["absolute"]),
        }
    return result


def evaluate_rows(model_rows: list[dict], horizons: list[int], bulletin_rows: list[dict] | None = None) -> dict:
    max_horizon = max(row["lead"] for row in model_rows) + 1
    unavailable = [horizon for horizon in horizons if horizon > max_horizon]
    if unavailable:
        raise CommandError(
            f"Requested horizons exceed available lead {max_horizon}: {unavailable}. No output was created."
        )

    bulletin_by_key = None
    if bulletin_rows is not None:
        bulletin_by_key = {_key(row): row for row in bulletin_rows}
        model_keys = {_key(row) for row in model_rows}
        if model_keys != set(bulletin_by_key):
            raise CommandError(
                "Bulletin A rows do not align exactly with model rows by sample_index, lead, and date. "
                "No output was created."
            )
        for row in model_rows:
            reference = bulletin_by_key[_key(row)]
            if any(
                not math.isclose(row[f"true_{axis}"], reference[f"true_{axis}"], rel_tol=0.0, abs_tol=1e-9)
                for axis in ["xpole", "ypole"]
            ):
                raise CommandError(
                    "Bulletin A and model files do not contain identical verifying observations. "
                    "No output was created."
                )

    by_horizon = {}
    for horizon in horizons:
        selected = [row for row in model_rows if row["lead"] < horizon]
        model_metrics = _metrics(selected)
        entry = {"model": model_metrics, "bulletin_a": None, "skill_score": None}
        if bulletin_by_key is not None:
            reference = [bulletin_by_key[_key(row)] for row in selected]
            reference_metrics = _metrics(reference)
            reference_rmse = reference_metrics["rmse_mas"]
            entry["bulletin_a"] = reference_metrics
            entry["skill_score"] = (
                1.0 - model_metrics["rmse_mas"] / reference_rmse
                if reference_rmse > 0.0
                else None
            )
        by_horizon[str(horizon)] = entry

    return {
        "metric_definition": {
            "rmse_mae": "Combined over xpole and ypole errors, in milliarcseconds.",
            "bulletin_a_skill_score": "1 - RMSE_model / RMSE_BulletinA; positive values favor Geo-chi-Former.",
            "lead_indexing": "CSV lead is zero-based; horizon H aggregates leads 0 through H-1.",
        },
        "overall_model": _metrics(model_rows),
        "horizons": by_horizon,
        "bulletin_a_reference_supplied": bulletin_by_key is not None,
    }


def run(args: argparse.Namespace) -> Path:
    predictions = resolve_path(args.predictions, base=Path.cwd())
    output = (
        resolve_path(args.output, base=Path.cwd())
        if args.output
        else predictions.parent / "independent_metrics.json"
    )
    model_rows = _read_rows(predictions)
    bulletin_rows = None
    if args.bulletin_a_predictions:
        bulletin_rows = _read_rows(resolve_path(args.bulletin_a_predictions, base=Path.cwd()))
    horizons = parse_horizons(args.horizons)
    result = evaluate_rows(model_rows, horizons, bulletin_rows)
    if output.parent.exists() and not output.parent.is_dir():
        raise CommandError(f"Output parent is not a directory: {output.parent}")
    if output.exists() and not args.overwrite:
        raise CommandError(f"Output file already exists: {output}\nPass --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = run(args)
    except CommandError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Evaluation complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
