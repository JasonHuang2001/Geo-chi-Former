"""Structural validation for locally supplied prepared data files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


class DataValidationError(ValueError):
    """Raised when a prepared CSV fails validation."""


def load_schema(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise DataValidationError(f"Schema must be a JSON object: {path}")
    return schema


def _fail(path: Path, message: str) -> None:
    raise DataValidationError(f"Invalid data in {path}: {message}")


def validate_csv(path: Path, schema_path: Path, *, strict_snapshot: bool = True) -> dict:
    schema = load_schema(schema_path)
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise DataValidationError(f"Could not read CSV {path}: {exc}") from exc

    expected_columns = list(schema["columns"])
    if list(frame.columns) != expected_columns:
        missing = [name for name in expected_columns if name not in frame.columns]
        extra = [name for name in frame.columns if name not in expected_columns]
        _fail(path, f"column order or membership differs; missing={missing}, extra={extra}")
    if frame.empty:
        _fail(path, "file has no data rows")
    if frame.isna().any().any():
        bad = frame.columns[frame.isna().any()].tolist()
        _fail(path, f"missing values found in columns {bad}")

    parsed_dates: dict[str, pd.Series] = {}
    for name in schema.get("date_columns", []):
        parsed = pd.to_datetime(frame[name], errors="coerce")
        if parsed.isna().any():
            _fail(path, f"invalid date values in {name}")
        if (parsed.dt.normalize() != parsed).any():
            _fail(path, f"non-daily timestamps in {name}")
        parsed_dates[name] = parsed

    for name in schema.get("numeric_columns", []):
        numeric = pd.to_numeric(frame[name], errors="coerce")
        if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            _fail(path, f"non-finite or non-numeric values in {name}")
    for name in schema.get("integer_columns", []):
        numeric = pd.to_numeric(frame[name], errors="coerce")
        if not np.equal(numeric.to_numpy(dtype=float), np.floor(numeric.to_numpy(dtype=float))).all():
            _fail(path, f"non-integer values in {name}")

    key = list(schema.get("primary_key", []))
    if key and frame.duplicated(key).any():
        _fail(path, f"duplicate primary-key rows for {key}")

    daily_column = schema.get("daily_date_column")
    if daily_column:
        dates = parsed_dates[daily_column]
        if not dates.is_monotonic_increasing:
            _fail(path, f"{daily_column} is not increasing")
        differences = dates.diff().dropna().dt.days
        if not (differences == 1).all():
            _fail(path, f"{daily_column} is not a continuous daily series")

    relationship = schema.get("lead_relationship")
    if relationship:
        issue = parsed_dates[relationship["issue_date_column"]]
        target = parsed_dates[relationship["target_date_column"]]
        leads = pd.to_numeric(frame[relationship["lead_column"]]).astype(int)
        if not (target == issue + pd.to_timedelta(leads, unit="D")).all():
            _fail(path, "target date does not equal issue date plus lead_day")
        expected_leads = list(relationship["allowed_leads"])
        grouped = frame.assign(_lead=leads).groupby(relationship["issue_date_column"])["_lead"]
        if any(sorted(group.tolist()) != expected_leads for _, group in grouped):
            _fail(path, "one or more issue dates do not contain exactly leads 0 through 13")

    snapshot = schema.get("paper_snapshot", {})
    if strict_snapshot:
        if len(frame) != int(snapshot["rows"]):
            _fail(path, f"row count {len(frame)} does not match paper snapshot {snapshot['rows']}")
        for key_name, expected in snapshot.items():
            if key_name.endswith("_min") or key_name.endswith("_max"):
                column, operation = key_name.rsplit("_", 1)
                actual = getattr(parsed_dates[column], operation)().strftime("%Y-%m-%d")
                if actual != expected:
                    _fail(path, f"{key_name}={actual}, expected {expected}")
        if "issue_dates" in snapshot:
            actual = parsed_dates["issue_date"].nunique()
            if actual != int(snapshot["issue_dates"]):
                _fail(path, f"issue_dates={actual}, expected {snapshot['issue_dates']}")

    return {
        "schema": str(schema_path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "strict_snapshot": bool(strict_snapshot),
        "status": "valid",
    }
