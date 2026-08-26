"""Shared configuration, validation, and command helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = REPOSITORY_ROOT / "checkpoints" / "paper_main" / "chiformer_forecasting_best.pth"
DEFAULT_EVALUATION_CONFIG = REPOSITORY_ROOT / "configs" / "paper_evaluation.json"
DEFAULT_INPUT_MANIFEST = REPOSITORY_ROOT / "data" / "paper_input_manifest.json"


class CommandError(RuntimeError):
    """Raised when a command cannot run safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise CommandError(f"Required JSON file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CommandError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise CommandError(f"Expected a JSON object in {path}")
    return value


def resolve_path(value: str | Path, *, base: Path = REPOSITORY_ROOT) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def require_file(
    path: Path,
    *,
    purpose: str,
    expected_sha256: str | None = None,
    instructions: str | None = None,
    verify_hash: bool = True,
) -> str:
    if not path.is_file():
        details = [
            f"Required input is missing: {path}",
            f"Purpose: {purpose}",
        ]
        if expected_sha256:
            details.append(f"Expected SHA-256: {expected_sha256}")
        if instructions:
            details.append(f"Source guidance: {instructions}")
        details.append("No output was created.")
        raise CommandError("\n".join(details))
    actual = sha256_file(path)
    if verify_hash and expected_sha256 and actual != expected_sha256.upper():
        raise CommandError(
            "\n".join(
                [
                    f"Input hash mismatch: {path}",
                    f"Purpose: {purpose}",
                    f"Expected SHA-256: {expected_sha256}",
                    f"Actual SHA-256:   {actual}",
                    "No output was created.",
                ]
            )
        )
    return actual


def load_checkpoint_configuration(checkpoint: Path) -> dict:
    checkpoint = checkpoint.resolve()
    checksums = read_sha256sums(checkpoint.parent / "SHA256SUMS")
    require_file(
        checkpoint,
        purpose="Geo-chi-Former model checkpoint",
        expected_sha256=checksums.get(checkpoint.name),
        verify_hash=True,
    )
    config_path = checkpoint.parent / "config.json"
    require_file(
        config_path,
        purpose="immutable training configuration",
        expected_sha256=checksums.get(config_path.name),
        verify_hash=True,
    )
    return load_json(config_path)


def read_sha256sums(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            values[parts[1].lstrip("*")] = parts[0].upper()
    return values


def apply_evaluation_overlay(config: dict, overlay: Mapping) -> dict:
    runtime_keys = {
        "metric_horizons",
        "use_fixed_test_forecast_dates",
        "test_forecast_start_date",
        "test_forecast_end_date",
        "test_forecast_stride",
        "eval_integrate_pm",
        "batch_size",
    }
    merged = dict(config)
    for key in runtime_keys:
        if key in overlay:
            merged[key] = overlay[key]
    return merged


def config_namespace(config: Mapping) -> SimpleNamespace:
    return SimpleNamespace(**dict(config))


def validate_prepared_inputs(
    config: Mapping,
    data_dir: Path,
    *,
    verify_hashes: bool = True,
    manifest_path: Path = DEFAULT_INPUT_MANIFEST,
) -> dict[str, dict[str, str | int]]:
    manifest = load_json(manifest_path)
    manifest_files = manifest.get("files", {})
    instructions = str(manifest.get("instructions", "See data/README.md."))
    requested = [str(config.get("data_path", "eop_data_xy_EAM.csv"))]
    if bool(config.get("use_future_observed_eam", False)):
        requested.append(str(config.get("future_eam_forecast_path", "eam14forecast_daily.csv")))

    validated: dict[str, dict[str, str | int]] = {}
    for relative_name in requested:
        path = (data_dir / relative_name).resolve()
        record = manifest_files.get(Path(relative_name).name, {})
        expected_hash = record.get("sha256")
        purpose = str(record.get("purpose", f"prepared model input {relative_name}"))
        actual_hash = require_file(
            path,
            purpose=purpose,
            expected_sha256=expected_hash,
            instructions=instructions,
            verify_hash=verify_hashes,
        )
        schema_report = None
        schema_name = record.get("schema")
        if schema_name:
            from utils.data_validation import DataValidationError, validate_csv

            schema_path = (REPOSITORY_ROOT / str(schema_name)).resolve()
            try:
                schema_report = validate_csv(path, schema_path, strict_snapshot=verify_hashes)
            except DataValidationError as exc:
                raise CommandError(f"{exc}\nNo output was created.") from exc
        validated[relative_name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual_hash,
            "schema_validation": schema_report,
        }
    return validated


def validate_output_directory(path: Path, *, overwrite: bool = False) -> None:
    if path.exists() and not path.is_dir():
        raise CommandError(f"Output path exists and is not a directory: {path}")
    if path.is_dir() and any(path.iterdir()) and not overwrite:
        raise CommandError(
            f"Output directory is not empty: {path}\n"
            "Choose a new directory or pass --overwrite. No output was changed."
        )


def assert_files_unchanged(before: Mapping[Path, str]) -> None:
    changed = []
    for path, expected_hash in before.items():
        if not path.is_file() or sha256_file(path) != expected_hash:
            changed.append(str(path))
    if changed:
        raise RuntimeError("Protected input changed during execution: " + ", ".join(changed))


def parse_horizons(value: str | Iterable[int]) -> list[int]:
    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
        horizons = [int(part) for part in values]
    else:
        horizons = [int(part) for part in value]
    if not horizons or any(horizon < 1 for horizon in horizons):
        raise CommandError("Metric horizons must be positive integers.")
    return sorted(set(horizons))
