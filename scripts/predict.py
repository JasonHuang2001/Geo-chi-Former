"""Stable checkpoint prediction entry point for Geo-chi-Former."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.runtime import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_EVALUATION_CONFIG,
    CommandError,
    apply_evaluation_overlay,
    assert_files_unchanged,
    config_namespace,
    load_checkpoint_configuration,
    load_json,
    require_file,
    resolve_path,
    sha256_file,
    validate_output_directory,
    validate_prepared_inputs,
)


DEFAULT_PAPER_UL_GAP_CACHE = REPOSITORY_ROOT / "data" / "reference" / "paper_ul_gap_test_L360_P30.npz"
PAPER_UL_GAP_CACHE_SHA256 = "B8D0DE60E6514DC2AB0813D0A3C13987B4DA13423F96135CAC1B551B7A67D04C"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Geo-chi-Former predictions for one issue date or the 309-date paper test set."
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Model checkpoint path.")
    parser.add_argument("--data-dir", required=True, help="Directory containing prepared input CSV files.")
    parser.add_argument(
        "--evaluation-config",
        default=str(DEFAULT_EVALUATION_CONFIG),
        help="Evaluation overlay JSON.",
    )
    parser.add_argument("--output-dir", required=True, help="New or empty prediction-output directory.")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--issue-date", help="Run one issue date in YYYY-MM-DD format.")
    selection.add_argument(
        "--all-paper-dates",
        action="store_true",
        help="Run the frozen 309-date paper evaluation protocol.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--ul-gap-cache",
        default=str(DEFAULT_PAPER_UL_GAP_CACHE),
        help="Frozen paper UL-gap intermediate; required for exact checkpoint regression.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--allow-unverified-inputs",
        action="store_true",
        help="Allow prepared files whose hashes differ from the frozen paper inputs.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    return parser


def _device_from_name(name, torch):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise CommandError("CUDA was requested but is not available. No output was created.")
    return torch.device(name)


def run(args: argparse.Namespace) -> Path:
    checkpoint = resolve_path(args.checkpoint, base=Path.cwd())
    data_dir = resolve_path(args.data_dir, base=Path.cwd())
    overlay_path = resolve_path(args.evaluation_config, base=Path.cwd())
    output_dir = resolve_path(args.output_dir, base=Path.cwd())
    training_config = load_checkpoint_configuration(checkpoint)
    overlay = load_json(overlay_path)
    config_dict = apply_evaluation_overlay(training_config, overlay)
    validated_inputs = validate_prepared_inputs(
        config_dict,
        data_dir,
        verify_hashes=not args.allow_unverified_inputs,
    )
    validate_output_directory(output_dir, overwrite=args.overwrite)

    protected = {checkpoint: sha256_file(checkpoint), checkpoint.parent / "config.json": sha256_file(checkpoint.parent / "config.json")}
    protected.update({Path(record["path"]): str(record["sha256"]) for record in validated_inputs.values()})
    ul_gap_cache = resolve_path(args.ul_gap_cache, base=Path.cwd())
    cache_hash = require_file(
        ul_gap_cache,
        purpose="frozen paper UL-gap regression intermediate",
        expected_sha256=PAPER_UL_GAP_CACHE_SHA256,
        instructions="See data/reference/README.md.",
        verify_hash=not args.allow_unverified_inputs,
    )
    protected[ul_gap_cache] = cache_hash

    import torch

    from data.EOP_loader import Dataset_EOP_ULGap
    from models.model_setup import apply_checkpoint_defaults, build_paper_model
    from utils.inference import evaluate_model_on_dataset

    config_dict["root_path"] = str(data_dir)
    config_dict["model_path"] = str(checkpoint)
    config_dict["save_dir"] = str(output_dir)
    config_dict["ul_gap_cache_dir"] = str(output_dir / "cache" / "eop_ul_gap")
    config_dict["ul_gap_cache_path"] = str(ul_gap_cache)
    if args.batch_size is not None:
        config_dict["batch_size"] = args.batch_size
    if args.issue_date:
        config_dict["use_fixed_test_forecast_dates"] = True
        config_dict["test_forecast_start_date"] = args.issue_date
        config_dict["test_forecast_end_date"] = args.issue_date
        expected_samples = 1
    else:
        expected_samples = int(overlay.get("expected_test_issue_dates", 309))

    config = config_namespace(config_dict)
    apply_checkpoint_defaults(config)
    device = _device_from_name(args.device, torch)
    model = build_paper_model(config, device)
    state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    dataset = Dataset_EOP_ULGap(config, flag="test")
    if len(dataset) != expected_samples:
        raise CommandError(
            f"Issue-date count mismatch: expected {expected_samples}, got {len(dataset)}. No prediction files were created."
        )

    result = evaluate_model_on_dataset(
        model=model,
        dataset=dataset,
        config=config,
        device=device,
        mode_name="test",
        save_dir=str(output_dir),
    )
    assert_files_unchanged(protected)
    manifest = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": protected[checkpoint],
        "data_files": validated_inputs,
        "ul_gap_cache": {
            "path": str(ul_gap_cache),
            "sha256": cache_hash,
        },
        "issue_date_count": len(dataset),
        "prediction_file": "test_predictions.csv",
        "metrics_file": "test_metrics.json",
        "device": str(device),
        "num_samples": result["num_samples"],
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return output_dir / "test_predictions.csv"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        predictions = run(args)
    except CommandError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Prediction complete: {predictions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
