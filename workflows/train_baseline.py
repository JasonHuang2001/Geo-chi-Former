"""Train and validate a paper comparison model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from data.EOP_loader import Dataset_EOP_ULGap
from models.DLinear import DLinear
from models.LSTM import LSTMForecaster
from models.PatchTST import PatchTST
from utils.neural_baseline_common import (
    AGG_EAM_COLS,
    NeuralBaselineTrainer,
    apply_neural_baseline_defaults,
    set_seed,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "baseline_runs.json"
MODEL_CLASSES = {
    "dlinear": DLinear,
    "lstm": LSTMForecaster,
    "patchtst": PatchTST,
}


def load_run_config(path: Path, model: str, target_space: str) -> SimpleNamespace:
    """Load one frozen paper baseline configuration."""
    document = json.loads(path.read_text(encoding="utf-8"))
    selected = None
    for run in document["runs"]:
        if run["model"].lower() == model and run["target_space"] == target_space:
            selected = run
            break
    if selected is None:
        raise ValueError(f"No baseline configuration for model={model}, target_space={target_space}")

    values = dict(document["common"])
    values.update(selected)
    values.update({
        "model_type": model,
        "root_path": "data",
        "data_path": "eop_data_xy_EAM.csv",
        "use_ul_gap_loader": True,
        "eam_cols": list(AGG_EAM_COLS),
        "scale_chi": False,
        "scale_eam": False,
        "scale_eam_rad_to_mas": True,
        "unit": "mas",
        "ul_gap_window": 1095,
        "ul_gap_periods": [365.25, 365.25 / 2],
        "ul_gap_huber_epsilon": 1.35,
        "precompute_ul_gap": True,
        "ul_gap_cache_dir": "data/.cache/eop_ul_gap",
        "use_doy_time_features": True,
        "use_doy_time_mark_features": True,
        "doy_time_periods": [365.25, 365.25 / 2, 365.25 / 3, 365.25 / 4],
        "train_start_date": "1993-01-01",
        "train_end_date": "2015-12-31",
        "val_start_date": "2016-01-01",
        "val_end_date": "2020-01-01",
        "warmup_epochs": 10,
        "lr_scheduler_type": "warmup_cosine",
        "min_lr_ratio": 0.01,
        "optimizer_type": "adamw",
        "gradient_clip_norm": 1.0,
        "patience": 20,
        "criterion_type": "HuberLoss",
        "huber_delta": 1.0,
        "dt": 1.0,
        "tc": 433.0,
        "q": 179.0,
    })
    return SimpleNamespace(**values)


def build_model(config: SimpleNamespace, device: torch.device) -> torch.nn.Module:
    """Build the selected comparison model."""
    model_type = str(config.model_type).lower()
    try:
        model_class = MODEL_CLASSES[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported baseline model: {model_type}") from exc
    return model_class(config).to(device)


def train(config: SimpleNamespace, device: torch.device) -> torch.nn.Module:
    """Train with per-epoch validation and restore the best checkpoint."""
    apply_neural_baseline_defaults(config, model_type=config.model_type)
    set_seed(config.seed)

    train_dataset = Dataset_EOP_ULGap(config, flag="train")
    val_dataset = Dataset_EOP_ULGap(config, flag="val")
    config.time_mark_names = getattr(train_dataset, "time_mark_names", [])
    apply_neural_baseline_defaults(config, model_type=config.model_type)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    print(f"[*] Train windows: {len(train_dataset)}, validation windows: {len(val_dataset)}")

    model = build_model(config, device)
    trainer = NeuralBaselineTrainer(model, train_loader, val_loader, config, device)
    return trainer.fit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_CLASSES), required=True)
    parser.add_argument("--target-space", choices=["pm", "chi"], required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_run_config(args.config, args.model, args.target_space)
    config.root_path = str(args.data_dir)
    device = torch.device(args.device)
    print(f"[*] Device: {device}")
    train(config, device)


if __name__ == "__main__":
    main()
