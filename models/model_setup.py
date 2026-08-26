"""Construct the published Geo-chi-Former architecture from its frozen config."""

from __future__ import annotations

from models.chiformer import Chiformer


TARGET_INDEX = {
    "Chi_x_no_long": 0,
    "Chi_y_no_long": 1,
    "Chi_x": 0,
    "Chi_y": 1,
}


def apply_checkpoint_defaults(config):
    """Normalize the few derived fields needed by the frozen checkpoint."""
    config.target = list(getattr(config, "target", ["Chi_x_no_long", "Chi_y_no_long"]))
    unknown = [name for name in config.target if name not in TARGET_INDEX]
    if unknown:
        raise ValueError(f"Unsupported checkpoint targets: {unknown}")
    config.ul_gap_target_indices = [TARGET_INDEX[name] for name in config.target]
    config.eam_cols = list(getattr(config, "eam_cols", []))
    config.future_observed_eam_cols = list(
        getattr(config, "future_observed_eam_cols", config.eam_cols)
    )
    config.future_eam_cols = list(getattr(config, "future_eam_cols", config.eam_cols))
    config.metric_horizons = list(getattr(config, "metric_horizons", [1, 7, 10, 14, 30]))
    config.batch_size = int(getattr(config, "batch_size", 16))

    required = (
        "seq_len", "pred_len", "patch_len", "input_size", "output_size",
        "target_slice", "observed_slice", "known_slice", "d_model", "n_heads",
    )
    missing = [name for name in required if not hasattr(config, name)]
    if missing:
        raise ValueError(f"Checkpoint configuration is missing required fields: {missing}")
    if not bool(getattr(config, "use_cross_variate_attention", True)):
        raise ValueError("This checkpoint requires cross-variate attention.")
    return config


def build_paper_model(config, device):
    apply_checkpoint_defaults(config)
    return Chiformer(config).to(device)
