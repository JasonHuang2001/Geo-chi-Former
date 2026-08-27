import math
import os

import torch
import numpy as np
from torch.utils.data import DataLoader

from data.EOP_loader import Dataset_EOP_ULGap
from models.chiformer import Chiformer
from utils.chiformer_trainer import ChiformerTrainer


AGG_EAM_COLS = [
    "aam_x", "aam_y",
    "oam_x", "oam_y",
    "ham_x", "ham_y",
    "slam_x", "slam_y",
]

UL_GAP_TARGET_INDEX = {
    "Chi_x_no_long": 0,
    "Chi_y_no_long": 1,
    "Chi_x": 0,
    "Chi_y": 1,
}


def _has_config_attr(config, name):
    return hasattr(config, name)


def _set_config_default(config, name, value):
    if not _has_config_attr(config, name):
        setattr(config, name, value)
    return getattr(config, name)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Invalid boolean string: {value!r}")
    return bool(value)


def _clean_name_part(value):
    text = str(value)
    for old, new in [
        ("Chi_", "chi"),
        ("_no_long", ""),
        (" ", ""),
        ("/", "-"),
        ("\\", "-"),
        (".", "p"),
    ]:
        text = text.replace(old, new)
    return text


def _target_name_tag(target):
    targets = list(target)
    cleaned = [_clean_name_part(name) for name in targets]
    if cleaned == ["chix", "chiy"]:
        return ""
    return "-".join(cleaned) if cleaned else "target"


def _float_name_tag(value):
    return _clean_name_part(f"{float(value):g}")


def _eam_name_tag(eam_cols):
    cols = list(eam_cols or [])
    if len(cols) == 0:
        return "EAM_none"

    tags = []
    used = set()
    short_names = {
        "aam": "a",
        "oam": "o",
        "ham": "h",
        "slam": "s",
        "eam": "e",
    }
    for base in ["aam", "oam", "ham", "slam", "eam"]:
        x_col = f"{base}_x"
        y_col = f"{base}_y"
        has_x = x_col in cols
        has_y = y_col in cols
        short = short_names[base]
        if has_x and has_y:
            tags.append(short)
            used.update([x_col, y_col])
        elif has_x:
            tags.append(f"{short}x")
            used.add(x_col)
        elif has_y:
            tags.append(f"{short}y")
            used.add(y_col)

    extras = [_clean_name_part(col) for col in cols if col not in used]
    tags.extend(extras)
    return "EAM_" + "-".join(tags)


def _curriculum_name_tag(config):
    if not _as_bool(getattr(config, "use_horizon_curriculum", False)):
        return None
    horizons = [int(h) for h in getattr(config, "curriculum_horizons", [])]
    if len(horizons) == 0:
        horizons = [int(getattr(config, "pred_len", 0))]
    return "curH" + "-".join(str(h) for h in horizons)


def build_chiformer_run_name(config):
    target_tag = _target_name_tag(getattr(config, "target", ["Chi_x_no_long", "Chi_y_no_long"]))
    eam_tag = _eam_name_tag(getattr(config, "eam_cols", AGG_EAM_COLS))
    hidden = int(getattr(config, "d_model", getattr(config, "hidden_size", 128)))
    time_tag = f"doy{int(getattr(config, 'time_feature_dim', 0))}"
    fut_obs_tag = "fObsEAM" if _as_bool(getattr(config, "use_future_observed_eam", False)) else "fObs0"
    fut_query_tag = "fEAMQ" if _as_bool(getattr(config, "use_future_eam_query", False)) else "fEAMQ0"
    cv_tag = None if _as_bool(getattr(config, "use_cross_variate_attention", True)) else "noCVA"
    rope_tag = None if _as_bool(getattr(config, "use_rope", True)) else "noRoPE"
    norm_tag = "TIN" if _as_bool(getattr(config, "use_target_instant_norm", True)) else "noTIN"
    var_tag = "varEmb" if _as_bool(getattr(config, "use_variate_embedding", False)) else "noVarEmb"
    query_tag = "tgtQ" if _as_bool(getattr(config, "use_target_query_embedding", False)) else "noTgtQ"
    axis_tag = "axisBias" if _as_bool(getattr(config, "use_axis_aware_bias", False)) else "noAxisBias"
    kv_shift_tag = f"KV_sh_{int(_as_bool(getattr(config, 'use_kv_shift', False)))}"
    curriculum_tag = _curriculum_name_tag(config)
    pm_tag = _float_name_tag(getattr(config, "lambda_pm", 0.0))
    parts = [
        getattr(config, "model_type", "chiformer"),
        getattr(config, "train_mode", "target_only"),
        target_tag,
        f"d{hidden}",
        time_tag,
        eam_tag,
        fut_obs_tag,
        fut_query_tag,
        cv_tag,
        rope_tag,
        norm_tag,
        var_tag,
        query_tag,
        axis_tag,
        kv_shift_tag,
        curriculum_tag,
        f"P_{int(getattr(config, 'patch_len', 14))}_{int(getattr(config, 'seq_len', 0))}",
        f"{int(getattr(config, 'pred_len', 14))}",
        f"fi_{int(getattr(config, 'chi_filter_window', 0))}",
        f"pm_{pm_tag}",
    ]
    return "_".join(str(part) for part in parts if part is not None and str(part))


class Configs:
    seed = 0

    # Data.
    root_path = "data/"
    data_path = "eop_data_xy_EAM.csv"
    seq_len = 360
    pred_len = 30
    use_ul_gap_loader = True
    features = ["Chi_x_no_long", "Chi_y_no_long"] + AGG_EAM_COLS
    target = ["Chi_x_no_long", "Chi_y_no_long"]
    eam_cols = AGG_EAM_COLS
    scale = False
    scale_chi = False
    scale_eam = False
    unit = "mas"
    ul_gap_window = 1095
    ul_gap_periods = [365.25, 365.25 / 2]
    ul_gap_huber_epsilon = 1.35
    ul_gap_min_fit_samples = 30
    precompute_ul_gap = True
    refresh_ul_gap_cache = False
    ul_gap_cache_dir = "data/.cache/eop_ul_gap"
    verbose_ul_gap_cache = True

    # Known calendar features.
    use_doy_time_features = True
    use_doy_time_mark_features = True
    doy_time_periods = [365.25, 365.25 / 2, 365.25 / 3, 365.25 / 4]
    time_feature_dim = len(doy_time_periods) * 2

    # Date split and optional filtering.
    apply_chi_filter = False
    chi_filter_window = 0
    chi_filter_method = "pm_causal_phase_corrected"
    chi_filter_phase_shift = chi_filter_window // 2
    train_start_date = "1993-01-01"
    train_end_date = "2015-12-31"
    val_start_date = "2016-01-01"
    val_end_date = "2020-01-01"
    # Optional forecast-side EAM.
    use_future_observed_eam = True
    future_observed_eam_cols = AGG_EAM_COLS
    use_future_eam_query = False
    future_eam_cols = AGG_EAM_COLS
    future_eam_forecast_path = "eam14forecast_daily.csv"
    future_eam_forecast_start_date = "2021-05-20"
    future_eam_available_len = 14
    # Component-wise calibration from 2021-2025 ETH/GFZ EAM forecast-analysis overlap.
    future_eam_noise_ratio = {
        "aam_x": 0.51676048,
        "aam_y": 0.35413006,
        "oam_x": 0.55954207,
        "oam_y": 0.53790316,
        "ham_x": 0.03001029,
        "ham_y": 0.04958461,
        "slam_x": 0.19647301,
        "slam_y": 0.21684768,
    }
    future_eam_noise_growth = {
        "aam_x": 0.15626575,
        "aam_y": 0.16135400,
        "oam_x": 0.08276080,
        "oam_y": 0.08686451,
        "ham_x": 0.28409593,
        "ham_y": 0.25651700,
        "slam_x": 0.34870299,
        "slam_y": 0.24664223,
    }
    future_eam_noise_seed = seed
    future_eam_abs_threshold = 1e-3

    # Model.
    model_type = "chiformer"
    patch_len = 8
    input_size = len(features) + time_feature_dim
    output_size = 2
    target_slice = [0, 1]
    observed_slice = list(range(2, 2 + len(AGG_EAM_COLS)))
    known_slice = list(range(2 + len(AGG_EAM_COLS), input_size))
    target_names = target
    observed_names = AGG_EAM_COLS
    known_names = []
    hidden_size = 128
    d_model = hidden_size
    n_heads = 4
    nhead = n_heads
    num_layers = 2
    d_ff = None
    dropout = 0.2
    smoothing_alpha = 1.0
    # KV shift for future-known information: K_i uses known_i, V_i uses known_{i+1}.
    use_kv_shift = True
    use_cross_variate_attention = True
    use_rope = True
    rope_base = 10000.0
    max_position_tokens = math.ceil((seq_len + pred_len) / patch_len) + 8
    use_target_query_embedding = True
    use_axis_aware_bias = True
    axis_same_bias = 0.2
    axis_cross_bias = -0.2
    axis_target_other_bias = 0.0
    axis_known_bias = 0.0
    use_target_instant_norm = True
    target_instant_norm_eps = 1e-6
    return_attn = False
    return_attn_detail = False

    # Optimization.
    train_mode = "joint"
    epochs = 150
    warmup_epochs = 10
    lr_scheduler_type = "warmup_cosine"
    min_lr_ratio = 0.01
    batch_size = 32 * 4
    optimizer_type = "adamw"
    learning_rate = 1e-4
    weight_decay = 1e-3
    gradient_clip_norm = 1.0
    patience = 20
    criterion_type = "HuberLoss"
    huber_delta = 1.0
    lambda_pm = 7.0

    # Expand the active loss horizon during training.
    use_horizon_curriculum = True
    curriculum_horizons = [patch_len, patch_len*2, patch_len*3, pred_len]
    curriculum_milestones = [0.2, 0.4, 0.7, 1.0]
    curriculum_apply_to_val = False

    dt = 1.0
    tc = 433.0
    q = 179.0

    # Output paths are derived from the configuration when left unset.
    run_name = None
    checkpoint_dir = None
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def refresh_output_paths(config, force=False):
    run_name = getattr(config, "run_name", None)
    if not run_name or force:
        run_name = build_chiformer_run_name(config)
        config.run_name = run_name
    if force or not getattr(config, "checkpoint_dir", None):
        config.checkpoint_dir = f"./checkpoints/{run_name}"


def apply_chiformer_defaults(config):
    _set_config_default(config, "model_type", "chiformer")
    config.patch_len = int(_set_config_default(config, "patch_len", 14))
    config.pred_len = int(_set_config_default(config, "pred_len", 14))
    config.use_ul_gap_loader = _as_bool(_set_config_default(config, "use_ul_gap_loader", True))
    config.scale_chi = _as_bool(_set_config_default(config, "scale_chi", False))
    config.scale_eam = _as_bool(_set_config_default(config, "scale_eam", False))
    config.target = list(getattr(config, "target", ["Chi_x_no_long", "Chi_y_no_long"]))
    unknown_targets = [name for name in config.target if name not in UL_GAP_TARGET_INDEX]
    if unknown_targets:
        raise ValueError(f"Unsupported Geo-chi-Former targets: {unknown_targets}")
    config.ul_gap_target_indices = [UL_GAP_TARGET_INDEX[name] for name in config.target]
    config.eam_cols = list(getattr(config, "eam_cols", AGG_EAM_COLS))
    if not _has_config_attr(config, "features"):
        config.features = list(config.target) + list(config.eam_cols)
    if not _has_config_attr(config, "future_observed_eam_cols"):
        config.future_observed_eam_cols = list(config.eam_cols)
    else:
        config.future_observed_eam_cols = list(config.future_observed_eam_cols)
    if not _has_config_attr(config, "future_eam_cols"):
        config.future_eam_cols = list(config.future_observed_eam_cols)
    else:
        config.future_eam_cols = list(config.future_eam_cols)
    if not _has_config_attr(config, "use_future_eam_query"):
        config.use_future_eam_query = _as_bool(getattr(config, "use_future_observed_eam", False))
    else:
        config.use_future_eam_query = _as_bool(config.use_future_eam_query)
    config.use_doy_time_mark_features = _as_bool(_set_config_default(config, "use_doy_time_mark_features", True))
    config.use_doy_time_features = _as_bool(_set_config_default(config, "use_doy_time_features", True))
    config.lambda_pm = float(getattr(config, "lambda_pm", 0.0))
    config.use_target_instant_norm = _as_bool(getattr(config, "use_target_instant_norm", True))
    config.target_instant_norm_eps = float(getattr(config, "target_instant_norm_eps", 1e-6))
    config.use_variate_embedding = _as_bool(getattr(config, "use_variate_embedding", True))
    config.use_target_query_embedding = _as_bool(getattr(config, "use_target_query_embedding", True))
    config.use_axis_aware_bias = _as_bool(getattr(config, "use_axis_aware_bias", True))
    config.use_cross_variate_attention = _as_bool(getattr(config, "use_cross_variate_attention", True))
    config.use_rope = _as_bool(getattr(config, "use_rope", True))
    default_max_positions = math.ceil((int(config.seq_len) + int(config.pred_len)) / int(config.patch_len)) + 8
    config.max_position_tokens = int(getattr(config, "max_position_tokens", default_max_positions))
    config.use_kv_shift = _as_bool(getattr(config, "use_kv_shift", True))
    config.axis_same_bias = float(getattr(config, "axis_same_bias", 0.2))
    config.axis_cross_bias = float(getattr(config, "axis_cross_bias", -0.1))
    config.axis_target_other_bias = float(getattr(config, "axis_target_other_bias", 0.0))
    config.axis_known_bias = float(getattr(config, "axis_known_bias", 0.0))
    config.ul_gap_window = int(getattr(config, "ul_gap_window", 1095))
    config.ul_gap_periods = list(getattr(config, "ul_gap_periods", [365.25, 365.25 / 2]))
    config.precompute_ul_gap = _as_bool(getattr(config, "precompute_ul_gap", True))
    config.refresh_ul_gap_cache = _as_bool(getattr(config, "refresh_ul_gap_cache", False))
    config.ul_gap_cache_dir = str(getattr(config, "ul_gap_cache_dir", "data/.cache/eop_ul_gap"))
    config.use_horizon_curriculum = _as_bool(getattr(config, "use_horizon_curriculum", False))
    config.curriculum_horizons = [int(h) for h in getattr(config, "curriculum_horizons", [config.pred_len])]
    config.curriculum_milestones = [float(m) for m in getattr(config, "curriculum_milestones", [1.0])]
    config.curriculum_apply_to_val = _as_bool(getattr(config, "curriculum_apply_to_val", False))

    time_dim = int(getattr(config, "time_feature_dim", len(getattr(config, "doy_time_periods", [])) * 2))
    observed_dim = len(config.eam_cols)
    if not _has_config_attr(config, "input_size"):
        config.input_size = len(config.target) + observed_dim + time_dim
    if hasattr(config, "output_size") and int(getattr(config, "output_size")) != len(config.target):
        raise ValueError(
            f"output_size={getattr(config, 'output_size')} does not match {len(config.target)} targets"
        )
    if not _has_config_attr(config, "output_size"):
        config.output_size = len(config.target)
    target_dim = len(config.target)
    if not _has_config_attr(config, "target_slice"):
        config.target_slice = list(range(target_dim))
    if not _has_config_attr(config, "observed_slice"):
        config.observed_slice = list(range(target_dim, target_dim + observed_dim))
    if not _has_config_attr(config, "known_slice"):
        config.known_slice = list(range(target_dim + observed_dim, config.input_size))
    if not _has_config_attr(config, "target_names"):
        config.target_names = list(config.target)
    else:
        config.target_names = list(config.target_names)
    if not _has_config_attr(config, "observed_names"):
        config.observed_names = list(config.eam_cols)
    else:
        config.observed_names = list(config.observed_names)
    if not _has_config_attr(config, "known_names"):
        config.known_names = list(getattr(config, "time_mark_names", []))
    else:
        config.known_names = list(config.known_names)
    if not _has_config_attr(config, "n_heads"):
        config.n_heads = int(getattr(config, "nhead", 4))
    if not _has_config_attr(config, "nhead"):
        config.nhead = int(config.n_heads)
    if not _has_config_attr(config, "d_model"):
        config.d_model = int(getattr(config, "hidden_size", 128))
    if not _has_config_attr(config, "hidden_size"):
        config.hidden_size = int(config.d_model)
    refresh_output_paths(config)


def build_model(config, device):
    return Chiformer(config).to(device)


def main():
    config = Configs()
    apply_chiformer_defaults(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Device: {device}")
    set_seed(config.seed)

    print("[*] Loading Dataset_EOP_ULGap...")
    train_dataset = Dataset_EOP_ULGap(config, flag="train")
    val_dataset = Dataset_EOP_ULGap(config, flag="val")
    if not _has_config_attr(config, "time_feature_dim"):
        config.time_feature_dim = train_dataset.time_marks.shape[-1]
    if not _has_config_attr(config, "time_mark_names"):
        config.time_mark_names = getattr(train_dataset, "time_mark_names", [])
    if not _has_config_attr(config, "observed_feature_indices"):
        config.observed_feature_indices = list(range(2, 2 + len(config.eam_cols)))
    apply_chiformer_defaults(config)
    if not _has_config_attr(config, "observed_feature_indices"):
        config.observed_feature_indices = list(range(2, 2 + len(config.eam_cols)))

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False)
    print(f" -> Train windows: {len(train_dataset)}, validation windows: {len(val_dataset)}")

    print("[*] Initializing Geo-chi-Former...")
    model = build_model(config, device)
    cross_keys = sum(1 for key in model.state_dict().keys() if "cross_variate" in key)
    print(
        f" -> Model: {model.__class__.__name__}, "
        f"use_cross_variate_attention={_as_bool(getattr(config, 'use_cross_variate_attention', True))}, "
        f"cross_variate_keys={cross_keys}"
    )
    print(f"[*] Training; checkpoint_dir={config.checkpoint_dir}")
    trainer = ChiformerTrainer(model, train_loader, val_loader, config, device)
    trainer.fit()


if __name__ == "__main__":
    main()
