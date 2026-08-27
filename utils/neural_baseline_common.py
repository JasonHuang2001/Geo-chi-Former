import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils.integrator import WilsonPhysicsLayer, integrate_dataset_chi_with_wilson


AGG_EAM_COLS = [
    "aam_x", "aam_y",
    "oam_x", "oam_y",
    "ham_x", "ham_y",
    "slam_x", "slam_y",
]


def has_config_attr(config, name):
    return hasattr(config, name)


def set_config_default(config, name, value):
    if not has_config_attr(config, name):
        setattr(config, name, value)
    return getattr(config, name)


def clean_name_part(value):
    text = str(value)
    for old, new in [(" ", ""), ("/", "-"), ("\\", "-"), (".", "p")]:
        text = text.replace(old, new)
    return text


def float_name_tag(value):
    return clean_name_part(f"{float(value):g}")


def eam_name_tag(eam_cols):
    cols = list(eam_cols or [])
    if len(cols) == 0:
        return "EAM_none"
    tags = []
    used = set()
    short_names = {"aam": "a", "oam": "o", "ham": "h", "slam": "s", "eam": "e"}
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
    tags.extend(clean_name_part(col) for col in cols if col not in used)
    return "EAM_" + "-".join(tags)


def build_baseline_run_name(config):
    model_type = str(getattr(config, "model_type", "baseline")).lower()
    target_space = str(getattr(config, "target_space", "chi")).lower()
    if target_space not in {"pm", "chi"}:
        raise ValueError(f"target_space must be 'pm' or 'chi', got {target_space}")
    target_tag = "chi2pm" if target_space == "chi" else "pm"
    hidden = int(getattr(config, "hidden_size", getattr(config, "d_model", 64)))
    time_tag = f"doy{int(getattr(config, 'time_feature_dim', 0))}"
    eam_tag = eam_name_tag(getattr(config, "eam_cols", [])) if bool(getattr(config, "use_eam", True)) else "EAM_none"
    revin_tag = "RevIN" if bool(getattr(config, "use_revin", True)) else "noRevIN"
    parts = [
        model_type,
        target_tag,
        f"d{hidden}",
        time_tag,
        eam_tag,
        revin_tag,
    ]
    if model_type == "patchtst":
        parts.append(f"P_{int(getattr(config, 'patch_len', 16))}_{int(getattr(config, 'stride', 8))}")
    parts.extend([
        f"S{int(getattr(config, 'seq_len', 0))}",
        f"H{int(getattr(config, 'pred_len', 0))}",
        f"pm{float_name_tag(getattr(config, 'lambda_pm', 0.0))}",
    ])
    return "_".join(str(part) for part in parts if part is not None and str(part))


def refresh_baseline_output_paths(config, force=False):
    run_name = getattr(config, "run_name", None)
    if not run_name or force:
        run_name = build_baseline_run_name(config)
        config.run_name = run_name
    if force or not getattr(config, "checkpoint_dir", None):
        config.checkpoint_dir = f"./checkpoints/{run_name}"


def apply_neural_baseline_defaults(config, model_type=None):
    if model_type is not None:
        config.model_type = str(model_type).lower()
    config.model_type = str(set_config_default(config, "model_type", "lstm")).lower()
    config.target_space = str(set_config_default(config, "target_space", "chi")).lower()
    if config.target_space not in {"pm", "chi"}:
        raise ValueError(f"target_space must be 'pm' or 'chi', got {config.target_space}")

    config.seq_len = int(set_config_default(config, "seq_len", 720))
    config.pred_len = int(set_config_default(config, "pred_len", 30))
    config.use_ul_gap_loader = bool(set_config_default(config, "use_ul_gap_loader", True))
    config.scale_chi = bool(set_config_default(config, "scale_chi", False))
    config.scale_eam = bool(set_config_default(config, "scale_eam", False))
    config.scale_eam_rad_to_mas = bool(set_config_default(config, "scale_eam_rad_to_mas", True))
    config.use_eam = bool(set_config_default(config, "use_eam", True))
    config.use_time = bool(set_config_default(config, "use_time", True))
    config.eam_cols = list(getattr(config, "eam_cols", AGG_EAM_COLS))
    config.future_eam_cols = list(getattr(config, "future_eam_cols", config.eam_cols))
    config.use_future_observed_eam = False
    config.use_doy_time_mark_features = bool(set_config_default(config, "use_doy_time_mark_features", config.use_time))
    config.use_doy_time_features = bool(set_config_default(config, "use_doy_time_features", config.use_time))
    config.doy_time_periods = list(getattr(config, "doy_time_periods", [365.25, 365.25 / 2, 365.25 / 3, 365.25 / 4]))
    time_dim = len(config.doy_time_periods) * 2 if config.use_time else 0
    config.time_feature_dim = int(getattr(config, "time_feature_dim", time_dim if config.use_time else 0))
    if not config.use_time:
        config.time_feature_dim = 0
        config.use_doy_time_mark_features = False
        config.use_doy_time_features = False

    target_names = ["xpole", "ypole"] if config.target_space == "pm" else ["Chi_x", "Chi_y"]
    config.target = list(getattr(config, "target", target_names))
    config.target_names = list(getattr(config, "target_names", target_names))
    config.observed_names = list(getattr(config, "observed_names", config.eam_cols if config.use_eam else []))
    config.known_names = list(getattr(config, "known_names", []))
    config.output_size = 2
    observed_dim = len(config.eam_cols) if config.use_eam else 0
    config.input_size = 2 + observed_dim + config.time_feature_dim
    config.target_slice = [0, 1]
    config.observed_slice = list(range(2, 2 + observed_dim))
    config.known_slice = list(range(2 + observed_dim, config.input_size))

    config.hidden_size = int(set_config_default(config, "hidden_size", 64))
    config.d_model = int(getattr(config, "d_model", config.hidden_size))
    config.n_heads = int(getattr(config, "n_heads", getattr(config, "nhead", 4)))
    config.nhead = int(getattr(config, "nhead", config.n_heads))
    config.num_layers = int(set_config_default(config, "num_layers", 2))
    config.dropout = float(set_config_default(config, "dropout", 0.2))
    config.use_revin = bool(set_config_default(config, "use_revin", True))
    config.patch_len = int(set_config_default(config, "patch_len", 16))
    config.stride = int(set_config_default(config, "stride", 8))
    config.pos_encoding_type = str(set_config_default(config, "pos_encoding_type", "learnable"))
    config.use_time_mark = False
    config.use_regime = False

    config.ul_gap_window = int(set_config_default(config, "ul_gap_window", 1095))
    config.ul_gap_periods = list(getattr(config, "ul_gap_periods", [365.25, 365.25 / 2]))
    config.precompute_ul_gap = bool(set_config_default(config, "precompute_ul_gap", True))
    config.refresh_ul_gap_cache = bool(set_config_default(config, "refresh_ul_gap_cache", False))
    config.ul_gap_cache_dir = str(set_config_default(config, "ul_gap_cache_dir", "data/.cache/eop_ul_gap"))
    config.verbose_ul_gap_cache = bool(set_config_default(config, "verbose_ul_gap_cache", True))
    config.lambda_pm = float(getattr(config, "lambda_pm", 0.0))
    config.gradient_clip_norm = float(getattr(config, "gradient_clip_norm", 1.0))
    config.warmup_epochs = int(getattr(config, "warmup_epochs", max(1, int(getattr(config, "epochs", 100)) // 10)))
    config.lr_scheduler_type = str(getattr(config, "lr_scheduler_type", "warmup_cosine"))
    config.min_lr_ratio = float(getattr(config, "min_lr_ratio", 0.01))
    refresh_baseline_output_paths(config)
    return config


def set_seed(seed):
    seed = int(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def config_to_dict(config):
    out = {}
    for key in dir(config):
        if key.startswith("__"):
            continue
        value = getattr(config, key)
        if callable(value):
            continue
        if hasattr(value, "tolist"):
            value = value.tolist()
        out[key] = value
    return out


class BaselineEarlyStopping:
    def __init__(self, patience=20, save_path="checkpoint.pth"):
        self.patience = int(patience)
        self.save_path = save_path
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.has_saved = False

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss:
            self.best_loss = float(val_loss)
            self.counter = 0
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            torch.save(model.state_dict(), self.save_path)
            self.has_saved = True
            return
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True


class NeuralBaselineTrainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.epochs = int(getattr(config, "epochs", 100))
        self.warmup_epochs = int(getattr(config, "warmup_epochs", max(1, self.epochs // 10)))
        self.lr_scheduler_type = getattr(config, "lr_scheduler_type", "warmup_cosine")
        self.min_lr_ratio = float(getattr(config, "min_lr_ratio", 0.01))
        self.gradient_clip_norm = float(getattr(config, "gradient_clip_norm", 1.0))
        self.lambda_pm = float(getattr(config, "lambda_pm", 0.0)) if str(getattr(config, "target_space", "chi")) == "chi" else 0.0
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)
        self.physics = self._build_physics() if self.lambda_pm > 0 else None
        self.checkpoint_dir = getattr(config, "checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        model_type = getattr(config, "model_type", "baseline")
        self.model_save_path = os.path.join(self.checkpoint_dir, f"{model_type}_forecasting_best.pth")
        self.early_stopping = BaselineEarlyStopping(
            patience=int(getattr(config, "patience", 20)),
            save_path=self.model_save_path,
        )
        self.train_loss_history = []
        self.val_loss_history = []
        self._save_config()

    def _build_physics(self):
        return WilsonPhysicsLayer(
            dt=float(getattr(self.config, "dt", 1.0)),
            tc=float(getattr(self.config, "tc", 433.0)),
            q=float(getattr(self.config, "q", 170.0)),
        ).to(self.device)

    def _build_criterion(self):
        criterion_type = str(getattr(self.config, "criterion_type", "MSE")).lower()
        huber_delta = float(getattr(self.config, "huber_delta", 1.0))
        if criterion_type in {"mse", "mseloss"}:
            return nn.MSELoss()
        if criterion_type in {"smoothl1loss", "smoothl1"}:
            return nn.SmoothL1Loss(beta=huber_delta)
        if criterion_type in {"huberloss", "huber"}:
            return nn.HuberLoss(delta=huber_delta)
        if criterion_type in {"l1loss", "l1", "mae"}:
            return nn.L1Loss()
        raise ValueError(f"Unsupported criterion_type: {criterion_type}")

    def _build_optimizer(self):
        optimizer_type = str(getattr(self.config, "optimizer_type", "adamw")).lower()
        lr = float(getattr(self.config, "learning_rate", 1e-3))
        weight_decay = float(getattr(self.config, "weight_decay", 0.0))
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if optimizer_type == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if optimizer_type == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        raise ValueError("optimizer_type must be 'adamw' or 'adam'")

    def _lr_lambda(self, epoch):
        scheduler_type = getattr(self, "lr_scheduler_type", "warmup_cosine")
        if scheduler_type == "none":
            scheduler_type = "constant"
        if scheduler_type == "constant":
            return 1.0
        if scheduler_type not in {"warmup_cosine", "cosine"}:
            raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}")
        use_warmup = scheduler_type.startswith("warmup_")
        if use_warmup and self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs
        start_epoch = self.warmup_epochs if use_warmup else 0
        total_anneal_epochs = max(1, self.epochs - start_epoch)
        progress = min(1.0, max(0.0, (epoch - start_epoch) / total_anneal_epochs))
        return max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    def _save_config(self):
        with open(os.path.join(self.checkpoint_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config_to_dict(self.config), f, ensure_ascii=False, indent=2)

    def _build_batch(self, batch):
        return build_model_batch(batch, self.config, self.device)

    def _compute_loss(self, pred, data):
        loss_main = self.criterion(pred, data["target_future"])
        loss_pm = pred.new_tensor(0.0)
        if self.lambda_pm > 0:
            pred_pm, _ = integrate_dataset_chi_with_wilson(self.physics, pred, data["p0"], data["chi0"])
            loss_pm = self.criterion(pred_pm, data["fut_pm"])
        return loss_main + self.lambda_pm * loss_pm, {
            "loss_main": float(loss_main.detach().cpu()),
            "loss_pm": float(loss_pm.detach().cpu()),
        }

    def _run_epoch(self, loader, train):
        self.model.train(train)
        total_loss = 0.0
        total_main = 0.0
        total_pm = 0.0
        count = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                data = self._build_batch(batch)
                pred = forward_baseline_model(self.model, data, self.config)
                loss, components = self._compute_loss(pred, data)
                if train:
                    loss.backward()
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                batch_size = int(data["target_future"].shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_main += components["loss_main"] * batch_size
                total_pm += components["loss_pm"] * batch_size
                count += batch_size
        denom = max(count, 1)
        return {
            "loss": total_loss / denom,
            "loss_main": total_main / denom,
            "loss_pm": total_pm / denom,
        }

    def fit(self):
        print(f"[*] Neural baseline training start: epochs={self.epochs}, checkpoint={self.model_save_path}")
        for epoch in range(self.epochs):
            train_metrics = self._run_epoch(self.train_loader, train=True)
            val_metrics = self._run_epoch(self.val_loader, train=False)
            self.scheduler.step()
            self.train_loss_history.append(train_metrics["loss"])
            self.val_loss_history.append(val_metrics["loss"])
            print(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"train={train_metrics['loss']:.6f} | val={val_metrics['loss']:.6f} | "
                f"main={val_metrics['loss_main']:.6f} | pm={val_metrics['loss_pm']:.6f}"
            )
            self.early_stopping(val_metrics["loss"], self.model)
            if self.early_stopping.early_stop:
                print("[*] Early stopping triggered.")
                break
        if self.early_stopping.has_saved:
            self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        return self.model


def build_model_batch(batch, config, device):
    batch = tuple(item.to(device) for item in batch)
    hist_chi = batch[0] + batch[4]
    fut_chi = batch[2] + batch[5]
    hist_pm = batch[10]
    fut_pm = batch[11]
    if str(getattr(config, "target_space", "chi")) == "pm":
        target_hist = hist_pm
        target_future = fut_pm
    else:
        target_hist = hist_chi
        target_future = fut_chi
    parts = [target_hist]
    if bool(getattr(config, "use_eam", True)):
        parts.append(batch[1])
    if bool(getattr(config, "use_time", True)):
        parts.append(batch[6])
    model_x = torch.cat(parts, dim=-1)
    return {
        "x": model_x,
        "target_future": target_future,
        "hist_pm": hist_pm,
        "fut_pm": fut_pm,
        "hist_chi": hist_chi,
        "fut_chi": fut_chi,
        "p0": batch[8],
        "chi0": batch[9],
    }


def forward_baseline_model(model, data, config):
    return model(data["x"], target_slice=[0, 1])
