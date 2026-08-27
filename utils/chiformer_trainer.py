import csv
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from utils.integrator import WilsonPhysicsLayer, integrate_dataset_chi_with_wilson
except ImportError:  # pragma: no cover - allow direct execution from utils
    from integrator import WilsonPhysicsLayer, integrate_dataset_chi_with_wilson


class ChiformerEarlyStopping:
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


class ChiformerTrainer:
    """Train Geo-chi-Former and select the best validation checkpoint.

    Dataset batches are mapped to historical chi targets, observed EAM,
    known calendar features, and future chi targets.
    """

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
        self.lambda_pm = float(getattr(config, "lambda_pm", 0.0))
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)
        self.physics = self._build_physics() if self.lambda_pm > 0 else None

        self.checkpoint_dir = getattr(config, "checkpoint_dir", "./checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        model_type = getattr(config, "model_type", "chiformer")
        self.model_save_path = os.path.join(self.checkpoint_dir, f"{model_type}_forecasting_best.pth")
        self.early_stopping = ChiformerEarlyStopping(
            patience=int(getattr(config, "patience", 20)),
            save_path=self.model_save_path,
        )
        self.train_loss_history = []
        self.val_loss_history = []
        self.training_log_rows = []
        self._save_config()

    def _build_physics(self):
        dt = float(getattr(self.config, "dt", 1.0))
        tc = float(getattr(self.config, "tc", 430.21))
        q = float(getattr(self.config, "q", 170.0))
        return WilsonPhysicsLayer(dt=dt, tc=tc, q=q).to(self.device)

    def _build_criterion(self):
        criterion_type = getattr(self.config, "criterion_type", "MSE")
        huber_delta = float(getattr(self.config, "huber_delta", 1.0))
        if criterion_type == "MSE":
            return nn.MSELoss()
        if criterion_type == "SmoothL1Loss":
            return nn.SmoothL1Loss(beta=huber_delta)
        if criterion_type == "HuberLoss":
            return nn.HuberLoss(delta=huber_delta)
        if criterion_type == "L1Loss":
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
        valid = {"warmup_cosine", "cosine", "constant"}
        if scheduler_type not in valid:
            raise ValueError(f"Unsupported lr_scheduler_type: {scheduler_type}; choose from {sorted(valid)}")

        if scheduler_type == "constant":
            return 1.0
        use_warmup = scheduler_type.startswith("warmup_")
        if use_warmup and self.warmup_epochs > 0 and epoch < self.warmup_epochs:
            return (epoch + 1) / self.warmup_epochs

        start_epoch = self.warmup_epochs if use_warmup else 0
        total_anneal_epochs = max(1, self.epochs - start_epoch)
        progress = min(1.0, max(0.0, (epoch - start_epoch) / total_anneal_epochs))
        return max(self.min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    @staticmethod
    def _config_to_dict(config):
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

    def _save_config(self):
        config_path = os.path.join(self.checkpoint_dir, "config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._config_to_dict(self.config), f, ensure_ascii=False, indent=2)

    def _is_ul_gap_batch(self, batch, dataset):
        if bool(getattr(self.config, "use_ul_gap_loader", False)):
            return True
        return len(batch) >= 12 and hasattr(dataset, "eam_cols") and hasattr(dataset, "chi_cols")

    def _build_ul_gap_batch(self, batch):
        (
            hist_chi_no_long,
            hist_eam,
            fut_chi_no_long,
            fut_eam,
            _hist_ul_gap,
            fut_ul_gap,
            hist_time_mark,
            fut_time_mark,
            p0_raw,
            chi0_raw,
            _hist_pm_raw,
            fut_pm_raw,
        ) = batch[:12]

        hist_chi_no_long = hist_chi_no_long.to(self.device)
        hist_eam = hist_eam.to(self.device)
        fut_chi_no_long = fut_chi_no_long.to(self.device)
        fut_eam = fut_eam.to(self.device)
        fut_ul_gap = fut_ul_gap.to(self.device)
        hist_time_mark = hist_time_mark.to(self.device)
        fut_time_mark = fut_time_mark.to(self.device)
        p0_raw = p0_raw.to(self.device)
        chi0_raw = chi0_raw.to(self.device)
        fut_pm_raw = fut_pm_raw.to(self.device)

        target_indices = list(getattr(self.config, "ul_gap_target_indices", [0, 1]))
        hist_chi_no_long = hist_chi_no_long[:, :, target_indices]
        fut_chi_no_long = fut_chi_no_long[:, :, target_indices]
        fut_ul_gap = fut_ul_gap[:, :, target_indices]

        model_x = torch.cat([hist_chi_no_long, hist_eam, hist_time_mark], dim=-1)
        future_known = fut_time_mark
        future_observed = None
        if bool(getattr(self.config, "use_future_observed_eam", False)):
            future_observed = fut_eam

        aux = {
            "fut_ul_gap": fut_ul_gap,
            "target_raw_chi": fut_chi_no_long + fut_ul_gap,
            "p0": p0_raw,
            "chi0": chi0_raw,
            "fut_pm": fut_pm_raw,
        }
        return model_x, future_known, future_observed, fut_chi_no_long, aux

    def _build_model_batch(self, batch, dataset):
        if self._is_ul_gap_batch(batch, dataset):
            return self._build_ul_gap_batch(batch)

        batch_x, _p0, _chi0, batch_x_mark, batch_y, batch_y_mark = batch[:6]
        batch_fut_x = batch[8] if len(batch) > 8 else None

        batch_x = batch_x.to(self.device)
        batch_x_mark = batch_x_mark.to(self.device)
        batch_y = batch_y.to(self.device)
        batch_y_mark = batch_y_mark.to(self.device)
        if batch_fut_x is not None:
            batch_fut_x = batch_fut_x.to(self.device)

        target_indices = list(getattr(dataset, "target_indices", getattr(self.config, "target_slice", [0, 1])))
        observed_cols = list(getattr(self.config, "observed_feature_indices", []))
        if len(observed_cols) == 0:
            observed_cols = [idx for idx in range(batch_x.shape[-1]) if idx not in target_indices]

        hist_target = batch_x[:, :, target_indices]
        hist_observed = batch_x[:, :, observed_cols] if len(observed_cols) > 0 else batch_x[:, :, :0]
        model_x = torch.cat([hist_target, hist_observed, batch_x_mark], dim=-1)
        future_known = batch_y_mark

        future_observed = None
        if bool(getattr(self.config, "use_future_observed_eam", False)):
            if batch_fut_x is None:
                raise ValueError("Dataset_EOP must return future EAM at batch[8] when use_future_observed_eam=True")
            if batch_fut_x.shape[-1] != hist_observed.shape[-1]:
                raise ValueError(
                    f"future-observed EAM must have {hist_observed.shape[-1]} channels, "
                    f"got {batch_fut_x.shape[-1]}"
                )
            future_observed = batch_fut_x

        aux = {
            "fut_ul_gap": None,
            "target_raw_chi": batch_y,
            "p0": None,
            "chi0": None,
            "fut_pm": None,
        }
        return model_x, future_known, future_observed, batch_y, aux

    @staticmethod
    def _extract_prediction(outputs):
        if isinstance(outputs, dict):
            return outputs["pred"]
        return outputs

    def _loss_horizon_for_epoch(self, epoch=None, train=True):
        pred_len = int(getattr(self.config, "pred_len", 0))
        if pred_len <= 0:
            return None
        if not bool(getattr(self.config, "use_horizon_curriculum", False)):
            return pred_len
        if not train and not bool(getattr(self.config, "curriculum_apply_to_val", False)):
            return pred_len

        horizons = [int(h) for h in getattr(self.config, "curriculum_horizons", [pred_len])]
        milestones = [float(m) for m in getattr(self.config, "curriculum_milestones", [1.0])]
        if len(horizons) != len(milestones):
            raise ValueError(
                "curriculum_horizons and curriculum_milestones must have equal length; "
                f"got {len(horizons)} and {len(milestones)}"
            )
        if len(horizons) == 0:
            return pred_len
        if any(h <= 0 for h in horizons):
            raise ValueError(f"curriculum_horizons must contain positive integers: {horizons}")
        if any(m <= 0 or m > 1 for m in milestones):
            raise ValueError(f"curriculum_milestones must lie in (0, 1]: {milestones}")
        if any(milestones[i] < milestones[i - 1] for i in range(1, len(milestones))):
            raise ValueError(f"curriculum_milestones must be nondecreasing: {milestones}")

        if epoch is None:
            epoch = self.epochs - 1
        progress = (int(epoch) + 1) / max(int(self.epochs), 1)
        active_horizon = horizons[-1]
        for horizon, milestone in zip(horizons, milestones):
            if progress <= milestone:
                active_horizon = horizon
                break
        return max(1, min(int(active_horizon), pred_len))

    @staticmethod
    def _slice_aux_for_horizon(aux, active_horizon):
        if aux is None or active_horizon is None:
            return aux
        out = dict(aux)
        for key in ["fut_ul_gap", "target_raw_chi", "fut_pm"]:
            value = out.get(key)
            if value is not None and hasattr(value, "dim") and value.dim() >= 2:
                out[key] = value[:, :active_horizon]
        return out

    def _compute_loss(self, pred_no_long, target_no_long, aux, active_horizon=None):
        if active_horizon is not None:
            active_horizon = max(1, min(int(active_horizon), pred_no_long.size(1), target_no_long.size(1)))
            pred_no_long = pred_no_long[:, :active_horizon]
            target_no_long = target_no_long[:, :active_horizon]
            aux = self._slice_aux_for_horizon(aux, active_horizon)
        else:
            active_horizon = int(pred_no_long.size(1))

        fut_ul_gap = aux.get("fut_ul_gap") if aux is not None else None
        if fut_ul_gap is not None:
            pred_raw_chi = pred_no_long + fut_ul_gap
            target_raw_chi = aux.get("target_raw_chi", target_no_long + fut_ul_gap)
        else:
            pred_raw_chi = pred_no_long
            target_raw_chi = aux.get("target_raw_chi", target_no_long) if aux is not None else target_no_long

        loss_chi = self.criterion(pred_raw_chi, target_raw_chi)
        loss_pm = pred_raw_chi.new_tensor(0.0)
        if self.lambda_pm > 0:
            if pred_raw_chi.size(-1) != 2:
                raise ValueError(
                    "PM loss requires raw chi with two channels [Chi_x, Chi_y]; "
                    f"got {pred_raw_chi.size(-1)}. Set lambda_pm=0 or train both channels."
                )
            if self.physics is None:
                self.physics = self._build_physics()
            required = ("p0", "chi0", "fut_pm")
            missing = [key for key in required if aux is None or aux.get(key) is None]
            if missing:
                raise ValueError(f"lambda_pm>0 requires UL-gap batch fields: {missing}")
            pm_pred, _ = integrate_dataset_chi_with_wilson(
                self.physics,
                pred_raw_chi,
                aux["p0"],
                aux["chi0"],
            )
            loss_pm = self.criterion(pm_pred, aux["fut_pm"])

        loss = loss_chi + self.lambda_pm * loss_pm
        components = {
            "loss_chi": float(loss_chi.detach().cpu()),
            "loss_pm": float(loss_pm.detach().cpu()),
            "active_horizon": float(active_horizon),
        }
        return loss, components

    def _run_epoch(self, loader, train, epoch=None):
        self.model.train(train)
        total_loss = 0.0
        total_loss_chi = 0.0
        total_loss_pm = 0.0
        total_horizon = 0.0
        count = 0
        active_horizon = self._loss_horizon_for_epoch(epoch=epoch, train=train)
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for batch in loader:
                model_x, future_known, future_observed, target, aux = self._build_model_batch(batch, loader.dataset)
                if train:
                    self.optimizer.zero_grad()
                outputs = self.model(
                    model_x,
                    future_known=future_known,
                    future_observed=future_observed,
                )
                pred = self._extract_prediction(outputs)
                loss, components = self._compute_loss(pred, target, aux, active_horizon=active_horizon)
                if train:
                    loss.backward()
                    if self.gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
                    self.optimizer.step()
                batch_size = int(target.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_loss_chi += components["loss_chi"] * batch_size
                total_loss_pm += components["loss_pm"] * batch_size
                total_horizon += components["active_horizon"] * batch_size
                count += batch_size
        denom = max(count, 1)
        return {
            "loss": total_loss / denom,
            "loss_chi": total_loss_chi / denom,
            "loss_pm": total_loss_pm / denom,
            "active_horizon": total_horizon / denom,
        }

    def _save_training_log(self):
        if not self.training_log_rows:
            return
        log_path = os.path.join(self.checkpoint_dir, "training_log.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.training_log_rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.training_log_rows)

    def fit(self):
        print("\n=== Geo-chi-Former training ===")
        start_time = time.time()
        for epoch in range(self.epochs):
            epoch_start = time.time()
            train_metrics = self._run_epoch(self.train_loader, train=True, epoch=epoch)
            val_metrics = self._run_epoch(self.val_loader, train=False, epoch=epoch)
            train_loss = train_metrics["loss"]
            val_loss = val_metrics["loss"]
            lr = self.optimizer.param_groups[0]["lr"]
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.training_log_rows.append({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_loss_chi": train_metrics["loss_chi"],
                "val_loss_chi": val_metrics["loss_chi"],
                "train_loss_pm": train_metrics["loss_pm"],
                "val_loss_pm": val_metrics["loss_pm"],
                "train_active_horizon": train_metrics["active_horizon"],
                "val_active_horizon": val_metrics["active_horizon"],
                "lr": lr,
            })
            print(
                f"Epoch {epoch + 1:03d} | Train: {train_loss:.6f} | "
                f"Val: {val_loss:.6f} | Chi: {val_metrics['loss_chi']:.6f} | "
                f"PM: {val_metrics['loss_pm']:.6f} | H: {train_metrics['active_horizon']:.0f} | LR: {lr:.6g} | "
                f"Time: {time.time() - epoch_start:.2f}s"
            )
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("-> Early stopping triggered.")
                break
            self.scheduler.step()

        if not self.early_stopping.has_saved:
            torch.save(self.model.state_dict(), self.model_save_path)
            self.early_stopping.has_saved = True
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "best_model.pth"))
        self._save_training_log()
        print(f"=== Training complete | {(time.time() - start_time) / 60:.2f} min ===")
        print(f"[*] Best model: {self.model_save_path}")
        return self.model
