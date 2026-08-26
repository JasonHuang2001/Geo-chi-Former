"""Checkpoint inference and numeric output."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.integrator import WilsonPhysicsLayer, integrate_dataset_chi_with_wilson


def _build_batch(batch, config, device):
    (
        hist_chi, hist_eam, future_chi, future_eam, hist_gap, future_gap,
        hist_time, future_time, pole_initial, chi_initial, hist_pm, future_pm,
    ) = batch[:12]
    target_indices = list(getattr(config, "ul_gap_target_indices", [0, 1]))
    hist_chi = hist_chi.to(device)[:, :, target_indices]
    future_chi = future_chi.to(device)[:, :, target_indices]
    future_gap = future_gap.to(device)[:, :, target_indices]
    model_input = torch.cat([hist_chi, hist_eam.to(device), hist_time.to(device)], dim=-1)
    future_observed = future_eam.to(device) if bool(getattr(config, "use_future_observed_eam", False)) else None
    auxiliary = {
        "hist_raw_chi": hist_chi + hist_gap.to(device)[:, :, target_indices],
        "future_gap": future_gap,
        "pole_initial": pole_initial.to(device),
        "chi_initial": chi_initial.to(device),
        "hist_pm": hist_pm.to(device),
        "future_pm": future_pm.to(device),
    }
    return model_input, future_time.to(device), future_observed, future_chi, auxiliary


def _metrics(prediction, truth, history_last):
    prediction = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    history_last = np.asarray(history_last, dtype=np.float64)
    error = prediction - truth
    result = {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
    }
    correlations = []
    mase = []
    for index in range(prediction.shape[-1]):
        predicted = prediction[:, :, index].reshape(-1)
        observed = truth[:, :, index].reshape(-1)
        if np.std(predicted) > 1e-8 and np.std(observed) > 1e-8:
            correlations.append(float(np.corrcoef(predicted, observed)[0, 1]))
        naive_error = np.mean(np.abs(truth[:, :, index] - history_last[:, index:index + 1]))
        if naive_error > 1e-8:
            mase.append(float(np.mean(np.abs(error[:, :, index])) / naive_error))
    result["CC"] = float(np.mean(correlations)) if correlations else 0.0
    result["MASE"] = float(np.mean(mase)) if mase else float("nan")
    return result


def _metric_report(pred_chi, true_chi, hist_chi, pred_pm, true_pm, hist_pm, horizons):
    report = {
        "raw_chi_overall": _metrics(pred_chi, true_chi, hist_chi),
        "pm_overall": _metrics(pred_pm, true_pm, hist_pm),
        "horizons": {},
    }
    for horizon in sorted({int(value) for value in horizons if 0 < int(value) <= pred_chi.shape[1]}):
        report["horizons"][str(horizon)] = {
            "raw_chi": _metrics(pred_chi[:, :horizon], true_chi[:, :horizon], hist_chi),
            "pm": _metrics(pred_pm[:, :horizon], true_pm[:, :horizon], hist_pm),
        }
    return report


def evaluate_model_on_dataset(model, dataset, config, device, mode_name, save_dir):
    """Run checkpoint inference and write predictions and metrics."""
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=int(config.batch_size), shuffle=False, drop_last=False)
    physics = WilsonPhysicsLayer(
        dt=float(getattr(config, "dt", 1.0)),
        tc=float(getattr(config, "tc", 433.0)),
        q=float(getattr(config, "q", 179.0)),
    ).to(device)
    model.eval()

    pred_chi_parts, true_chi_parts, hist_chi_parts = [], [], []
    pred_pm_parts, true_pm_parts, hist_pm_parts = [], [], []
    rows = []
    sample_offset = 0
    with torch.no_grad():
        for batch in loader:
            model_input, future_known, future_observed, target_short, aux = _build_batch(batch, config, device)
            output = model(model_input, future_known=future_known, future_observed=future_observed)
            predicted_short = output["pred"] if isinstance(output, dict) else output
            predicted_chi = predicted_short + aux["future_gap"]
            true_chi = target_short + aux["future_gap"]
            predicted_pm, _ = integrate_dataset_chi_with_wilson(
                physics, predicted_chi, aux["pole_initial"], aux["chi_initial"]
            )

            arrays = [value.detach().cpu().numpy() for value in (
                predicted_chi, true_chi, aux["hist_raw_chi"], predicted_pm,
                aux["future_pm"], aux["hist_pm"],
            )]
            pred_chi, true_chi_np, hist_chi, pred_pm, true_pm, hist_pm = arrays
            pred_chi_parts.append(pred_chi)
            true_chi_parts.append(true_chi_np)
            hist_chi_parts.append(hist_chi[:, -1, :])
            pred_pm_parts.append(pred_pm)
            true_pm_parts.append(true_pm)
            hist_pm_parts.append(hist_pm[:, -1, :])

            for local_index in range(pred_chi.shape[0]):
                sample_index = sample_offset + local_index
                dates = dataset.sample_future_dates(sample_index)
                for lead, date in enumerate(dates):
                    rows.append({
                        "sample_index": sample_index, "lead": lead, "date": str(date)[:10],
                        "pred_raw_chi_x": float(pred_chi[local_index, lead, 0]),
                        "pred_raw_chi_y": float(pred_chi[local_index, lead, 1]),
                        "true_raw_chi_x": float(true_chi_np[local_index, lead, 0]),
                        "true_raw_chi_y": float(true_chi_np[local_index, lead, 1]),
                        "pred_xpole": float(pred_pm[local_index, lead, 0]),
                        "pred_ypole": float(pred_pm[local_index, lead, 1]),
                        "true_xpole": float(true_pm[local_index, lead, 0]),
                        "true_ypole": float(true_pm[local_index, lead, 1]),
                    })
            sample_offset += pred_chi.shape[0]

    combined = [np.concatenate(parts, axis=0) for parts in (
        pred_chi_parts, true_chi_parts, hist_chi_parts,
        pred_pm_parts, true_pm_parts, hist_pm_parts,
    )]
    metrics = _metric_report(*combined, horizons=getattr(config, "metric_horizons", []))
    with (output_dir / f"{mode_name}_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / f"{mode_name}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"metrics": metrics, "num_samples": len(dataset), "save_dir": str(output_dir)}
