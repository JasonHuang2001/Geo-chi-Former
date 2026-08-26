"""Small EOP/EAM helpers shared by the checkpoint data loader."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


DEFAULT_FUTURE_EAM_NOISE_RATIO = {
    "aam_x": 0.51676048, "aam_y": 0.35413006,
    "oam_x": 0.55954207, "oam_y": 0.53790316,
    "ham_x": 0.03001029, "ham_y": 0.04958461,
    "slam_x": 0.19647301, "slam_y": 0.21684768,
}

DEFAULT_FUTURE_EAM_NOISE_GROWTH = {
    "aam_x": 0.15626575, "aam_y": 0.16135400,
    "oam_x": 0.08276080, "oam_y": 0.08686451,
    "ham_x": 0.28409593, "ham_y": 0.25651700,
    "slam_x": 0.34870299, "slam_y": 0.24664223,
}


def get_chi_wilson_numpy(x, y, dt=1.0, tc=433.0, q=179.0):
    """Apply the discrete Wilson inversion from polar motion to excitation."""
    pole = np.asarray(x) - 1j * np.asarray(y)
    sigma_c = 2 * np.pi / tc
    sigma_cw = sigma_c * (1 + 1j / (2 * q))
    front = (1j / (2 * sigma_cw * dt)) * np.exp(1j * np.pi * dt / tc)
    step = np.exp(1j * sigma_cw * dt)
    aligned = pole[1:-1]
    chi = front * (pole[2:] + (1 - step) * aligned - step * pole[:-2])
    return chi.real, chi.imag, aligned.real, -aligned.imag


def resolve_future_eam_parameter(value, columns, name):
    """Normalize scalar, sequence, or per-column EAM parameters to one vector."""
    columns = list(columns)
    if isinstance(value, Mapping):
        missing = [column for column in columns if column not in value]
        if missing:
            raise ValueError(f"{name} is missing component parameters: {missing}")
        result = np.asarray([float(value[column]) for column in columns], dtype=np.float32)
    elif isinstance(value, (list, tuple, np.ndarray)):
        result = np.asarray(value, dtype=np.float32)
        if result.ndim != 1 or result.shape[0] != len(columns):
            raise ValueError(f"{name} must have length {len(columns)}; got shape={result.shape}")
    else:
        result = np.full(len(columns), float(value), dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values: {result}")
    return result
