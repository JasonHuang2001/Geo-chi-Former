import torch
import torch.nn as nn
import numpy as np


class WilsonPhysicsLayer(nn.Module):
    """Discrete Wilson reconstruction with day-based ``dt``/``tc`` and quality factor ``q``."""

    def __init__(self, dt, tc, q):
        super().__init__()
        self.dt = dt
        sigma_c = 2 * np.pi / tc
        sigma_cw = sigma_c * (1 + 1j / (2 * q))

        term_exp_1 = np.exp(1j * np.pi * dt / tc)
        factor_A = (-1j * sigma_cw * dt / 2) * term_exp_1
        factor_B = np.exp(1j * sigma_cw * dt)

        self.register_buffer('A_r', torch.tensor(factor_A.real, dtype=torch.float32))
        self.register_buffer('A_i', torch.tensor(factor_A.imag, dtype=torch.float32))
        self.register_buffer('B_r', torch.tensor(factor_B.real, dtype=torch.float32))
        self.register_buffer('B_i', torch.tensor(factor_B.imag, dtype=torch.float32))

    def complex_mul(self, r1, i1, r2, i2):
        return r1 * r2 - i1 * i2, r1 * i2 + i1 * r2

    def forward(self, chi_phy, p0, chi0):
        """Reconstruct polar motion from excitation."""
        _, steps, _ = chi_phy.shape
        if steps == 0:
            return chi_phy.new_empty(chi_phy.shape), chi_phy

        out_dtype = chi_phy.dtype
        dtype = torch.float64 if out_dtype == torch.float64 else torch.float32
        device = chi_phy.device
        complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64

        a = torch.complex(self.A_r.to(dtype=dtype), self.A_i.to(dtype=dtype)).to(complex_dtype)
        b = torch.complex(self.B_r.to(dtype=dtype), self.B_i.to(dtype=dtype)).to(complex_dtype)

        chi_real = chi_phy[..., 0].to(dtype=dtype)
        chi_imag = chi_phy[..., 1].to(dtype=dtype)
        chi = torch.complex(chi_real, chi_imag).to(complex_dtype)
        prev_chi = torch.cat(
            [torch.complex(chi0[:, 0].to(dtype=dtype), chi0[:, 1].to(dtype=dtype)).to(complex_dtype).unsqueeze(1), chi[:, :-1]],
            dim=1,
        )
        forcing = a * (chi + prev_chi)

        powers = b ** torch.arange(steps, device=device, dtype=dtype)
        weighted_forcing = forcing / powers.unsqueeze(0)
        p_initial = torch.complex(p0[:, 0].to(dtype=dtype), p0[:, 1].to(dtype=dtype)).to(complex_dtype)
        p_complex = powers.unsqueeze(0) * (b * p_initial.unsqueeze(1) + torch.cumsum(weighted_forcing, dim=1))

        p_out = torch.stack((p_complex.real, p_complex.imag), dim=2).to(out_dtype)
        return p_out, chi_phy


def integrate_dataset_chi_with_wilson(physics, chi_future, pole_last, chi_last):
    """Integrate Dataset_EOP/Matlab-convention chi back to physical PM.

    Dataset_EOP and the LS-CATS helpers invert pole motion with p = x - i*y.
    WilsonPhysicsLayer internally represents pole motion as p = x + i*y.
    Convert only the pole y sign at the boundary; chi_future and chi_last stay
    in the inversion convention used to create them.
    """
    pole_last_internal = torch.stack((pole_last[..., 0], -pole_last[..., 1]), dim=-1)
    pole_internal, chi_out = physics(chi_future, pole_last_internal, chi_last)
    pole_physical = torch.stack((pole_internal[..., 0], -pole_internal[..., 1]), dim=-1)
    return pole_physical, chi_out


def integrate_matlab_chi_with_wilson(physics, chi_future, pole_last, chi_last):
    return integrate_dataset_chi_with_wilson(physics, chi_future, pole_last, chi_last)


def invert_dataset_pole_with_wilson(physics, pole_future, pole_last, chi_last):
    """Invert the exact discrete recurrence used by ``WilsonPhysicsLayer``.

    Inputs and outputs use the Dataset_EOP convention: physical pole motion is
    represented as ``(x, y)`` while chi is represented as the real and
    imaginary parts associated with ``p = x - i*y``. This function is the
    algebraic inverse of :func:`integrate_dataset_chi_with_wilson`; it is
    distinct from the centred finite-difference chi estimator used when
    preprocessing a long observed PM series.
    """
    if pole_future.ndim != 3 or pole_future.size(-1) != 2:
        raise ValueError(f"pole_future must be [B, T, 2], got {tuple(pole_future.shape)}")
    if pole_last.ndim != 2 or pole_last.size(-1) != 2:
        raise ValueError(f"pole_last must be [B, 2], got {tuple(pole_last.shape)}")
    if chi_last.ndim != 2 or chi_last.size(-1) != 2:
        raise ValueError(f"chi_last must be [B, 2], got {tuple(chi_last.shape)}")
    if pole_future.size(0) != pole_last.size(0) or pole_future.size(0) != chi_last.size(0):
        raise ValueError("pole_future, pole_last, and chi_last must share the batch dimension")
    if pole_future.size(1) == 0:
        return pole_future.new_empty(pole_future.shape)

    out_dtype = pole_future.dtype
    dtype = torch.float64 if out_dtype == torch.float64 else torch.float32
    complex_dtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    a = torch.complex(physics.A_r.to(dtype=dtype), physics.A_i.to(dtype=dtype)).to(complex_dtype)
    b = torch.complex(physics.B_r.to(dtype=dtype), physics.B_i.to(dtype=dtype)).to(complex_dtype)

    pole_internal = torch.complex(
        pole_future[..., 0].to(dtype=dtype),
        -pole_future[..., 1].to(dtype=dtype),
    ).to(complex_dtype)
    previous_pole = torch.complex(
        pole_last[..., 0].to(dtype=dtype),
        -pole_last[..., 1].to(dtype=dtype),
    ).to(complex_dtype)
    previous_chi = torch.complex(
        chi_last[..., 0].to(dtype=dtype),
        chi_last[..., 1].to(dtype=dtype),
    ).to(complex_dtype)

    chi_steps = []
    for step in range(pole_internal.size(1)):
        current_pole = pole_internal[:, step]
        current_chi = (current_pole - b * previous_pole) / a - previous_chi
        chi_steps.append(current_chi)
        previous_pole = current_pole
        previous_chi = current_chi
    chi = torch.stack(chi_steps, dim=1)
    return torch.stack((chi.real, chi.imag), dim=-1).to(out_dtype)
