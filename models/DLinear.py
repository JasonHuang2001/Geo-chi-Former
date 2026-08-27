import torch
import torch.nn as nn

from models.common_fft_modules import RevIN


class MovingAverage(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = int(kernel_size)
        if self.kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {self.kernel_size}")
        self.avg = nn.AvgPool1d(kernel_size=self.kernel_size, stride=1, padding=0)

    def forward(self, x):
        """
        x: [B, L, C]
        return: [B, L, C]
        """
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        if left > 0:
            front = x[:, 0:1, :].repeat(1, left, 1)
        else:
            front = x[:, :0, :]
        if right > 0:
            end = x[:, -1:, :].repeat(1, right, 1)
        else:
            end = x[:, :0, :]
        x_pad = torch.cat([front, x, end], dim=1)
        return self.avg(x_pad.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = MovingAverage(kernel_size)

    def forward(self, x):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class DLinear(nn.Module):
    """
    DLinear baseline with series decomposition and direct linear forecasting.

    The implementation follows the channel-independent DLinear formulation:
    each channel is decomposed into seasonal and trend components, projected
    from seq_len to pred_len, then target channels are selected for evaluation.
    """

    def __init__(self, config):
        super().__init__()
        self.seq_len = int(config.seq_len)
        self.pred_len = int(config.pred_len)
        self.input_size = int(config.input_size)
        self.output_size = int(config.output_size)
        self.target_slice = getattr(config, "target_slice", None)
        self.moving_avg = int(getattr(config, "moving_avg", 25))
        self.individual = bool(getattr(config, "individual", True))
        self.use_revin = bool(getattr(config, "use_revin", True))

        self.decomposition = SeriesDecomposition(self.moving_avg)
        if self.use_revin:
            self.revin = RevIN(num_features=self.input_size, eps=1e-5, affine=True)

        if self.individual:
            self.linear_seasonal = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.input_size)]
            )
            self.linear_trend = nn.ModuleList(
                [nn.Linear(self.seq_len, self.pred_len) for _ in range(self.input_size)]
            )
        else:
            self.linear_seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.linear_trend = nn.Linear(self.seq_len, self.pred_len)
        self._reset_parameters()

    def _reset_parameters(self):
        modules = list(self.linear_seasonal) + list(self.linear_trend) if self.individual else [
            self.linear_seasonal,
            self.linear_trend,
        ]
        for module in modules:
            nn.init.constant_(module.weight, 1.0 / max(1, self.seq_len))
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _project(self, seasonal, trend):
        seasonal = seasonal.permute(0, 2, 1)  # [B, C, L]
        trend = trend.permute(0, 2, 1)
        if self.individual:
            seasonal_out = []
            trend_out = []
            for channel in range(self.input_size):
                seasonal_out.append(self.linear_seasonal[channel](seasonal[:, channel, :]))
                trend_out.append(self.linear_trend[channel](trend[:, channel, :]))
            seasonal_out = torch.stack(seasonal_out, dim=1)
            trend_out = torch.stack(trend_out, dim=1)
        else:
            seasonal_out = self.linear_seasonal(seasonal)
            trend_out = self.linear_trend(trend)
        return (seasonal_out + trend_out).permute(0, 2, 1).contiguous()

    def forward(self, x, target_slice=None):
        """
        x: [B, seq_len, input_size]
        return: [B, pred_len, output_size]
        """
        if x.dim() != 3:
            raise ValueError(f"x must be [B, L, C], got shape={tuple(x.shape)}")
        if x.size(1) != self.seq_len:
            raise ValueError(f"x length must be seq_len={self.seq_len}, got {x.size(1)}")
        if x.size(2) != self.input_size:
            raise ValueError(f"x channels must be input_size={self.input_size}, got {x.size(2)}")

        if self.use_revin:
            x = self.revin(x, "norm")

        seasonal, trend = self.decomposition(x)
        pred = self._project(seasonal, trend)

        final_target_slice = target_slice if target_slice is not None else self.target_slice
        if final_target_slice is not None:
            pred = pred[:, :, final_target_slice]
        elif self.output_size < self.input_size:
            pred = pred[:, :, -self.output_size:]

        if self.use_revin:
            pred = self.revin(pred, "denorm", target_slice=final_target_slice)
        return pred
