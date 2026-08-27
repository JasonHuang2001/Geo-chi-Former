import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.common_fft_modules import RevIN, SinusoidalPositionalEncoding, RoPETransformerEncoderLayer

class PatchTST(nn.Module):
    """Channel-independent PatchTST forecasting baseline."""
    def __init__(self, config):
        super(PatchTST, self).__init__()
        # Required dimensions.
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.hidden_size = config.hidden_size

        # Optional settings keep the baseline configuration backward compatible.
        self.patch_len = getattr(config, 'patch_len', 16)
        self.stride = getattr(config, 'stride', 8)
        self.num_layers = getattr(config, 'num_layers', 2)
        self.nhead = getattr(config, 'nhead', 4)
        self.dropout = getattr(config, 'dropout', 0.2)
        self.use_revin = getattr(config, 'use_revin', True)
        self.pos_encoding_type = getattr(config, 'pos_encoding_type', 'learnable')
        self.rope_base = float(getattr(config, 'rope_base', 10000.0))
        self.rope_scale = float(getattr(config, 'rope_scale', 1.0))
        self.target_slice = getattr(config, 'target_slice', None)
        self.use_time_mark = bool(getattr(config, 'use_time_mark', False))
        self.time_feature_dim = int(getattr(config, 'time_feature_dim', 0))
        self.use_regime = bool(getattr(config, 'use_regime', False))
        self.regime_dim = int(getattr(config, 'regime_dim', 8))
        if self.use_regime and self.regime_dim <= 0:
            raise ValueError(f"regime_dim must be positive, got {self.regime_dim}")

        # Patch geometry.
        self.pad_len = (self.stride - (self.seq_len - self.patch_len) % self.stride) % self.stride
        self.num_patches = (self.seq_len + self.pad_len - self.patch_len) // self.stride + 1

        if self.use_revin:
            self.revin = RevIN(num_features=self.input_size, eps=1e-5, affine=True)

        # Patch embeddings.
        self.patch_embedding = nn.Linear(self.patch_len, self.hidden_size)
        if self.use_time_mark:
            if self.time_feature_dim <= 0:
                raise ValueError("config.time_feature_dim must be positive when use_time_mark=True")
            self.time_patch_embedding = nn.Linear(self.patch_len * self.time_feature_dim, self.hidden_size)
        else:
            self.time_patch_embedding = None
        if self.pos_encoding_type == 'learnable':
            self.positional_encoding = nn.Parameter(torch.randn(1, self.num_patches, self.hidden_size))
        elif self.pos_encoding_type == 'sin':
            self.positional_encoding = SinusoidalPositionalEncoding(self.hidden_size)
        elif self.pos_encoding_type == 'rope':
            self.positional_encoding = None
        else:
            raise ValueError(f"Unsupported pos_encoding_type: {self.pos_encoding_type}")
        if self.use_regime:
            self.regime_encoder = nn.Sequential(
                nn.LayerNorm(self.regime_dim),
                nn.Linear(self.regime_dim, self.hidden_size * 2),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size * 2, self.hidden_size),
            )

        # Transformer encoder.
        if self.pos_encoding_type == 'rope':
            self.transformer_layers = nn.ModuleList([
                RoPETransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=self.nhead,
                    dim_feedforward=self.hidden_size * 4,
                    dropout=self.dropout,
                    rope_base=self.rope_base,
                    rope_scale=self.rope_scale
                ) for _ in range(self.num_layers)
            ])
            self.transformer = None
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_size,
                nhead=self.nhead,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.dropout,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
            self.transformer_layers = None

        # Flatten encoded patches and project to the forecast horizon.
        head_in_features = self.num_patches * self.hidden_size
        if self.use_regime:
            head_in_features += self.hidden_size
        self.head = nn.Linear(head_in_features, self.pred_len)

    def forward(self, x, x_mark=None, target_slice=None, regime=None):
        """
        x shape: (batch_size, seq_len, input_size)
        x_mark shape: (batch_size, seq_len, time_feature_dim)
        regime shape: (batch_size, regime_dim), optional history summary at t0
        """
        B, L, M = x.shape
        if self.use_time_mark and x_mark is None:
            raise ValueError("x_mark is required when use_time_mark=True")
        if self.use_regime:
            if regime is None:
                raise ValueError("regime with shape [B, regime_dim] is required when use_regime=True")
            if regime.dim() != 2 or regime.size(0) != B or regime.size(1) != self.regime_dim:
                raise ValueError(
                    f"regime shape={tuple(regime.shape)} does not match ({B}, {self.regime_dim})"
                )

        if self.use_revin:
            x = self.revin(x, 'norm')

        # Pad the history to a complete final patch.
        if self.pad_len > 0:
            x = F.pad(x.permute(0, 2, 1), (0, self.pad_len), mode='replicate').permute(0, 2, 1)
            if self.use_time_mark:
                x_mark = F.pad(x_mark.permute(0, 2, 1), (0, self.pad_len), mode='replicate').permute(0, 2, 1)

        # Treat each channel as an independent sequence.
        x = x.permute(0, 2, 1).contiguous().view(B * M, -1)

        # Form and embed patches.
        patches = x.unfold(1, self.patch_len, self.stride)
        enc_in = self.patch_embedding(patches)
        if self.use_time_mark:
            time_patches = x_mark.unfold(1, self.patch_len, self.stride)
            time_patches = time_patches.permute(0, 1, 3, 2).contiguous().view(B, self.num_patches, -1)
            time_emb = self.time_patch_embedding(time_patches)
            enc_in = enc_in + time_emb.repeat_interleave(M, dim=0)

        # Add positional information.
        if self.pos_encoding_type == 'learnable':
            enc_in = enc_in + self.positional_encoding
        elif self.pos_encoding_type == 'sin':
            enc_in = self.positional_encoding(enc_in)

        # Encode patches and forecast each channel.
        if self.pos_encoding_type == 'rope':
            enc_out = enc_in
            for layer in self.transformer_layers:
                enc_out = layer(enc_out)
        else:
            enc_out = self.transformer(enc_in)
        enc_out = enc_out.view(B * M, -1)   #(Batch * M, num_patch * hidden)
        if self.use_regime:
            # regime_emb: [B, hidden_size] -> [B*M, hidden_size]
            # Condition the forecast head without repeating over time.
            regime_emb = self.regime_encoder(regime).repeat_interleave(M, dim=0)
            enc_out = torch.cat([enc_out, regime_emb], dim=-1)
        dec_out = self.head(enc_out)

        # Restore [batch, horizon, channel] layout.
        dec_out = dec_out.view(B, M, self.pred_len).permute(0, 2, 1).contiguous()

        # Select target channels explicitly when covariates are present.
        final_target_slice = target_slice if target_slice is not None else self.target_slice
        if final_target_slice is not None:
            dec_out = dec_out[:, :, final_target_slice]
        elif self.output_size < self.input_size:
            dec_out = dec_out[:, :, -self.output_size:]

        if self.use_revin:
            dec_out = self.revin(dec_out, 'denorm', target_slice=final_target_slice)

        return dec_out
