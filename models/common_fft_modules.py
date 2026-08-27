import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, self.num_features))
        self._mean = None
        self._stdev = None

    def _resolve_feature_selector(self, x, target_slice):
        if target_slice is not None:
            return target_slice
        if x.shape[-1] == self.num_features:
            return slice(None)
        return list(range(self.num_features - x.shape[-1], self.num_features))

    def forward(self, x, mode, target_slice=None):
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._stdev = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False).detach() + self.eps)
            x = (x - self._mean) / self._stdev
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        if mode == "denorm":
            if self._mean is None or self._stdev is None:
                raise RuntimeError("RevIN denorm was called before norm")
            selector = self._resolve_feature_selector(x, target_slice)
            mean = self._mean[:, :, selector]
            stdev = self._stdev[:, :, selector]
            if self.affine:
                weight = self.affine_weight[:, :, selector]
                bias = self.affine_bias[:, :, selector]
                x = (x - bias) / (weight + self.eps * self.eps)
            return x * stdev + mean
        raise ValueError(f"Unsupported RevIN mode: {mode}")


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        d_model = int(d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(dtype=x.dtype, device=x.device)


class RoPETransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="gelu",
        rope_base=10000.0,
        rope_scale=1.0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        if self.d_model % self.nhead != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by nhead={self.nhead}")
        self.head_dim = self.d_model // self.nhead
        self.rope_base = float(rope_base)
        self.rope_scale = float(rope_scale)

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.linear1 = nn.Linear(self.d_model, int(dim_feedforward))
        self.linear2 = nn.Linear(int(dim_feedforward), self.d_model)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def _project_heads(self, proj, src):
        bsz, seq_len, _ = src.shape
        return proj(src).view(bsz, seq_len, self.nhead, self.head_dim)

    def _rope_cos_sin(self, seq_len, device, dtype):
        rotary_dim = (self.head_dim // 2) * 2
        if rotary_dim == 0:
            return None, None, rotary_dim
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim))
        pos = torch.arange(seq_len, device=device, dtype=dtype) * self.rope_scale
        freqs = torch.einsum("i,j->ij", pos, inv_freq)
        return freqs.cos()[None, :, None, :], freqs.sin()[None, :, None, :], rotary_dim

    def _apply_rope(self, x, cos, sin, rotary_dim):
        if rotary_dim == 0:
            return x
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        x_even = x_rot[..., 0::2]
        x_odd = x_rot[..., 1::2]
        rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)
        if x_pass.numel() == 0:
            return rotated
        return torch.cat([rotated, x_pass], dim=-1)

    def _self_attention(self, src, src_mask=None, src_key_padding_mask=None):
        bsz, seq_len, _ = src.shape
        q = self._project_heads(self.q_proj, src)
        k = self._project_heads(self.k_proj, src)
        v = self._project_heads(self.v_proj, src)
        cos, sin, rotary_dim = self._rope_cos_sin(seq_len, src.device, src.dtype)
        q = self._apply_rope(q, cos, sin, rotary_dim)
        k = self._apply_rope(k, cos, sin, rotary_dim)

        scores = torch.einsum("blhd,bshd->bhls", q, k) / math.sqrt(self.head_dim)
        if src_mask is not None:
            scores = scores + src_mask
        if src_key_padding_mask is not None:
            mask = src_key_padding_mask[:, None, None, :].to(dtype=torch.bool)
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhls,bshd->blhd", attn, v).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = src + self.dropout(self._self_attention(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask))
        src = self.norm1(src)
        ff = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(ff)
        return self.norm2(src)
