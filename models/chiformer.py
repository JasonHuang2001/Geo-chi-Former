"""Geo-chi-Former model for excitation-domain forecasting.

The model combines variable-wise patch embeddings, causal cross-time
attention, cross-variate attention, known-covariate KV shift, and recursive
next-patch decoding. RoPE is limited to cross-time attention. Cross-variate
attention concatenates source variables, shifts known values from K_i/V_i to
K_i/V_{i+1}, smooths scores before softmax, and can apply axis-aware bias.
Inputs use shape ``[batch, time, variables]``.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F




def _to_index(idx, default):
    """Normalize a configured slice or index list."""
    if idx is None:
        return default
    if isinstance(idx, slice):
        return idx
    return list(idx)


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


def _normalize_position_encoding_mode(value):
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "learned": "learnable",
        "learnable": "learnable",
        "absolute": "learnable",
        "sin": "sinusoidal",
        "sincos": "sinusoidal",
        "sin_cos": "sinusoidal",
        "sinusoidal": "sinusoidal",
    }
    if mode not in aliases:
        raise ValueError(
            f"Invalid position_encoding_mode={value!r}; "
            "expected 'learnable' or 'sinusoidal'"
        )
    return aliases[mode]


def _num_channels_from_index(idx, total):
    if isinstance(idx, slice):
        return len(range(*idx.indices(total)))
    return len(idx)


def _pad_to_multiple(x, multiple, dim=1):
    """Zero-pad one dimension to a multiple and return the pad length."""
    length = x.size(dim)
    pad_len = (multiple - length % multiple) % multiple
    if pad_len == 0:
        return x, 0

    pad_shape = list(x.shape)
    pad_shape[dim] = pad_len
    pad = torch.zeros(*pad_shape, dtype=x.dtype, device=x.device)
    x = torch.cat([x, pad], dim=dim)
    return x, pad_len


def _attention_intervention_matches(intervention, attention_type, layer_index=None, group=None):
    if intervention is None:
        return False
    if str(intervention.get("attention_type", "")).strip().lower() != attention_type:
        return False
    requested_layer = intervention.get("layer")
    if requested_layer is not None and int(requested_layer) != int(layer_index):
        return False
    requested_group = intervention.get("group")
    if requested_group is not None and str(requested_group).strip().lower() != str(group).strip().lower():
        return False
    return True


def _renormalize_attention(attn):
    denom = attn.sum(dim=-1, keepdim=True)
    return torch.where(denom > 0, attn / denom.clamp_min(1e-12), attn)


def _apply_attention_intervention(attn, intervention, attention_type, layer_index=None, group=None):
    """Apply an analysis-only intervention to effective attention weights."""
    if not _attention_intervention_matches(
        intervention,
        attention_type=attention_type,
        layer_index=layer_index,
        group=group,
    ):
        return attn

    mode = str(intervention.get("mode", "")).strip().lower()
    changed = attn.clone()
    head = intervention.get("head")
    head_slice = slice(None) if head is None else int(head)

    if mode == "disable_head":
        changed[..., head_slice, :, :] = 0
        return changed

    if mode == "uniform_head":
        selected = changed[..., head_slice, :, :]
        support = selected > 0
        uniform = support.to(selected.dtype)
        uniform = _renormalize_attention(uniform)
        changed[..., head_slice, :, :] = uniform
        return changed

    sources = [int(idx) for idx in intervention.get("source_indices", [])]
    if mode in {"mask_sources", "keep_sources"}:
        if not sources:
            raise ValueError(f"{mode} requires non-empty source_indices")
        if mode == "mask_sources":
            changed[..., head_slice, :, sources] = 0
        else:
            keep_mask = torch.zeros(changed.size(-1), dtype=torch.bool, device=changed.device)
            keep_mask[sources] = True
            changed[..., head_slice, :, ~keep_mask] = 0
        return _renormalize_attention(changed)

    raise ValueError(f"Unsupported attention intervention mode: {mode!r}")


class RotaryEmbedding(nn.Module):
    """Rotary position embedding for cross-time queries and keys."""

    def __init__(self, dim, base=10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE head_dim must be even; got {dim}")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q, k):
        """
        q/k: [B, H, N, Dh]
        """
        n = q.size(-2)
        device = q.device
        pos = torch.arange(n, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("n,d->nd", pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos()[None, None, :, :].to(dtype=q.dtype)
        sin = emb.sin()[None, None, :, :].to(dtype=q.dtype)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k


class PatchEmbedding(nn.Module):
    """Embed per-variable patches with a shared linear projection."""

    def __init__(
        self,
        patch_len,
        d_model,
        dropout=0.0,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.proj = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, L, C]
        return: [B, N, C, D]
        """
        b, l, c = x.shape
        x, _ = _pad_to_multiple(x, self.patch_len, dim=1)
        n = x.size(1) // self.patch_len
        x = x.reshape(b, n, self.patch_len, c).permute(0, 1, 3, 2).contiguous()
        x = self.proj(x)
        return self.dropout(x)


class LearnableTemporalPositionEmbedding(nn.Module):
    """Learnable absolute position embedding for Cross-Time tokens."""

    def __init__(self, max_len, d_model, dropout=0.0):
        super().__init__()
        self.max_len = int(max_len)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.max_len, d_model))
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        n = x.size(1)
        if n > self.max_len:
            raise ValueError(
                f"Cross-Time token length {n} exceeds max_position_tokens={self.max_len}"
            )
        return self.dropout(x + self.pos_embed[:, :n, :].to(dtype=x.dtype))


class SinusoidalTemporalPositionEmbedding(nn.Module):
    """Fixed sinusoidal absolute position embedding for Cross-Time tokens."""

    def __init__(self, max_len, d_model, dropout=0.0):
        super().__init__()
        self.max_len = int(max_len)
        position = torch.arange(self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pos_embed = torch.zeros(1, self.max_len, d_model, dtype=torch.float32)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term[:pos_embed[:, :, 1::2].shape[-1]])
        self.register_buffer("pos_embed", pos_embed, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n = x.size(1)
        if n > self.max_len:
            raise ValueError(
                f"Cross-Time token length {n} exceeds max_position_tokens={self.max_len}"
            )
        return self.dropout(x + self.pos_embed[:, :n, :].to(device=x.device, dtype=x.dtype))


class FeedForward(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff=None,
        dropout=0.1,
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class CausalSelfAttentionRoPE(nn.Module):
    """Causal cross-time self-attention with optional RoPE."""

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        rope_base=10000.0,
        use_rope=True,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.use_rope = _as_bool(use_rope)
        self.rope = RotaryEmbedding(self.head_dim, base=rope_base) if self.use_rope else None

    def forward(
        self,
        x,
        return_attn=False,
        attention_intervention=None,
        layer_index=None,
        group=None,
    ):
        """
        x: [Bv, N, D], where Bv = B * C_variable
        return: [Bv, N, D]
        """
        b, n, d = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.rope is not None:
            q, k = self.rope(q, k)

        raw_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        causal_mask = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
        final_scores = raw_scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))
        attn = F.softmax(final_scores, dim=-1)
        attn = self.attn_drop(attn)
        attn = _apply_attention_intervention(
            attn,
            attention_intervention,
            attention_type="cross_time",
            layer_index=layer_index,
            group=group,
        )

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n, d)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        if not return_attn:
            return out
        return out, {
            "raw_scores": raw_scores,
            "final_scores": final_scores,
            "attn_weights": attn,
        }


class CrossTimeBlock(nn.Module):
    """Apply shared causal attention within each variable's patch sequence."""

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout,
        rope_base,
        use_rope=True,
        max_position_tokens=1024,
        position_encoding_mode="learnable",
    ):
        super().__init__()
        self.use_rope = _as_bool(use_rope)
        self.position_encoding_mode = _normalize_position_encoding_mode(position_encoding_mode)
        if self.use_rope:
            self.learned_pos = None
            self.sinusoidal_pos = None
        elif self.position_encoding_mode == "learnable":
            self.learned_pos = LearnableTemporalPositionEmbedding(
                max_position_tokens, d_model, dropout=dropout
            )
            self.sinusoidal_pos = None
        else:
            self.learned_pos = None
            self.sinusoidal_pos = SinusoidalTemporalPositionEmbedding(
                max_position_tokens, d_model, dropout=dropout
            )
        self.attn = CausalSelfAttentionRoPE(
            d_model,
            n_heads,
            dropout=dropout,
            rope_base=rope_base,
            use_rope=self.use_rope,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @property
    def position_encoding(self):
        if self.learned_pos is not None:
            return self.learned_pos
        return self.sinusoidal_pos

    def forward_one_group(
        self,
        x,
        group,
        return_attn=False,
        attention_intervention=None,
        layer_index=None,
        capture_mode="last_query",
    ):
        """
        x: [B, N, C, D]
        return: [B, N, C, D]
        """
        if x is None or x.size(2) == 0:
            return x, None
        b, n, c, d = x.shape
        # Attend over time independently for each variable.
        z = x.permute(0, 2, 1, 3).reshape(b * c, n, d)
        if self.position_encoding is not None:
            z = self.position_encoding(z)
        attn_result = self.attn(
            z,
            return_attn=return_attn,
            attention_intervention=attention_intervention,
            layer_index=layer_index,
            group=group,
        )
        if return_attn:
            attn_out, detail = attn_result
        else:
            attn_out = attn_result
            detail = None
        z = self.norm1(z + attn_out)
        z = self.norm2(z + self.ffn(z))
        z = z.reshape(b, c, n, d).permute(0, 2, 1, 3).contiguous()
        if detail is not None:
            reshaped = {}
            for key, value in detail.items():
                value = value.reshape(b, c, self.attn.n_heads, n, n)
                if capture_mode == "last_query":
                    value = value[:, :, :, -1:, :]
                elif capture_mode != "full":
                    raise ValueError(
                        f"Unsupported cross_time_capture={capture_mode!r}; "
                        "expected 'last_query' or 'full'"
                    )
                reshaped[key] = value
            detail = reshaped
        return z, detail

    def forward(
        self,
        h_tgt,
        h_obs,
        h_knw,
        return_attn=False,
        attention_intervention=None,
        layer_index=None,
        capture_mode="last_query",
    ):
        h_tgt, target_detail = self.forward_one_group(
            h_tgt,
            "target",
            return_attn=return_attn,
            attention_intervention=attention_intervention,
            layer_index=layer_index,
            capture_mode=capture_mode,
        )
        h_obs, observed_detail = self.forward_one_group(
            h_obs,
            "observed",
            return_attn=return_attn,
            attention_intervention=attention_intervention,
            layer_index=layer_index,
            capture_mode=capture_mode,
        )
        h_knw, known_detail = self.forward_one_group(
            h_knw,
            "known",
            return_attn=return_attn,
            attention_intervention=attention_intervention,
            layer_index=layer_index,
            capture_mode=capture_mode,
        )
        detail = None
        if return_attn:
            detail = {
                "target": target_detail,
                "observed": observed_detail,
                "known": known_detail,
            }
        return h_tgt, h_obs, h_knw, detail


class SmoothedCrossVariateAttention(nn.Module):
    """Cross-variate attention with optional KV shift and score smoothing.

    Target tokens query target, observed, and known-variable tokens. Known
    values require ``Nt + 1`` tokens when KV shift is enabled.
    """

    def __init__(
        self,
        d_model,
        n_heads,
        dropout=0.1,
        smoothing_alpha=0.2,
        use_kv_shift=True,
        return_attn=False,
        c_target=None,
        axis_bias=None,
        use_target_query_embedding=False,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        if not (0.0 < smoothing_alpha <= 1.0):
            raise ValueError("smoothing_alpha must be in (0, 1]")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.smoothing_alpha = smoothing_alpha
        self.use_kv_shift = _as_bool(use_kv_shift)
        self.return_attn = return_attn
        self.c_target = c_target

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        if use_target_query_embedding:
            if c_target is None:
                raise ValueError("c_target is required when target query embeddings are enabled")
            self.target_query_embed = nn.Parameter(torch.zeros(c_target, d_model))
            nn.init.trunc_normal_(self.target_query_embed, std=0.02)
        else:
            self.target_query_embed = None

        if axis_bias is None:
            self.axis_bias = None
        else:
            self.axis_bias = nn.Parameter(axis_bias.detach().clone().float())

    @staticmethod
    def _ensure_known_len(h_knw, need_len):
        """Repeat the final known token when KV shift needs extra context."""
        if h_knw.size(1) >= need_len:
            return h_knw
        pad_len = need_len - h_knw.size(1)
        pad = h_knw[:, -1:, :, :].expand(-1, pad_len, -1, -1)
        return torch.cat([h_knw, pad], dim=1)

    def _smooth_scores(self, scores):
        """
        scores: [B, Nt, H, Ct, C_all]
        return: [B, Nt, H, Ct, C_all]
        """
        alpha = self.smoothing_alpha
        if alpha >= 1.0 or scores.size(1) <= 1:
            return scores
        smoothed = [scores[:, 0]]
        prev = scores[:, 0]
        for i in range(1, scores.size(1)):
            prev = alpha * scores[:, i] + (1.0 - alpha) * prev
            smoothed.append(prev)
        return torch.stack(smoothed, dim=1)

    def forward(
        self,
        h_tgt,
        h_obs,
        h_knw,
        return_attn=None,
        attention_intervention=None,
        layer_index=None,
    ):
        b, nt, ct, d = h_tgt.shape
        return_attn = self.return_attn if return_attn is None else return_attn

        pieces_k = [h_tgt]
        pieces_v = [h_tgt]

        if h_obs is not None and h_obs.size(2) > 0:
            # Observed covariates align only with historical target patches.
            if h_obs.size(1) < nt:
                pad_len = nt - h_obs.size(1)
                pad = torch.zeros(b, pad_len, h_obs.size(2), d, dtype=h_obs.dtype, device=h_obs.device)
                h_obs = torch.cat([h_obs, pad], dim=1)
            pieces_k.append(h_obs[:, :nt])
            pieces_v.append(h_obs[:, :nt])

        if h_knw is not None and h_knw.size(2) > 0:
            need_len = nt + 1 if self.use_kv_shift else nt
            h_knw = self._ensure_known_len(h_knw, need_len)
            pieces_k.append(h_knw[:, :nt])
            if self.use_kv_shift:
                pieces_v.append(h_knw[:, 1:nt + 1])
            else:
                pieces_v.append(h_knw[:, :nt])

        # Standard alignment uses K_i/V_i; KV shift uses K_i/V_{i+1}.
        h_k = torch.cat(pieces_k, dim=2)
        h_v = torch.cat(pieces_v, dim=2)
        c_all = h_k.size(2)

        q_input = h_tgt
        if self.target_query_embed is not None:
            if self.target_query_embed.size(0) != ct:
                raise ValueError(
                    f"target_query_embed has {self.target_query_embed.size(0)} channels; expected {ct}"
                )
            q_input = q_input + self.target_query_embed[None, None, :, :]

        q = self.q_proj(q_input).reshape(b, nt, ct, self.n_heads, self.head_dim)
        k = self.k_proj(h_k).reshape(b, nt, c_all, self.n_heads, self.head_dim)
        v = self.v_proj(h_v).reshape(b, nt, c_all, self.n_heads, self.head_dim)

        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        # Patch-wise scores: [B, Nt, H, Ct, C_all].
        content_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        axis_bias = torch.zeros_like(content_scores)
        if self.axis_bias is not None:
            if self.axis_bias.shape != (ct, c_all):
                raise ValueError(
                    f"axis_bias must have shape {(ct, c_all)}; got {tuple(self.axis_bias.shape)}"
                )
            axis_bias = self.axis_bias[None, None, None, :, :].to(
                dtype=content_scores.dtype,
                device=content_scores.device,
            ).expand_as(content_scores)
        final_scores = content_scores + axis_bias
        smoothed_scores = self._smooth_scores(final_scores)
        attn = F.softmax(smoothed_scores, dim=-1)
        attn = self.attn_drop(attn)
        attn = _apply_attention_intervention(
            attn,
            attention_intervention,
            attention_type="cross_variate",
            layer_index=layer_index,
        )

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2, 4).reshape(b, nt, ct, d)
        out = self.out_proj(out)
        out = self.proj_drop(out)

        detail = None
        if return_attn:
            detail = {
                "attn_weights": attn,
                "content_scores": content_scores,
                "axis_bias": axis_bias,
                "final_scores": final_scores,
                "raw_scores": final_scores,
                "smoothed_scores": smoothed_scores,
            }
        return out, detail


class CrossVariateBlock(nn.Module):
    """Update target tokens through cross-variate attention."""

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout,
        smoothing_alpha,
        use_kv_shift,
        c_target,
        axis_bias,
        use_target_query_embedding,
    ):
        super().__init__()
        self.attn = SmoothedCrossVariateAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            smoothing_alpha=smoothing_alpha,
            use_kv_shift=use_kv_shift,
            c_target=c_target,
            axis_bias=axis_bias,
            use_target_query_embedding=use_target_query_embedding,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff=d_ff, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        h_tgt,
        h_obs,
        h_knw,
        return_attn=False,
        attention_intervention=None,
        layer_index=None,
    ):
        attn_out, detail = self.attn(
            h_tgt,
            h_obs,
            h_knw,
            return_attn=return_attn,
            attention_intervention=attention_intervention,
            layer_index=layer_index,
        )
        h_tgt = self.norm1(h_tgt + attn_out)
        h_tgt = self.norm2(h_tgt + self.ffn(h_tgt))
        return h_tgt, detail


class ChiformerLayer(nn.Module):
    """Compose cross-time and cross-variate attention."""

    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout,
        rope_base,
        use_rope,
        max_position_tokens,
        position_encoding_mode,
        smoothing_alpha,
        use_kv_shift,
        use_cross_variate_attention,
        c_target,
        axis_bias,
        use_target_query_embedding,
    ):
        super().__init__()
        self.cross_time = CrossTimeBlock(
            d_model,
            n_heads,
            d_ff,
            dropout,
            rope_base,
            use_rope=use_rope,
            max_position_tokens=max_position_tokens,
            position_encoding_mode=position_encoding_mode,
        )
        self.use_cross_variate_attention = _as_bool(use_cross_variate_attention)
        if self.use_cross_variate_attention:
            self.cross_variate = CrossVariateBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                smoothing_alpha=smoothing_alpha,
                use_kv_shift=use_kv_shift,
                c_target=c_target,
                axis_bias=axis_bias,
                use_target_query_embedding=use_target_query_embedding,
            )
        else:
            self.cross_variate = None

    def forward(
        self,
        h_tgt,
        h_obs,
        h_knw,
        return_attn=False,
        attention_intervention=None,
        layer_index=None,
        cross_time_capture="last_query",
    ):
        h_tgt, h_obs, h_knw, cross_time_detail = self.cross_time(
            h_tgt,
            h_obs,
            h_knw,
            return_attn=return_attn,
            attention_intervention=attention_intervention,
            layer_index=layer_index,
            capture_mode=cross_time_capture,
        )
        if self.cross_variate is None:
            return h_tgt, h_obs, h_knw, {
                "cross_time": cross_time_detail,
                "cross_variate": None,
            }
        h_tgt, cross_variate_detail = self.cross_variate(
            h_tgt,
            h_obs,
            h_knw,
            return_attn=return_attn,
            attention_intervention=attention_intervention,
            layer_index=layer_index,
        )
        return h_tgt, h_obs, h_knw, {
            "cross_time": cross_time_detail,
            "cross_variate": cross_variate_detail,
        }


class Chiformer(nn.Module):
    """Geo-chi-Former excitation forecaster.

    Required configuration fields are ``seq_len``, ``pred_len``, and
    ``input_size``. Other architecture settings have defaults in this class.
    """

    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = int(getattr(configs, "seq_len"))
        self.pred_len = int(getattr(configs, "pred_len"))
        self.input_size = int(getattr(configs, "input_size"))

        self.patch_len = int(getattr(configs, "patch_len", 16))
        self.d_model = int(getattr(configs, "d_model", getattr(configs, "hidden_size", 128)))
        self.n_heads = int(getattr(configs, "n_heads", getattr(configs, "nhead", 4)))
        self.num_layers = int(getattr(configs, "num_layers", 2))
        self.d_ff = getattr(configs, "d_ff", None)
        self.dropout = float(getattr(configs, "dropout", 0.1))
        self.smoothing_alpha = float(getattr(configs, "smoothing_alpha", 0.2))
        self.use_kv_shift = _as_bool(getattr(configs, "use_kv_shift", True))
        self.use_cross_variate_attention = _as_bool(getattr(configs, "use_cross_variate_attention", True))
        self.use_rope = _as_bool(getattr(configs, "use_rope", True))
        self.position_encoding_mode = _normalize_position_encoding_mode(
            getattr(configs, "position_encoding_mode", "learnable")
        )
        self.rope_base = float(getattr(configs, "rope_base", 10000.0))
        default_max_positions = math.ceil((self.seq_len + self.pred_len) / self.patch_len) + 8
        self.max_position_tokens = int(getattr(configs, "max_position_tokens", default_max_positions))
        self.return_attn_default = _as_bool(getattr(configs, "return_attn", False))
        self.use_target_instant_norm = _as_bool(getattr(configs, "use_target_instant_norm", False))
        self.target_instant_norm_eps = float(getattr(configs, "target_instant_norm_eps", 1e-6))

        self.target_index = _to_index(getattr(configs, "target_slice", [0, 1]), slice(0, 2))
        self.observed_index = _to_index(getattr(configs, "observed_slice", []), slice(0, 0))
        self.known_index = _to_index(getattr(configs, "known_slice", []), slice(0, 0))

        self.c_target = _num_channels_from_index(self.target_index, self.input_size)
        self.c_observed = _num_channels_from_index(self.observed_index, self.input_size)
        self.c_known = _num_channels_from_index(self.known_index, self.input_size)
        self.output_size = int(getattr(configs, "output_size", self.c_target))
        if self.output_size != self.c_target:
            raise ValueError(
                "output_size must equal the number of target channels; "
                f"output_size={self.output_size}, c_target={self.c_target}"
            )

        self.patch_embed = PatchEmbedding(self.patch_len, self.d_model, dropout=self.dropout)

        # Preserve source identity during cross-variate attention.
        self.use_variate_embedding = _as_bool(getattr(configs, "use_variate_embedding", False))
        total_vars = self.c_target + self.c_observed + self.c_known
        if self.use_variate_embedding and total_vars > 0:
            self.variate_embed = nn.Parameter(torch.zeros(total_vars, self.d_model))
            nn.init.trunc_normal_(self.variate_embed, std=0.02)
        else:
            self.variate_embed = None

        self.use_target_query_embedding = _as_bool(getattr(configs, "use_target_query_embedding", False))
        self.use_axis_aware_bias = _as_bool(getattr(configs, "use_axis_aware_bias", False))
        axis_bias = (
            self._build_axis_bias(configs, total_vars)
            if self.use_axis_aware_bias and self.use_cross_variate_attention
            else None
        )

        self.layers = nn.ModuleList([
            ChiformerLayer(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                rope_base=self.rope_base,
                use_rope=self.use_rope,
                max_position_tokens=self.max_position_tokens,
                position_encoding_mode=self.position_encoding_mode,
                smoothing_alpha=self.smoothing_alpha,
                use_kv_shift=self.use_kv_shift,
                use_cross_variate_attention=self.use_cross_variate_attention,
                c_target=self.c_target,
                axis_bias=axis_bias,
                use_target_query_embedding=self.use_target_query_embedding,
            )
            for _ in range(self.num_layers)
        ])
        self.projection = nn.Linear(self.d_model, self.patch_len)

    @staticmethod
    def _index_to_list(index, total):
        if isinstance(index, slice):
            return list(range(*index.indices(total)))
        return list(index)

    @staticmethod
    def _axis_from_name(name):
        name = str(name).lower()
        if "_x" in name or name.endswith("x"):
            return 0
        if "_y" in name or name.endswith("y"):
            return 1
        return None

    def _build_axis_bias(self, configs, total_vars):
        same_bias = float(getattr(configs, "axis_same_bias", 0.2))
        cross_bias = float(getattr(configs, "axis_cross_bias", 0.0))
        target_other_bias = float(getattr(configs, "axis_target_other_bias", 0.0))
        known_bias = float(getattr(configs, "axis_known_bias", 0.0))

        target_names = list(getattr(configs, "target_names", []))
        observed_names = list(getattr(configs, "observed_names", []))
        known_names = list(getattr(configs, "known_names", []))
        if len(target_names) < self.c_target:
            target_names.extend([f"target_{idx}" for idx in range(len(target_names), self.c_target)])
        if len(observed_names) < self.c_observed:
            observed_names.extend([f"observed_{idx}" for idx in range(len(observed_names), self.c_observed)])
        if len(known_names) < self.c_known:
            known_names.extend([f"known_{idx}" for idx in range(len(known_names), self.c_known)])

        source_groups = []
        source_axes = []
        for idx, name in enumerate(target_names[:self.c_target]):
            axis = self._axis_from_name(name)
            if axis is None and idx < 2:
                axis = idx
            source_groups.append("target")
            source_axes.append(axis)
        for idx, name in enumerate(observed_names[:self.c_observed]):
            axis = self._axis_from_name(name)
            if axis is None and self.c_target >= 2:
                axis = idx % 2
            source_groups.append("observed")
            source_axes.append(axis)
        for _name in known_names[:self.c_known]:
            source_groups.append("known")
            source_axes.append(None)

        if len(source_groups) != total_vars:
            raise ValueError(
                f"axis-bias source count {len(source_groups)} does not match total_vars={total_vars}"
            )

        bias = torch.zeros(self.c_target, total_vars, dtype=torch.float32)
        for query_idx in range(self.c_target):
            query_axis = query_idx if query_idx < 2 else None
            for source_idx, (group, source_axis) in enumerate(zip(source_groups, source_axes)):
                if group == "known" or query_axis is None or source_axis is None:
                    value = known_bias
                elif group == "target" and source_idx != query_idx:
                    value = target_other_bias
                elif source_axis == query_axis:
                    value = same_bias
                else:
                    value = cross_bias
                bias[query_idx, source_idx] = value
        return bias

    def _target_instant_stats(self, target_hist):
        mean = target_hist.mean(dim=1, keepdim=True)
        var = target_hist.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.target_instant_norm_eps)
        return {"mean": mean, "std": std}

    def _denorm_target_prediction(self, pred, stats):
        if stats is None:
            return pred
        return pred * stats["std"] + stats["mean"]

    def _norm_future_observed(self, future_observed, stats):
        if future_observed is None or stats is None or self.c_observed <= 0:
            return future_observed
        if future_observed.size(-1) != self.c_observed:
            raise ValueError(
                f"future_observed must have {self.c_observed} channels; "
                f"got {future_observed.size(-1)}"
            )
        out = future_observed.clone()
        for pos in range(self.c_observed):
            target_axis = pos % self.c_target
            out[:, :, pos] = (
                out[:, :, pos] - stats["mean"][:, :, target_axis]
            ) / stats["std"][:, :, target_axis]
        return out

    def _apply_target_instant_norm(self, x):
        """Normalize targets, cycle observed channels over target axes, and leave known values unchanged."""
        if not self.use_target_instant_norm:
            return x, None

        if self.c_target < 1:
            raise ValueError("Target instant normalization requires at least one target channel")

        out = x.clone()
        target_cols = self._index_to_list(self.target_index, self.input_size)
        target_hist = x[:, :, target_cols]
        stats = self._target_instant_stats(target_hist)
        out[:, :, target_cols] = (target_hist - stats["mean"]) / stats["std"]

        observed_cols = self._index_to_list(self.observed_index, self.input_size)
        for pos, col in enumerate(observed_cols):
            target_axis = pos % self.c_target
            out[:, :, col] = (
                x[:, :, col] - stats["mean"][:, :, target_axis]
            ) / stats["std"][:, :, target_axis]

        return out, stats

    def _add_variate_embedding(
        self,
        h_tgt,
        h_obs,
        h_knw,
    ):
        if self.variate_embed is None:
            return h_tgt, h_obs, h_knw
        start = 0
        h_tgt = h_tgt + self.variate_embed[start:start + self.c_target][None, None, :, :]
        start += self.c_target
        if h_obs is not None and self.c_observed > 0:
            h_obs = h_obs + self.variate_embed[start:start + self.c_observed][None, None, :, :]
        start += self.c_observed
        if h_knw is not None and self.c_known > 0:
            h_knw = h_knw + self.variate_embed[start:start + self.c_known][None, None, :, :]
        return h_tgt, h_obs, h_knw

    def _split_inputs(self, x, future_known):
        """
        x: [B, L, C]
        future_known: [B, pred_len, C_known] or None
        """
        target_hist = x[:, :, self.target_index]
        observed_hist = x[:, :, self.observed_index] if self.c_observed > 0 else None
        known_hist = x[:, :, self.known_index] if self.c_known > 0 else None

        if self.c_known > 0:
            if future_known is None:
                # Repeat the final known value when future covariates are absent.
                last = known_hist[:, -1:, :]
                future_known = last.expand(-1, self.pred_len, -1)
            elif future_known.size(-1) != self.c_known:
                raise ValueError(
                    f"future_known must have {self.c_known} channels; got {future_known.size(-1)}"
                )
            known_full = torch.cat([known_hist, future_known], dim=1)
        else:
            known_full = None

        return target_hist, observed_hist, known_full

    def _encode_tokens(
        self,
        target_seq,
        observed_seq,
        known_full,
        return_attn=False,
        attention_intervention=None,
        cross_time_capture="last_query",
    ):
        """
        target_seq:   [B, Lt, Ct]
        observed_seq: [B, Lo, Co] or None
        known_full:   [B, Lk, Ck] or None
        return:
            h_tgt: [B, Nt, Ct, D]
            details: dict of attention detail lists
        """
        h_tgt = self.patch_embed(target_seq)
        h_obs = self.patch_embed(observed_seq) if observed_seq is not None and self.c_observed > 0 else None
        h_knw = self.patch_embed(known_full) if known_full is not None and self.c_known > 0 else None
        h_tgt, h_obs, h_knw = self._add_variate_embedding(h_tgt, h_obs, h_knw)

        cross_variate_keys = (
            "attn_weights",
            "content_scores",
            "axis_bias",
            "final_scores",
            "raw_scores",
            "smoothed_scores",
        )
        details = {
            "cross_time": [],
            "cross_variate": {key: [] for key in cross_variate_keys},
        }
        for key in cross_variate_keys:
            details[key] = details["cross_variate"][key]

        for layer_index, layer in enumerate(self.layers):
            h_tgt, h_obs, h_knw, layer_detail = layer(
                h_tgt,
                h_obs,
                h_knw,
                return_attn=return_attn,
                attention_intervention=attention_intervention,
                layer_index=layer_index,
                cross_time_capture=cross_time_capture,
            )
            if return_attn and layer_detail is not None:
                details["cross_time"].append(layer_detail["cross_time"])
                cross_variate_detail = layer_detail["cross_variate"]
                if cross_variate_detail is not None:
                    for key in cross_variate_keys:
                        details["cross_variate"][key].append(cross_variate_detail[key])
        return h_tgt, details

    def _predict_next_patch(
        self,
        target_seq,
        observed_seq,
        known_full,
        return_attn=False,
        attention_intervention=None,
        cross_time_capture="last_query",
    ):
        """Predict the next target patch from the final encoded token."""
        h_tgt, details = self._encode_tokens(
            target_seq,
            observed_seq,
            known_full,
            return_attn=return_attn,
            attention_intervention=attention_intervention,
            cross_time_capture=cross_time_capture,
        )
        last_token = h_tgt[:, -1]
        next_patch = self.projection(last_token)
        next_patch = next_patch.permute(0, 2, 1).contiguous()
        return next_patch, details

    def forward(
        self,
        x,
        future_known=None,
        future_observed=None,
        return_attn=None,
        attention_capture=None,
        cross_time_capture="last_query",
        attention_intervention=None,
    ):
        """Forecast ``pred_len`` target steps by recursive patch decoding.

        ``future_known`` and ``future_observed`` use shape
        ``[batch, pred_len, channels]``. ``attention_capture`` accepts
        ``last_rollout`` or ``all_rollouts``; cross-time capture accepts
        ``last_query`` or ``full``.
        """
        if x.dim() != 3:
            raise ValueError(f"x must have shape [B, L, C]; got {tuple(x.shape)}")
        if x.size(-1) != self.input_size:
            raise ValueError(f"x must have {self.input_size} channels; got {x.size(-1)}")

        return_attn = self.return_attn_default if return_attn is None else _as_bool(return_attn)
        if attention_capture not in {None, "last_rollout", "all_rollouts"}:
            raise ValueError(
                f"Unsupported attention_capture={attention_capture!r}; "
                "expected None, 'last_rollout', or 'all_rollouts'"
            )
        capture_all_rollouts = attention_capture == "all_rollouts"
        collect_attention = return_attn or attention_capture in {"last_rollout", "all_rollouts"}
        x, norm_stats = self._apply_target_instant_norm(x)
        future_observed = self._norm_future_observed(future_observed, norm_stats)
        target_hist, observed_hist, known_full = self._split_inputs(x, future_known)
        cur_target = target_hist
        cur_observed = observed_hist
        preds = []
        last_details = {}
        rollout_attention = []

        generated = 0
        while generated < self.pred_len:
            next_patch, details = self._predict_next_patch(
                cur_target,
                cur_observed,
                known_full,
                return_attn=collect_attention,
                attention_intervention=attention_intervention,
                cross_time_capture=cross_time_capture,
            )
            take = min(self.patch_len, self.pred_len - generated)
            preds.append(next_patch[:, :take, :])
            if capture_all_rollouts:
                rollout_attention.append({
                    "lead_start": generated + 1,
                    "lead_end": generated + take,
                    "cross_time": details["cross_time"],
                    "cross_variate": details["cross_variate"],
                })

            # Append the full patch before the next decoding step.
            cur_target = torch.cat([cur_target, next_patch], dim=1)

            # Use zeros when future observed covariates are unavailable.
            if self.c_observed > 0:
                if future_observed is not None:
                    obs_patch = future_observed[:, generated:generated + self.patch_len, :]
                    if obs_patch.size(1) < self.patch_len:
                        pad_len = self.patch_len - obs_patch.size(1)
                        pad = torch.zeros(
                            obs_patch.size(0), pad_len, obs_patch.size(2),
                            dtype=obs_patch.dtype, device=obs_patch.device,
                        )
                        obs_patch = torch.cat([obs_patch, pad], dim=1)
                else:
                    obs_patch = torch.zeros(
                        x.size(0), self.patch_len, self.c_observed,
                        dtype=x.dtype, device=x.device,
                    )
                cur_observed = torch.cat([cur_observed, obs_patch], dim=1) if cur_observed is not None else obs_patch

            generated += take
            if collect_attention:
                last_details = details

        pred = torch.cat(preds, dim=1)[:, :self.pred_len, :]
        pred = self._denorm_target_prediction(pred, norm_stats)
        if collect_attention:
            outputs = {"pred": pred, **last_details}
            if capture_all_rollouts:
                outputs["rollout_attention"] = rollout_attention
            return outputs
        return pred

    @torch.no_grad()
    def forward_next_patches(
        self,
        x,
        future_known=None,
        future_observed=None,
        return_attn=False,
    ):
        """Predict one next patch per historical target token."""
        x, norm_stats = self._apply_target_instant_norm(x)
        future_observed = self._norm_future_observed(future_observed, norm_stats)
        target_hist, observed_hist, known_full = self._split_inputs(x, future_known)
        h_tgt, details = self._encode_tokens(
            target_hist,
            observed_hist,
            known_full,
            return_attn=return_attn,
        )
        out = self.projection(h_tgt)
        out = out.permute(0, 1, 3, 2).contiguous()
        if norm_stats is not None:
            out = out * norm_stats["std"].unsqueeze(1) + norm_stats["mean"].unsqueeze(1)
        if return_attn:
            return {"next_patches": out, **details}
        return out
