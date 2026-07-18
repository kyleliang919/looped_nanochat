"""
Unified Flash Attention interface with automatic FA3/SDPA switching.

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
to PyTorch SDPA on incompatible CUDA GPUs, MPS, and CPU.

Usage (drop-in replacement for FA3):
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on CUDA GPUs
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3."""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are currently compiled for Hopper (sm90), Ada (sm89) and Ampere (sm80/sm86)
        # Blackwell (sm100) needs SDPA fallback until FA3 is recompiled or FA4 is released
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel, has_kernel
        # The varunneal kernel obtains better results for H100/Hopper
        if major == 9:
            hf_kernel = "varunneal/flash-attention-3"
            return get_kernel(hf_kernel).flash_attn_interface
        else:
            hf_kernel = "kernels-community/flash-attn3"
            if has_kernel(hf_kernel):
                return get_kernel(hf_kernel).flash_attn_interface
            else:
                return None

    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
_override_impl = None


def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


# =============================================================================
# SDPA helpers
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    q, k, v are (B, H, T, D) format.
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)


def sdpa_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """Always use PyTorch SDPA. Unlike the external FA3 wrapper, this supports
    Q/K head dimensions that differ from V, which the routing loop uses to encode
    an additive low-rank logit bias without materializing an attention map.
    """
    assert causal, "nanochat only uses this helper for causal attention"
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)


# =============================================================================
# FlexAttention routing path
# =============================================================================
# The routing loop needs softmax((q.k + gate*(rq.rk)) / sqrt(D)) @ v. Encoding
# that via width-2D concatenated Q/K forces plain SDPA off the fused kernel
# (2-3x slower). FlexAttention keeps a fused, compiled Triton kernel even at the
# wider head dim, so we route the training path through it. Equivalence:
#   qa = cat(q, rq), ka = cat(k, gate*rk) => qa.ka = q.k + gate*(rq.rk)
#   with softmax scale = 1/sqrt(D) this is exactly the additive-bias attention.
_flex_attention = None
_flex_block_mask_cache = {}


def _get_flex_attention():
    global _flex_attention
    if _flex_attention is None:
        from torch.nn.attention.flex_attention import flex_attention
        # dynamic=True so varying sequence lengths (as in eval over many tasks)
        # don't trigger a recompile per shape and fall back to the unfused
        # (full-scores-materializing) eager path.
        _flex_attention = torch.compile(flex_attention, dynamic=True)
    return _flex_attention


def _get_routing_block_mask(T, window, device):
    """Causal (optionally left-sliding-window) block mask, cached by (T, window)."""
    from torch.nn.attention.flex_attention import create_block_mask
    key = (T, window, str(device))
    bm = _flex_block_mask_cache.get(key)
    if bm is None:
        if window is None or window < 0 or window >= T:
            def mask_mod(b, h, qi, ki):
                return qi >= ki
        else:
            def mask_mod(b, h, qi, ki):
                return (qi >= ki) & (qi - ki <= window)
        bm = create_block_mask(mask_mod, None, None, T, T, device=device)
        _flex_block_mask_cache[key] = bm
    return bm


def flex_routing_attn_func(q, k, v, route_q, route_k, gate, window_size=(-1, -1)):
    """Causal routing attention via FlexAttention.

    q, k, v, route_q, route_k are (B, T, H, D) (route_k may share k's head count).
    gate is a scalar tensor. Computes, causally (with optional left window):
        softmax((q.k^T + gate * route_q.route_k^T) / sqrt(D)) @ v
    Returns (B, T, H, D).
    """
    import math
    B, T, H, D = q.shape
    # Fold the gate into the routing keys, then concatenate along head_dim so the
    # dot product carries the additive bias. Softmax scale stays 1/sqrt(D).
    q_aug = torch.cat((q, route_q), dim=-1)                    # (B, T, H, 2D)
    k_aug = torch.cat((k, gate.to(k.dtype) * route_k), dim=-1)  # (B, T, Hkv, 2D)
    # FlexAttention wants (B, H, T, D)
    q_aug = q_aug.transpose(1, 2)
    k_aug = k_aug.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q_aug.size(1) != k_aug.size(1)
    window = None if window_size[0] is None or window_size[0] < 0 else int(window_size[0])
    block_mask = _get_routing_block_mask(T, window, q.device)
    flex = _get_flex_attention()
    y = flex(q_aug, k_aug, v, block_mask=block_mask, scale=1.0 / math.sqrt(D),
             enable_gqa=enable_gqa)
    return y.transpose(1, 2)

# =============================================================================
# Public API: Same interface as FA3
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).

    Args:
        q, k, v: Tensors of shape (B, T, H, D)
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.

    Args:
        q: Queries, shape (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        causal: Whether to use causal masking
        window_size: (left, right) sliding window. -1 means unlimited.

    Returns:
        Output tensor of shape (B, T_new, H, D)
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
    sdpa_attn_func=sdpa_attn_func,
    flex_routing_attn_func=flex_routing_attn_func,
)
