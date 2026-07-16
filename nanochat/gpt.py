"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"

    # Experimental two-pass causal routing loop. Pass 1 discovers deep Q/K routing
    # features; pass 2 reuses them as an additive attention-logit prior.
    routing_loop: bool = False
    # Connectivity from pass-1 layers to pass-2 layers:
    # last | same | reverse | progressive | offset:<fraction-or-layers>
    routing_pattern: str = "progressive"
    routing_gate_init: float = 0.05
    routing_detach: bool = False
    first_pass_loss_weight: float = 0.1


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

def build_routing_source_map(pattern, n_layer):
    """Map each pass-2 layer to one pass-1 source layer.

    Keeping one source per target lets us represent the routing prior exactly by
    concatenating source Q/K features to pass-2 Q/K, without storing an NxN map.
    """
    if n_layer <= 0:
        return []
    pattern = pattern.lower().strip()
    if pattern == "last":
        return [n_layer - 1] * n_layer
    if pattern == "same":
        return list(range(n_layer))
    if pattern == "reverse":
        return list(reversed(range(n_layer)))
    if pattern == "progressive":
        # Three coarse abstraction levels: middle, upper-middle, final.
        anchors = sorted(set([
            round(0.50 * (n_layer - 1)),
            round(0.75 * (n_layer - 1)),
            n_layer - 1,
        ]))
        return [anchors[min(len(anchors) - 1, (i * len(anchors)) // n_layer)] for i in range(n_layer)]
    if pattern.startswith("offset:"):
        raw = pattern.split(":", 1)[1]
        value = float(raw)
        offset = round(value * (n_layer - 1)) if abs(value) < 1 else round(value)
        return [min(n_layer - 1, max(0, i + offset)) for i in range(n_layer)]
    raise ValueError(f"Unknown routing_pattern={pattern!r}")

def inverse_softplus(x):
    # Stable enough for the small positive initialization values used here.
    x = max(float(x), 1e-8)
    return torch.log(torch.expm1(torch.as_tensor(x, dtype=torch.float32)))

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(
        self, x, ve, cos_sin, window_size, kv_cache,
        capture_routing=False, routing_qk=None, routing_gate_logit=None,
    ):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2
        captured_qk = (q, k) if capture_routing else None

        if routing_qk is None:
            if kv_cache is None:
                y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
            else:
                k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
                y = flash_attn.flash_attn_with_kvcache(
                    q, k_cache, v_cache, k=k, v=v,
                    cache_seqlens=kv_cache.cache_seqlens,
                    causal=True, window_size=window_size,
                )
                if self.layer_idx == kv_cache.n_layers - 1:
                    kv_cache.advance(T)
        else:
            # Exact low-rank additive bias without an NxN attention map. For D=head_dim:
            #   softmax((q2 k2^T + g qr kr^T) / sqrt(D))
            # equals ordinary attention over concatenated, rescaled Q/K of width 2D.
            route_q, route_k = routing_qk
            gate = F.softplus(routing_gate_logit.float()).to(dtype=q.dtype)
            base_scale = 2.0 ** 0.25
            route_scale = base_scale * torch.sqrt(gate.clamp_min(1e-8))
            q_aug = torch.cat((base_scale * q, route_scale * route_q), dim=-1)

            if kv_cache is None:
                k_aug = torch.cat((base_scale * k, route_scale * route_k), dim=-1)
                y = flash_attn.sdpa_attn_func(
                    q_aug, k_aug, v, causal=True, window_size=window_size
                )
            else:
                # Keep a normal pass-2 KV cache. We only concatenate the full keys
                # transiently, reading route_k from the pass-1 source-layer cache.
                pos = kv_cache.get_pos()
                k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
                k_cache[:, pos:pos+T] = k
                v_cache[:, pos:pos+T] = v
                end_pos = pos + T
                k_full = k_cache[:, :end_pos]
                v_full = v_cache[:, :end_pos]
                assert route_k.size(1) == end_pos, (route_k.shape, end_pos)
                k_aug = torch.cat((base_scale * k_full, route_scale * route_k), dim=-1)
                y = flash_attn.sdpa_attn_func(
                    q_aug, k_aug, v_full, causal=True, window_size=window_size
                )
                if self.layer_idx == kv_cache.n_layers - 1:
                    kv_cache.advance(T)

        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y, captured_qk


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(
        self, x, ve, cos_sin, window_size, kv_cache,
        capture_routing=False, routing_qk=None, routing_gate_logit=None,
    ):
        attn_out, captured_qk = self.attn(
            norm(x), ve, cos_sin, window_size, kv_cache,
            capture_routing=capture_routing,
            routing_qk=routing_qk,
            routing_gate_logit=routing_gate_logit,
        )
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x, captured_qk


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        if config.routing_loop:
            self.routing_source_for_target = build_routing_source_map(config.routing_pattern, config.n_layer)
            self.routing_source_layers = tuple(sorted(set(self.routing_source_for_target)))
            self.routing_gate_logits = nn.Parameter(torch.zeros(config.n_layer))
            self.pass_embeddings = nn.Parameter(torch.zeros(2, config.n_embd))
        else:
            self.routing_source_for_target = []
            self.routing_source_layers = ()
            self.register_parameter("routing_gate_logits", None)
            self.register_parameter("pass_embeddings", None)
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized 
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)
        if self.config.routing_loop:
            torch.nn.init.zeros_(self.pass_embeddings)
            gate_logit = inverse_softplus(self.config.routing_gate_init).item()
            torch.nn.init.constant_(self.routing_gate_logits, gate_logit)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        if self.config.routing_loop:
            # Two trunks; pass-2 QK width is doubled by the routing features.
            num_flops_per_token = 12 * self.num_matmul_params() + 3 * attn_flops
        else:
            num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = (4 * self.num_matmul_params() + 3 * attn_flops) if self.config.routing_loop else (2 * self.num_matmul_params() + attn_flops)
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = (4 * self.num_matmul_params() * num_tokens + 3 * attn_flops) if self.config.routing_loop else (2 * self.num_matmul_params() * num_tokens + attn_flops)
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        multiplier = 2 if self.config.routing_loop else 1
        return multiplier * self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        # Pass 1 reads K/V; pass 2 reads its own K/V plus the selected pass-1 K.
        return total * (2.5 if self.config.routing_loop else 1)

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        routing = 0 if not self.config.routing_loop else self.routing_gate_logits.numel() + self.pass_embeddings.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars + routing
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'routing': routing,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        routing_params = [] if not self.config.routing_loop else [self.routing_gate_logits, self.pass_embeddings]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params) + len(routing_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        if routing_params:
            param_groups.append(dict(kind='adamw', params=routing_params, lr=scalar_lr * 0.1, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def _embed_tokens(self, idx, kv_cache, pass_idx):
        B, T = idx.size()
        x = self.transformer.wte(idx).to(COMPUTE_DTYPE)
        x = norm(x)
        if self.config.routing_loop:
            x = x + self.pass_embeddings[pass_idx].to(x.dtype)

        if kv_cache is None:
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear
        return x

    def _run_pass(self, idx, cos_sin, pass_idx, kv_cache=None, routing_states=None, pass1_cache=None):
        x = self._embed_tokens(idx, kv_cache, pass_idx)
        x0 = x
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2
        x_backout = None
        captured = {}
        source_set = set(self.routing_source_layers)

        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            routing_qk = None
            routing_gate_logit = None
            if pass_idx == 1:
                source = self.routing_source_for_target[i]
                route_q, route_k_current = routing_states[source]
                if self.config.routing_detach:
                    route_q = route_q.detach()
                    route_k_current = route_k_current.detach()
                if kv_cache is None:
                    route_k = route_k_current
                else:
                    route_k_cache, _ = pass1_cache.get_layer_cache(source)
                    route_k = route_k_cache[:, :pass1_cache.get_pos()]
                    if self.config.routing_detach:
                        route_k = route_k.detach()
                routing_qk = (route_q, route_k)
                routing_gate_logit = self.routing_gate_logits[i]

            x, route = block(
                x, ve, cos_sin, self.window_sizes[i], kv_cache,
                capture_routing=(pass_idx == 0 and i in source_set),
                routing_qk=routing_qk, routing_gate_logit=routing_gate_logit,
            )
            if route is not None:
                captured[i] = route
            if i == backout_layer:
                x_backout = x

        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        return norm(x), captured

    def _logits(self, x):
        softcap = 15
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        return softcap * torch.tanh(logits / softcap)

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_pass_losses=False):
        B, T = idx.size()
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device
        assert self.cos.dtype == COMPUTE_DTYPE

        if self.config.routing_loop and kv_cache is not None:
            assert hasattr(kv_cache, "pass1") and hasattr(kv_cache, "pass2"), "routing_loop requires RoutingLoopKVCache"
            assert kv_cache.pass1.get_pos() == kv_cache.pass2.get_pos()
            T0 = kv_cache.pass1.get_pos()
            cache1, cache2 = kv_cache.pass1, kv_cache.pass2
        else:
            T0 = 0 if kv_cache is None else kv_cache.get_pos()
            cache1, cache2 = kv_cache, None
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

        if not self.config.routing_loop:
            x, _ = self._run_pass(idx, cos_sin, pass_idx=0, kv_cache=cache1)
            logits = self._logits(x)
            if targets is None:
                return logits
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

        x1, routing_states = self._run_pass(idx, cos_sin, pass_idx=0, kv_cache=cache1)
        x2, _ = self._run_pass(
            idx, cos_sin, pass_idx=1, kv_cache=cache2,
            routing_states=routing_states, pass1_cache=cache1,
        )
        logits2 = self._logits(x2)
        if targets is None:
            return logits2

        loss2 = F.cross_entropy(logits2.view(-1, logits2.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        need_loss1 = self.config.first_pass_loss_weight > 0 and (self.training or return_pass_losses)
        if need_loss1:
            logits1 = self._logits(x1)
            loss1 = F.cross_entropy(logits1.view(-1, logits1.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        else:
            loss1 = loss2.detach() * 0
        # Validation/evaluation should report the final pass only so BPB remains comparable.
        loss = loss2 + self.config.first_pass_loss_weight * loss1 if self.training else loss2
        if return_pass_losses:
            return loss, {"pass1": loss1.detach(), "pass2": loss2.detach()}
        return loss

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
