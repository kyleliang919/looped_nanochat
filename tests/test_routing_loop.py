import math

import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig, build_routing_source_map
from nanochat.engine import RoutingLoopKVCache
from nanochat.flash_attention import flash_attn


def _explicit_causal_attention(q, k, v, route_q, route_k, gate):
    """Reference implementation for the additive routing prior."""
    # Inputs are nanochat layout: (B, T, H, D).
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    route_q = route_q.transpose(1, 2)
    route_k = route_k.transpose(1, 2)
    d = q.size(-1)
    logits = q @ k.transpose(-2, -1) / math.sqrt(d)
    logits = logits + gate * (route_q @ route_k.transpose(-2, -1) / math.sqrt(d))
    t = q.size(-2)
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    logits = logits.masked_fill(~causal, float("-inf"))
    return (F.softmax(logits, dim=-1) @ v).transpose(1, 2)


def test_routing_source_maps():
    assert build_routing_source_map("last", 4) == [3, 3, 3, 3]
    assert build_routing_source_map("same", 4) == [0, 1, 2, 3]
    assert build_routing_source_map("reverse", 4) == [3, 2, 1, 0]
    assert build_routing_source_map("offset:1", 4) == [1, 2, 3, 3]
    assert len(build_routing_source_map("progressive", 12)) == 12


def test_augmented_qk_matches_additive_bias():
    torch.manual_seed(0)
    b, t, h, d = 2, 7, 3, 8
    q = torch.randn(b, t, h, d)
    k = torch.randn(b, t, h, d)
    v = torch.randn(b, t, h, d)
    route_q = torch.randn(b, t, h, d)
    route_k = torch.randn(b, t, h, d)
    gate = torch.tensor(0.37)

    # Same rescaling used by nanochat.gpt.CausalSelfAttention.
    base_scale = 2.0 ** 0.25
    route_scale = base_scale * torch.sqrt(gate)
    q_aug = torch.cat((base_scale * q, route_scale * route_q), dim=-1)
    k_aug = torch.cat((base_scale * k, route_scale * route_k), dim=-1)
    actual = flash_attn.sdpa_attn_func(q_aug, k_aug, v, causal=True, window_size=(-1, 0))
    expected = _explicit_causal_attention(q, k, v, route_q, route_k, gate)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def _tiny_model(routing_detach=False):
    config = GPTConfig(
        sequence_len=32,
        vocab_size=64,
        n_layer=4,
        n_head=4,
        n_kv_head=4,
        n_embd=32,
        window_pattern="L",
        routing_loop=True,
        routing_pattern="progressive",
        routing_gate_init=0.1,
        routing_detach=routing_detach,
        first_pass_loss_weight=0.0,
    )
    model = GPT(config, pad_vocab_size_to=1)
    model.init_weights()
    # nanochat intentionally zero-initializes output projections. Make this unit
    # test exercise routing gradients immediately instead of after optimizer step 1.
    with torch.no_grad():
        for block in model.transformer.h:
            torch.nn.init.normal_(block.attn.c_proj.weight, std=0.02)
            torch.nn.init.normal_(block.mlp.c_proj.weight, std=0.02)
    return model


def test_routing_loop_is_end_to_end_differentiable():
    torch.manual_seed(1)
    model = _tiny_model(routing_detach=False)
    idx = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))
    loss = model(idx, targets)
    loss.backward()

    assert model.routing_gate_logits.grad is not None
    assert torch.isfinite(model.routing_gate_logits.grad).all()
    assert model.routing_gate_logits.grad.abs().sum() > 0

    source = model.routing_source_layers[-1]
    source_q_grad = model.transformer.h[source].attn.c_q.weight.grad
    assert source_q_grad is not None
    assert source_q_grad.abs().sum() > 0


def test_cached_prefill_and_decode_match_full_forward():
    torch.manual_seed(2)
    model = _tiny_model().eval()
    ids = torch.randint(0, model.config.vocab_size, (1, 7))
    m = model.config
    cache_kwargs = dict(
        batch_size=1,
        num_heads=m.n_kv_head,
        seq_len=16,
        head_dim=m.n_embd // m.n_head,
        num_layers=m.n_layer,
        device=model.get_device(),
        dtype=model.transformer.wte.weight.dtype,
    )

    with torch.no_grad():
        full = model(ids)

        cache = RoutingLoopKVCache(**cache_kwargs)
        prefill = model(ids, kv_cache=cache)
        torch.testing.assert_close(prefill, full, atol=3e-5, rtol=3e-5)

        cache = RoutingLoopKVCache(**cache_kwargs)
        model(ids[:, :-1], kv_cache=cache)
        decoded_last = model(ids[:, -1:], kv_cache=cache)
        torch.testing.assert_close(decoded_last, full[:, -1:], atol=3e-5, rtol=3e-5)
