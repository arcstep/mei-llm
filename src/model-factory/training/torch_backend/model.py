"""PyTorch numerical backend of the canonical MLX SAN backbone.

Full-sequence training and explicit activation/KV Q/DQ; no deployment cache
or SDK parity claim. Parameter names and shapes match the MLX float model.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from .inference_math import InferenceLinear, linear as inference_linear

ARCH = Path(__file__).resolve().parents[3] / "architecture/mei-1.2-51m"
_spec = importlib.util.spec_from_file_location("mei_vocab_reference_config", ARCH / "config.py")
_config = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _config
_spec.loader.exec_module(_config)
NeedleZhConfig = _config.NeedleZhConfig


def normal(*shape):
    return nn.Parameter(torch.randn(shape) * 0.02)


def linear(n, m, std=0.02, bias=False):
    layer = InferenceLinear(n, m, bias=bias)
    nn.init.normal_(layer.weight, std=std)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


def rms_unit(x, eps=1e-6):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def shift_right(x, offset):
    if offset == 0:
        return x
    if offset >= x.shape[-2]:
        return torch.zeros_like(x)
    return torch.cat((torch.zeros_like(x[..., :offset, :]), x[..., :-offset, :]), dim=-2)


def walsh_hadamard(x):
    n = x.shape[-1]
    if n <= 0 or n & (n - 1):
        raise ValueError("WHT requires a power of two")
    width = 1
    while width < n:
        pairs = x.reshape(*x.shape[:-1], n // (2 * width), 2, width)
        left, right = pairs[..., 0, :], pairs[..., 1, :]
        x = torch.cat((left + right, left - right), dim=-1).reshape(x.shape)
        width *= 2
    return x / math.sqrt(n)


def engram_indices(tokens, orders, heads, slots):
    # Explicit uint32 wrap: signed int64 multiplication itself cannot overflow
    # here (the largest product is below 2**57). CUDA supports these operations.
    cols = []
    for oi, order in enumerate(orders):
        for head in range(heads):
            acc = torch.full_like(tokens, (0x9E3779B9 * (oi * heads + head + 1)) & 0xFFFFFFFF)
            for j in range(order):
                shifted = shift_right(tokens[..., None], j)[..., 0]
                acc = ((acc ^ shifted) * 0x01000193) & 0xFFFFFFFF
            cols.append((acc ^ (acc >> 15)) % slots)
    return torch.stack(cols, dim=-1)


def sinkhorn(logits, iters=20):
    k = logits.float()
    for _ in range(iters):
        k = k - k.logsumexp(-1, keepdim=True)
        k = k - k.logsumexp(-2, keepdim=True)
    return k.exp()


class ZCRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return ((1 + self.scale) * xf / (xf.square().mean(-1, keepdim=True) + self.eps).sqrt()).to(x.dtype)


class FixedWalshHadamardMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d, self.n = d, 1 << (d - 1).bit_length()
        self.d1 = nn.Parameter(torch.ones(self.n))
        self.d2 = nn.Parameter(torch.ones(self.n))
        self.d3 = nn.Parameter(torch.full((self.n,), 0.02))

    def forward(self, x):
        z = F.pad(x, (0, self.n - self.d))
        z = walsh_hadamard(self.d1.to(z.dtype) * z)
        z = walsh_hadamard(F.silu(self.d2.to(z.dtype) * z))
        return (self.d3.to(z.dtype) * z)[..., :self.d]


class GroupedAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qat_activation_ste = False
        d, h = cfg.d_model, cfg.head_dim
        self.q_proj = linear(d, cfg.n_heads * h)
        self.k_proj = linear(d, cfg.n_kv_heads * h)
        self.v_proj = linear(d, cfg.n_kv_heads * h)
        self.gate_proj = linear(d, d)
        self.o_proj = linear(d, d, 0.02 / math.sqrt(2 * cfg.n_layers))
        self.q_norm, self.k_norm = ZCRMSNorm(h, cfg.rms_eps), ZCRMSNorm(h, cfg.rms_eps)

    def forward(self, x, rope, cache_context=None):
        if self.qat_activation_ste:
            from training.qat.cq2_torch_51m import activation_int8
            x = activation_int8(x)
        b, t, _ = x.shape
        cfg = self.cfg
        q = self.q_norm(self.q_proj(x).reshape(b, t, cfg.n_heads, cfg.head_dim).transpose(1, 2))
        k = self.k_norm(self.k_proj(x).reshape(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2))
        v = self.v_proj(x).reshape(b, t, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        cos, sin = rope
        def rotate(z):
            a, c = z.chunk(2, dim=-1)
            return torch.cat((a * cos - c * sin, c * cos + a * sin), dim=-1).to(z.dtype)
        q, k = rotate(q), rotate(k)
        int8_cache = cache_context is not None and cache_context["storage"] == "int8"
        if self.qat_activation_ste and not int8_cache:
            k, v = activation_int8(k), activation_int8(v)
        present = None
        if cache_context is not None:
            from .cache import active_cache_view, merge_cache
            k, v, present = merge_cache(k, v, cache_context["cache"],
                                        storage=cache_context["storage"], write_index=cache_context["write_index"])
            k, v, attention_mask = active_cache_view(k, v, cache_context["mask"])
        else:
            attention_mask = None
        repeat = cfg.n_heads // cfg.n_kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        if cache_context is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k.to(q.dtype), v.to(q.dtype),
                    attn_mask=attention_mask, is_causal=False)
        out = out.transpose(1, 2).reshape(b, t, cfg.d_model)
        out = out * self.gate_proj(x).sigmoid()
        if self.qat_activation_ste:
            out = activation_int8(out)
        result = self.o_proj(out)
        return (result, present) if cache_context is not None else result


class Engram(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.orders, self.slots = cfg.engram_orders, cfg.engram_slots
        self.qat_activation_ste = False
        self.heads = max(1, cfg.d_model // (len(self.orders) * 128))
        tables = len(self.orders) * self.heads
        sub_dim = max(1, cfg.d_model // tables)
        self.tables = normal(tables, self.slots, sub_dim)
        self.key_proj = linear(tables * sub_dim, cfg.d_model)
        self.value_proj = linear(tables * sub_dim, cfg.d_model, 0.02 / math.sqrt(2 * cfg.n_layers))
        self.taps = nn.Parameter(torch.cat((torch.ones(1, cfg.d_model), torch.zeros(cfg.engram_conv_taps - 1, cfg.d_model))))

    def forward(self, tokens):
        indices = engram_indices(tokens, self.orders, self.heads, self.slots)
        pos = torch.arange(tokens.shape[1], device=tokens.device)[None, :, None]
        fetched = []
        for i in range(self.tables.shape[0]):
            ok = pos >= self.orders[i // self.heads] - 1
            fetched.append(self.tables[i][indices[:, :, i]] * ok)
        e = torch.cat(fetched, dim=-1)
        if self.qat_activation_ste:
            from training.qat.cq2_torch_51m import activation_int8
            e = activation_int8(e)
        key, value = self.key_proj(e), self.value_proj(e)
        mixed = torch.zeros_like(value)
        for j in range(self.taps.shape[0]):
            offset = j * max(self.orders)
            mixed = mixed + self.taps[j] * shift_right(value, offset) * (pos >= offset)
        return key, mixed


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d = cfg.d_model
        self.attn_norm = ZCRMSNorm(self.d, cfg.rms_eps)
        self.attn = GroupedAttention(cfg)
        self.post_attn_norm = ZCRMSNorm(self.d, cfg.rms_eps)
        self.attn_gate = nn.Parameter(torch.tensor(0.0))
        self.mlp_norm = ZCRMSNorm(self.d, cfg.rms_eps)
        self.mlp = FixedWalshHadamardMLP(self.d)

    def forward(self, x, rope, engram=None, cache_context=None):
        if engram is not None:
            key, value = engram
            alpha = ((rms_unit(x) * rms_unit(key)).sum(-1) / math.sqrt(self.d)).sigmoid()
            x = x + (alpha[..., None] * value.float()).to(x.dtype)
        attention = self.attn(self.attn_norm(x), rope, cache_context=cache_context)
        present = None
        if cache_context is not None:
            attention, present = attention
        x = x + self.attn_gate.sigmoid().to(x.dtype) * self.post_attn_norm(attention)
        result = x + self.mlp(self.mlp_norm(x))
        return (result, present) if cache_context is not None else result


class NeedleZh(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = ZCRMSNorm(cfg.d_model, cfg.rms_eps)
        self.engrams = nn.ModuleList([Engram(cfg) for _ in cfg.engram_layers])
        n, d, layers = cfg.mhc_lanes, cfg.d_model, cfg.n_layers
        self.mhc_phi_pre, self.mhc_phi_post = normal(layers, n * d, n), normal(layers, n * d, n)
        self.mhc_phi_res = normal(layers, n * d, n * n)
        self.mhc_b_pre, self.mhc_b_post = nn.Parameter(torch.zeros(layers, n)), nn.Parameter(torch.zeros(layers, n))
        self.mhc_b_res = nn.Parameter((4 * torch.eye(n)).expand(layers, n, n).clone())
        self.mhc_a_pre = nn.Parameter(torch.full((layers,), 0.01))
        self.mhc_a_post = nn.Parameter(torch.full((layers,), 0.01))
        self.mhc_a_res = nn.Parameter(torch.full((layers,), 0.01))
        self.conf_probes = normal(cfg.conf_probes, d)
        self.conf_proj = linear(cfg.conf_probes * d, 1, bias=True)
        self.lm_head = None if cfg.tie_embeddings else linear(d, cfg.vocab_size)
        lane = torch.eye(n)[torch.arange(layers) % n]
        self.register_buffer("pre_off", 8 * lane - 4, persistent=False)
        self.register_buffer("post_off", -4 * (1 - lane), persistent=False)
        self.checkpoint_blocks = False
        self.qat_activation_ste = False

    def enable_qat(self, enabled=True):
        self.qat_activation_ste = bool(enabled)
        for block in self.blocks:
            block.attn.qat_activation_ste = bool(enabled)
        for engram in self.engrams:
            engram.qat_activation_ste = bool(enabled)

    def layer(self, i, lanes, rope, engram, cache_context=None):
        cfg, n = self.cfg, self.cfg.mhc_lanes
        b, t = lanes.shape[:2]
        # Explicit FP32 islands: .float() alone does not stop autocast matmul.
        # The block between these islands retains the outer precision policy.
        with torch.autocast(device_type=lanes.device.type, enabled=False):
            xf = lanes.float()
            nx = rms_unit(lanes.reshape(b, t, n * cfg.d_model))
            hpre = (self.mhc_a_pre[i] * (nx @ self.mhc_phi_pre[i].float()) + self.mhc_b_pre[i] + self.pre_off[i]).sigmoid()
            u = torch.einsum("btn,btnc->btc", hpre, xf).to(lanes.dtype)
        block = self.blocks[i](u, rope, engram, cache_context=cache_context)
        present = None
        if cache_context is not None:
            block, present = block
        delta = block - u
        with torch.autocast(device_type=lanes.device.type, enabled=False):
            hpost = 2 * (self.mhc_a_post[i] * (nx @ self.mhc_phi_post[i].float()) + self.mhc_b_post[i] + self.post_off[i]).sigmoid()
            residual = (nx @ self.mhc_phi_res[i].float()).reshape(b, t, n, n)
            hres = sinkhorn(self.mhc_a_res[i] * residual + self.mhc_b_res[i], cfg.sinkhorn_iters)
            result = (torch.einsum("btij,btjc->btic", hres, xf) + hpost[..., None] * delta.float()[:, :, None, :]).to(lanes.dtype)
        return (result, present) if cache_context is not None else result

    def forward(self, tokens, return_confidence=False, return_cells=False, *,
                position_ids=None, cache=None, cache_position_ids=None,
                cache_write_index=None, engram_prefix_ids=None, return_cache=False,
                cache_storage="float"):
        cfg = self.cfg
        b, t = tokens.shape
        cached = bool(return_cache or cache is not None)
        if cached and self.training:
            raise ValueError("mutable inference cache cannot enter a training forward")
        if cache_storage not in {"float", "int8"} or (cache_storage == "int8" and not self.qat_activation_ste):
            raise ValueError("int8 cache requires the explicit QAT activation/KV contract")
        if cache is not None and len(cache) != cfg.n_layers:
            raise ValueError("cache layer count differs")
        x = self.embed(tokens) * math.sqrt(cfg.d_model)
        dims = torch.arange(0, cfg.head_dim, 2, device=x.device).float()
        cache_len = 0 if cache is None else cache[0][0].shape[2]
        positions = (torch.arange(t, device=x.device) + cache_len if position_ids is None else
                     torch.as_tensor(position_ids, device=x.device, dtype=torch.long))
        if positions.shape != (t,) or (cached and (int(positions.min()) < 0 or int(positions.max()) >= cfg.max_seq_len)):
            raise ValueError("invalid inference positions")
        angles = torch.outer(positions.float(), 1 / (cfg.rope_theta ** (dims / cfg.head_dim)))
        rope = angles.cos()[None, None], angles.sin()[None, None]
        engram_tokens = tokens
        if engram_prefix_ids is not None:
            prefix = torch.as_tensor(engram_prefix_ids, device=tokens.device, dtype=tokens.dtype).reshape(b,-1)
            engram_tokens = torch.cat((prefix,tokens),dim=1)
        engrams = {layer: tuple(value[:,-t:,:] for value in self.engrams[site](engram_tokens))
                   for site, layer in enumerate(cfg.engram_layers)}
        mask = None
        if cached:
            if cache is not None and cache_position_ids is None:
                old_positions = torch.arange(cache_len, device=x.device)
            else:
                old_positions = torch.as_tensor([] if cache_position_ids is None else cache_position_ids,
                                                device=x.device, dtype=torch.long)
            key_positions = (old_positions if cache_write_index is not None else torch.cat((old_positions,positions)))
            if cache is None: key_positions = positions
            mask = (positions[:,None] >= key_positions[None,:])[None,None,:,:]
        new_cache = [] if cached else None
        lanes = x[:, :, None].expand(b, t, cfg.mhc_lanes, cfg.d_model)
        cells = [x] if return_cells else None
        for i in range(cfg.n_layers):
            if cached:
                context = {"cache": None if cache is None else cache[i], "storage":cache_storage,
                           "mask":mask, "write_index":cache_write_index}
                lanes, present = self.layer(i,lanes,rope,engrams.get(i),cache_context=context)
                new_cache.append(present)
            elif self.checkpoint_blocks and self.training:
                from torch.utils.checkpoint import checkpoint
                lanes = checkpoint(self.layer, i, lanes, rope, engrams.get(i), use_reentrant=False)
            else:
                lanes = self.layer(i, lanes, rope, engrams.get(i))
            if cells is not None:
                cells.append(lanes.mean(2))
        hidden = self.final_norm(lanes.mean(2))
        weight = self.embed.weight if cfg.tie_embeddings else self.lm_head.weight
        lm_input = hidden
        if self.qat_activation_ste:
            from training.qat.cq2_torch_51m import activation_int8
            lm_input = activation_int8(hidden)
        block_rows = getattr(self,"inference_block_rows",0) if not self.training and not torch.is_grad_enabled() else 0
        out = {"logits": inference_linear(lm_input.float(), weight,block_rows=block_rows), "hidden": hidden}
        if cached:
            out["cache"] = new_cache
        if cells is not None:
            out["cells"] = cells
        if return_confidence:
            last = hidden[:, -1]
            scores = ((last.float() @ self.conf_probes.T) / math.sqrt(cfg.d_model)).softmax(-1)
            pooled = (scores[:, :, None] * last[:, None, :]).flatten(1)
            out["confidence_logit"] = self.conf_proj(pooled).squeeze(-1).float()
        return out
