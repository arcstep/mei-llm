"""Decode-only MLX/Metal operators for the exact MEI 1.0 51M contract.

The numerical oracle remains :mod:`architecture`.  These operators cover the
fixed deployed decode shape (batch=1, token=1, d_model=512, four mHC lanes)
without reintroducing the historical dense Hadamard matrix.  Unsupported
shapes must fall back to the reference graph.
"""

from __future__ import annotations

import mlx.core as mx

_D_MODEL = 512
_LANES = 4
_MHC_WIDTH = _D_MODEL * _LANES
_WHT_SCALE = 0.04419417382415922


def _kernel(name: str, inputs: list[str], outputs: list[str], source: str):
    kwargs = {
        "name": name,
        "input_names": inputs,
        "output_names": outputs,
        "source": source,
    }
    try:
        return mx.fast.metal_kernel(
            **kwargs,
            compile_options={"math_mode": "safe"},
        )
    except TypeError:
        return mx.fast.metal_kernel(**kwargs)


def fused_zcrms_norm(x: mx.array, scale: mx.array, eps: float) -> mx.array:
    """Use MLX's fused RMS kernel with this model's zero-centred scale."""

    return mx.fast.rms_norm(x, (1.0 + scale).astype(x.dtype), eps)


def fused_gqa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    mask: mx.array | str | None,
) -> mx.array:
    """Run native 8Q/4KV grouped-query scaled-dot-product attention."""

    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


_QK_NORM_ROPE_SOURCE = r"""
    uint group = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    bool is_q = group < 8;
    uint head = is_q ? group : group - 8;
    uint base = head * 64;
    float a = is_q ? float(q_raw[base + lane]) : float(k_raw[base + lane]);
    float b = is_q
        ? float(q_raw[base + lane + 32])
        : float(k_raw[base + lane + 32]);
    float inv_rms = metal::rsqrt(
        simd_sum(a * a + b * b) / 64.0f + 1e-6f);
    float scale_a = is_q ? float(q_scale[lane]) : float(k_scale[lane]);
    float scale_b = is_q
        ? float(q_scale[lane + 32])
        : float(k_scale[lane + 32]);
    a *= inv_rms * (1.0f + scale_a);
    b *= inv_rms * (1.0f + scale_b);

    uint rope_index = uint(position[0]) * 32 + lane;
    float c = float(rope_cos[rope_index]);
    float s = float(rope_sin[rope_index]);
    if (is_q) {
        q[base + lane] = T(a * c - b * s);
        q[base + lane + 32] = T(b * c + a * s);
    } else {
        k[base + lane] = T(a * c - b * s);
        k[base + lane + 32] = T(b * c + a * s);
    }
"""

_qk_norm_rope_kernel = None


def fused_qk_norm_rope(
    q_raw: mx.array,
    k_raw: mx.array,
    q_scale: mx.array,
    k_scale: mx.array,
    rope: tuple[mx.array, mx.array],
    position: mx.array,
) -> tuple[mx.array, mx.array]:
    global _qk_norm_rope_kernel
    if _qk_norm_rope_kernel is None:
        _qk_norm_rope_kernel = _kernel(
            "mei_51m_qk_norm_rope_8q4kv_f32",
            [
                "q_raw",
                "k_raw",
                "q_scale",
                "k_scale",
                "rope_cos",
                "rope_sin",
                "position",
            ],
            ["q", "k"],
            _QK_NORM_ROPE_SOURCE,
        )
    return tuple(
        _qk_norm_rope_kernel(
            inputs=[
                q_raw,
                k_raw,
                q_scale,
                k_scale,
                rope[0],
                rope[1],
                position.reshape(-1),
            ],
            template=[("T", mx.float32)],
            grid=(12 * 32, 1, 1),
            threadgroup=(32, 1, 1),
            output_shapes=[(1, 8, 1, 64), (1, 4, 1, 64)],
            output_dtypes=[q_raw.dtype, k_raw.dtype],
        )
    )


_FIXED_WHT_MLP_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float values[512];
    values[tid] = float(x[tid]) * float(d1[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint width = 1; width < 512; width <<= 1) {
        uint base = tid & ~(2 * width - 1);
        uint lane = tid & (width - 1);
        float left = values[base + lane];
        float right = values[base + width + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[tid] = (tid & width) ? left - right : left + right;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float z = values[tid] * 0.04419417382415922f * float(d2[tid]);
    values[tid] = z / (1.0f + metal::exp(-z));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint width = 1; width < 512; width <<= 1) {
        uint base = tid & ~(2 * width - 1);
        uint lane = tid & (width - 1);
        float left = values[base + lane];
        float right = values[base + width + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[tid] = (tid & width) ? left - right : left + right;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[tid] = T(values[tid] * 0.04419417382415922f * float(d3[tid]));
"""

_fixed_wht_mlp_kernel = None


def fused_fixed_wht_mlp(
    x: mx.array,
    d1: mx.array,
    d2: mx.array,
    d3: mx.array,
) -> mx.array:
    """Fuse the two algorithmic 512-wide WHT passes and diagonal gates."""

    global _fixed_wht_mlp_kernel
    if _fixed_wht_mlp_kernel is None:
        _fixed_wht_mlp_kernel = _kernel(
            "mei_51m_fixed_wht_mlp_512_f32",
            ["x", "d1", "d2", "d3"],
            ["out"],
            _FIXED_WHT_MLP_SOURCE,
        )
    return _fixed_wht_mlp_kernel(
        inputs=[x.reshape(-1), d1, d2, d3],
        template=[("T", mx.float32)],
        grid=(_D_MODEL, 1, 1),
        threadgroup=(_D_MODEL, 1, 1),
        output_shapes=[tuple(x.shape)],
        output_dtypes=[x.dtype],
    )[0]


_MHC_COEFF_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    threadgroup float dots[64];
    threadgroup float sums[64];
    float dot = 0.0f;
    float ss = 0.0f;
    for (uint i = tid; i < 2048; i += 64) {
        float value = float(lanes[i]);
        ss += value * value;
        dot += value * float(phi[group * 2048 + i]);
    }
    dots[tid] = dot;
    sums[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float dot_sum = 0.0f;
        float square_sum = 0.0f;
        for (uint i = 0; i < 64; ++i) {
            dot_sum += dots[i];
            square_sum += sums[i];
        }
        float normalized = dot_sum
            * metal::rsqrt(square_sum / 2048.0f + 1e-6f);
        if (group < 4) {
            float z = float(a_pre[0]) * normalized
                + float(b_pre[group]) + float(off_pre[group]);
            coeff[group] = 1.0f / (1.0f + metal::exp(-z));
        } else if (group < 8) {
            uint index = group - 4;
            float z = float(a_post[0]) * normalized
                + float(b_post[index]) + float(off_post[index]);
            coeff[group] = 2.0f / (1.0f + metal::exp(-z));
        } else {
            uint index = group - 8;
            coeff[group] = float(a_res[0]) * normalized + float(b_res[index]);
        }
    }
"""

_MHC_PRE_MIX_20_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float matrix[16];
    threadgroup float row_norm[4];
    threadgroup float col_norm[4];
    if (tid < 4) {
        hpost[tid] = coeff[4 + tid];
    }
    if (tid < 16) {
        matrix[tid] = coeff[8 + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint iteration = 0; iteration < 20; ++iteration) {
        if (tid < 4) {
            uint row = tid;
            float maximum = -INFINITY;
            for (uint col = 0; col < 4; ++col) {
                maximum = metal::max(maximum, matrix[row * 4 + col]);
            }
            float sum = 0.0f;
            for (uint col = 0; col < 4; ++col) {
                sum += metal::exp(matrix[row * 4 + col] - maximum);
            }
            row_norm[row] = maximum + metal::log(sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 16) {
            matrix[tid] -= row_norm[tid / 4];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 4) {
            uint col = tid;
            float maximum = -INFINITY;
            for (uint row = 0; row < 4; ++row) {
                maximum = metal::max(maximum, matrix[row * 4 + col]);
            }
            float sum = 0.0f;
            for (uint row = 0; row < 4; ++row) {
                sum += metal::exp(matrix[row * 4 + col] - maximum);
            }
            col_norm[col] = maximum + metal::log(sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 16) {
            matrix[tid] -= col_norm[tid % 4];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 16) {
        hres[tid] = metal::exp(matrix[tid]);
    }
    for (uint dim = tid; dim < 512; dim += 64) {
        float value = 0.0f;
        for (uint lane = 0; lane < 4; ++lane) {
            value += float(coeff[lane]) * float(lanes[lane * 512 + dim]);
        }
        u[dim] = T(value);
    }
"""

_mhc_coeff_kernel = None
_mhc_pre_mix_kernel = None


def fused_mhc_coeff(
    lanes: mx.array,
    phi: mx.array,
    a_pre: mx.array,
    a_post: mx.array,
    a_res: mx.array,
    b_pre: mx.array,
    b_post: mx.array,
    b_res: mx.array,
    off_pre: mx.array,
    off_post: mx.array,
) -> mx.array:
    global _mhc_coeff_kernel
    if _mhc_coeff_kernel is None:
        _mhc_coeff_kernel = _kernel(
            "mei_51m_mhc_coeff_4x512_f32",
            [
                "lanes",
                "phi",
                "a_pre",
                "a_post",
                "a_res",
                "b_pre",
                "b_post",
                "b_res",
                "off_pre",
                "off_post",
            ],
            ["coeff"],
            _MHC_COEFF_SOURCE,
        )
    return _mhc_coeff_kernel(
        inputs=[
            lanes,
            phi,
            a_pre.reshape(-1),
            a_post.reshape(-1),
            a_res.reshape(-1),
            b_pre,
            b_post,
            b_res,
            off_pre,
            off_post,
        ],
        template=[],
        grid=(24 * 64, 1, 1),
        threadgroup=(64, 1, 1),
        output_shapes=[(24,)],
        output_dtypes=[lanes.dtype],
    )[0]


def fused_mhc_pre_from_coeff(
    lanes: mx.array,
    coeff: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    global _mhc_pre_mix_kernel
    if _mhc_pre_mix_kernel is None:
        _mhc_pre_mix_kernel = _kernel(
            "mei_51m_mhc_pre_mix_sinkhorn20_4x512_f32",
            ["lanes", "coeff"],
            ["u", "hpost", "hres"],
            _MHC_PRE_MIX_20_SOURCE,
        )
    return tuple(
        _mhc_pre_mix_kernel(
            inputs=[lanes, coeff],
            template=[("T", mx.float32)],
            grid=(64, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(1, 1, _D_MODEL), (4,), (16,)],
            output_dtypes=[lanes.dtype, lanes.dtype, lanes.dtype],
        )
    )


def fused_mhc_pre(
    lanes: mx.array,
    phi: mx.array,
    a_pre: mx.array,
    a_post: mx.array,
    a_res: mx.array,
    b_pre: mx.array,
    b_post: mx.array,
    b_res: mx.array,
    off_pre: mx.array,
    off_post: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    coeff = fused_mhc_coeff(
        lanes,
        phi,
        a_pre,
        a_post,
        a_res,
        b_pre,
        b_post,
        b_res,
        off_pre,
        off_post,
    )
    return fused_mhc_pre_from_coeff(lanes, coeff)


_BLOCK_MHC_POST_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float reduction[512];
    threadgroup float state[512];
    threadgroup float values[512];
    threadgroup float inv_rms;

    float attn_value = float(attn[tid]);
    reduction[tid] = attn_value * attn_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 512; ++i) {
            total += reduction[i];
        }
        inv_rms = metal::rsqrt(total / 512.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gate = 1.0f / (1.0f + metal::exp(-float(attn_gate[0])));
    float state_value = float(skip[tid])
        + gate * attn_value * inv_rms * (1.0f + float(post_scale[tid]));
    state[tid] = state_value;
    reduction[tid] = state_value * state_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 512; ++i) {
            total += reduction[i];
        }
        inv_rms = metal::rsqrt(total / 512.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    values[tid] = state_value * inv_rms
        * (1.0f + float(mlp_scale[tid])) * float(d1[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 1; width < 512; width <<= 1) {
        uint base = tid & ~(2 * width - 1);
        uint lane = tid & (width - 1);
        float left = values[base + lane];
        float right = values[base + width + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[tid] = (tid & width) ? left - right : left + right;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float z = values[tid] * 0.04419417382415922f * float(d2[tid]);
    values[tid] = z / (1.0f + metal::exp(-z));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 1; width < 512; width <<= 1) {
        uint base = tid & ~(2 * width - 1);
        uint lane = tid & (width - 1);
        float left = values[base + lane];
        float right = values[base + width + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[tid] = (tid & width) ? left - right : left + right;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float delta = state_value
        + values[tid] * 0.04419417382415922f * float(d3[tid])
        - float(u[tid]);
    for (uint row = 0; row < 4; ++row) {
        float value = float(hpost[row]) * delta;
        for (uint col = 0; col < 4; ++col) {
            value += float(hres[row * 4 + col])
                * float(lanes[col * 512 + tid]);
        }
        out[row * 512 + tid] = T(value);
    }
"""

_block_mhc_post_kernel = None


def fused_block_mhc_post(
    skip: mx.array,
    attn: mx.array,
    attn_gate: mx.array,
    post_scale: mx.array,
    mlp_scale: mx.array,
    d1: mx.array,
    d2: mx.array,
    d3: mx.array,
    u: mx.array,
    lanes: mx.array,
    hpost: mx.array,
    hres: mx.array,
) -> mx.array:
    """Fuse the decode block tail, fixed WHT MLP and mHC post-mix."""

    global _block_mhc_post_kernel
    if _block_mhc_post_kernel is None:
        _block_mhc_post_kernel = _kernel(
            "mei_51m_fixed_wht_block_mhc_post_f32",
            [
                "skip",
                "attn",
                "attn_gate",
                "post_scale",
                "mlp_scale",
                "d1",
                "d2",
                "d3",
                "u",
                "lanes",
                "hpost",
                "hres",
            ],
            ["out"],
            _BLOCK_MHC_POST_SOURCE,
        )
    return _block_mhc_post_kernel(
        inputs=[
            skip,
            attn,
            attn_gate.reshape(-1),
            post_scale,
            mlp_scale,
            d1,
            d2,
            d3,
            u,
            lanes,
            hpost,
            hres,
        ],
        template=[("T", mx.float32)],
        grid=(_D_MODEL, 1, 1),
        threadgroup=(_D_MODEL, 1, 1),
        output_shapes=[(1, 1, _LANES, _D_MODEL)],
        output_dtypes=[skip.dtype],
    )[0]


_ENGRAM_TWO_FULL_TAPS_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint site = threadgroup_position_in_grid.x;
    threadgroup uint indices[16];
    if (tid < 16) {
        uint tap = tid >> 2;
        uint table = tid & 3;
        uint position = 12 - tap * 3;
        uint order = table < 2 ? 2 : 3;
        uint acc = 0x9E3779B9u * (table + 1);
        for (uint j = 0; j < order; ++j) {
            acc = (acc ^ uint(tokens[position - j])) * 0x01000193u;
        }
        acc = acc ^ (acc >> 15);
        indices[tid] = acc % 8192u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float key_acc = 0.0f;
    float value_acc = 0.0f;
    for (uint i = 0; i < 512; ++i) {
        uint table = i >> 7;
        uint sub = i & 127;
        float key_weight = site == 0
            ? float(key0[tid * 512 + i]) : float(key1[tid * 512 + i]);
        float value_weight = site == 0
            ? float(value0[tid * 512 + i]) : float(value1[tid * 512 + i]);
        for (uint tap = 0; tap < 4; ++tap) {
            uint table_index = (table * 8192 + indices[tap * 4 + table])
                * 128 + sub;
            float embedding = site == 0
                ? float(tables0[table_index]) : float(tables1[table_index]);
            if (tap == 0) {
                key_acc += embedding * key_weight;
            }
            float tap_weight = site == 0
                ? float(taps0[tap * 512 + tid])
                : float(taps1[tap * 512 + tid]);
            value_acc += tap_weight * embedding * value_weight;
        }
    }
    keys[site * 512 + tid] = T(key_acc);
    values[site * 512 + tid] = T(value_acc);
"""

_engram_two_full_taps_kernel = None


def fused_engram_two_full_taps(
    tokens: mx.array,
    first,
    second,
) -> tuple[mx.array, mx.array]:
    """Fuse both Engram sites with all four trained dilation-3 taps.

    ``tokens`` is the contract's 12-token history plus the current decode
    token.  The sampled value positions are 12, 9, 6 and 3, so every order-2/3 ngram
    is valid under the reference boundary mask.
    """

    global _engram_two_full_taps_kernel
    if _engram_two_full_taps_kernel is None:
        _engram_two_full_taps_kernel = _kernel(
            "mei_51m_engram_two_sites_four_taps_decode_f32",
            [
                "tokens",
                "tables0",
                "tables1",
                "key0",
                "key1",
                "value0",
                "value1",
                "taps0",
                "taps1",
            ],
            ["keys", "values"],
            _ENGRAM_TWO_FULL_TAPS_SOURCE,
        )
    keys, values = _engram_two_full_taps_kernel(
        inputs=[
            tokens.reshape(-1),
            first.tables,
            second.tables,
            first.key_proj.weight,
            second.key_proj.weight,
            first.value_proj.weight,
            second.value_proj.weight,
            first.taps,
            second.taps,
        ],
        template=[("T", mx.float32)],
        grid=(2 * _D_MODEL, 1, 1),
        threadgroup=(_D_MODEL, 1, 1),
        output_shapes=[(2, 1, 1, _D_MODEL), (2, 1, 1, _D_MODEL)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return keys, values


def supports_decode_fusion(
    *,
    batch: int,
    tokens: int,
    lanes: int,
    d_model: int,
    dtype,
) -> bool:
    return (
        batch == 1
        and tokens == 1
        and lanes == _LANES
        and d_model == _D_MODEL
        and dtype == mx.float32
    )
