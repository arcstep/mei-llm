"""Decode-only MLX/Metal fused operators for mei-1.0-58m.

The reference graph remains the numerical oracle. These kernels only cover the
fixed deployed decode shape (batch=1, token=1, d_model=512, four mHC lanes).
Unsupported shapes must fall back to architecture.py's reference operators.
"""

from __future__ import annotations

import mlx.core as mx

_D_MODEL = 512
_LANES = 4
_LANE_WIDTH = _D_MODEL * _LANES


def fused_zcrms_norm(x: mx.array, scale: mx.array, eps: float) -> mx.array:
    """Use MLX's fused RMS kernel with this model's zero-centred scale."""
    weight = (1.0 + scale).astype(x.dtype)
    return mx.fast.rms_norm(x, weight, eps)


def fused_gqa(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    *,
    scale: float,
    mask: mx.array | str | None,
) -> mx.array:
    """Fused SDPA supports native 8Q/4KV grouped-query attention."""
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)


_QKVG_ROPE_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    threadgroup float projected[64];
    threadgroup float inv_rms;

    if (group < 8) {
        uint base = group * 64;
        float acc = 0.0f;
        for (uint i = 0; i < 512; ++i) {
            acc += float(x[i]) * float(q_weight[(base + tid) * 512 + i]);
        }
        projected[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float ss = 0.0f;
            for (uint i = 0; i < 64; ++i) {
                ss += projected[i] * projected[i];
            }
            inv_rms = metal::rsqrt(ss / 64.0f + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        projected[tid] *= inv_rms * (1.0f + float(q_scale[tid]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint hidx = tid & 31;
        uint rope_idx = uint(position[0]) * 32 + hidx;
        float a = projected[hidx];
        float b = projected[hidx + 32];
        float c = float(rope_cos[rope_idx]);
        float s = float(rope_sin[rope_idx]);
        q[base + tid] = tid < 32 ? a * c - b * s : b * c + a * s;
    } else if (group < 12) {
        uint base = (group - 8) * 64;
        float acc = 0.0f;
        for (uint i = 0; i < 512; ++i) {
            acc += float(x[i]) * float(k_weight[(base + tid) * 512 + i]);
        }
        projected[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float ss = 0.0f;
            for (uint i = 0; i < 64; ++i) {
                ss += projected[i] * projected[i];
            }
            inv_rms = metal::rsqrt(ss / 64.0f + 1e-6f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        projected[tid] *= inv_rms * (1.0f + float(k_scale[tid]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint hidx = tid & 31;
        uint rope_idx = uint(position[0]) * 32 + hidx;
        float a = projected[hidx];
        float b = projected[hidx + 32];
        float c = float(rope_cos[rope_idx]);
        float s = float(rope_sin[rope_idx]);
        k[base + tid] = tid < 32 ? a * c - b * s : b * c + a * s;
    } else if (group < 16) {
        uint base = (group - 12) * 64;
        float acc = 0.0f;
        for (uint i = 0; i < 512; ++i) {
            acc += float(x[i]) * float(v_weight[(base + tid) * 512 + i]);
        }
        v[base + tid] = acc;
    } else {
        uint base = (group - 16) * 64;
        float acc = 0.0f;
        for (uint i = 0; i < 512; ++i) {
            acc += float(x[i]) * float(gate_weight[(base + tid) * 512 + i]);
        }
        gate[base + tid] = acc;
    }
"""


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


_MLP_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float hidden[512];

    float acc = 0.0f;
    for (uint i = 0; i < 512; ++i) {
        acc += float(x[i]) * float(d1[i]) * float(h[i * 512 + tid]);
    }
    float a = acc * float(d2[tid]);
    hidden[tid] = a / (1.0f + metal::exp(-a));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc2 = 0.0f;
    for (uint i = 0; i < 512; ++i) {
        acc2 += hidden[i] * float(h[i * 512 + tid]);
    }
    out[tid] = T(acc2 * float(d3[tid]));
"""

_MLP_MHC_POST_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    threadgroup float state_values[512];
    threadgroup float normed[512];
    threadgroup float hidden[512];
    threadgroup float partials[16];
    threadgroup float inv_rms;
    threadgroup float gate;

    float value = float(attn[tid]);
    float square_sum = simd_sum(value * value);
    if (lane == 0) {
        partials[simd_group] = square_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float partial = lane < 16 ? partials[lane] : 0.0f;
        float total = simd_sum(partial);
        if (lane == 0) {
            inv_rms = metal::rsqrt(total / 512.0f + 1e-6f);
            float z = float(attn_gate[0]);
            gate = 1.0f / (1.0f + metal::exp(-z));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float state_value = float(skip[tid])
        + gate * value * inv_rms * (1.0f + float(post_scale[tid]));
    state_values[tid] = state_value;

    square_sum = simd_sum(state_value * state_value);
    if (lane == 0) {
        partials[simd_group] = square_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float partial = lane < 16 ? partials[lane] : 0.0f;
        float total = simd_sum(partial);
        if (lane == 0) {
            inv_rms = metal::rsqrt(total / 512.0f + 1e-6f);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    normed[tid] = state_value * inv_rms * (1.0f + float(mlp_scale[tid]))
        * float(d1[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = 0; i < 512; ++i) {
        acc += normed[i] * float(h[i * 512 + tid]);
    }
    float a = acc * float(d2[tid]);
    hidden[tid] = a / (1.0f + metal::exp(-a));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc2 = 0.0f;
    for (uint i = 0; i < 512; ++i) {
        acc2 += hidden[i] * float(h[i * 512 + tid]);
    }
    float residual = state_values[tid] + acc2 * float(d3[tid])
        - float(u[tid]);
    for (uint row = 0; row < 4; ++row) {
        float value = float(hpost[row]) * residual;
        for (uint col = 0; col < 4; ++col) {
            value += float(hres[row * 4 + col])
                * float(lanes[col * 512 + tid]);
        }
        out[row * 512 + tid] = T(value);
    }
"""

_MLP_MHC_TRANSITION_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float hidden[512];
    threadgroup float scratch[512];
    threadgroup float next_lanes[2048];
    threadgroup float coeff[24];
    threadgroup float row_norm[4];
    threadgroup float col_norm[4];
    threadgroup float inv_rms;

    float acc = 0.0f;
    for (uint i = 0; i < 512; ++i) {
        acc += float(x[i]) * float(d1[i]) * float(h[i * 512 + tid]);
    }
    float activation = acc * float(d2[tid]);
    hidden[tid] = activation / (1.0f + metal::exp(-activation));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float acc2 = 0.0f;
    for (uint i = 0; i < 512; ++i) {
        acc2 += hidden[i] * float(h[i * 512 + tid]);
    }
    float residual = float(state[tid]) + acc2 * float(d3[tid])
        - float(u[tid]);
    float square_sum = 0.0f;
    for (uint row = 0; row < 4; ++row) {
        float value = float(hpost[row]) * residual;
        for (uint col = 0; col < 4; ++col) {
            value += float(hres[row * 4 + col])
                * float(lanes[col * 512 + tid]);
        }
        next_lanes[row * 512 + tid] = value;
        out_lanes[row * 512 + tid] = T(value);
        square_sum += value * value;
    }
    scratch[tid] = square_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 512; ++i) {
            total += scratch[i];
        }
        inv_rms = metal::rsqrt(total / 2048.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 384) {
        uint output = tid >> 4;
        uint lane = tid & 15;
        float dot = 0.0f;
        for (uint i = lane; i < 2048; i += 16) {
            dot += next_lanes[i] * float(phi[output * 2048 + i]);
        }
        scratch[tid] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 384 && (tid & 15) == 0) {
        uint output = tid >> 4;
        float dot = 0.0f;
        for (uint i = 0; i < 16; ++i) {
            dot += scratch[tid + i];
        }
        float normalized = dot * inv_rms;
        if (output < 4) {
            float z = float(a_pre[0]) * normalized
                + float(b_pre[output]) + float(off_pre[output]);
            coeff[output] = 1.0f / (1.0f + metal::exp(-z));
        } else if (output < 8) {
            uint index = output - 4;
            float z = float(a_post[0]) * normalized
                + float(b_post[index]) + float(off_post[index]);
            coeff[output] = 2.0f / (1.0f + metal::exp(-z));
        } else {
            uint index = output - 8;
            coeff[output] = float(a_res[0]) * normalized + float(b_res[index]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid < 4) {
        next_hpost[tid] = coeff[4 + tid];
    }
    if (tid < 16) {
        scratch[tid] = coeff[8 + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint iteration = 0; iteration < 8; ++iteration) {
        if (tid < 4) {
            uint row = tid;
            float maximum = -INFINITY;
            for (uint col = 0; col < 4; ++col) {
                maximum = metal::max(maximum, scratch[row * 4 + col]);
            }
            float sum = 0.0f;
            for (uint col = 0; col < 4; ++col) {
                sum += metal::exp(scratch[row * 4 + col] - maximum);
            }
            row_norm[row] = maximum + metal::log(sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 16) {
            scratch[tid] -= row_norm[tid / 4];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 4) {
            uint col = tid;
            float maximum = -INFINITY;
            for (uint row = 0; row < 4; ++row) {
                maximum = metal::max(maximum, scratch[row * 4 + col]);
            }
            float sum = 0.0f;
            for (uint row = 0; row < 4; ++row) {
                sum += metal::exp(scratch[row * 4 + col] - maximum);
            }
            col_norm[col] = maximum + metal::log(sum);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 16) {
            scratch[tid] -= col_norm[tid % 4];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 16) {
        next_hres[tid] = metal::exp(scratch[tid]);
    }
    float mixed = 0.0f;
    for (uint row = 0; row < 4; ++row) {
        mixed += coeff[row] * next_lanes[row * 512 + tid];
    }
    next_u[tid] = T(mixed);
"""


_MHC_PRE_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float reduction[512];
    threadgroup float inv_rms;
    threadgroup float gates[4];

    float ss = 0.0f;
    for (uint lane = 0; lane < 4; ++lane) {
        float value = float(lanes[lane * 512 + tid]);
        ss += value * value;
    }
    reduction[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        float total = 0.0f;
        for (uint i = 0; i < 512; ++i) {
            total += reduction[i];
        }
        inv_rms = metal::rsqrt(total / 2048.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint lane = 0; lane < 4; ++lane) {
        nx[lane * 512 + tid] =
            T(float(lanes[lane * 512 + tid]) * inv_rms);
    }
    if (tid < 4) {
        float dot = 0.0f;
        for (uint i = 0; i < 2048; ++i) {
            dot += float(lanes[i]) * inv_rms * float(phi[i * 4 + tid]);
        }
        float z = float(a[0]) * dot + float(b[tid]) + float(off[tid]);
        gates[tid] = 1.0f / (1.0f + metal::exp(-z));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mixed = 0.0f;
    for (uint lane = 0; lane < 4; ++lane) {
        mixed += gates[lane] * float(lanes[lane * 512 + tid]);
    }
    u[tid] = T(mixed);
"""


_MHC_POST_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float hpost[4];
    threadgroup float matrix[16];

    if (tid < 4) {
        float dot = 0.0f;
        for (uint i = 0; i < 2048; ++i) {
            dot += float(nx[i]) * float(phi_post[i * 4 + tid]);
        }
        float z = float(a_post[0]) * dot
            + float(b_post[tid]) + float(off_post[tid]);
        hpost[tid] = 2.0f / (1.0f + metal::exp(-z));
    }
    if (tid < 16) {
        float dot = 0.0f;
        for (uint i = 0; i < 2048; ++i) {
            dot += float(nx[i]) * float(phi_res[i * 16 + tid]);
        }
        matrix[tid] = float(a_res[0]) * dot + float(b_res[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid == 0) {
        for (uint iteration = 0; iteration < 8; ++iteration) {
            for (uint row = 0; row < 4; ++row) {
                float maximum = -INFINITY;
                for (uint col = 0; col < 4; ++col) {
                    maximum = metal::max(maximum, matrix[row * 4 + col]);
                }
                float sum = 0.0f;
                for (uint col = 0; col < 4; ++col) {
                    sum += metal::exp(matrix[row * 4 + col] - maximum);
                }
                float normalizer = maximum + metal::log(sum);
                for (uint col = 0; col < 4; ++col) {
                    matrix[row * 4 + col] -= normalizer;
                }
            }
            for (uint col = 0; col < 4; ++col) {
                float maximum = -INFINITY;
                for (uint row = 0; row < 4; ++row) {
                    maximum = metal::max(maximum, matrix[row * 4 + col]);
                }
                float sum = 0.0f;
                for (uint row = 0; row < 4; ++row) {
                    sum += metal::exp(matrix[row * 4 + col] - maximum);
                }
                float normalizer = maximum + metal::log(sum);
                for (uint row = 0; row < 4; ++row) {
                    matrix[row * 4 + col] -= normalizer;
                }
            }
        }
        for (uint i = 0; i < 16; ++i) {
            matrix[i] = metal::exp(matrix[i]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint row = 0; row < 4; ++row) {
        float value = hpost[row] * float(y[tid]);
        for (uint col = 0; col < 4; ++col) {
            value += matrix[row * 4 + col]
                * float(lanes[col * 512 + tid]);
        }
        out[row * 512 + tid] = T(value);
    }
"""

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


_MHC_PRE_MIX_SOURCE = r"""
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
    for (uint iteration = 0; iteration < 8; ++iteration) {
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
    float value = 0.0f;
    for (uint lane = 0; lane < 4; ++lane) {
        value += float(coeff[lane]) * float(lanes[lane * 512 + tid]);
    }
    u[tid] = T(value);
"""

_MHC_PERSISTENT_PRE_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    threadgroup float partial[64];
    threadgroup float inv_rms;
    threadgroup float dot_total;
    threadgroup float local_coeff[24];
    threadgroup float matrix[16];
    threadgroup float row_norm[4];
    threadgroup float col_norm[4];
    threadgroup uint ready;

    uint output = group;
    float dot = 0.0f;
    float ss = 0.0f;
    for (uint i = tid; i < 2048; i += 64) {
        float value = float(lanes[i]);
        dot += value * float(phi[output * 2048 + i]);
        ss += value * value;
    }
    partial[tid] = dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total_dot = 0.0f;
        for (uint i = 0; i < 64; ++i) {
            total_dot += partial[i];
        }
        dot_total = total_dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    partial[tid] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total_ss = 0.0f;
        for (uint i = 0; i < 64; ++i) {
            total_ss += partial[i];
        }
        inv_rms = metal::rsqrt(total_ss / 2048.0f + 1e-6f);
        float normalized = dot_total * inv_rms;
        float result;
        if (output < 4) {
            float z = float(a_pre[0]) * normalized
                + float(b_pre[output]) + float(off_pre[output]);
            result = 1.0f / (1.0f + metal::exp(-z));
        } else if (output < 8) {
            uint index = output - 4;
            float z = float(a_post[0]) * normalized
                + float(b_post[index]) + float(off_post[index]);
            result = 2.0f / (1.0f + metal::exp(-z));
        } else {
            uint index = output - 8;
            result = float(a_res[0]) * normalized + float(b_res[index]);
        }
        atomic_store_explicit(&coeff[output], result, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    if (tid == 0) {
        uint old = atomic_fetch_add_explicit(&sync[0], 1u, memory_order_relaxed);
        if (old == 23u) {
            atomic_store_explicit(&sync[1], 1u, memory_order_relaxed);
        } else {
            while (atomic_load_explicit(&sync[1], memory_order_relaxed) == 0u) {}
        }
        ready = 1u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    if (group == 0 && ready == 1u) {
        if (tid < 24) {
            local_coeff[tid] = atomic_load_explicit(
                &coeff[tid], memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 4) {
            atomic_store_explicit(
                &hpost[tid], local_coeff[4 + tid], memory_order_relaxed);
        }
        if (tid < 16) {
            matrix[tid] = local_coeff[8 + tid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint iteration = 0; iteration < 8; ++iteration) {
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
            atomic_store_explicit(
                &hres[tid], metal::exp(matrix[tid]), memory_order_relaxed);
        }
        for (uint channel = tid; channel < 512; channel += 64) {
            float mixed = 0.0f;
            for (uint lane = 0; lane < 4; ++lane) {
                mixed += local_coeff[lane]
                    * float(lanes[lane * 512 + channel]);
            }
            atomic_store_explicit(&u[channel], mixed, memory_order_relaxed);
        }
    }
"""


_MHC_POST_MIX_SOURCE = r"""
    uint tid = thread_position_in_grid.x;
    for (uint row = 0; row < 4; ++row) {
        float value = float(hpost[row]) * float(y[tid]);
        for (uint col = 0; col < 4; ++col) {
            value += float(hres[row * 4 + col])
                * float(lanes[col * 512 + tid]);
        }
        out[row * 512 + tid] = T(value);
    }
"""

_MHC_POST_NEXT_COEFF_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    threadgroup float dots[64];
    threadgroup float sums[64];
    float dot = 0.0f;
    float ss = 0.0f;
    for (uint i = tid; i < 2048; i += 64) {
        uint row = i / 512;
        uint channel = i % 512;
        float value = float(hpost[row]) * float(y[channel]);
        for (uint col = 0; col < 4; ++col) {
            value += float(hres[row * 4 + col])
                * float(lanes[col * 512 + channel]);
        }
        if (group == 0) {
            out[i] = T(value);
        }
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


_BLOCK_TAIL_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float reduction[512];
    threadgroup float state[512];
    threadgroup float hidden[512];
    threadgroup float inv_rms;

    float attn_value = float(attn[tid]);
    reduction[tid] = attn_value * attn_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float ss = 0.0f;
        for (uint i = 0; i < 512; ++i) {
            ss += reduction[i];
        }
        inv_rms = metal::rsqrt(ss / 512.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gate = 1.0f / (1.0f + metal::exp(-float(attn_gate[0])));
    float state_value = float(skip[tid])
        + gate * attn_value * inv_rms * (1.0f + float(post_scale[tid]));
    state[tid] = state_value;
    reduction[tid] = state_value * state_value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float ss = 0.0f;
        for (uint i = 0; i < 512; ++i) {
            ss += reduction[i];
        }
        inv_rms = metal::rsqrt(ss / 512.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc = 0.0f;
    for (uint i = 0; i < 512; ++i) {
        float z = state[i] * inv_rms * (1.0f + float(mlp_scale[i]))
            * float(d1[i]);
        acc += z * float(h[i * 512 + tid]);
    }
    float a = acc * float(d2[tid]);
    hidden[tid] = a / (1.0f + metal::exp(-a));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float acc2 = 0.0f;
    for (uint i = 0; i < 512; ++i) {
        acc2 += hidden[i] * float(h[i * 512 + tid]);
    }
    out[tid] = T(state[tid] + acc2 * float(d3[tid]));
"""

_ENGRAM_TWO_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint site = threadgroup_position_in_grid.x;
    threadgroup uint indices[4];
    if (tid < 4) {
        uint acc = 0x9E3779B9u * (tid + 1);
        acc = (acc ^ uint(tokens[2])) * 0x01000193u;
        acc = (acc ^ uint(tokens[1])) * 0x01000193u;
        if (tid >= 2) {
            acc = (acc ^ uint(tokens[0])) * 0x01000193u;
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
        uint table_index = (table * 8192 + indices[table]) * 128 + sub;
        float embedding = site == 0
            ? float(tables0[table_index]) : float(tables1[table_index]);
        float key_weight = site == 0
            ? float(key0[tid * 512 + i]) : float(key1[tid * 512 + i]);
        float value_weight = site == 0
            ? float(value0[tid * 512 + i]) : float(value1[tid * 512 + i]);
        key_acc += embedding * key_weight;
        value_acc += embedding * value_weight;
    }
    keys[site * 512 + tid] = T(key_acc);
    float tap = site == 0 ? float(taps0[tid]) : float(taps1[tid]);
    values[site * 512 + tid] = T(value_acc * tap);
"""


_qkvg_rope_kernel = None
_qk_norm_rope_kernel = None
_mlp_kernel = None
_mlp_mhc_post_kernel = None
_mlp_mhc_transition_kernel = None
_mhc_coeff_kernel = None
_mhc_pre_mix_kernel = None
_mhc_persistent_pre_kernel = None
_mhc_post_mix_kernel = None
_mhc_post_next_coeff_kernel = None
_block_tail_kernel = None
_engram_two_kernel = None


def _kernel(name: str, inputs: list[str], outputs: list[str], source: str):
    return mx.fast.metal_kernel(
        name=name,
        input_names=inputs,
        output_names=outputs,
        source=source,
        compile_options={"math_mode": "safe"},
    )


def fused_qkvg_rope(
    x: mx.array,
    q_weight: mx.array,
    k_weight: mx.array,
    v_weight: mx.array,
    gate_weight: mx.array,
    q_scale: mx.array,
    k_scale: mx.array,
    rope: tuple[mx.array, mx.array],
    position: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    global _qkvg_rope_kernel
    if _qkvg_rope_kernel is None:
        _qkvg_rope_kernel = _kernel(
            "mei_qkvg_norm_rope_8q4kv_f32",
            [
                "x",
                "q_weight",
                "k_weight",
                "v_weight",
                "gate_weight",
                "q_scale",
                "k_scale",
                "rope_cos",
                "rope_sin",
                "position",
            ],
            ["q", "k", "v", "gate"],
            _QKVG_ROPE_SOURCE,
        )
    return tuple(
        _qkvg_rope_kernel(
            inputs=[
                x.reshape(-1),
                q_weight,
                k_weight,
                v_weight,
                gate_weight,
                q_scale,
                k_scale,
                rope[0],
                rope[1],
                position.reshape(-1),
            ],
            template=[],
            grid=(24 * 64, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[
                (1, 8, 1, 64),
                (1, 4, 1, 64),
                (1, 4, 1, 64),
                (1, 1, 512),
            ],
            output_dtypes=[mx.float32, mx.float32, mx.float32, mx.float32],
        )
    )


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
            "mei_qk_norm_rope_8q4kv_f32",
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


def fused_hadamard_mlp(
    x: mx.array,
    h: mx.array,
    d1: mx.array,
    d2: mx.array,
    d3: mx.array,
) -> mx.array:
    global _mlp_kernel
    if _mlp_kernel is None:
        _mlp_kernel = _kernel(
            "mei_hadamard_mlp_512_f32",
            ["x", "h", "d1", "d2", "d3"],
            ["out"],
            _MLP_SOURCE,
        )
    return _mlp_kernel(
        inputs=[x.reshape(-1), h, d1, d2, d3],
        template=[("T", mx.float32)],
        grid=(_D_MODEL, 1, 1),
        threadgroup=(_D_MODEL, 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def fused_hadamard_mlp_mhc_post(
    skip: mx.array,
    attn: mx.array,
    attn_gate: mx.array,
    post_scale: mx.array,
    mlp_scale: mx.array,
    h: mx.array,
    d1: mx.array,
    d2: mx.array,
    d3: mx.array,
    u: mx.array,
    lanes: mx.array,
    hpost: mx.array,
    hres: mx.array,
) -> mx.array:
    global _mlp_mhc_post_kernel
    if _mlp_mhc_post_kernel is None:
        _mlp_mhc_post_kernel = _kernel(
            "mei_hadamard_mlp_mhc_post_512_f32",
            [
                "skip",
                "attn",
                "attn_gate",
                "post_scale",
                "mlp_scale",
                "h",
                "d1",
                "d2",
                "d3",
                "u",
                "lanes",
                "hpost",
                "hres",
            ],
            ["out"],
            _MLP_MHC_POST_SOURCE,
        )
    return _mlp_mhc_post_kernel(
        inputs=[
            skip,
            attn,
            attn_gate.reshape(-1),
            post_scale,
            mlp_scale,
            h,
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


def fused_hadamard_mlp_mhc_transition(
    x: mx.array,
    h: mx.array,
    d1: mx.array,
    d2: mx.array,
    d3: mx.array,
    state: mx.array,
    u: mx.array,
    lanes: mx.array,
    hpost: mx.array,
    hres: mx.array,
    phi: mx.array,
    a_pre: mx.array,
    a_post: mx.array,
    a_res: mx.array,
    b_pre: mx.array,
    b_post: mx.array,
    b_res: mx.array,
    off_pre: mx.array,
    off_post: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    global _mlp_mhc_transition_kernel
    if _mlp_mhc_transition_kernel is None:
        _mlp_mhc_transition_kernel = _kernel(
            "mei_hadamard_mlp_mhc_transition_512_f32",
            [
                "x",
                "h",
                "d1",
                "d2",
                "d3",
                "state",
                "u",
                "lanes",
                "hpost",
                "hres",
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
            ["out_lanes", "next_u", "next_hpost", "next_hres"],
            _MLP_MHC_TRANSITION_SOURCE,
        )
    return tuple(
        _mlp_mhc_transition_kernel(
            inputs=[
                x,
                h,
                d1,
                d2,
                d3,
                state,
                u,
                lanes,
                hpost,
                hres,
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
            template=[("T", mx.float32)],
            grid=(_D_MODEL, 1, 1),
            threadgroup=(_D_MODEL, 1, 1),
            output_shapes=[
                (1, 1, _LANES, _D_MODEL),
                (1, 1, _D_MODEL),
                (4,),
                (16,),
            ],
            output_dtypes=[x.dtype, x.dtype, x.dtype, x.dtype],
        )
    )


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
            "mei_mhc_coeff_4x512_f32",
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
            "mei_mhc_pre_mix_sinkhorn_4x512_f32",
            ["lanes", "coeff"],
            ["u", "hpost", "hres"],
            _MHC_PRE_MIX_SOURCE,
        )
    return tuple(
        _mhc_pre_mix_kernel(
            inputs=[lanes, coeff],
            template=[("T", mx.float32)],
            grid=(_D_MODEL, 1, 1),
            threadgroup=(_D_MODEL, 1, 1),
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


def fused_mhc_persistent_pre(
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
    global _mhc_persistent_pre_kernel
    if _mhc_persistent_pre_kernel is None:
        _mhc_persistent_pre_kernel = mx.fast.metal_kernel(
            name="mei_mhc_persistent_pre_4x512_f32",
            input_names=[
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
            output_names=["sync", "coeff", "u", "hpost", "hres"],
            source=_MHC_PERSISTENT_PRE_SOURCE,
            atomic_outputs=True,
            compile_options={"math_mode": "safe"},
        )
    _, _, u, hpost, hres = _mhc_persistent_pre_kernel(
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
        output_shapes=[(2,), (24,), (1, 1, 512), (4,), (16,)],
        output_dtypes=[mx.uint32, mx.float32, mx.float32, mx.float32, mx.float32],
        init_value=0,
    )
    return u, hpost, hres


def fused_mhc_post(
    lanes: mx.array,
    y: mx.array,
    hpost: mx.array,
    hres: mx.array,
) -> mx.array:
    global _mhc_post_mix_kernel
    if _mhc_post_mix_kernel is None:
        _mhc_post_mix_kernel = _kernel(
            "mei_mhc_post_mix_4x512_f32",
            ["lanes", "y", "hpost", "hres"],
            ["out"],
            _MHC_POST_MIX_SOURCE,
        )
    return _mhc_post_mix_kernel(
        inputs=[lanes, y.reshape(-1), hpost, hres],
        template=[("T", mx.float32)],
        grid=(_D_MODEL, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, 1, _LANES, _D_MODEL)],
        output_dtypes=[lanes.dtype],
    )[0]


def fused_mhc_post_next_coeff(
    lanes: mx.array,
    y: mx.array,
    hpost: mx.array,
    hres: mx.array,
    phi: mx.array,
    a_pre: mx.array,
    a_post: mx.array,
    a_res: mx.array,
    b_pre: mx.array,
    b_post: mx.array,
    b_res: mx.array,
    off_pre: mx.array,
    off_post: mx.array,
) -> tuple[mx.array, mx.array]:
    global _mhc_post_next_coeff_kernel
    if _mhc_post_next_coeff_kernel is None:
        _mhc_post_next_coeff_kernel = _kernel(
            "mei_mhc_post_next_coeff_4x512_f32",
            [
                "lanes",
                "y",
                "hpost",
                "hres",
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
            ["out", "coeff"],
            _MHC_POST_NEXT_COEFF_SOURCE,
        )
    return tuple(
        _mhc_post_next_coeff_kernel(
            inputs=[
                lanes,
                y.reshape(-1),
                hpost,
                hres,
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
            template=[("T", mx.float32)],
            grid=(24 * 64, 1, 1),
            threadgroup=(64, 1, 1),
            output_shapes=[(1, 1, 4, 512), (24,)],
            output_dtypes=[lanes.dtype, lanes.dtype],
        )
    )


def fused_block_tail(
    skip: mx.array,
    attn: mx.array,
    attn_gate: mx.array,
    post_scale: mx.array,
    mlp_scale: mx.array,
    h: mx.array,
    d1: mx.array,
    d2: mx.array,
    d3: mx.array,
) -> mx.array:
    global _block_tail_kernel
    if _block_tail_kernel is None:
        _block_tail_kernel = _kernel(
            "mei_block_tail_mlp_512_f32",
            [
                "skip",
                "attn",
                "attn_gate",
                "post_scale",
                "mlp_scale",
                "h",
                "d1",
                "d2",
                "d3",
            ],
            ["out"],
            _BLOCK_TAIL_SOURCE,
        )
    return _block_tail_kernel(
        inputs=[
            skip,
            attn,
            attn_gate.reshape(-1),
            post_scale,
            mlp_scale,
            h,
            d1,
            d2,
            d3,
        ],
        template=[("T", mx.float32)],
        grid=(_D_MODEL, 1, 1),
        threadgroup=(_D_MODEL, 1, 1),
        output_shapes=[skip.shape],
        output_dtypes=[skip.dtype],
    )[0]


def fused_engram_two(tokens: mx.array, first, second) -> tuple[mx.array, mx.array]:
    """Fuse two deployed Engram sites for a single token plus two-token prefix."""
    global _engram_two_kernel
    if _engram_two_kernel is None:
        _engram_two_kernel = _kernel(
            "mei_engram_two_sites_decode_f32",
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
            _ENGRAM_TWO_SOURCE,
        )
    keys, values = _engram_two_kernel(
        inputs=[
            tokens.reshape(-1),
            first.tables,
            second.tables,
            first.key_proj.weight,
            second.key_proj.weight,
            first.value_proj.weight,
            second.value_proj.weight,
            first.taps[0],
            second.taps[0],
        ],
        template=[("T", mx.float32)],
        grid=(2 * _D_MODEL, 1, 1),
        threadgroup=(_D_MODEL, 1, 1),
        output_shapes=[(2, 1, 1, _D_MODEL), (2, 1, 1, _D_MODEL)],
        output_dtypes=[mx.float32, mx.float32],
    )
    return keys, values


def supports_decode_fusion(*, batch: int, tokens: int, lanes: int, d_model: int, dtype) -> bool:
    return (
        batch == 1
        and tokens == 1
        and lanes == _LANES
        and d_model == _D_MODEL
        and dtype == mx.float32
    )
