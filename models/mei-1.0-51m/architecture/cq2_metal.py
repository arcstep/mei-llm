"""Resident packed CQ2/CQ4 Metal primitives for MEI 1.0 51M.

The portable format stores 128-value groups in the Walsh-Hadamard domain.
For a row-major matrix whose input width is divisible by 128,

    dot(H(q), x) == dot(q, H(x))

so GEMV/GEMM can read the 2/4-bit codebook values directly without ever
materializing a dequantized weight matrix.  This module deliberately handles
only pure-Q2 or pure-Q4 tensors; mixed-group containers remain supported by
the portable reference readers and fail closed here until a prefix-offset
directory is present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import numpy as np

GROUP_SIZE = 128
_Q2 = (-1.5104176, -0.45278, 0.45278, 1.5104176)
_Q4 = (
    -2.732589,
    -2.069018,
    -1.618046,
    -1.256231,
    -0.94234,
    -0.656759,
    -0.388055,
    -0.128396,
    0.128396,
    0.388055,
    0.656759,
    0.94234,
    1.256231,
    1.618046,
    2.069018,
    2.732589,
)


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


_PREPARE_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint group = threadgroup_position_in_grid.x;
    uint index = group * 128 + tid;
    threadgroup float values[128];
    values[tid] = float(x[index]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 1; width < 128; width <<= 1) {
        uint base = tid & ~(2 * width - 1);
        uint lane = tid & (width - 1);
        float left = values[base + lane];
        float right = values[base + width + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[tid] = (tid & width) ? left - right : left + right;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[index] = values[tid] * 0.08838834764831843f;
"""

_PREPARE_ZCRMS_512_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    uint lane = thread_index_in_simdgroup;
    uint simd_group = simdgroup_index_in_threadgroup;
    threadgroup float values[512];
    threadgroup float partials[4];
    threadgroup float inv_rms;
    float value[4];
    float sum_sq = 0.0f;
    for (uint group = 0; group < 4; ++group) {
        uint local_index = group * 128 + tid;
        value[group] = float(x[row * 512 + local_index]);
        sum_sq += value[group] * value[group];
    }
    float local_sum = simd_sum(sum_sq);
    if (lane == 0) partials[simd_group] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group == 0) {
        float part = lane < 4 ? partials[lane] : 0.0f;
        float total = simd_sum(part);
        if (lane == 0) inv_rms = metal::rsqrt(total / 512.0f + 1e-6f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint group = 0; group < 4; ++group) {
        uint local_index = group * 128 + tid;
        values[local_index] = value[group] * inv_rms * (1.0f + float(scale[local_index]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 1; width < 128; width <<= 1) {
        float left[4];
        float right[4];
        uint local_base = tid & ~(2 * width - 1);
        uint pair_lane = tid & (width - 1);
        for (uint group = 0; group < 4; ++group) {
            uint base = group * 128 + local_base;
            left[group] = values[base + pair_lane];
            right[group] = values[base + width + pair_lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint group = 0; group < 4; ++group) {
            values[group * 128 + tid] = (tid & width)
                ? left[group] - right[group]
                : left[group] + right[group];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint group = 0; group < 4; ++group) {
        uint local_index = group * 128 + tid;
        out[row * 512 + local_index] = values[local_index] * 0.08838834764831843f;
    }
"""

_PREPARE_GATED_512_SOURCE = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint row = threadgroup_position_in_grid.x;
    threadgroup float values[512];
    for (uint group = 0; group < 4; ++group) {
        uint local_index = group * 128 + tid;
        uint index = row * 512 + local_index;
        float gate_value = float(gate[index]);
        values[local_index] = float(x[index]) / (1.0f + metal::exp(-gate_value));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 1; width < 128; width <<= 1) {
        float left[4];
        float right[4];
        uint local_base = tid & ~(2 * width - 1);
        uint pair_lane = tid & (width - 1);
        for (uint group = 0; group < 4; ++group) {
            uint base = group * 128 + local_base;
            left[group] = values[base + pair_lane];
            right[group] = values[base + width + pair_lane];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint group = 0; group < 4; ++group) {
            values[group * 128 + tid] = (tid & width)
                ? left[group] - right[group]
                : left[group] + right[group];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint group = 0; group < 4; ++group) {
        uint local_index = group * 128 + tid;
        out[row * 512 + local_index] = values[local_index] * 0.08838834764831843f;
    }
"""

_ENGRAM_INDICES_4_TAPS_SOURCE = r"""
    uint tid = thread_position_in_grid.x;
    if (tid >= 16) return;
    uint tap = tid >> 2;
    uint table = tid & 3;
    uint position = 12 - tap * 3;
    uint order = table < 2 ? 2 : 3;
    uint acc = 0x9E3779B9u * (table + 1);
    for (uint offset = 0; offset < order; ++offset) {
        acc = (acc ^ uint(tokens[position - offset])) * 0x01000193u;
    }
    acc = acc ^ (acc >> 15);
    out[tid] = int(acc % 8192u);
"""

def _codebook_source(bits: int) -> str:
    values = _Q2 if bits == 2 else _Q4
    literal = ", ".join(f"{value:.9g}f" for value in values)
    packed_bytes = GROUP_SIZE * bits // 8
    unpack = (
        """
            uint activation = activation_base + packed_index * 4;
            float x0 = float(xhat[activation]);
            float x1 = float(xhat[activation + 1]);
            float x2 = float(xhat[activation + 2]);
            float x3 = float(xhat[activation + 3]);
            float high = ((packed & 2u) ? x0 : -x0)
                + ((packed & 8u) ? x1 : -x1)
                + ((packed & 32u) ? x2 : -x2)
                + ((packed & 128u) ? x3 : -x3);
            float low = ((packed & 1u) ? x0 : -x0)
                + ((packed & 4u) ? x1 : -x1)
                + ((packed & 16u) ? x2 : -x2)
                + ((packed & 64u) ? x3 : -x3);
            high_acc += high;
            low_acc += low;
        """
        if bits == 2
        else """
            uint activation = activation_base + packed_index * 2;
            group_acc += codebook[packed & 15u] * float(xhat[activation]);
            group_acc += codebook[(packed >> 4) & 15u] * float(xhat[activation + 1]);
        """
    )
    bitplane_accumulators = "float high_acc = 0.0f; float low_acc = 0.0f;" if bits == 2 else ""
    bitplane_finish = (
        "group_acc = 0.9815988f * high_acc + 0.5288188f * low_acc;"
        if bits == 2
        else ""
    )
    return f"""
    uint output = thread_position_in_grid.x;
    uint batch = uint(dims[0]);
    uint rows = uint(dims[1]);
    uint cols = uint(dims[2]);
    if (output >= batch * rows) return;
    uint sample = output / rows;
    uint row = output - sample * rows;
    uint groups_per_row = cols / 128;
    float codebook[{len(values)}] = {{{literal}}};
    float acc = 0.0f;
    for (uint local_group = 0; local_group < groups_per_row; ++local_group) {{
        uint group = row * groups_per_row + local_group;
        uint packed_base = group * {packed_bytes};
        uint activation_base = sample * cols + local_group * 128;
        float group_acc = 0.0f;
        {bitplane_accumulators}
        for (uint packed_index = 0; packed_index < {packed_bytes}; ++packed_index) {{
            uint packed = uint(data[packed_base + packed_index]);
            {unpack}
        }}
        {bitplane_finish}
        acc += group_acc * float(scales[group]);
    }}
    out[output] = acc;
"""


def _codebook_shared_activation_source(bits: int) -> str:
    """One output row per lane, with the activation tile shared by 32 rows."""

    values = _Q2 if bits == 2 else _Q4
    literal = ", ".join(f"{value:.9g}f" for value in values)
    packed_bytes = GROUP_SIZE * bits // 8
    unpack = (
        """
            uint activation_index = packed_index * 4;
            float x0 = activation[activation_index];
            float x1 = activation[activation_index + 1];
            float x2 = activation[activation_index + 2];
            float x3 = activation[activation_index + 3];
            float high = ((packed & 2u) ? x0 : -x0)
                + ((packed & 8u) ? x1 : -x1)
                + ((packed & 32u) ? x2 : -x2)
                + ((packed & 128u) ? x3 : -x3);
            float low = ((packed & 1u) ? x0 : -x0)
                + ((packed & 4u) ? x1 : -x1)
                + ((packed & 16u) ? x2 : -x2)
                + ((packed & 64u) ? x3 : -x3);
            high_acc += high;
            low_acc += low;
        """
        if bits == 2
        else """
            uint activation_index = packed_index * 2;
            group_acc += codebook[packed & 15u] * activation[activation_index];
            group_acc += codebook[(packed >> 4) & 15u] * activation[activation_index + 1];
        """
    )
    bitplane_accumulators = "float high_acc = 0.0f; float low_acc = 0.0f;" if bits == 2 else ""
    bitplane_finish = (
        "group_acc = 0.9815988f * high_acc + 0.5288188f * low_acc;"
        if bits == 2
        else ""
    )
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint output = thread_position_in_grid.x;
    uint batch = uint(dims[0]);
    uint rows = uint(dims[1]);
    uint cols = uint(dims[2]);
    if (output >= batch * rows) return;
    uint sample = output / rows;
    uint row = output - sample * rows;
    uint groups_per_row = cols / 128;
    float codebook[{len(values)}] = {{{literal}}};
    threadgroup float activation[128];
    float acc = 0.0f;
    for (uint local_group = 0; local_group < groups_per_row; ++local_group) {{
        uint activation_base = sample * cols + local_group * 128;
        for (uint index = tid; index < 128; index += 32) {{
            activation[index] = float(xhat[activation_base + index]);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint group = row * groups_per_row + local_group;
        uint packed_base = group * {packed_bytes};
        float group_acc = 0.0f;
        {bitplane_accumulators}
        for (uint packed_index = 0; packed_index < {packed_bytes}; ++packed_index) {{
            uint packed = uint(data[packed_base + packed_index]);
            {unpack}
        }}
        {bitplane_finish}
        acc += group_acc * float(scales[group]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    out[output] = acc;
"""


def _q2_bitplane_shared_activation_source() -> str:
    return r"""
    uint tid = thread_position_in_threadgroup.x;
    uint output = thread_position_in_grid.x;
    uint batch = uint(dims[0]);
    uint rows = uint(dims[1]);
    uint cols = uint(dims[2]);
    if (output >= batch * rows) return;
    uint sample = output / rows;
    uint row = output - sample * rows;
    uint groups_per_row = cols / 128;
    threadgroup float activation[128];
    float acc = 0.0f;
    for (uint local_group = 0; local_group < groups_per_row; ++local_group) {
        uint activation_base = sample * cols + local_group * 128;
        for (uint index = tid; index < 128; index += 32) {
            activation[index] = float(xhat[activation_base + index]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint group = row * groups_per_row + local_group;
        uint plane_base = group * 32;
        float high_acc = 0.0f;
        float low_acc = 0.0f;
        for (uint packed_index = 0; packed_index < 16; ++packed_index) {
            uint low = uint(data[plane_base + packed_index]);
            uint high = uint(data[plane_base + 16 + packed_index]);
            uint activation_index = packed_index * 8;
            for (uint bit = 0; bit < 8; ++bit) {
                float value = activation[activation_index + bit];
                low_acc += (low & (1u << bit)) ? value : -value;
                high_acc += (high & (1u << bit)) ? value : -value;
            }
        }
        float group_acc = 0.9815988f * high_acc + 0.5288188f * low_acc;
        acc += group_acc * float(scales[group]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    out[output] = acc;
"""


def _gather_source(bits: int) -> str:
    values = _Q2 if bits == 2 else _Q4
    literal = ", ".join(f"{value:.9g}f" for value in values)
    shift = "((tid & 3) * 2)" if bits == 2 else "((tid & 1) * 4)"
    divisor = "4" if bits == 2 else "2"
    mask = "3" if bits == 2 else "15"
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint requested_group = threadgroup_position_in_grid.x;
    uint count = uint(dims[0]);
    uint rows = uint(dims[1]);
    uint cols = uint(dims[2]);
    uint groups_per_row = cols / 128;
    uint request = requested_group / groups_per_row;
    uint local_group = requested_group - request * groups_per_row;
    if (request >= count) return;
    uint row = min(uint(indices[request]), rows - 1);
    uint group = row * groups_per_row + local_group;
    uint packed_base = group * {GROUP_SIZE * bits // 8};
    float codebook[{len(values)}] = {{{literal}}};
    uint code = (uint(data[packed_base + tid / {divisor}]) >> {shift}) & {mask};
    threadgroup float values[128];
    values[tid] = codebook[code] * float(scales[group]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint width = 1; width < 128; width <<= 1) {{
        uint base = tid & ~(2 * width - 1);
        uint lane = tid & (width - 1);
        float left = values[base + lane];
        float right = values[base + width + lane];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        values[tid] = (tid & width) ? left - right : left + right;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}
    uint out_index = request * cols + local_group * 128 + tid;
    out[out_index] = values[tid] * 0.08838834764831843f;
"""


def _gather_transformed_source(bits: int) -> str:
    values = _Q2 if bits == 2 else _Q4
    literal = ", ".join(f"{value:.9g}f" for value in values)
    shift = "((tid & 3) * 2)" if bits == 2 else "((tid & 1) * 4)"
    divisor = "4" if bits == 2 else "2"
    mask = "3" if bits == 2 else "15"
    return f"""
    uint tid = thread_position_in_threadgroup.x;
    uint requested_group = threadgroup_position_in_grid.x;
    uint count = uint(dims[0]);
    uint rows = uint(dims[1]);
    uint cols = uint(dims[2]);
    uint groups_per_row = cols / 128;
    uint request = requested_group / groups_per_row;
    uint local_group = requested_group - request * groups_per_row;
    if (request >= count) return;
    uint row = min(uint(indices[request]), rows - 1);
    uint group = row * groups_per_row + local_group;
    uint packed_base = group * {GROUP_SIZE * bits // 8};
    float codebook[{len(values)}] = {{{literal}}};
    uint code = (uint(data[packed_base + tid / {divisor}]) >> {shift}) & {mask};
    uint out_index = request * cols + local_group * 128 + tid;
    out[out_index] = codebook[code] * float(scales[group]);
"""


_prepare_kernel = None
_prepare_zcrms_512_kernel = None
_prepare_gated_512_kernel = None
_engram_indices_4_taps_kernel = None
_linear_kernels: dict[tuple[int, str], Any] = {}
_gather_kernels: dict[int, Any] = {}
_gather_transformed_kernels: dict[int, Any] = {}


@dataclass(frozen=True)
class PackedActivation:
    values: mx.array
    leading_shape: tuple[int, ...]
    width: int
    layout: str


class PackedCqMatrix:
    """A pure-Q2/Q4 row-major matrix resident in packed Metal buffers."""

    def __init__(
        self,
        *,
        name: str,
        shape: tuple[int, ...],
        bits: int,
        data: bytes,
        scales_f16: bytes,
        bit_map: bytes,
    ) -> None:
        if bits not in {2, 4}:
            raise ValueError("packed Metal tensor must be pure Q2 or Q4")
        if len(shape) < 2 or shape[-1] % GROUP_SIZE:
            raise ValueError("packed Metal tensor requires a row width divisible by 128")
        self.name = str(name)
        self.shape = tuple(int(value) for value in shape)
        self.bits = int(bits)
        self.cols = self.shape[-1]
        self.rows = math.prod(self.shape[:-1])
        self.groups = self.rows * (self.cols // GROUP_SIZE)
        expected_data = self.groups * GROUP_SIZE * self.bits // 8
        if len(data) != expected_data or len(scales_f16) != self.groups * 2:
            raise ValueError(f"{self.name} packed payload length mismatch")
        expected_map = (self.groups + 7) // 8
        if len(bit_map) != expected_map:
            raise ValueError(f"{self.name} bit map length mismatch")
        active = [
            (bit_map[group // 8] >> (group % 8)) & 1
            for group in range(self.groups)
        ]
        if self.bits == 2 and any(active):
            raise ValueError(f"{self.name} is mixed/Q4, not pure Q2")
        if self.bits == 4 and not all(active):
            raise ValueError(f"{self.name} is mixed/Q2, not pure Q4")
        data_np = np.frombuffer(data, dtype=np.uint8).copy()
        scales_np = np.frombuffer(scales_f16, dtype="<f2").copy()
        if not np.isfinite(scales_np).all() or not np.all(scales_np > 0):
            raise ValueError(f"{self.name} scale must be finite and positive")
        self.data = mx.array(data_np)
        self.scales = mx.array(scales_np)
        self.linear_data = None
        self._codes_dropped = False
        mx.eval(self.data, self.scales)

    @classmethod
    def from_container(cls, container, entry):
        data = container.blob[entry.data_offset : entry.data_offset + entry.data_nbytes]
        scales = container.blob[
            entry.scales_offset : entry.scales_offset + entry.scales_nbytes
        ]
        bit_map = container.blob[
            entry.bit_map_offset : entry.bit_map_offset + entry.bit_map_nbytes
        ]
        return cls(
            name=entry.name,
            shape=entry.shape,
            bits=2 if entry.dtype == "cq2" else 4,
            data=data,
            scales_f16=scales,
            bit_map=bit_map,
        )

    @classmethod
    def concatenate_rows(cls, name: str, parts: list["PackedCqMatrix"]):
        if not parts:
            raise ValueError("packed concatenation requires at least one matrix")
        bits = parts[0].bits
        cols = parts[0].cols
        if any(part.bits != bits or part.cols != cols for part in parts):
            raise ValueError("packed concatenation dtype/width mismatch")
        result = object.__new__(cls)
        result.name = str(name)
        result.bits = bits
        result.cols = cols
        result.rows = sum(part.rows for part in parts)
        result.shape = (result.rows, result.cols)
        result.groups = result.rows * (result.cols // GROUP_SIZE)
        result.data = mx.concatenate([part.data for part in parts])
        result.scales = mx.concatenate([part.scales for part in parts])
        result.linear_data = None
        result._codes_dropped = False
        mx.eval(result.data, result.scales)
        return result

    def enable_q2_bitplane_linear(self, *, drop_codes: bool) -> None:
        """Reorder Q2 codes into equal-size low/high bitplanes for GEMV."""

        if self.bits != 2 or self.linear_data is not None or self._codes_dropped:
            raise ValueError(f"{self.name} cannot enable Q2 bitplane linear layout")
        packed = np.asarray(self.data, dtype=np.uint8).reshape(self.groups, 32)
        shifts = np.asarray([0, 2, 4, 6], dtype=np.uint8)
        expanded = ((packed[:, :, None] >> shifts[None, None, :]) & 3).reshape(
            self.groups,
            GROUP_SIZE,
        )
        low = np.packbits(expanded & 1, axis=1, bitorder="little")
        high = np.packbits((expanded >> 1) & 1, axis=1, bitorder="little")
        planes = np.concatenate([low, high], axis=1).reshape(-1)
        if int(planes.size) != int(self.data.size):
            raise ValueError(f"{self.name} Q2 bitplane byte count mismatch")
        self.linear_data = mx.array(planes)
        if drop_codes:
            self.data = mx.array(np.empty((0,), dtype=np.uint8))
            self._codes_dropped = True
        mx.eval(self.linear_data, self.data)

    def prepare(self, x: mx.array) -> PackedActivation:
        global _prepare_kernel
        width = int(x.shape[-1])
        if width % GROUP_SIZE:
            raise ValueError("packed activation width must be divisible by 128")
        leading = tuple(int(value) for value in x.shape[:-1])
        flat = x.astype(mx.float32).reshape(-1, width)
        if _prepare_kernel is None:
            _prepare_kernel = _kernel(
                "mei_51m_cq_prepare_wht128_f32",
                ["x"],
                ["out"],
                _PREPARE_SOURCE,
            )
        transformed = _prepare_kernel(
            inputs=[flat],
            template=[],
            grid=(int(flat.size), 1, 1),
            threadgroup=(GROUP_SIZE, 1, 1),
            output_shapes=[tuple(flat.shape)],
            output_dtypes=[mx.float32],
        )[0]
        return PackedActivation(transformed, leading, width, "wht")

    @staticmethod
    def engram_indices_4_taps(tokens: mx.array) -> mx.array:
        """Exact order-2/3, dilation-3 indices for 12 history + current."""

        global _engram_indices_4_taps_kernel
        if tuple(tokens.shape) != (1, 13) or tokens.dtype != mx.int32:
            raise ValueError("packed Engram index kernel requires int32 [1,13]")
        if _engram_indices_4_taps_kernel is None:
            _engram_indices_4_taps_kernel = _kernel(
                "mei_51m_engram_indices_order23_d3_taps4",
                ["tokens"],
                ["out"],
                _ENGRAM_INDICES_4_TAPS_SOURCE,
            )
        return _engram_indices_4_taps_kernel(
            inputs=[tokens.reshape(-1)],
            template=[],
            grid=(16, 1, 1),
            threadgroup=(16, 1, 1),
            output_shapes=[(1, 4, 4)],
            output_dtypes=[mx.int32],
        )[0]

    def prepare_zcrms(self, x: mx.array, scale: mx.array) -> PackedActivation:
        """Fuse 512-wide zero-centred RMS normalization and four WHT groups."""

        global _prepare_zcrms_512_kernel
        if self.cols != 512 or int(x.shape[-1]) != 512:
            raise ValueError("packed ZCRMS preparation requires width 512")
        leading = tuple(int(value) for value in x.shape[:-1])
        flat = x.astype(mx.float32).reshape(-1, 512)
        if _prepare_zcrms_512_kernel is None:
            _prepare_zcrms_512_kernel = _kernel(
                "mei_51m_cq_prepare_zcrms_wht512_f32",
                ["x", "scale"],
                ["out"],
                _PREPARE_ZCRMS_512_SOURCE,
            )
        transformed = _prepare_zcrms_512_kernel(
            inputs=[flat, scale],
            template=[],
            grid=(int(flat.shape[0]) * GROUP_SIZE, 1, 1),
            threadgroup=(GROUP_SIZE, 1, 1),
            output_shapes=[tuple(flat.shape)],
            output_dtypes=[mx.float32],
        )[0]
        return PackedActivation(transformed, leading, 512, "wht")

    def prepare_gated(self, x: mx.array, gate: mx.array) -> PackedActivation:
        """Fuse sigmoid gating and four 128-wide WHT activation groups."""

        global _prepare_gated_512_kernel
        if self.cols != 512 or int(x.shape[-1]) != 512 or tuple(x.shape) != tuple(gate.shape):
            raise ValueError("packed gated preparation requires matching width-512 tensors")
        leading = tuple(int(value) for value in x.shape[:-1])
        flat_x = x.astype(mx.float32).reshape(-1, 512)
        flat_gate = gate.astype(mx.float32).reshape(-1, 512)
        if _prepare_gated_512_kernel is None:
            _prepare_gated_512_kernel = _kernel(
                "mei_51m_cq_prepare_sigmoid_gate_wht512_f32",
                ["x", "gate"],
                ["out"],
                _PREPARE_GATED_512_SOURCE,
            )
        transformed = _prepare_gated_512_kernel(
            inputs=[flat_x, flat_gate],
            template=[],
            grid=(int(flat_x.shape[0]) * GROUP_SIZE, 1, 1),
            threadgroup=(GROUP_SIZE, 1, 1),
            output_shapes=[tuple(flat_x.shape)],
            output_dtypes=[mx.float32],
        )[0]
        return PackedActivation(transformed, leading, 512, "wht")

    def linear_prepared(self, prepared: PackedActivation) -> mx.array:
        if prepared.width != self.cols:
            raise ValueError(f"{self.name} input width mismatch")
        if prepared.layout != "wht":
            raise ValueError(f"{self.name} prepared activation layout mismatch")
        layout = (
            "bitplane32"
            if self.bits == 2 and self.linear_data is not None and self.rows % 32 == 0
            else "shared32"
            if self.rows % 32 == 0
            else "scalar"
        )
        kernel_key = (self.bits, layout)
        kernel = _linear_kernels.get(kernel_key)
        if kernel is None:
            kernel = _kernel(
                f"mei_51m_cq{self.bits}_linear_wht128_{layout}_f32",
                ["data", "scales", "xhat", "dims"],
                ["out"],
                (
                    _q2_bitplane_shared_activation_source()
                    if layout == "bitplane32"
                    else _codebook_shared_activation_source(self.bits)
                    if layout == "shared32"
                    else _codebook_source(self.bits)
                ),
            )
            _linear_kernels[kernel_key] = kernel
        batch = math.prod(prepared.leading_shape) if prepared.leading_shape else 1
        dims = mx.array([batch, self.rows, self.cols], dtype=mx.uint32)
        output = kernel(
            inputs=[
                self.linear_data if layout == "bitplane32" else self.data,
                self.scales,
                prepared.values,
                dims,
            ],
            template=[],
            grid=(batch * self.rows, 1, 1),
            threadgroup=(32 if layout in {"bitplane32", "shared32"} else 64, 1, 1),
            output_shapes=[(batch, self.rows)],
            output_dtypes=[mx.float32],
        )[0]
        return output.reshape((*prepared.leading_shape, self.rows))

    def linear(self, x: mx.array) -> mx.array:
        return self.linear_prepared(self.prepare(x))

    def linear_transformed(self, xhat: mx.array) -> mx.array:
        """Multiply an activation already represented in orthonormal WHT space."""

        width = int(xhat.shape[-1])
        prepared = PackedActivation(
            xhat.astype(mx.float32).reshape(-1, width),
            tuple(int(value) for value in xhat.shape[:-1]),
            width,
            "wht",
        )
        return self.linear_prepared(prepared)

    def gather_rows(self, indices: mx.array) -> mx.array:
        if self._codes_dropped:
            raise ValueError(f"{self.name} packed codes were released from a linear-only matrix")
        kernel = _gather_kernels.get(self.bits)
        if kernel is None:
            kernel = _kernel(
                f"mei_51m_cq{self.bits}_gather_wht128_f32",
                ["data", "scales", "indices", "dims"],
                ["out"],
                _gather_source(self.bits),
            )
            _gather_kernels[self.bits] = kernel
        leading = tuple(int(value) for value in indices.shape)
        flat_indices = indices.astype(mx.uint32).reshape(-1)
        count = int(flat_indices.size)
        dims = mx.array([count, self.rows, self.cols], dtype=mx.uint32)
        output = kernel(
            inputs=[self.data, self.scales, flat_indices, dims],
            template=[],
            grid=(count * self.cols, 1, 1),
            threadgroup=(GROUP_SIZE, 1, 1),
            output_shapes=[(count, self.cols)],
            output_dtypes=[mx.float32],
        )[0]
        return output.reshape((*leading, self.cols))

    def gather_transformed_rows(self, indices: mx.array) -> mx.array:
        """Gather rows in stored WHT/codebook space without inverse WHT."""

        if self._codes_dropped:
            raise ValueError(f"{self.name} packed codes were released from a linear-only matrix")

        kernel = _gather_transformed_kernels.get(self.bits)
        if kernel is None:
            kernel = _kernel(
                f"mei_51m_cq{self.bits}_gather_transformed_f32",
                ["data", "scales", "indices", "dims"],
                ["out"],
                _gather_transformed_source(self.bits),
            )
            _gather_transformed_kernels[self.bits] = kernel
        leading = tuple(int(value) for value in indices.shape)
        flat_indices = indices.astype(mx.uint32).reshape(-1)
        count = int(flat_indices.size)
        dims = mx.array([count, self.rows, self.cols], dtype=mx.uint32)
        output = kernel(
            inputs=[self.data, self.scales, flat_indices, dims],
            template=[],
            grid=(count * self.cols, 1, 1),
            threadgroup=(GROUP_SIZE, 1, 1),
            output_shapes=[(count, self.cols)],
            output_dtypes=[mx.float32],
        )[0]
        return output.reshape((*leading, self.cols))

    @property
    def resident_bytes(self) -> int:
        linear = 0 if self.linear_data is None else int(self.linear_data.size)
        return int(self.data.size) + linear + int(self.scales.size) * 2
