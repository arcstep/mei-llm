"""Portable CQ2 primitives and ``MEICQ201`` tensor-container reader.

The math intentionally mirrors ``platform/_shared/rust/mei-sdk-core/src/cq2.rs``.  It is a
MEI format and makes no `.cact` binary-compatibility claim.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import SdkError
from .json_schema_lite import loads_strict

QUANT_MATH_ID = "mei-cq-v2-g128-wht-codebook"
CONTAINER_MAGIC = b"MEICQ201"
CONTAINER_VERSION = 2
GROUP_SIZE = 128
Q2_CODEBOOK = (-1.5104176, -0.45278, 0.45278, 1.5104176)
Q4_CODEBOOK = (
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
_TENSOR_NAME_RE = re.compile(r"^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$")


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def _wht(values: list[float]) -> None:
    stride = 1
    while stride < GROUP_SIZE:
        for start in range(0, GROUP_SIZE, stride * 2):
            for index in range(start, start + stride):
                left = values[index]
                right = values[index + stride]
                values[index] = _f32(left + right)
                values[index + stride] = _f32(left - right)
        stride *= 2
    scale = _f32(1.0 / math.sqrt(GROUP_SIZE))
    for index, value in enumerate(values):
        values[index] = _f32(value * scale)


def _f32_to_f16(value: float) -> int:
    bits = struct.unpack("<I", struct.pack("<f", _f32(value)))[0]
    sign = (bits >> 16) & 0x8000
    source_exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    if source_exponent == 0xFF:
        return sign | 0x7C00 | (0 if mantissa == 0 else 0x0200)
    exponent = source_exponent - 127 + 15
    if exponent <= 0:
        if exponent < -10:
            return sign
        normalized = mantissa | 0x800000
        shift = 14 - exponent
        half = normalized >> shift
        remainder = normalized & ((1 << shift) - 1)
        halfway = 1 << (shift - 1)
        if remainder > halfway or (remainder == halfway and half & 1):
            half += 1
        return sign | half
    if exponent >= 31:
        return sign | 0x7C00
    half = sign | (exponent << 10) | (mantissa >> 13)
    remainder = mantissa & 0x1FFF
    if remainder > 0x1000 or (remainder == 0x1000 and half & 1):
        half = (half + 1) & 0xFFFF
    return half


def _f16_to_f32(value: int) -> float:
    sign = (value & 0x8000) << 16
    exponent = (value >> 10) & 0x1F
    mantissa = value & 0x03FF
    if exponent == 0:
        if mantissa == 0:
            bits = sign
        else:
            exponent_value = -14
            while mantissa & 0x0400 == 0:
                mantissa <<= 1
                exponent_value -= 1
            mantissa &= 0x03FF
            bits = sign | ((exponent_value + 127) << 23) | (mantissa << 13)
    elif exponent == 31:
        bits = sign | 0x7F800000 | (mantissa << 13)
    else:
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13)
    return struct.unpack("<f", struct.pack("<I", bits))[0]


@dataclass(frozen=True)
class CqTensor:
    n_values: int
    data: bytes
    scales_f16: tuple[int, ...]
    bit_map: bytes

    @property
    def groups(self) -> int:
        return (self.n_values + GROUP_SIZE - 1) // GROUP_SIZE

    def bits_for_group(self, group: int) -> int:
        if group < 0 or group >= self.groups:
            raise SdkError("invalid_argument", "group out of range")
        if group // 8 >= len(self.bit_map):
            raise SdkError("package_invalid", "CQ2 bit_map is truncated")
        return 4 if (self.bit_map[group // 8] >> (group % 8)) & 1 else 2


def _nearest(value: float, codebook: tuple[float, ...]) -> int:
    best = 0
    distance = math.inf
    for index, candidate in enumerate(codebook):
        current = abs(_f32(value) - _f32(candidate))
        if current < distance:
            best = index
            distance = current
    return best


def _pack_codes(codes: list[int], bits: int) -> bytes:
    output = bytearray()
    per_byte = 8 // bits
    for start in range(0, len(codes), per_byte):
        byte = 0
        for offset, code in enumerate(codes[start : start + per_byte]):
            byte |= (int(code) & ((1 << bits) - 1)) << (offset * bits)
        output.append(byte)
    return bytes(output)


def quantize(values: Iterable[float], group_bits: list[int]) -> CqTensor:
    """Quantize without expanding a NumPy tensor into millions of Python floats."""

    if hasattr(values, "reshape") and hasattr(values, "size"):
        flat = values.reshape(-1)  # type: ignore[attr-defined]
        n_values = int(flat.size)

        def group_values(start: int) -> list[float]:
            return [_f32(value) for value in flat[start : start + GROUP_SIZE].tolist()]

    else:
        source = [_f32(value) for value in values]
        n_values = len(source)

        def group_values(start: int) -> list[float]:
            return source[start : start + GROUP_SIZE]

    if not n_values:
        raise SdkError("invalid_argument", "CQ2 tensor is empty")
    groups = (n_values + GROUP_SIZE - 1) // GROUP_SIZE
    if len(group_bits) != groups or any(bits not in {2, 4} for bits in group_bits):
        raise SdkError("invalid_argument", "CQ2 requires one 2/4-bit selector per group")
    data = bytearray()
    scales: list[int] = []
    bit_map = bytearray((groups + 7) // 8)
    for group in range(groups):
        start = group * GROUP_SIZE
        block = group_values(start)
        if any(not math.isfinite(value) for value in block):
            raise SdkError("invalid_argument", "CQ2 tensor values must be finite")
        block += [0.0] * (GROUP_SIZE - len(block))
        _wht(block)
        squares = _f32(0.0)
        for value in block:
            squares = _f32(squares + _f32(value * value))
        scale = _f32(math.sqrt(_f32(squares / GROUP_SIZE)))
        scale = max(scale, _f32(1.17549435e-38))
        # f32 minimum-positive underflows to binary16 zero.  CQ2 requires a
        # finite positive stored scale, so clamp after conversion as Rust and
        # JS do for all-zero/subnormal groups.
        scale_f16 = max(1, _f32_to_f16(scale))
        if scale_f16 & 0x7C00 == 0x7C00:
            raise SdkError("invalid_argument", "CQ2 scale is not representable as finite f16")
        scales.append(scale_f16)
        # Codes must be selected against the exact binary16 scale stored in
        # the package.  This is the Rust/JS contract and matters for small
        # groups close to an f16 rounding boundary.
        quant_scale = _f16_to_f32(scale_f16)
        bits = group_bits[group]
        if bits == 4:
            bit_map[group // 8] |= 1 << (group % 8)
        codebook = Q2_CODEBOOK if bits == 2 else Q4_CODEBOOK
        codes = [_nearest(_f32(value / quant_scale), codebook) for value in block]
        data.extend(_pack_codes(codes, bits))
    return CqTensor(n_values, bytes(data), tuple(scales), bytes(bit_map))


@dataclass(frozen=True)
class TensorToPack:
    """Language-neutral input for a portable tensor container."""

    name: str
    shape: tuple[int, ...]
    values: Any
    role: str
    dtype: str
    group_bits: tuple[int, ...] | None = None


def _safe_tensor_bytes(values: Any, dtype: str, n_params: int) -> bytes:
    if hasattr(values, "reshape") and hasattr(values, "size"):
        flat = values.reshape(-1)
        if int(flat.size) != n_params:
            raise SdkError("invalid_argument", "safe tensor value count mismatch")
        source = flat.tolist()
    else:
        source = list(values)
    if len(source) != n_params:
        raise SdkError("invalid_argument", "safe tensor value count mismatch")
    if dtype == "i8":
        integers = []
        for value in source:
            numeric = int(value)
            if numeric != value or numeric < -128 or numeric > 127:
                raise SdkError("invalid_argument", "i8 tensor value is out of range")
            integers.append(numeric)
        return struct.pack("<" + "b" * len(integers), *integers)
    floats = [float(value) for value in source]
    if any(not math.isfinite(value) for value in floats):
        raise SdkError("invalid_argument", "safe tensor values must be finite")
    code = "e" if dtype == "f16" else "f"
    return struct.pack("<" + code * len(floats), *floats)


def build_tensor_container(tensors: Iterable[TensorToPack]) -> tuple[bytes, list[dict[str, Any]]]:
    """Build one gap-free ``MEICQ201`` container and manifest directory."""

    rows = list(tensors)
    if not rows:
        raise SdkError("invalid_argument", "tensor container is empty")
    seen: set[str] = set()
    entries: list[dict[str, Any]] = []
    sections: list[dict[str, bytes]] = []
    for tensor in rows:
        name = str(tensor.name)
        if not _TENSOR_NAME_RE.fullmatch(name) or name in seen:
            raise SdkError("duplicate_tensor" if name in seen else "invalid_argument", name)
        seen.add(name)
        shape = tuple(int(value) for value in tensor.shape)
        if any(value <= 0 for value in shape):
            raise SdkError("invalid_argument", f"{name}.shape is invalid")
        n_params = math.prod(shape) if shape else 1
        role = str(tensor.role)
        if role not in {
            "lm",
            "contrastive",
            "mw_disposition",
            "confidence",
            "narration_adapter",
        }:
            raise SdkError("invalid_argument", f"{name}.role is invalid")
        dtype = str(tensor.dtype)
        if dtype not in {"cq2", "cq4", "f16", "f32", "i8"}:
            raise SdkError("invalid_argument", f"{name}.dtype is invalid")
        entry: dict[str, Any] = {
            "name": name,
            "role": role,
            "shape": list(shape),
            "n_params": n_params,
            "dtype": dtype,
        }
        if dtype in {"cq2", "cq4"}:
            groups = (n_params + GROUP_SIZE - 1) // GROUP_SIZE
            bits = list(tensor.group_bits or ((4,) * groups if dtype == "cq4" else (2,) * groups))
            if dtype == "cq4" and any(value != 4 for value in bits):
                raise SdkError("invalid_argument", f"{name} cq4 requires all groups to be 4-bit")
            if dtype == "cq2" and bits and all(value == 4 for value in bits):
                raise SdkError("invalid_argument", f"{name} all-q4 data must use cq4 dtype")
            packed = quantize(tensor.values, bits)
            scales = b"".join(struct.pack("<H", value) for value in packed.scales_f16)
            sections.append({"data": packed.data, "scales": scales, "bit_map": packed.bit_map})
            entry.update(
                {
                    "group_size": GROUP_SIZE,
                    "transform": "wht",
                    "codebook": (
                        "gaussian-lloyd-q4-v1" if dtype == "cq4" else "gaussian-lloyd-q2-v1"
                    ),
                }
            )
        else:
            sections.append({"data": _safe_tensor_bytes(tensor.values, dtype, n_params)})
            entry.update({"transform": "none", "codebook": "none"})
        entries.append(entry)

    materialized = entries
    for _ in range(32):
        header = json.dumps(
            {"quant_math_id": QUANT_MATH_ID, "tensors": materialized},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        cursor = 16 + len(header)
        updated: list[dict[str, Any]] = []
        payload = bytearray()
        for entry, section_map in zip(entries, sections, strict=True):
            row = dict(entry)
            for field in ("data", "scales", "bit_map"):
                section = section_map.get(field)
                if section is None:
                    continue
                row[field] = {"offset": cursor, "nbytes": len(section)}
                cursor += len(section)
                payload.extend(section)
            updated.append(row)
        if updated == materialized:
            blob = CONTAINER_MAGIC + struct.pack("<II", CONTAINER_VERSION, len(header)) + header + payload
            parsed = TensorContainer.parse(blob)
            if list(parsed.entries) != [row.name for row in rows]:
                raise SdkError("package_invalid", "tensor order changed during container build")
            return blob, updated
        materialized = updated
    raise SdkError("package_invalid", "tensor container header offsets did not converge")


def dequantize(tensor: CqTensor) -> list[float]:
    groups = tensor.groups
    expected_map = (groups + 7) // 8
    if len(tensor.scales_f16) != groups or len(tensor.bit_map) != expected_map:
        raise SdkError("package_invalid", "CQ2 metadata length mismatch")
    if groups % 8 and tensor.bit_map[-1] >> (groups % 8):
        raise SdkError("package_invalid", "CQ2 bit_map has non-zero padding bits")
    cursor = 0
    output: list[float] = []
    for group in range(groups):
        bits = tensor.bits_for_group(group)
        byte_count = GROUP_SIZE * bits // 8
        packed = tensor.data[cursor : cursor + byte_count]
        if len(packed) != byte_count:
            raise SdkError("package_invalid", "CQ2 data is truncated")
        cursor += byte_count
        mask = (1 << bits) - 1
        codebook = Q2_CODEBOOK if bits == 2 else Q4_CODEBOOK
        scale = _f16_to_f32(tensor.scales_f16[group])
        if not math.isfinite(scale) or scale <= 0:
            raise SdkError("package_invalid", "CQ2 scale must be finite and positive")
        block = []
        for index in range(GROUP_SIZE):
            code = (packed[index * bits // 8] >> ((index % (8 // bits)) * bits)) & mask
            block.append(_f32(_f32(codebook[code]) * scale))
        _wht(block)
        output.extend(block)
    if cursor != len(tensor.data):
        raise SdkError("package_invalid", "CQ2 data has trailing bytes")
    return output[: tensor.n_values]


@dataclass(frozen=True)
class TensorEntry:
    name: str
    shape: tuple[int, ...]
    n_params: int
    dtype: str
    data_offset: int
    data_nbytes: int
    scales_offset: int = 0
    scales_nbytes: int = 0
    bit_map_offset: int = 0
    bit_map_nbytes: int = 0
    group_size: int = GROUP_SIZE
    role: str | None = None
    transform: str = "none"
    codebook: str = "none"


class TensorContainer:
    def __init__(self, blob: bytes, entries: dict[str, TensorEntry]):
        self.blob = blob
        self.entries = entries
        self.quant_math_id = QUANT_MATH_ID

    @classmethod
    def parse(cls, blob: bytes) -> "TensorContainer":
        if len(blob) < 16:
            raise SdkError("package_invalid", "tensor container is truncated")
        if blob[:8] != CONTAINER_MAGIC:
            raise SdkError("package_invalid", "tensor container magic mismatch")
        version, header_len = struct.unpack_from("<II", blob, 8)
        if version != CONTAINER_VERSION:
            raise SdkError("package_invalid", f"unsupported CQ2 container version {version}")
        header_end = 16 + header_len
        if header_end > len(blob):
            raise SdkError("package_invalid", "truncated tensor header")
        try:
            metadata = loads_strict(blob[16:header_end].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise SdkError("invalid_json", str(exc)) from exc
        if not isinstance(metadata, dict):
            raise SdkError("package_invalid", "tensor container header must be an object")
        if set(metadata) != {"quant_math_id", "tensors"}:
            raise SdkError("package_invalid", "tensor container header fields do not match v2")
        if metadata.get("quant_math_id") != QUANT_MATH_ID:
            raise SdkError("package_invalid", "CQ2 quant_math_id mismatch")
        tensors = metadata.get("tensors")
        if not isinstance(tensors, list) or not tensors:
            raise SdkError("package_invalid", "tensor container tensors must be a non-empty array")
        entries: dict[str, TensorEntry] = {}
        ranges: list[tuple[int, int, str]] = []
        allowed_entry_keys = {
            "name",
            "shape",
            "n_params",
            "dtype",
            "data",
            "scales",
            "bit_map",
            "group_size",
            "role",
            "transform",
            "codebook",
        }
        for raw in tensors:
            if not isinstance(raw, dict) or not set(raw) <= allowed_entry_keys:
                raise SdkError("package_invalid", "invalid tensor header entry")
            name_raw = raw.get("name")
            if not isinstance(name_raw, str):
                raise SdkError("package_invalid", "tensor name must be a string")
            name = name_raw
            if not name or name in entries:
                raise SdkError("duplicate_tensor", name)
            shape_raw = raw.get("shape")
            if (
                not isinstance(shape_raw, list)
                or any(
                    not isinstance(value, int) or isinstance(value, bool) or value <= 0
                    for value in shape_raw
                )
            ):
                raise SdkError("package_invalid", f"{name}.shape is invalid")
            shape = tuple(shape_raw)
            n_params_raw = raw.get("n_params")
            if (
                not isinstance(n_params_raw, int)
                or isinstance(n_params_raw, bool)
                or n_params_raw <= 0
            ):
                raise SdkError("package_invalid", f"{name}.n_params is invalid")
            n_params = n_params_raw
            # Canonical 51M contains scalar ``blocks.N.attn_gate`` tensors.
            # JSON shape=[] has the standard empty product of one.
            if math.prod(shape) != n_params:
                raise SdkError("package_invalid", f"{name}.shape mismatch")
            dtype = str(raw.get("dtype") or "")
            if dtype not in {"cq2", "cq4", "f16", "f32", "i8"}:
                raise SdkError("package_invalid", f"unsupported tensor dtype {dtype}")
            for required_key in ("role", "transform", "codebook"):
                if required_key not in raw:
                    raise SdkError(
                        "package_invalid", f"{name}.{required_key} is required in v2 header"
                    )
            data = _range(raw.get("data"), blob, name + ".data", ranges)
            scales = _range(raw.get("scales"), blob, name + ".scales", ranges, optional=True)
            bit_map = _range(raw.get("bit_map"), blob, name + ".bit_map", ranges, optional=True)
            if dtype in {"cq2", "cq4"}:
                if raw.get("group_size") != GROUP_SIZE:
                    raise SdkError("package_invalid", f"{name}.group_size must be 128")
                if "scales" not in raw or "bit_map" not in raw:
                    raise SdkError("package_invalid", f"{name} CQ metadata is required")
                expected_transform = "wht"
                expected_codebook = (
                    "gaussian-lloyd-q2-v1" if dtype == "cq2" else "gaussian-lloyd-q4-v1"
                )
                if raw.get("transform", expected_transform) != expected_transform:
                    raise SdkError("package_invalid", f"{name}.transform mismatch")
                if raw.get("codebook", expected_codebook) != expected_codebook:
                    raise SdkError("package_invalid", f"{name}.codebook mismatch")
                group_size = GROUP_SIZE
            else:
                if any(key in raw for key in ("scales", "bit_map", "group_size")):
                    raise SdkError("package_invalid", f"{name} safe dtype has CQ metadata")
                if raw.get("transform", "none") != "none" or raw.get("codebook", "none") != "none":
                    raise SdkError("package_invalid", f"{name} safe dtype metadata mismatch")
                group_size = GROUP_SIZE
            role = raw.get("role")
            if role not in {
                "lm",
                "contrastive",
                "mw_disposition",
                "confidence",
                "narration_adapter",
            }:
                raise SdkError("package_invalid", f"{name}.role is invalid")
            entries[name] = TensorEntry(
                name=name,
                shape=shape,
                n_params=n_params,
                dtype=dtype,
                data_offset=data[0],
                data_nbytes=data[1],
                scales_offset=scales[0],
                scales_nbytes=scales[1],
                bit_map_offset=bit_map[0],
                bit_map_nbytes=bit_map[1],
                group_size=group_size,
                role=role,
                transform=str(raw["transform"]),
                codebook=str(raw["codebook"]),
            )
        ranges.sort()
        cursor = header_end
        for start, end, label in ranges:
            if start < cursor:
                raise SdkError("package_range_invalid", f"{label} overlaps prior container data")
            if start != cursor:
                raise SdkError("package_range_invalid", f"gap before {label}")
            cursor = end
        if cursor != len(blob):
            raise SdkError("package_range_invalid", "tensor container has trailing or missing data")
        for entry in entries.values():
            _validate_entry_payload(blob, entry)
        return cls(blob, entries)

    @classmethod
    def load(cls, path: str | Path) -> "TensorContainer":
        return cls.parse(Path(path).read_bytes())

    def dequantize(self, name: str) -> list[float]:
        entry = self.entries.get(name)
        if entry is None:
            raise SdkError("package_invalid", f"missing tensor {name}")
        data = self.blob[entry.data_offset : entry.data_offset + entry.data_nbytes]
        if entry.dtype in {"cq2", "cq4"}:
            scales_blob = self.blob[
                entry.scales_offset : entry.scales_offset + entry.scales_nbytes
            ]
            if len(scales_blob) % 2:
                raise SdkError("package_invalid", f"{name}.scales is truncated")
            scales = tuple(
                struct.unpack_from("<H", scales_blob, offset)[0]
                for offset in range(0, len(scales_blob), 2)
            )
            bit_map = self.blob[
                entry.bit_map_offset : entry.bit_map_offset + entry.bit_map_nbytes
            ]
            values = dequantize(CqTensor(entry.n_params, data, scales, bit_map))
            if entry.dtype == "cq2" and all(
                (bit_map[group // 8] >> (group % 8)) & 1
                for group in range((entry.n_params + GROUP_SIZE - 1) // GROUP_SIZE)
            ):
                raise SdkError("package_invalid", f"{name} claims cq2 but contains only q4 groups")
            if entry.dtype == "cq4" and any(
                not ((bit_map[group // 8] >> (group % 8)) & 1)
                for group in range((entry.n_params + GROUP_SIZE - 1) // GROUP_SIZE)
            ):
                raise SdkError("package_invalid", f"{name} cq4 bit_map contains q2 groups")
            return values
        if entry.dtype == "f32":
            expected = entry.n_params * 4
            if len(data) != expected:
                raise SdkError("package_invalid", f"{name}.data length mismatch")
            return list(struct.unpack("<" + "f" * entry.n_params, data))
        if entry.dtype == "f16":
            expected = entry.n_params * 2
            if len(data) != expected:
                raise SdkError("package_invalid", f"{name}.data length mismatch")
            return [struct.unpack_from("<e", data, offset)[0] for offset in range(0, len(data), 2)]
        if len(data) != entry.n_params:
            raise SdkError("package_invalid", f"{name}.data length mismatch")
        return [float(value) for value in struct.unpack("<" + "b" * entry.n_params, data)]


def _range(
    raw: Any,
    blob: bytes,
    label: str,
    ranges: list[tuple[int, int, str]],
    *,
    optional: bool = False,
) -> tuple[int, int]:
    if raw is None and optional:
        return 0, 0
    if not isinstance(raw, dict):
        raise SdkError("package_range_invalid", f"missing {label}")
    offset = raw.get("offset")
    nbytes = raw.get("nbytes")
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or not isinstance(nbytes, int)
        or isinstance(nbytes, bool)
        or offset < 0
        or nbytes < 0
        or offset + nbytes > len(blob)
    ):
        raise SdkError("package_range_invalid", label)
    if nbytes:
        ranges.append((offset, offset + nbytes, label))
    return offset, nbytes


def _validate_entry_payload(blob: bytes, entry: TensorEntry) -> None:
    data = blob[entry.data_offset : entry.data_offset + entry.data_nbytes]
    if entry.dtype in {"cq2", "cq4"}:
        groups = (entry.n_params + GROUP_SIZE - 1) // GROUP_SIZE
        if entry.scales_nbytes != groups * 2:
            raise SdkError("package_invalid", f"{entry.name}.scales length mismatch")
        if entry.bit_map_nbytes != (groups + 7) // 8:
            raise SdkError("package_invalid", f"{entry.name}.bit_map length mismatch")
        bit_map = blob[
            entry.bit_map_offset : entry.bit_map_offset + entry.bit_map_nbytes
        ]
        if groups % 8 and bit_map[-1] >> (groups % 8):
            raise SdkError("package_invalid", f"{entry.name}.bit_map has non-zero padding bits")
        q4_groups = sum(
            (bit_map[group // 8] >> (group % 8)) & 1 for group in range(groups)
        )
        expected_data = groups * 32 + q4_groups * 32
        if len(data) != expected_data:
            raise SdkError("package_invalid", f"{entry.name}.data/bit_map length mismatch")
        if entry.dtype == "cq4" and q4_groups != groups:
            raise SdkError("package_invalid", f"{entry.name} cq4 contains q2 groups")
        if entry.dtype == "cq2" and q4_groups == groups:
            raise SdkError("package_invalid", f"{entry.name} cq2 cannot be all-q4")
        scales = blob[
            entry.scales_offset : entry.scales_offset + entry.scales_nbytes
        ]
        for offset in range(0, len(scales), 2):
            scale = _f16_to_f32(struct.unpack_from("<H", scales, offset)[0])
            if not math.isfinite(scale) or scale <= 0:
                raise SdkError("package_invalid", f"{entry.name} has invalid CQ scale")
        return
    expected = entry.n_params * {"f16": 2, "f32": 4, "i8": 1}[entry.dtype]
    if len(data) != expected:
        raise SdkError("package_invalid", f"{entry.name}.data length mismatch")
    if entry.scales_nbytes or entry.bit_map_nbytes:
        raise SdkError("package_invalid", f"{entry.name} safe dtype has CQ metadata")
    if entry.dtype == "f16":
        values = (struct.unpack_from("<e", data, offset)[0] for offset in range(0, len(data), 2))
    elif entry.dtype == "f32":
        values = (struct.unpack_from("<f", data, offset)[0] for offset in range(0, len(data), 4))
    else:
        return
    if any(not math.isfinite(value) for value in values):
        raise SdkError("package_invalid", f"{entry.name} contains NaN/Inf")
