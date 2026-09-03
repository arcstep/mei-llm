"""Shared 51M pack math: block Q2/Q4, STE, activation/KV fake-quant, binary package.

This is the deployment-kernel math. QAT fake-quant, Python dequant, and the
Rust/WASM kernels must bit-match these codes and scales.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

QUANT_MATH_ID = "mei-qpack-v1-block64-q4s7-q2u4"
PACK_MAGIC = b"MEIQPK01"
PACK_VERSION = 1
BLOCK_SIZE = 64
ALIGN = 64
Q4_LEVELS = 7
Q4_CODE_MIN = -8
Q4_CODE_MAX = 7
Q2_LEVELS = 1.5  # codes 0..3 reconstruct as (code - 1.5) * scale, scale = max_abs / 1.5
ENDIANNESS = "little"
WEIGHTS_FORMAT = "mei-q4-packed-v1"

FP32_BYTES = 4
FP16_BYTES = 2

ALLOWED_PRODUCTS = ("mei-1.0-51m", "mei-1.0-51m")

Q4_BASELINE_PAYLOAD_BYTES = 51_463_797 * 4 / 8  # 25_731_898.5 → ceil in reports
Q4_PACKAGE_BUDGET_BYTES = 30 * 1024 * 1024
CQ2_AVG_BITS = 3.0
CQ2_RAW_PAYLOAD_BUDGET = 19.30 * 1024 * 1024


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_bytes(n_params: int, bits: int) -> int:
    if bits >= 32:
        return int(n_params) * 4
    if bits == 16:
        return int(n_params) * 2
    return (int(n_params) * int(bits) + 7) // 8


def scale_bytes(n_params: int, block_size: int = BLOCK_SIZE, bits: int = 4) -> int:
    if bits >= 16:
        return 0
    n_blocks = (int(n_params) + block_size - 1) // block_size
    return n_blocks * 4


def pad_n(n: int, align: int = ALIGN) -> int:
    return (n + align - 1) // align * align


def _as_f32(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float32).reshape(-1)


def block_view(flat: np.ndarray, block_size: int = BLOCK_SIZE) -> tuple[np.ndarray, int]:
    n = int(flat.size)
    n_blocks = (n + block_size - 1) // block_size
    padded = np.zeros((n_blocks * block_size,), dtype=np.float32)
    padded[:n] = flat.astype(np.float32, copy=False)
    return padded.reshape(n_blocks, block_size), n


def quantize_q4_codes(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    abs_max = np.maximum(np.max(np.abs(block), axis=-1), 1e-8).astype(np.float32)
    scale = abs_max / float(Q4_LEVELS)
    q = np.clip(np.round(block / scale[:, None]), Q4_CODE_MIN, Q4_CODE_MAX).astype(np.int8)
    return q, scale


def dequant_q4_codes(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return codes.astype(np.float32) * scale[:, None]


def quantize_q2_codes(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    abs_max = np.maximum(np.max(np.abs(block), axis=-1), 1e-8).astype(np.float32)
    scale = abs_max / float(Q2_LEVELS)
    q = np.clip(np.round(block / scale[:, None] + Q2_LEVELS), 0, 3).astype(np.uint8)
    return q, scale


def dequant_q2_codes(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (codes.astype(np.float32) - Q2_LEVELS) * scale[:, None]


def fake_quant_block(arr, bits: int, block_size: int = BLOCK_SIZE) -> np.ndarray:
    """Reconstruct arr with the same block math as the packed kernel (numpy)."""
    src = np.asarray(arr, dtype=np.float32)
    shape = src.shape
    flat = src.reshape(-1)
    blocks, n = block_view(flat, block_size)
    if bits <= 2:
        codes, scale = quantize_q2_codes(blocks)
        recon = dequant_q2_codes(codes, scale).reshape(-1)[:n]
    elif bits <= 4:
        codes, scale = quantize_q4_codes(blocks)
        recon = dequant_q4_codes(codes, scale).reshape(-1)[:n]
    elif bits <= 16:
        recon = flat.astype(np.float16).astype(np.float32)
    else:
        recon = flat
    return recon.reshape(shape)


def reconstruction_mse(arr, bits: int, block_size: int = BLOCK_SIZE) -> float:
    src = np.asarray(arr, dtype=np.float32)
    recon = fake_quant_block(src, bits, block_size)
    return float(np.mean((src - recon) ** 2))


def pack_nibbles(codes: np.ndarray) -> bytes:
    """Pack int8 codes in [-8, 7] as unsigned nibbles (code+8)."""
    u = (codes.astype(np.int16) + 8).clip(0, 15).astype(np.uint8).reshape(-1)
    if u.size % 2:
        u = np.concatenate([u, np.zeros((1,), dtype=np.uint8)])
    lo = u[0::2]
    hi = u[1::2]
    return np.bitwise_or(lo, np.left_shift(hi.astype(np.uint8), 4)).tobytes()


def unpack_nibbles(blob: bytes, n: int) -> np.ndarray:
    raw = np.frombuffer(blob, dtype=np.uint8)
    need = (n + 1) // 2
    if raw.size < need:
        raise ValueError(f"nibble payload short: have={raw.size} need={need}")
    lo = np.bitwise_and(raw[:need], 0x0F).astype(np.int8)
    hi = np.right_shift(raw[:need], 4).astype(np.int8)
    codes = np.empty((need * 2,), dtype=np.int8)
    codes[0::2] = lo - 8
    codes[1::2] = hi - 8
    return codes[:n]


def pack_twobit(codes: np.ndarray) -> bytes:
    u = codes.astype(np.uint8).reshape(-1) & 0x03
    pad = (4 - (u.size % 4)) % 4
    if pad:
        u = np.concatenate([u, np.zeros((pad,), dtype=np.uint8)])
    packed = u[0::4] | (u[1::4] << 2) | (u[2::4] << 4) | (u[3::4] << 6)
    return packed.astype(np.uint8).tobytes()


def unpack_twobit(blob: bytes, n: int) -> np.ndarray:
    raw = np.frombuffer(blob, dtype=np.uint8)
    need = (n + 3) // 4
    if raw.size < need:
        raise ValueError(f"q2 payload short: have={raw.size} need={need}")
    codes = np.empty((need * 4,), dtype=np.uint8)
    chunk = raw[:need]
    codes[0::4] = chunk & 0x03
    codes[1::4] = (chunk >> 2) & 0x03
    codes[2::4] = (chunk >> 4) & 0x03
    codes[3::4] = (chunk >> 6) & 0x03
    return codes[:n]


@dataclass
class TensorEntry:
    name: str
    shape: list[int]
    n_params: int
    bits: int
    block_size: int
    n_blocks: int
    packed_offset: int
    packed_nbytes: int
    scale_offset: int
    scale_nbytes: int
    dtype: str = "float32"


@dataclass
class PackHeader:
    magic: str
    version: int
    quant_math_id: str
    endianness: str
    block_size: int
    n_tensors: int
    tensors: list[TensorEntry] = field(default_factory=list)


def _align_offset(offset: int) -> int:
    return pad_n(offset, ALIGN)


def pack_tensor_payload(arr: np.ndarray, bits: int, block_size: int = BLOCK_SIZE) -> tuple[bytes, bytes, TensorEntry]:
    src = np.asarray(arr, dtype=np.float32)
    shape = [int(x) for x in src.shape]
    n = int(src.size)
    if bits >= 32:
        blob = src.astype("<f4", copy=False).tobytes()
        entry = TensorEntry(
            name="",
            shape=shape,
            n_params=n,
            bits=32,
            block_size=block_size,
            n_blocks=0,
            packed_offset=0,
            packed_nbytes=len(blob),
            scale_offset=0,
            scale_nbytes=0,
        )
        return blob, b"", entry
    blocks, _n = block_view(src.reshape(-1), block_size)
    n_blocks = int(blocks.shape[0])
    if bits <= 2:
        codes, scale = quantize_q2_codes(blocks)
        packed = pack_twobit(codes.reshape(-1)[: n_blocks * block_size])
    else:
        codes, scale = quantize_q4_codes(blocks)
        packed = pack_nibbles(codes.reshape(-1)[: n_blocks * block_size])
    scales = scale.astype("<f4", copy=False).tobytes()
    entry = TensorEntry(
        name="",
        shape=shape,
        n_params=n,
        bits=int(bits),
        block_size=block_size,
        n_blocks=n_blocks,
        packed_offset=0,
        packed_nbytes=len(packed),
        scale_offset=0,
        scale_nbytes=len(scales),
    )
    return packed, scales, entry


def unpack_tensor_payload(packed: bytes, scales: bytes, entry: TensorEntry) -> np.ndarray:
    n = int(entry.n_params)
    shape = tuple(int(x) for x in entry.shape)
    if entry.bits >= 32:
        arr = np.frombuffer(packed, dtype="<f4", count=n)
        return arr.reshape(shape).astype(np.float32, copy=True)
    n_blocks = int(entry.n_blocks)
    scale = np.frombuffer(scales, dtype="<f4", count=n_blocks)
    if entry.bits <= 2:
        codes = unpack_twobit(packed, n_blocks * entry.block_size)
        recon = dequant_q2_codes(codes.reshape(n_blocks, entry.block_size), scale)
    else:
        codes = unpack_nibbles(packed, n_blocks * entry.block_size)
        recon = dequant_q4_codes(codes.reshape(n_blocks, entry.block_size), scale)
    return recon.reshape(-1)[:n].reshape(shape).astype(np.float32, copy=True)


def build_pack_bytes(tensors: dict[str, np.ndarray], bit_map: dict[str, int]) -> tuple[bytes, PackHeader]:
    """Serialize named tensors. bit_map may be per-tensor int or missing (default 4)."""
    blobs: list[tuple[str, bytes, bytes, TensorEntry]] = []
    for name, arr in tensors.items():
        bits = int(bit_map.get(name, 4))
        packed, scales, entry = pack_tensor_payload(arr, bits)
        entry.name = name
        blobs.append((name, packed, scales, entry))

    def serialize_header(entries_in: list[TensorEntry]) -> bytes:
        payload = {
            "magic": PACK_MAGIC.decode("ascii"),
            "version": PACK_VERSION,
            "quant_math_id": QUANT_MATH_ID,
            "endianness": ENDIANNESS,
            "block_size": BLOCK_SIZE,
            "n_tensors": len(entries_in),
            "tensors": [asdict(item) for item in entries_in],
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def entry_with_offsets(src: TensorEntry, packed_off: int, scale_off: int) -> TensorEntry:
        return TensorEntry(
            name=src.name,
            shape=src.shape,
            n_params=src.n_params,
            bits=src.bits,
            block_size=src.block_size,
            n_blocks=src.n_blocks,
            packed_offset=packed_off,
            packed_nbytes=src.packed_nbytes,
            scale_offset=scale_off,
            scale_nbytes=src.scale_nbytes,
            dtype=src.dtype,
        )

    def layout(payload_start: int) -> tuple[list[TensorEntry], list[bytes]]:
        cursor = payload_start
        filled: list[TensorEntry] = []
        payload_blobs: list[bytes] = []
        for _name, packed, scales, entry in blobs:
            scale_off = cursor
            cursor += entry.scale_nbytes
            packed_off = cursor
            cursor += entry.packed_nbytes
            next_cur = _align_offset(cursor)
            filled.append(entry_with_offsets(entry, packed_off, scale_off))
            payload_blobs.append(scales + packed + (b"\x00" * (next_cur - cursor)))
            cursor = next_cur
        return filled, payload_blobs

    # Reserve an aligned header region large enough that offsets never depend on JSON width.
    dummy = [entry_with_offsets(item[3], 10**12, 10**12) for item in blobs]
    probe = serialize_header(dummy)
    payload_start = _align_offset(8 + 4 + 4 + len(probe) + 256)
    filled, payload_blobs = layout(payload_start)
    header_json = serialize_header(filled)
    prefix_len = 8 + 4 + 4 + len(header_json)
    if prefix_len > payload_start:
        payload_start = _align_offset(prefix_len)
        filled, payload_blobs = layout(payload_start)
        header_json = serialize_header(filled)
        prefix_len = 8 + 4 + 4 + len(header_json)
    pad = payload_start - prefix_len
    out = bytearray()
    out.extend(PACK_MAGIC)
    out.extend(struct.pack("<II", PACK_VERSION, len(header_json)))
    out.extend(header_json)
    out.extend(b"\x00" * pad)
    for blob in payload_blobs:
        out.extend(blob)
    header = PackHeader(
        magic=PACK_MAGIC.decode("ascii"),
        version=PACK_VERSION,
        quant_math_id=QUANT_MATH_ID,
        endianness=ENDIANNESS,
        block_size=BLOCK_SIZE,
        n_tensors=len(filled),
        tensors=filled,
    )
    return bytes(out), header


def parse_pack_bytes(blob: bytes) -> tuple[PackHeader, dict[str, np.ndarray]]:
    if blob[:8] != PACK_MAGIC:
        raise ValueError(f"bad pack magic: {blob[:8]!r}")
    version, header_len = struct.unpack_from("<II", blob, 8)
    if version != PACK_VERSION:
        raise ValueError(f"unsupported pack version {version}")
    header_raw = blob[16 : 16 + header_len]
    meta = json.loads(header_raw.decode("utf-8"))
    if meta.get("quant_math_id") != QUANT_MATH_ID:
        raise ValueError(f"quant_math_id mismatch: {meta.get('quant_math_id')}")
    tensors: list[TensorEntry] = []
    arrays: dict[str, np.ndarray] = {}
    for row in meta.get("tensors") or []:
        entry = TensorEntry(
            name=str(row["name"]),
            shape=[int(x) for x in row["shape"]],
            n_params=int(row["n_params"]),
            bits=int(row["bits"]),
            block_size=int(row.get("block_size") or BLOCK_SIZE),
            n_blocks=int(row.get("n_blocks") or 0),
            packed_offset=int(row["packed_offset"]),
            packed_nbytes=int(row["packed_nbytes"]),
            scale_offset=int(row["scale_offset"]),
            scale_nbytes=int(row["scale_nbytes"]),
            dtype=str(row.get("dtype") or "float32"),
        )
        tensors.append(entry)
        packed = blob[entry.packed_offset : entry.packed_offset + entry.packed_nbytes]
        scales = blob[entry.scale_offset : entry.scale_offset + entry.scale_nbytes] if entry.scale_nbytes else b""
        arrays[entry.name] = unpack_tensor_payload(packed, scales, entry)
    header = PackHeader(
        magic=str(meta.get("magic")),
        version=int(meta.get("version") or version),
        quant_math_id=str(meta.get("quant_math_id")),
        endianness=str(meta.get("endianness") or ENDIANNESS),
        block_size=int(meta.get("block_size") or BLOCK_SIZE),
        n_tensors=int(meta.get("n_tensors") or len(tensors)),
        tensors=tensors,
    )
    return header, arrays


def load_pack_file(path: Path) -> tuple[PackHeader, dict[str, np.ndarray]]:
    return parse_pack_bytes(Path(path).read_bytes())


def component_of(name: str) -> str:
    if name.startswith("embed") or name.startswith("lm_head"):
        return "embedding"
    if ".attn_norm" in name or ".mlp_norm" in name or ".post_attn_norm" in name or name.startswith("final_norm"):
        return "norm"
    if name.startswith("mhc_") or "mhc" in name:
        return "mhc"
    if name.startswith("engrams") or ".engrams" in name:
        return "engram"
    if name.startswith("contrastive"):
        return "retrieval"
    if name.startswith("conf_v2") or name.startswith("conf_"):
        return "confidence"
    if ".attn." in name:
        return "attention"
    if ".mlp." in name:
        return "mlp"
    return "other"


def default_bits_for(name: str, n_params: int) -> int:
    """Tiny vectors stay f32; large tensors default Q4. Embedding is always Q4."""
    if n_params < BLOCK_SIZE:
        return 32
    if component_of(name) == "norm" and n_params <= 512:
        return 32
    return 4


def per_block_scores(name: str, arr: np.ndarray, block_size: int = BLOCK_SIZE) -> list[dict[str, Any]]:
    src = np.asarray(arr, dtype=np.float32).reshape(-1)
    n = int(src.size)
    if n < block_size:
        return []
    blocks, _ = block_view(src, block_size)
    n_full = n // block_size
    if n_full <= 0:
        return []
    use = blocks[:n_full]
    q4_codes, q4_scale = quantize_q4_codes(use)
    q2_codes, q2_scale = quantize_q2_codes(use)
    recon4 = dequant_q4_codes(q4_codes, q4_scale)
    recon2 = dequant_q2_codes(q2_codes, q2_scale)
    mse4 = np.mean((use - recon4) ** 2, axis=1)
    mse2 = np.mean((use - recon2) ** 2, axis=1)
    extra = np.maximum(mse2 - mse4, 0.0)
    bytes_saved = float(block_size) * (4 - 2) / 8.0
    rows = []
    for i in range(n_full):
        loss = float(extra[i])
        rows.append(
            {
                "name": name,
                "block": i,
                "n_params": block_size,
                "mse_q4": float(mse4[i]),
                "mse_q2": float(mse2[i]),
                "quality_loss": loss,
                "bytes_saved": bytes_saved,
                "loss_per_byte": loss / bytes_saved if bytes_saved else math.inf,
            }
        )
    return rows


def propose_mixed_bit_map(
    tensors: dict[str, np.ndarray],
    *,
    target_avg_bits: float = CQ2_AVG_BITS,
    embed_bits: int = 4,
) -> dict[str, Any]:
    """Assign Q2 to lowest loss-per-byte blocks until average bits <= target.

    Embedding stays Q4. Returns per-tensor bits plus optional per-block overrides
    for mixed tensors. This is a *candidate*; product_final stays false until QAT.
    """
    total = sum(int(np.asarray(v).size) for v in tensors.values())
    # bits_i * n_i / total <= target  =>  2*n_q2 + 4*n_q4 + 32*n_f32 <= target * total
    # Treat f32 as fixed. Remaining large tensors are 4 or 2.
    f32_params = 0
    q_candidates: list[dict[str, Any]] = []
    tensor_bits: dict[str, int] = {}
    block_bits: dict[str, list[int]] = {}
    for name, arr in tensors.items():
        n = int(np.asarray(arr).size)
        if component_of(name) == "embedding":
            tensor_bits[name] = embed_bits
            continue
        if default_bits_for(name, n) == 32:
            tensor_bits[name] = 32
            f32_params += n
            continue
        tensor_bits[name] = 4
        q_candidates.extend(per_block_scores(name, arr))

    q_params = total - f32_params - sum(
        int(np.asarray(tensors[n]).size) for n, b in tensor_bits.items() if b == embed_bits and component_of(n) == "embedding"
    )
    embed_params = sum(int(np.asarray(v).size) for n, v in tensors.items() if component_of(n) == "embedding")
    # Weighted average: (embed*4 + f32*32 + q2*2 + q4*4) / total
    # We cannot change embed/f32. Need avg <= 3.0:
    # (4*embed + 32*f32 + 2*q2 + 4*(q_rest-q2)) / total <= 3
    # 4*embed + 32*f32 + 4*q_rest - 2*q2 <= 3 * total
    q_rest = total - embed_params - f32_params
    rhs = target_avg_bits * total - 4 * embed_params - 32 * f32_params - 4 * q_rest
    # -2 * q2 <= rhs - wait: 4*embed + 32*f32 + 4*q_rest - 2*q2 <= 3*total
    # -2*q2 <= 3*total - 4*embed - 32*f32 - 4*q_rest
    # q2 >= (4*embed + 32*f32 + 4*q_rest - 3*total) / 2
    need_q2 = (4 * embed_params + 32 * f32_params + 4 * q_rest - target_avg_bits * total) / 2.0
    need_q2 = max(0.0, need_q2)

    ranked = sorted(q_candidates, key=lambda row: (float(row["loss_per_byte"]), row["name"], row["block"]))
    assigned_q2 = 0
    chosen_keys: set[tuple[str, int]] = set()
    for row in ranked:
        if assigned_q2 >= need_q2:
            break
        chosen_keys.add((row["name"], int(row["block"])))
        assigned_q2 += int(row["n_params"])

    per_tensor_q2: dict[str, int] = {}
    for name, arr in tensors.items():
        if tensor_bits.get(name) != 4:
            continue
        n = int(np.asarray(arr).size)
        n_blocks = n // BLOCK_SIZE
        bits_list = [2 if (name, i) in chosen_keys else 4 for i in range(n_blocks)]
        n_q2 = sum(1 for b in bits_list if b == 2) * BLOCK_SIZE
        per_tensor_q2[name] = n_q2
        if n_q2 == 0:
            tensor_bits[name] = 4
        elif n_q2 >= (n_blocks * BLOCK_SIZE) and n % BLOCK_SIZE == 0:
            tensor_bits[name] = 2
        else:
            tensor_bits[name] = 4  # mixed stored as Q4 tensor + block map
            block_bits[name] = bits_list + ([4] if n % BLOCK_SIZE else [])

    raw_payload = 0.0
    weighted_bits = 0.0
    for name, arr in tensors.items():
        n = int(np.asarray(arr).size)
        bits = int(tensor_bits[name])
        if name in block_bits:
            nb = block_bits[name]
            n_q2 = sum(1 for b in nb if b == 2) * BLOCK_SIZE
            n_q4 = n - n_q2
            raw_payload += payload_bytes(n_q2, 2) + payload_bytes(n_q4, 4)
            weighted_bits += 2 * n_q2 + 4 * n_q4
        else:
            raw_payload += payload_bytes(n, bits)
            weighted_bits += bits * n
    avg_bits = weighted_bits / max(total, 1)
    quality_mean = (
        float(np.mean([row["loss_per_byte"] for row in ranked[: max(1, len(chosen_keys))]])) if chosen_keys else 0.0
    )
    meets_size = raw_payload <= CQ2_RAW_PAYLOAD_BUDGET + 1 and avg_bits <= target_avg_bits + 1e-6
    # Reconstruction-quality gate: mixed is blocked for product until QAT recovers quality.
    return {
        "kind": "q2q4-product-candidate-bit-map",
        "candidate": True,
        "product_final": False,
        "quality_blocked": True,
        "quality_blocked_reason": (
            "CQ2 mixed map meets the byte budget only by putting ~50% of non-embedding "
            "params at 2-bit. Quality is not recovered until QAT + three-runtime parity."
        ),
        "qat_mandatory": True,
        "quant_math_id": QUANT_MATH_ID,
        "block_size": BLOCK_SIZE,
        "embed_bits": embed_bits,
        "target_avg_bits": target_avg_bits,
        "params": total,
        "embed_params": embed_params,
        "f32_params": f32_params,
        "q2_params": int(assigned_q2),
        "need_q2_params": int(math.ceil(need_q2)),
        "avg_bits": avg_bits,
        "raw_payload_bytes": int(raw_payload),
        "raw_payload_mb": raw_payload / (1024 * 1024),
        "budget_raw_payload_mb": CQ2_RAW_PAYLOAD_BUDGET / (1024 * 1024),
        "meets_size_budget": bool(meets_size),
        "tensor_bits": tensor_bits,
        "mixed_block_tensors": {k: int(sum(1 for b in v if b == 2)) for k, v in block_bits.items()},
        "n_mixed_tensors": len(block_bits),
        "mean_chosen_loss_per_byte": quality_mean,
        "ranking_rule": "quality_loss / bytes_saved, lowest first; embedding stays Q4; tiny tensors stay f32",
    }


def coarsen_mixed_candidate_to_tensor_bits(candidate: dict[str, Any], sizes: dict[str, int]) -> dict[str, Any]:
    """Turn block-mixed candidate counts into packable per-tensor {2,4,32} bits.

    The pack kernel stores one bit-width per tensor. Whole-tensor Q2 is a
    conservative coarsening of the candidate's per-block Q2 assignment.
    """
    tb = {str(k): int(v) for k, v in (candidate.get("tensor_bits") or {}).items()}
    mixed = candidate.get("mixed_block_tensors") or {}
    need = int(candidate.get("need_q2_params") or 0)
    assigned = sum(int(sizes.get(name, 0)) for name, bits in tb.items() if int(bits) == 2)
    ranked = sorted(((str(k), int(v)) for k, v in mixed.items()), key=lambda kv: (-int(kv[1]), kv[0]))
    promoted: list[dict[str, Any]] = []
    for name, n_blocks in ranked:
        if assigned >= need:
            break
        if int(tb.get(name, 4)) == 2:
            continue
        n = int(sizes.get(name) or 0)
        if n <= 0:
            continue
        tb[name] = 2
        assigned += n
        promoted.append({"name": name, "n_params": n, "q2_blocks_in_candidate": n_blocks})
    for name, n in sizes.items():
        bits = int(tb.get(name, 4))
        if bits not in (2, 4, 32):
            tb[name] = 4
        elif name not in tb:
            tb[name] = 4
    raw = 0
    weighted = 0.0
    total = sum(int(n) for n in sizes.values()) or int(candidate.get("params") or 1)
    for name, n in sizes.items():
        bits = int(tb.get(name, 4))
        raw += payload_bytes(int(n), bits)
        weighted += bits * int(n)
    out = dict(candidate)
    out.update(
        {
            "kind": "q2q4-product-tensor-bits-coarsened",
            "candidate": False,
            "product_final": False,
            "quality_blocked": True,
            "coarsened_from_mixed_blocks": True,
            "coarsen_rule": (
                "Whole-tensor Q2 in descending candidate Q2-block count until need_q2_params; "
                "the pack kernel is per-tensor, not per-block."
            ),
            "q2_params_whole_tensor": assigned,
            "promoted_tensors": promoted,
            "n_promoted": len(promoted),
            "tensor_bits": tb,
            "raw_payload_bytes": int(raw),
            "raw_payload_mb": raw / (1024 * 1024),
            "avg_bits": weighted / max(total, 1),
            "meets_size_budget": bool(raw <= CQ2_RAW_PAYLOAD_BUDGET + 1),
        }
    )
    return out


def q4_baseline_map(tensors: dict[str, np.ndarray]) -> dict[str, Any]:
    bits: dict[str, int] = {}
    total = 0
    raw = 0
    for name, arr in tensors.items():
        n = int(np.asarray(arr).size)
        total += n
        b = 32 if default_bits_for(name, n) == 32 else 4
        bits[name] = b
        raw += payload_bytes(n, b)
    return {
        "kind": "q4-baseline-bit-map",
        "candidate": False,
        "product_final": False,
        "qat_mandatory": True,
        "quant_math_id": QUANT_MATH_ID,
        "block_size": BLOCK_SIZE,
        "params": total,
        "avg_bits": 4.0,
        "raw_payload_bytes": int(raw),
        "raw_payload_mb": raw / (1024 * 1024),
        "package_budget_mb": Q4_PACKAGE_BUDGET_BYTES / (1024 * 1024),
        "tensor_bits": bits,
        "rule": "All large tensors Q4 block-64; embedding Q4; tiny/norm vectors f32.",
    }


# ---------------------------------------------------------------------------
# MLX STE / activation-KV (imported by quant_ops_51m)
# ---------------------------------------------------------------------------

STE_IMPLEMENTED = True
KERNEL_FEASIBILITY_RECORDED = True
ACTIVATION_KV_BITS = 8


def mlx_fake_quant_weight(arr, bits: int = 4, block_size: int = BLOCK_SIZE):
    import mlx.core as mx

    xf = arr.astype(mx.float32)
    flat = xf.reshape(-1)
    n = int(flat.size)
    n_blocks = (n + block_size - 1) // block_size
    pad = n_blocks * block_size - n
    if pad:
        flat = mx.concatenate([flat, mx.zeros((pad,), dtype=mx.float32)])
    blocks = flat.reshape((n_blocks, block_size))
    abs_max = mx.maximum(mx.max(mx.abs(blocks), axis=-1), 1e-8)
    if bits <= 2:
        scale = abs_max / float(Q2_LEVELS)
        q = mx.clip(mx.round(blocks / scale[:, None] + Q2_LEVELS), 0, 3)
        recon = (q - Q2_LEVELS) * scale[:, None]
    else:
        scale = abs_max / float(Q4_LEVELS)
        q = mx.clip(mx.round(blocks / scale[:, None]), Q4_CODE_MIN, Q4_CODE_MAX)
        recon = q * scale[:, None]
    out = recon.reshape(-1)[:n].reshape(xf.shape)
    return out.astype(arr.dtype)


def mlx_ste_quantize(arr, bits: int = 4, block_size: int = BLOCK_SIZE):
    """Straight-through estimator: forward fake-quant, backward identity."""
    import mlx.core as mx

    xf = arr.astype(mx.float32)
    q = mlx_fake_quant_weight(xf, bits=bits, block_size=block_size)
    return (xf + mx.stop_gradient(q - xf)).astype(arr.dtype)


def mlx_fake_quant_activation(arr, bits: int = ACTIVATION_KV_BITS):
    import mlx.core as mx

    xf = arr.astype(mx.float32)
    levels = float((1 << (bits - 1)) - 1)
    # Needle fake_quant_act uses group_size=x.shape[-1]: one scale for each
    # vector on the last axis, never one scale for the whole tensor.
    abs_max = mx.max(mx.abs(xf), axis=-1, keepdims=True)
    scale = mx.where(abs_max > 0, abs_max / levels, 1.0)
    q = mx.clip(mx.round(xf / scale), -levels - 1, levels)
    return (q * scale).astype(arr.dtype)


def mlx_ste_activation(arr, bits: int = ACTIVATION_KV_BITS):
    import mlx.core as mx

    xf = arr.astype(mx.float32)
    q = mlx_fake_quant_activation(xf, bits=bits)
    return (xf + mx.stop_gradient(q - xf)).astype(arr.dtype)
