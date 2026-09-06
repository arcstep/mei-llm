#!/usr/bin/env python3
"""Pack a 51M QAT FP32 master into a new quantized package.

Never overwrites artifacts/mei-1.2-51m/legacy/mei-1.0-51m/exp-00300m/models/base/mei-1.0-51m-base-scratch300m-v1 or the PTQ q4-v1 package.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from common.identity_51m import (
    ARCHITECTURE_ID,
    ARCHITECTURE_SPEC,
    EXPECTED_PARAMS,
    JOBS_DIR,
    MODEL_ID,
    Q4_BASELINE_MAP_NAME,
    QAT_MANDATORY_NOTE,
    QAT_Q4_MODEL_ID,
    QAT_Q4_PACKAGE_DIR,
    QAT_Q4_PACKAGE_ID,
    QAT_Q4_RELEASE_PATH,
    QAT_Q4_WEIGHTS_PATH,
    RELEASE_PATH,
    ROOT,
    WEIGHTS_PATH,
    fail,
    load_json,
    sha256_file,
    validate_release,
    write_json,
)
from release.pack_q4_51m import export_vocab
from release.quant_pack_51m import (
    BLOCK_SIZE,
    ENDIANNESS,
    Q4_PACKAGE_BUDGET_BYTES,
    QUANT_MATH_ID,
    WEIGHTS_FORMAT,
    build_pack_bytes,
    sha256_bytes,
)
from tokenizer import ZhTokenizerV1


def write_manifest(
    *,
    package_id: str,
    parent_id: str,
    weights_name: str,
    weights_sha: str,
    tok_name: str,
    tok_sha256: str,
    vocab_name: str,
    vocab_sha: str,
    header,
    bit_map: dict,
    parent_sha: str,
    scheme: str,
    status: str,
) -> dict:
    from config import NeedleZhConfig

    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    directory = []
    for item in header.tensors:
        directory.append(
            {
                "name": item.name,
                "shape": item.shape,
                "n_params": item.n_params,
                "bits": item.bits,
                "block_size": item.block_size,
                "n_blocks": item.n_blocks,
                "packed_offset": item.packed_offset,
                "packed_nbytes": item.packed_nbytes,
                "scale_offset": item.scale_offset,
                "scale_nbytes": item.scale_nbytes,
                "dtype": item.dtype,
            }
        )
    bit_blob = json.dumps(bit_map.get("tensor_bits") or {}, sort_keys=True).encode("utf-8")
    return {
        "package_format": "mei-model-package-v1",
        "product": "mei-1.0-51m",
        "package_id": package_id,
        "runtime_min": "mei-runtime-abi-1",
        "parent_package_id": parent_id,
        "parent_weights_sha256": parent_sha,
        "quant_math_id": QUANT_MATH_ID,
        "endianness": ENDIANNESS,
        "bit_map_sha256": sha256_bytes(bit_blob),
        "architecture": {
            "id": ARCHITECTURE_ID,
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "head_dim": cfg.head_dim,
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "rope_theta": cfg.rope_theta,
            "engram_layers": list(cfg.engram_layers),
            "engram_orders": list(cfg.engram_orders),
            "engram_slots": cfg.engram_slots,
            "engram_conv_taps": cfg.engram_conv_taps,
            "mhc_lanes": cfg.mhc_lanes,
            "sinkhorn_iters": cfg.sinkhorn_iters,
            "tie_embeddings": cfg.tie_embeddings,
            "rms_eps": cfg.rms_eps,
            "conf_probes": cfg.conf_probes,
            "kv_window": cfg.kv_window,
        },
        "tokenizer": {
            "id": "zh-24k-v1",
            "file": tok_name,
            "sha256": tok_sha256,
            "vocab_file": vocab_name,
            "vocab_sha256": vocab_sha,
        },
        "weights": {
            "file": weights_name,
            "format": WEIGHTS_FORMAT,
            "sha256": weights_sha,
            "quantization": {
                "scheme": scheme,
                "status": status,
                "block_size": BLOCK_SIZE,
                "quant_math_id": QUANT_MATH_ID,
            },
            "directory": directory,
            "payload_bytes": int(bit_map.get("raw_payload_bytes") or 0),
        },
        "heads": {
            "lm": {"present": True, "trained": True, "status": "ready"},
            "contrastive": {"present": False, "trained": False, "status": "missing"},
            "mw_disposition": {"present": False, "trained": False, "status": "missing"},
            "confidence": {"present": True, "trained": False, "status": "untrained"},
        },
        "release_class": "experimental",
        "qat_mandatory": True,
        "note": QAT_MANDATORY_NOTE,
    }


def numpy_from_master(weights: Path) -> dict[str, np.ndarray]:
    from architecture import NeedleZh, count_params
    from common.checkpoint import flatten_params, load_params
    from config import NeedleZhConfig

    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    model = NeedleZh(cfg)
    import mlx.core as mx

    mx.eval(model.parameters())
    load_params(model, weights, strict=True)
    mx.eval(model.parameters())
    if count_params(model) != EXPECTED_PARAMS:
        raise RuntimeError("QAT master param count mismatch")
    return {name: np.array(val, dtype=np.float32) for name, val in flatten_params(model).items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, default=QAT_Q4_WEIGHTS_PATH)
    parser.add_argument("--out-dir", type=Path, default=QAT_Q4_PACKAGE_DIR)
    parser.add_argument("--package-id", default=QAT_Q4_PACKAGE_ID)
    parser.add_argument("--parent-id", default=MODEL_ID)
    parser.add_argument(
        "--parent-weights-sha256",
        default=None,
        help="Explicit parent master hash for derived QAT/SFT packages.",
    )
    parser.add_argument("--bit-map", type=Path, default=None)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--scheme", default="q4")
    parser.add_argument("--budget-bytes", type=int, default=int(Q4_PACKAGE_BUDGET_BYTES))
    parser.add_argument(
        "--budget-raw-bytes",
        type=int,
        default=None,
        help="Optional raw weight payload ceiling (CQ2: 19.3MiB). Independent of tokenizer files.",
    )
    parser.add_argument("--receipt-name", default="qat-q4-package-receipt.json")
    args = parser.parse_args()
    error = validate_release(load_json(RELEASE_PATH), WEIGHTS_PATH)
    if error:
        return fail(error)
    if not args.master.is_file():
        receipt = {
            "ok": False,
            "skipped": True,
            "reason": f"missing QAT master {args.master}",
            "package_id": args.package_id,
        }
        blocked = write_json(args.jobs_dir / args.receipt_name, receipt)
        if blocked:
            return fail(blocked)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return 2

    dest: Path = args.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    tensors = numpy_from_master(args.master)
    bit_map_path = args.bit_map or (args.jobs_dir / Q4_BASELINE_MAP_NAME)
    bit_map = load_json(bit_map_path)
    tensor_bits = bit_map.get("tensor_bits") or {}
    packed, header = build_pack_bytes(tensors, tensor_bits)
    weights_path = dest / "weights.q4"
    weights_path.write_bytes(packed)
    tok_src = ROOT / "tokenizer" / "zh-24k-v1" / "zh-24k-v1.model"
    tok_dest = dest / "tokenizer.model"
    shutil.copy2(tok_src, tok_dest)
    tok = ZhTokenizerV1(tok_dest)
    vocab = export_vocab(tok)
    vocab_path = dest / "tokenizer.vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False) + "\n", encoding="utf-8")
    parent_sha = str(args.parent_weights_sha256 or "")
    if not parent_sha:
        parent_sha = sha256_file(WEIGHTS_PATH)
    if not args.parent_weights_sha256 and QAT_Q4_RELEASE_PATH.is_file():
        parent_sha = str(load_json(QAT_Q4_RELEASE_PATH).get("parent_weights_sha256") or parent_sha)
    manifest = write_manifest(
        package_id=args.package_id,
        parent_id=args.parent_id,
        weights_name="weights.q4",
        weights_sha=sha256_file(weights_path),
        tok_name="tokenizer.model",
        tok_sha256=sha256_file(tok_dest),
        vocab_name="tokenizer.vocab.json",
        vocab_sha=sha256_file(vocab_path),
        header=header,
        bit_map=bit_map,
        parent_sha=parent_sha,
        scheme=args.scheme,
        status="qat",
    )
    package_bytes = weights_path.stat().st_size + tok_dest.stat().st_size + vocab_path.stat().st_size
    raw_over = False
    if args.budget_raw_bytes is not None:
        raw_over = int(bit_map.get("raw_payload_bytes") or weights_path.stat().st_size) > int(args.budget_raw_bytes)
    over = package_bytes > int(args.budget_bytes) or raw_over
    (dest / "mei-model.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    receipt = {
        "ok": not over,
        "package_id": args.package_id,
        "model_id": QAT_Q4_MODEL_ID if args.package_id == QAT_Q4_PACKAGE_ID else args.package_id,
        "out_dir": str(dest),
        "master": str(args.master),
        "master_sha256": sha256_file(args.master),
        "parent_model_id": args.parent_id,
        "parent_weights_sha256": parent_sha,
        "quant_math_id": QUANT_MATH_ID,
        "bit_map": str(bit_map_path),
        "bit_map_sha256": manifest["bit_map_sha256"],
        "weights_bytes": weights_path.stat().st_size,
        "weights_mb": weights_path.stat().st_size / (1024 * 1024),
        "package_files_bytes": package_bytes,
        "package_files_mb": package_bytes / (1024 * 1024),
        "budget_bytes": int(args.budget_bytes),
        "budget_raw_bytes": args.budget_raw_bytes,
        "raw_payload_bytes": bit_map.get("raw_payload_bytes"),
        "within_budget": not over,
        "within_raw_budget": not raw_over,
        "n_tensors": header.n_tensors,
        "qat_mandatory": True,
        "immutable_parent_untouched": True,
    }
    blocked = write_json(args.jobs_dir / args.receipt_name, receipt)
    if blocked:
        return fail(blocked)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    if over:
        return fail(f"package {package_bytes} bytes exceeds budget {args.budget_bytes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
