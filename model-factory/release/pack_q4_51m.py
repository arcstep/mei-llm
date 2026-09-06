#!/usr/bin/env python3
"""Pack the immutable 51M float base into a loadable Q4 model package.

Does not overwrite the frozen Base. Writes the cycle-bound package artifact.
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
    Q4_PACKAGE_DIR,
    Q4_PACKAGE_ID,
    QAT_MANDATORY_NOTE,
    RELEASE_PATH,
    ROOT,
    WEIGHTS_PATH,
    fail,
    load_json,
    sha256_file,
    validate_release,
    write_json,
)
from release.quant_pack_51m import (
    BLOCK_SIZE,
    ENDIANNESS,
    Q4_PACKAGE_BUDGET_BYTES,
    QUANT_MATH_ID,
    WEIGHTS_FORMAT,
    build_pack_bytes,
    q4_baseline_map,
    sha256_bytes,
)
from tokenizer import ZhTokenizerV1


def _params_to_numpy(model) -> dict[str, np.ndarray]:
    from common.checkpoint import flatten_params

    return {name: np.array(val, dtype=np.float32) for name, val in flatten_params(model).items()}


def load_51m_numpy() -> tuple[object, dict[str, np.ndarray]]:
    import mlx.core as mx

    from architecture import NeedleZh, count_params
    from common.checkpoint import load_params
    from config import NeedleZhConfig

    cfg = NeedleZhConfig.from_spec(ARCHITECTURE_SPEC)
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    load_params(model, WEIGHTS_PATH, strict=True)
    mx.eval(model.parameters())
    if count_params(model) != EXPECTED_PARAMS:
        raise RuntimeError("loaded param count mismatch")
    return model, _params_to_numpy(model)


def export_vocab(tok: ZhTokenizerV1) -> dict:
    pieces = []
    for idx in range(tok.vocab_size):
        piece = tok.sp.id_to_piece(idx)
        try:
            score = float(tok.sp.get_score(idx))
        except Exception:
            score = 0.0
        pieces.append({"id": idx, "piece": piece, "score": score})
    return {
        "tokenizer_id": "zh-24k-v1",
        "vocab_size": tok.vocab_size,
        "pad_id": tok.pad_id,
        "eos_id": tok.eos_id,
        "bos_id": tok.bos_id,
        "unk_id": tok.unk_id,
        "model_sha256": tok.model_sha256,
        "pieces": pieces,
    }


def write_manifest(
    dest: Path,
    *,
    weights_name: str,
    weights_sha: str,
    tok_name: str,
    tok_sha256: str,
    vocab_name: str,
    vocab_sha: str,
    header,
    bit_map: dict,
    parent_sha: str,
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
        "package_id": Q4_PACKAGE_ID,
        "runtime_min": "mei-runtime-abi-1",
        "parent_package_id": MODEL_ID,
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
                "scheme": "q4",
                "status": "diagnostic",
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Q4_PACKAGE_DIR)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    args = parser.parse_args()
    release = load_json(RELEASE_PATH)
    error = validate_release(release, WEIGHTS_PATH)
    if error:
        return fail(error)

    dest: Path = args.out_dir
    dest.mkdir(parents=True, exist_ok=True)
    _model, tensors = load_51m_numpy()
    baseline = q4_baseline_map(tensors)
    jobs_map = args.jobs_dir / Q4_BASELINE_MAP_NAME
    if jobs_map.is_file():
        baseline = {**load_json(jobs_map), **{"tensor_bits": baseline["tensor_bits"], "raw_payload_bytes": baseline["raw_payload_bytes"], "raw_payload_mb": baseline["raw_payload_mb"]}}

    packed, header = build_pack_bytes(tensors, baseline["tensor_bits"])
    weights_path = dest / "weights.q4"
    weights_path.write_bytes(packed)
    tok_src = next(parent for parent in Path(__file__).resolve().parents if (parent / "CURRENT.json").is_file()) / "models/mei-1.2-51m/tokenizer/zh-24k-v1.model"
    tok_dest = dest / "tokenizer.model"
    shutil.copy2(tok_src, tok_dest)
    tok = ZhTokenizerV1(tok_dest)
    vocab = export_vocab(tok)
    vocab_path = dest / "tokenizer.vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest = write_manifest(
        dest,
        weights_name="weights.q4",
        weights_sha=sha256_file(weights_path),
        tok_name="tokenizer.model",
        tok_sha256=sha256_file(tok_dest),
        vocab_name="tokenizer.vocab.json",
        vocab_sha=sha256_file(vocab_path),
        header=header,
        bit_map=baseline,
        parent_sha=sha256_file(WEIGHTS_PATH),
    )
    package_bytes = weights_path.stat().st_size + tok_dest.stat().st_size + vocab_path.stat().st_size
    over = package_bytes > Q4_PACKAGE_BUDGET_BYTES
    receipt = {
        "ok": not over,
        "package_id": Q4_PACKAGE_ID,
        "out_dir": str(dest),
        "weights_bytes": weights_path.stat().st_size,
        "weights_mb": weights_path.stat().st_size / (1024 * 1024),
        "package_files_bytes": package_bytes,
        "package_files_mb": package_bytes / (1024 * 1024),
        "budget_mb": Q4_PACKAGE_BUDGET_BYTES / (1024 * 1024),
        "within_30mb_budget": not over,
        "n_tensors": header.n_tensors,
        "quant_math_id": QUANT_MATH_ID,
        "qat_mandatory": True,
        "product_final": False,
    }
    (dest / "mei-model.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    blocked = write_json(args.jobs_dir / "q4-package-receipt.json", {**receipt, "weights_sha256": manifest["weights"]["sha256"]})
    if blocked:
        return fail(blocked)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    if over:
        return fail(f"Q4 package {package_bytes} bytes exceeds 30MB budget")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
