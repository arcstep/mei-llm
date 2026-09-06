#!/usr/bin/env python3
"""Export the final 51M LM and all independent heads as package v2.

This is the only product CQ2 exporter.  The historical block64 uniform packers
remain diagnostic/legacy and must never be labelled ``mei-cq-v2``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from common._repo import ARCHITECTURE_DIR, ROOT, TOKENIZER_DIR, architecture_contracts
from training.qat.cq2_policy_51m import lm_storage_dtype, uniform_group_bits

sys.path.insert(0, str(ROOT / "src/platform/python-sdk"))
sys.path.insert(0, str(ROOT / "src/platform/_shared/runtime"))

from mei_sdk.cq2 import (  # noqa: E402
    GROUP_SIZE,
    QUANT_MATH_ID,
    TensorContainer,
    TensorToPack,
    build_tensor_container,
)
from mei_sdk.package import load_package  # noqa: E402
from mei_sdk.shared import ToolIndex, index_fingerprint  # noqa: E402


PACKAGE_LIMIT = 18 * 1024 * 1024
# 词表随冻结指针（新链 zh-24k-v3；旧链重打包时同样遵循当前指针——指针即权威）
from common._repo import frozen_tokenizer_path  # noqa: E402

_FROZEN_TOK = frozen_tokenizer_path()
TOKENIZER_MODEL = _FROZEN_TOK
TOKENIZER_MANIFEST = _FROZEN_TOK.parent / f"tokenizer-{_FROZEN_TOK.name.removesuffix('.model')}-manifest.json"
MW_CODEBOOK = ROOT / "cycles/mei-1.1-51m/exp-00300m/corpus/sft-suite/historical-notebook-releases/recipes/mw-disposition-codebook-v1.json"
EXPECTED_HEADS: dict[str, dict[str, tuple[int, ...]]] = {
    "contrastive": {
        "heads.contrastive.tok_probes": (4, 512),
        "heads.contrastive.lay_probes": (4, 512),
        "heads.contrastive.proj.weight": (128, 2048),
    },
    "mw_disposition": {
        "heads.mw_disposition.proj.weight": (20, 512),
        "heads.mw_disposition.proj.bias": (20,),
    },
    "confidence": {
        "heads.confidence.cell_probes": (8, 512),
        "heads.confidence.proj.weight": (1, 4096),
        "heads.confidence.proj.bias": (1,),
    },
    "narration_adapter": {
        "heads.narration_adapter.down.weight": (16, 512),
        "heads.narration_adapter.up.weight": (24000, 16),
    },
}
RUNTIME_QUANTIZATION = {
    "weight_math_id": QUANT_MATH_ID,
    "activation_semantics_id": (
        "int8-symmetric-per-last-axis-vector-qdq-forward_identity-backward-v2"
    ),
    "activation_sites": [
        "engram_input",
        "attention_input",
        "attention_output",
        "lm_head_input",
    ],
    "activation_q_quantized": False,
    "activation_reconstruction_dtype": "f32",
    "kv_semantics_id": (
        "mei-int8-kv-per-head-vector-qdq-forward_identity-backward-v1"
    ),
    "kv_sites": ["attention_key_after_rope", "attention_value"],
    "kv_storage_dtype": "int8",
    "kv_scale_granularity": "per-head-vector",
    "int8_code_min": -128,
    "int8_code_max": 127,
    "scale_denominator": 127,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value
    )


def _load_inputs(master_path: Path, heads_path: Path) -> tuple[list[TensorToPack], list[str]]:
    contracts = architecture_contracts()
    expected = list(contracts["weight_contract"]["tensor_order"])
    master = np.load(master_path, allow_pickle=False)
    actual_names = list(master.files)
    expected_names = [str(row["name"]) for row in expected]
    if actual_names != expected_names:
        raise RuntimeError("FP32 master does not match the ordered 400-tensor weight contract")
    tensors: list[TensorToPack] = []
    for row in expected:
        name = str(row["name"])
        shape = tuple(int(value) for value in row["shape"])
        array = np.asarray(master[name], dtype=np.float32)
        if tuple(array.shape) != shape:
            raise RuntimeError(f"LM tensor shape mismatch: {name}")
        dtype = lm_storage_dtype(name, shape)
        bits = uniform_group_bits(name, shape)
        tensors.append(TensorToPack(name, shape, array, "lm", dtype, bits))

    heads = np.load(heads_path, allow_pickle=False)
    expected_head_names = [
        name
        for role in ("contrastive", "mw_disposition", "confidence", "narration_adapter")
        for name in EXPECTED_HEADS[role]
    ]
    if list(heads.files) != expected_head_names:
        missing = sorted(set(expected_head_names) - set(heads.files))
        extra = sorted(set(heads.files) - set(expected_head_names))
        raise RuntimeError(f"portable head tensor contract mismatch missing={missing} extra={extra}")
    for role in ("contrastive", "mw_disposition", "confidence", "narration_adapter"):
        for name, shape in EXPECTED_HEADS[role].items():
            array = np.asarray(heads[name], dtype=np.float32)
            if tuple(array.shape) != shape:
                raise RuntimeError(f"head tensor shape mismatch: {name}")
            tensors.append(
                TensorToPack(
                    name,
                    shape,
                    array,
                    role,
                    "cq2" if role == "narration_adapter" else "f16",
                )
            )
    return tensors, expected_head_names


def _head_hash(container: TensorContainer) -> str:
    digest = hashlib.sha256()
    for name in sorted(
        name for name, entry in container.entries.items() if entry.role == "contrastive"
    ):
        entry = container.entries[name]
        digest.update(name.encode("utf-8"))
        for offset, nbytes in (
            (entry.data_offset, entry.data_nbytes),
            (entry.scales_offset, entry.scales_nbytes),
            (entry.bit_map_offset, entry.bit_map_nbytes),
        ):
            if nbytes:
                digest.update(container.blob[offset : offset + nbytes])
    return digest.hexdigest()


def _rewrite_index(
    source_path: Path,
    output_path: Path,
    *,
    model_hash: str,
    head_hash: str,
    tokenizer_hash: str,
) -> None:
    source = ToolIndex.load(source_path)
    if not source.records:
        raise RuntimeError("final package requires a non-empty trained tool index")
    target = ToolIndex(
        model_hash=model_hash,
        head_hash=head_hash,
        tokenizer_hash=tokenizer_hash,
        serializer=source.serializer,
    )
    target.records = source.records
    target.catalog_hash = source.catalog_hash
    combined_schema = _sha_bytes(
        _canonical({name: row.schema_hash for name, row in sorted(target.records.items())})
    )
    target.fingerprint = index_fingerprint(
        model_hash=model_hash,
        head_hash=head_hash,
        tokenizer_hash=tokenizer_hash,
        serializer=target.serializer,
        schema_hash=combined_schema,
        catalog_hash=target.catalog_hash,
    )
    target.save(output_path)
    ToolIndex.load(output_path, expected_fingerprint=target.fingerprint)


def _directory_sha(directory: list[dict[str, Any]], role: str) -> str:
    return _sha_bytes(_canonical([row for row in directory if row["role"] == role]))


def _write_receipts(
    root: Path,
    *,
    package_id: str,
    contracts: dict[str, str],
    container_sha: str,
    directory: list[dict[str, Any]],
    evidence_path: Path,
) -> dict[str, tuple[str, str]]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if set(evidence) != {
        "lm",
        "contrastive",
        "mw_disposition",
        "confidence",
        "narration_adapter",
    }:
        raise RuntimeError("stage evidence must contain exactly five independent components")
    outputs: dict[str, tuple[str, str]] = {}
    for component in (
        "lm",
        "contrastive",
        "mw_disposition",
        "confidence",
        "narration_adapter",
    ):
        row = evidence[component]
        fingerprint = row.get("stage_fingerprint_sha256")
        if row.get("status") != "passed" or not _is_sha(fingerprint):
            raise RuntimeError(f"{component} stage evidence is not passed and fingerprinted")
        receipt = {
            "schema": "mei-training-receipt-v2",
            "product": "mei-1.2-51m",
            "package_id": package_id,
            "component": component,
            "stage_id": str(row.get("stage_id") or component),
            "stage_fingerprint_sha256": fingerprint,
            "status": "passed",
            "contracts": contracts,
            "tensor_container_sha256": container_sha,
            "tensor_directory_sha256": _directory_sha(directory, component),
        }
        payload = _canonical(receipt)
        filename = f"training-receipt-{component}.json"
        (root / filename).write_bytes(payload)
        outputs[component] = (filename, _sha_bytes(payload))
    return outputs


def _lm_prefixes(names: list[str]) -> list[str]:
    candidates = [
        "embed",
        "blocks",
        "final_norm",
        "engrams",
        "mhc_phi_pre",
        "mhc_phi_post",
        "mhc_phi_res",
        "mhc_b_pre",
        "mhc_b_post",
        "mhc_b_res",
        "mhc_a_pre",
        "mhc_a_post",
        "mhc_a_res",
        "conf_probes",
        "conf_proj",
    ]
    covered = {
        name
        for prefix in candidates
        for name in names
        if name == prefix or name.startswith(prefix + ".")
    }
    if covered != set(names):
        raise RuntimeError("LM tensor prefix inventory is incomplete")
    return candidates


def _manifest_bytes_with_size(manifest: dict[str, Any], payload_bytes: int) -> bytes:
    for _ in range(32):
        encoded = _canonical(manifest)
        measured = payload_bytes + len(encoded)
        if manifest["resources"]["package_bytes"] == measured:
            return encoded
        manifest["resources"]["package_bytes"] = measured
    raise RuntimeError("manifest package_bytes did not converge")


def export(args: argparse.Namespace) -> dict[str, Any]:
    if args.out_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing package: {args.out_dir}")
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    tensors, _head_names = _load_inputs(args.master, args.heads)
    blob, directory = build_tensor_container(tensors)
    container_sha = _sha_bytes(blob)
    container = TensorContainer.parse(blob)
    contracts_raw = architecture_contracts()
    contracts = {
        key: str(contracts_raw[key])
        for key in (
            "weight_contract_sha256",
            "runtime_profile_sha256",
            "training_aux_sha256",
        )
    }
    spec = json.loads((ARCHITECTURE_DIR / "spec/model.json").read_text(encoding="utf-8"))
    arch = spec["architecture"]
    profile = spec["runtime_profile"]
    tokenizer_meta = json.loads(TOKENIZER_MANIFEST.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="mei-cq2-v2-", dir=args.out_dir.parent) as temp_name:
        temp = Path(temp_name)
        (temp / "tensors.bin").write_bytes(blob)
        shutil.copyfile(TOKENIZER_MODEL, temp / "tokenizer.model")
        shutil.copyfile(MW_CODEBOOK, temp / "mw-disposition-codebook-v1.json")
        calibration_source = getattr(args, "retrieval_calibration", None)
        if calibration_source is not None:
            calibration = json.loads(Path(calibration_source).read_text(encoding="utf-8"))
        else:
            calibration = {
                "calibration_id": "mei-retrieval-platt-v1",
                "scale": 1.0,
                "bias": 0.0,
                "discard_threshold": 0.0,
                "expand_threshold": 0.0,
                "validated": False,
                "fallback_scan_all_eligible": True,
                "validation_manifest_sha256": "0" * 64,
                "catalog_sizes": [10, 20, 50],
                "metrics": {"status": "calibration_missing"},
            }
        (temp / "retrieval-calibration.json").write_text(
            json.dumps(calibration, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        _rewrite_index(
            args.tool_index,
            temp / "tool-index.json",
            model_hash=container_sha,
            head_hash=_head_hash(container),
            tokenizer_hash=_sha_file(TOKENIZER_MODEL),
        )
        receipts = _write_receipts(
            temp,
            package_id=args.package_id,
            contracts=contracts,
            container_sha=container_sha,
            directory=directory,
            evidence_path=args.stage_evidence,
        )

        file_roles = {
            "tensors.bin": "tensor_container",
            "tokenizer.model": "tokenizer",
            "tool-index.json": "tool_index",
            "mw-disposition-codebook-v1.json": "head_codebook",
            "retrieval-calibration.json": "auxiliary",
            **{filename: "training_receipt" for filename, _sha in receipts.values()},
        }
        files = [
            {
                "path": name,
                "sha256": _sha_file(temp / name),
                "nbytes": (temp / name).stat().st_size,
                "role": role,
            }
            for name, role in sorted(file_roles.items())
        ]
        architecture = {
            "id": spec["architecture_id"],
            "d_model": arch["d_model"],
            "n_layers": arch["n_layers"],
            "n_heads": arch["n_heads"],
            "n_kv_heads": arch["n_kv_heads"],
            "head_dim": arch["d_model"] // arch["n_heads"],
            "vocab_size": arch["vocab_size"],
            "max_seq_len": arch["max_seq_len"],
            "parameter_count": 51_463_797,
            "rope_theta": arch["rope_theta"],
            "engram_layers": arch["engram_layers"],
            "engram_orders": arch["engram_orders"],
            "engram_slots": arch["engram_slots"],
            "engram_conv_taps": arch["engram_conv_taps"],
            "mhc_lanes": arch["mhc_lanes"],
            "sinkhorn_iters": arch["sinkhorn_iters"],
            "tie_embeddings": arch["tie_embeddings"],
            "rms_eps": 1e-6,
            "conf_probes": int(spec["training_aux"]["confidence_probes"]),
            "mlp": arch["mlp"],
            "confidence_head": arch["confidence_head"],
        }
        runtime_profile = {
            "max_context_tokens": profile["max_context_tokens"],
            "default_profile": profile["default_profile"],
            "stable_prefix_profiles": profile["stable_prefix_profiles"],
            "ordinary_window_policy": profile["ordinary_window_policy"],
            "default_output_tokens": profile["default_output_max_tokens"],
            "candidate_batch_size": profile["tool_batch_size"],
            "context_packer_id": profile["context_packer_id"],
            "retrieval_batch_policy_id": profile["retrieval_batch_policy_id"],
            "prompt_framing_id": profile["prompt_framing_id"],
            "assistant_suffix": profile["assistant_suffix"],
            "kv_dtype": "i8",
            "activation_dtype": "i8",
        }
        manifest = {
            "package_format": "mei-model-package-v2",
            "product": "mei-1.2-51m",
            "package_id": args.package_id,
            "runtime_min": "mei-runtime-abi-2",
            "parent_package_id": args.parent_package_id,
            "contracts": contracts,
            "architecture": architecture,
            "runtime_profile": runtime_profile,
            "retrieval_calibration": calibration,
            "runtime_quantization": RUNTIME_QUANTIZATION,
            "tokenizer": {
                "id": "zh-24k-v1",
                "file": "tokenizer.model",
                "sha256": _sha_file(temp / "tokenizer.model"),
                "pad_id": tokenizer_meta["pad_id"],
                "eos_id": tokenizer_meta["eos_id"],
                "bos_id": tokenizer_meta["bos_id"],
                "unk_id": tokenizer_meta["unk_id"],
            },
            "tensor_container": {
                "file": "tensors.bin",
                "format": "mei-cq-tensor-v2",
                "sha256": container_sha,
                "payload_bytes": len(blob),
                "quant_math_id": QUANT_MATH_ID,
                "directory": directory,
            },
            "files": files,
            "heads": {
                "lm": {
                    "present": True,
                    "trained": True,
                    "status": "ready",
                    "tensor_prefixes": _lm_prefixes(
                        [name for name, entry in container.entries.items() if entry.role == "lm"]
                    ),
                    "training_receipt_sha256": receipts["lm"][1],
                },
                "contrastive": {
                    "present": True,
                    "trained": True,
                    "status": "ready",
                    "tensor_prefixes": ["heads.contrastive"],
                    "training_receipt_sha256": receipts["contrastive"][1],
                },
                "mw_disposition": {
                    "present": True,
                    "trained": True,
                    "status": "ready",
                    "tensor_prefixes": ["heads.mw_disposition"],
                    "training_receipt_sha256": receipts["mw_disposition"][1],
                    "label_codebook_sha256": _sha_file(temp / "mw-disposition-codebook-v1.json"),
                },
                "confidence": {
                    "present": True,
                    "trained": True,
                    "status": "ready",
                    "tensor_prefixes": ["heads.confidence"],
                    "training_receipt_sha256": receipts["confidence"][1],
                },
                "narration_adapter": {
                    "present": True,
                    "trained": True,
                    "status": "ready",
                    "tensor_prefixes": ["heads.narration_adapter"],
                    "training_receipt_sha256": receipts["narration_adapter"][1],
                },
            },
            "capabilities": {
                "retrieval": True,
                "full_call": True,
                "mw_disposition": True,
                "confidence": True,
                "multi_step": True,
                "narration": True,
            },
            "training_receipts": [
                receipts[name][1]
                for name in (
                    "lm",
                    "contrastive",
                    "mw_disposition",
                    "confidence",
                    "narration_adapter",
                )
            ],
            "resources": {
                "package_bytes": 0,
                "rust_session_peak_bytes": 0,
                "wasm_heap_peak_bytes": 0,
            },
            "release_class": "candidate",
        }
        payload_bytes = sum(row["nbytes"] for row in files)
        (temp / "mei-model.json").write_bytes(_manifest_bytes_with_size(manifest, payload_bytes))
        package = load_package(temp)
        if not package.tensor_identity_complete():
            raise RuntimeError("exported LM tensor directory is not the canonical 51M identity")
        temp.replace(args.out_dir)

    package_bytes = sum(path.stat().st_size for path in args.out_dir.iterdir() if path.is_file())
    return {
        "ok": True,
        "package_id": args.package_id,
        "package_dir": str(args.out_dir),
        "tensor_container_sha256": container_sha,
        "tensor_count": len(directory),
        "lm_tensor_count": sum(row["role"] == "lm" for row in directory),
        "mtp_tensor_count": sum("mtp" in row["name"].lower() for row in directory),
        "package_bytes": package_bytes,
        "package_limit_bytes": PACKAGE_LIMIT,
        "package_within_limit": package_bytes <= PACKAGE_LIMIT,
        "package_export_complete": True,
        # Export is one stage in the lifecycle.  Only the final productization
        # auditor may assert process_complete after runtime, parity, resource,
        # MTP and reporting stages all have terminal receipts.
        "process_complete": False,
        "release_eligible": False,
        "release_ineligible_reasons": [
            *([] if package_bytes <= PACKAGE_LIMIT else ["package_size"]),
            "resource_measurement_pending",
            "portable_runtime_execution_pending",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--tool-index", type=Path, required=True)
    parser.add_argument("--retrieval-calibration", type=Path)
    parser.add_argument("--stage-evidence", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--package-id", required=True)
    parser.add_argument("--parent-package-id", default=None)
    args = parser.parse_args()
    report = export(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
