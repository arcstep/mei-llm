from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import SdkError
from .json_schema_lite import ContractSchemaError, loads_strict, validate_instance
from .version import SPEC_DIR, sdk_versions

REQUIRED_HEADS = ("lm", "contrastive", "mw_disposition", "confidence")
OPTIONAL_HEADS = ("narration_adapter",)
ALL_HEADS = REQUIRED_HEADS + OPTIONAL_HEADS
MW_LABEL_CODEBOOK_SHA256 = "913d2c4bca8a796c9baddb9a79539cc703aa6420af4260be2c5ca0e0d5a68d40"
_CONTRACT_SCHEMA = "mei-architecture-contract-v1"
_CANONICAL_ARCHITECTURE: dict[str, Any] = {
    "id": "mei-1.0-51m-arch-v1",
    "d_model": 512,
    "n_layers": 27,
    "n_heads": 8,
    "n_kv_heads": 4,
    "head_dim": 64,
    "vocab_size": 24_000,
    "max_seq_len": 2_048,
    "parameter_count": 51_463_797,
    "rope_theta": 100_000.0,
    "engram_layers": [2, 15],
    "engram_orders": [2, 3],
    "engram_slots": 8_192,
    "engram_conv_taps": 4,
    "mhc_lanes": 4,
    "sinkhorn_iters": 20,
    "tie_embeddings": True,
    "rms_eps": 1e-6,
    "conf_probes": 8,
    "mlp": "FixedWalshHadamardMLP",
    "confidence_head": True,
}
_CANONICAL_RUNTIME_PROFILE = {
    "max_context_tokens": 2_048,
    "stable_prefix_tokens": 1_024,
    "rolling_window_tokens": 256,
    "default_output_tokens": 128,
    "kv_dtype": "i8",
    "activation_dtype": "i8",
}
_CANONICAL_RUNTIME_QUANTIZATION = {
    "weight_math_id": "mei-cq-v2-g128-wht-codebook",
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


def _canonical_weight_tensors() -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Expand the immutable 400-tensor deployed 51M geometry.

    This mirrors the architecture contract without importing MLX or repository
    training code into the portable SDK.  Scalar attention gates intentionally
    use shape ``()`` and therefore have one parameter.
    """

    d_model = 512
    n_layers = 27
    head_dim = 64
    kv_dim = 256
    lanes = 4
    lane_width = lanes * d_model
    tensors: list[tuple[str, tuple[int, ...]]] = []

    def add(name: str, *shape: int) -> None:
        tensors.append((name, tuple(shape)))

    add("embed.weight", 24_000, d_model)
    for layer in range(n_layers):
        prefix = f"blocks.{layer}"
        add(f"{prefix}.attn_norm.scale", d_model)
        add(f"{prefix}.attn.q_proj.weight", d_model, d_model)
        add(f"{prefix}.attn.k_proj.weight", kv_dim, d_model)
        add(f"{prefix}.attn.v_proj.weight", kv_dim, d_model)
        add(f"{prefix}.attn.gate_proj.weight", d_model, d_model)
        add(f"{prefix}.attn.o_proj.weight", d_model, d_model)
        add(f"{prefix}.attn.q_norm.scale", head_dim)
        add(f"{prefix}.attn.k_norm.scale", head_dim)
        add(f"{prefix}.post_attn_norm.scale", d_model)
        add(f"{prefix}.attn_gate")
        add(f"{prefix}.mlp_norm.scale", d_model)
        add(f"{prefix}.mlp.d1", d_model)
        add(f"{prefix}.mlp.d2", d_model)
        add(f"{prefix}.mlp.d3", d_model)
    add("final_norm.scale", d_model)
    for site in range(2):
        prefix = f"engrams.{site}"
        add(f"{prefix}.tables", 4, 8_192, 128)
        add(f"{prefix}.key_proj.weight", d_model, 512)
        add(f"{prefix}.value_proj.weight", d_model, 512)
        add(f"{prefix}.taps", 4, d_model)
    add("mhc_phi_pre", n_layers, lane_width, lanes)
    add("mhc_phi_post", n_layers, lane_width, lanes)
    add("mhc_phi_res", n_layers, lane_width, lanes * lanes)
    add("mhc_b_pre", n_layers, lanes)
    add("mhc_b_post", n_layers, lanes)
    add("mhc_b_res", n_layers, lanes, lanes)
    add("mhc_a_pre", n_layers)
    add("mhc_a_post", n_layers)
    add("mhc_a_res", n_layers)
    add("conf_probes", 8, d_model)
    add("conf_proj.weight", 1, 8 * d_model)
    add("conf_proj.bias", 1)
    return tuple(tensors)


_CANONICAL_LM_TENSORS = _canonical_weight_tensors()


def _portable_lm_dtype(name: str) -> str:
    if name == "embed.weight" or name.startswith("mhc_"):
        return "cq4"
    if (
        name.startswith("engrams.") and not name.endswith(".taps")
    ) or (".attn." in name and name.endswith("_proj.weight")):
        return "cq2"
    return "f16"


def _canonical_digest(value: Any) -> str:
    body = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


_CANONICAL_WEIGHT_CONTRACT = {
    "schema": _CONTRACT_SCHEMA,
    "kind": "weight_geometry",
    "architecture_id": "mei-1.0-51m-arch-v1",
    "tokenizer": {"id": "zh-24k-v1", "pad_id": 0, "eos_id": 1, "bos_id": 2, "unk_id": 3},
    "tensor_order": [
        {"name": name, "shape": list(shape)} for name, shape in _CANONICAL_LM_TENSORS
    ],
    "trainable_params": 51_463_797,
    "topology": {
        "tie_embeddings": True,
        "mlp": "FixedWalshHadamardMLP",
        "rope_theta": 100_000.0,
        "engram_layers": [2, 15],
        "engram_orders": [2, 3],
        "sinkhorn_iters": 20,
        "fixed_non_parameters": [
            "walsh_hadamard_transform",
            "mhc_pre_off",
            "mhc_post_off",
        ],
    },
}
_CANONICAL_WEIGHT_CONTRACT_SHA256 = _canonical_digest(_CANONICAL_WEIGHT_CONTRACT)
_CANONICAL_RUNTIME_CONTRACT_SHA256 = _canonical_digest(
    {
        "schema": _CONTRACT_SCHEMA,
        "kind": "runtime_profile",
        "architecture_id": "mei-1.0-51m-arch-v1",
        "max_context_tokens": 2_048,
        "stable_prefix_max_tokens": 1_024,
        "ordinary_window_tokens": 256,
        "default_output_max_tokens": 128,
        "kv_cache_dtype": "int8",
        "activation_dtype": "int8",
    }
)
_CANONICAL_TRAINING_AUX_SHA256 = _canonical_digest(
    {
        "schema": _CONTRACT_SCHEMA,
        "kind": "training_aux",
        "architecture_id": "mei-1.0-51m-arch-v1",
        "mtp": {
            "mode": "training_only_ablation",
            "enabled_by_default": False,
            "export": False,
        },
        "confidence_probes": 8,
    }
)
_READY_AUX_HEAD_CONTRACTS: dict[str, dict[str, tuple[int, ...]]] = {
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


@dataclass
class HeadStatus:
    present: bool
    trained: bool
    status: str
    tensor_prefixes: tuple[str, ...] = ()
    training_receipt_sha256: str | None = None
    label_codebook_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "trained": self.trained,
            "status": self.status,
            "tensor_prefixes": list(self.tensor_prefixes),
            "training_receipt_sha256": self.training_receipt_sha256,
            "label_codebook_sha256": self.label_codebook_sha256,
        }


@dataclass
class HeadReport:
    lm: HeadStatus
    contrastive: HeadStatus
    mw_disposition: HeadStatus
    confidence: HeadStatus
    narration_adapter: HeadStatus

    def as_dict(self) -> dict[str, Any]:
        return {
            "lm": self.lm.as_dict(),
            "contrastive": self.contrastive.as_dict(),
            "mw_disposition": self.mw_disposition.as_dict(),
            "confidence": self.confidence.as_dict(),
            "narration_adapter": self.narration_adapter.as_dict(),
        }

    def missing(self) -> list[str]:
        out: list[str] = []
        for name in REQUIRED_HEADS:
            head: HeadStatus = getattr(self, name)
            if head.status != "ready" or not head.present or not head.trained:
                out.append(name)
        return out


@dataclass
class ModelPackage:
    path: Path
    manifest: dict[str, Any]
    heads: HeadReport
    verified_hashes: bool = False
    resource_measurement_verified: bool = False
    warnings: list[str] = field(default_factory=list)
    compatibility: dict[str, Any] = field(default_factory=dict)

    @property
    def package_id(self) -> str:
        return str(self.manifest["package_id"])

    def packed_structure_ready(self) -> bool:
        if self.manifest.get("package_format") == "mei-model-package-v2":
            container = self.manifest.get("tensor_container") or {}
            required_capabilities = {
                "retrieval",
                "full_call",
                "mw_disposition",
                "confidence",
                "multi_step",
            }
            capabilities = self.manifest.get("capabilities") or {}
            return (
                container.get("format") == "mei-cq-tensor-v2"
                and container.get("quant_math_id") == "mei-cq-v2-g128-wht-codebook"
                and self.tensor_identity_complete()
                and self.portable_quantization_policy_complete()
                and self.tool_index_payload_complete()
                and not self.heads.missing()
                and frozenset(capabilities) in {
                    frozenset(required_capabilities),
                    frozenset(required_capabilities | {"narration"}),
                }
                and all(capabilities.get(name) is True for name in required_capabilities)
            )
        weights = self.manifest.get("weights") or {}
        fmt = str(weights.get("format") or "")
        scheme = str((weights.get("quantization") or {}).get("scheme") or "")
        return fmt == "mei-q4-packed-v1" and scheme in {
            "q4",
            "cq2",
            "qat-q4",
            "qat-cq2",
        }

    def tool_index_payload_complete(self) -> bool:
        if self.manifest.get("package_format") != "mei-model-package-v2":
            return False
        rows = [
            row
            for row in (self.manifest.get("files") or [])
            if isinstance(row, dict) and row.get("role") == "tool_index"
        ]
        return len(rows) == 1

    def portable_quantization_policy_complete(self) -> bool:
        if not self.tensor_identity_complete():
            return False
        entries = ((self.manifest.get("tensor_container") or {}).get("directory") or [])
        lm_entries = [
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("role") == "lm"
        ]
        return bool(lm_entries) and all(
            entry.get("dtype") == _portable_lm_dtype(str(entry.get("name") or ""))
            for entry in lm_entries
        )

    def runtime_quantization_complete(self) -> bool:
        return (
            self.manifest.get("package_format") == "mei-model-package-v2"
            and self.manifest.get("runtime_quantization")
            == _CANONICAL_RUNTIME_QUANTIZATION
        )

    def packed_inference_ready(self) -> bool:
        return self.verified_hashes and self.packed_structure_ready()

    def tensor_identity_complete(self) -> bool:
        if self.manifest.get("package_format") != "mei-model-package-v2":
            return False
        contracts = self.manifest.get("contracts") or {}
        if contracts != {
            "weight_contract_sha256": _CANONICAL_WEIGHT_CONTRACT_SHA256,
            "runtime_profile_sha256": _CANONICAL_RUNTIME_CONTRACT_SHA256,
            "training_aux_sha256": _CANONICAL_TRAINING_AUX_SHA256,
        }:
            return False
        architecture = self.manifest.get("architecture") or {}
        if architecture != _CANONICAL_ARCHITECTURE:
            return False
        runtime_profile = self.manifest.get("runtime_profile") or {}
        if runtime_profile != _CANONICAL_RUNTIME_PROFILE:
            return False
        if (self.manifest.get("tokenizer") or {}).get("id") != "zh-24k-v1":
            return False
        entries = ((self.manifest.get("tensor_container") or {}).get("directory") or [])
        actual_lm = tuple(
            (str(entry.get("name") or ""), tuple(entry.get("shape") or ()))
            for entry in entries
            if isinstance(entry, dict) and entry.get("role") == "lm"
        )
        lm_params = sum(math.prod(shape) for _, shape in actual_lm)
        has_mtp = any(
            any(
                component == "mtp" or component.startswith("mtp_")
                for component in str(entry.get("name") or "").lower().split(".")
            )
            for entry in entries
            if isinstance(entry, dict)
        )
        expected = int((self.manifest.get("architecture") or {}).get("parameter_count") or 0)
        return (
            actual_lm == _CANONICAL_LM_TENSORS
            and lm_params == expected == 51_463_797
            and len(actual_lm) == 400
            and not has_mtp
        )

    def capabilities(self) -> dict[str, Any]:
        missing = self.heads.missing()
        declared = dict(self.manifest.get("capabilities") or {})
        is_v2 = self.manifest.get("package_format") == "mei-model-package-v2"
        v2_trusted = (
            is_v2
            and self.verified_hashes
            and self.tensor_identity_complete()
            and self.runtime_quantization_complete()
        )
        ready = {
            name: getattr(self.heads, name).status == "ready"
            and getattr(self.heads, name).present
            and getattr(self.heads, name).trained
            for name in REQUIRED_HEADS
        }
        effective = {
            "retrieval": bool(declared.get("retrieval")) and v2_trusted and ready["contrastive"],
            "full_call": bool(declared.get("full_call")) and v2_trusted and ready["lm"],
            "mw_disposition": bool(declared.get("mw_disposition"))
            and v2_trusted
            and ready["mw_disposition"],
            "confidence": bool(declared.get("confidence")) and v2_trusted and ready["confidence"],
            "multi_step": bool(declared.get("multi_step"))
            and v2_trusted
            and ready["lm"],
        }
        narration = self.heads.narration_adapter
        effective["narration"] = bool(declared.get("narration")) and v2_trusted and (
            narration.status == "ready" and narration.present and narration.trained
        )
        diagnostic_inference = self.packed_inference_ready()
        resources = self.manifest.get("resources") or {}
        resource_limits_reported = (
            is_v2
            and self.verified_hashes
            and self.resource_measurement_verified
            and int(resources.get("package_bytes") or 0) <= 18 * 1024 * 1024
            and int(resources.get("rust_session_peak_bytes") or 0) <= 64 * 1024 * 1024
            and int(resources.get("wasm_heap_peak_bytes") or 0) <= 96 * 1024 * 1024
        )
        # Python/MLX is the numerical oracle. It can attest mechanism coverage
        # for a complete verified v2 package, but never edge int8/resource
        # qualification; that remains a separate Rust/WASM measurement gate.
        implementation_complete = bool(
            is_v2
            and self.packed_inference_ready()
            and self.runtime_quantization_complete()
            and not missing
            and all(effective[name] for name in ("retrieval", "full_call", "mw_disposition", "confidence", "multi_step"))
        )
        product_ready = False
        resource_eligible = False
        legacy = not is_v2
        capabilities = {
            "package_id": self.package_id,
            "package_format": self.manifest.get("package_format"),
            "release_class": self.manifest.get("release_class"),
            "inference": legacy and diagnostic_inference,
            "diagnostic_inference": diagnostic_inference,
            "protocol": True,
            "heads": self.heads.as_dict(),
            "missing_or_untrained_heads": missing,
            "hash_verified": self.verified_hashes,
            "inference_payload_verified": self.verified_hashes,
            "resource_measurement_verified": self.resource_measurement_verified,
            "resource_limits_reported": resource_limits_reported,
            "tensor_identity_complete": self.tensor_identity_complete(),
            "portable_quantization_policy_complete": self.portable_quantization_policy_complete(),
            "runtime_quantization_complete": self.runtime_quantization_complete(),
            "tool_index_payload_complete": self.tool_index_payload_complete(),
            "compatibility_mode": (
                "v1-read-only-degraded" if legacy else "v2-native"
            ),
            "read_only": legacy,
            "degraded": legacy or not product_ready,
            "implementation_complete": implementation_complete,
            "open_capabilities": (
                ["v2-native"]
                if legacy
                else ["int8-bounded-kv", "trusted-resource-measurement"]
            ),
            "product_ready": product_ready,
            "resource_eligible": resource_eligible,
            "release_eligible": False,
            "quantized_only": diagnostic_inference,
            "compatibility": dict(self.compatibility),
            "declared_capabilities": declared,
            "versions": sdk_versions(),
        }
        capabilities.update(effective)
        return capabilities


def _head(raw: dict[str, Any], name: str) -> HeadStatus:
    if name not in raw:
        raise SdkError("package_invalid", f"heads.{name} must be listed explicitly")
    item = raw[name]
    if not isinstance(item, dict):
        raise SdkError("package_invalid", f"heads.{name} must be an object")
    return HeadStatus(
        present=bool(item.get("present")),
        trained=bool(item.get("trained")),
        status=str(item.get("status") or "missing"),
        tensor_prefixes=tuple(str(value) for value in (item.get("tensor_prefixes") or [])),
        training_receipt_sha256=(
            str(item["training_receipt_sha256"])
            if item.get("training_receipt_sha256") is not None
            else None
        ),
        label_codebook_sha256=(
            str(item["label_codebook_sha256"])
            if item.get("label_codebook_sha256") is not None
            else None
        ),
    )


def _optional_head(raw: dict[str, Any], name: str) -> HeadStatus:
    if name not in raw:
        return HeadStatus(False, False, "missing")
    return _head(raw, name)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_V2_SCHEMA: dict[str, Any] | None = None
_RESOURCE_RECEIPT_SCHEMA: dict[str, Any] | None = None


def _validate_against_v2_schema(manifest: dict[str, Any]) -> None:
    global _V2_SCHEMA
    if _V2_SCHEMA is None:
        try:
            loaded = loads_strict(
                (SPEC_DIR / "model-package-v2.schema.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as exc:
            raise SdkError("package_invalid", f"cannot load package v2 schema: {exc}") from exc
        if not isinstance(loaded, dict):
            raise SdkError("package_invalid", "package v2 schema root is not an object")
        _V2_SCHEMA = loaded
    try:
        validate_instance(manifest, _V2_SCHEMA)
    except ContractSchemaError as exc:
        raise SdkError("package_invalid", f"model-package-v2 JSON Schema: {exc}") from exc


def _safe_file(root: Path, rel: Any, *, label: str) -> Path:
    if (
        not isinstance(rel, str)
        or not rel
        or "\\" in rel
        or "\x00" in rel
        or rel.startswith("/")
        or any(part in {"", ".", ".."} for part in rel.split("/"))
    ):
        raise SdkError("package_path_unsafe", f"{label} must be a safe relative path")
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SdkError("package_path_unsafe", f"{label} escapes package root") from exc
    return candidate


def _reject_symlink_components(root: Path, rel: str) -> None:
    current = root
    for part in rel.split("/"):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise SdkError("file_not_found", f"missing package file: {rel}") from exc
        if stat.S_ISLNK(mode):
            raise SdkError("package_path_unsafe", f"package symlink is forbidden: {rel}")


def _collect_payload_files(root: Path) -> set[str]:
    found: set[str] = set()

    def visit(directory: Path) -> None:
        try:
            children = list(directory.iterdir())
        except OSError as exc:
            raise SdkError("file_not_found", f"cannot read package directory: {directory}") from exc
        for child in children:
            try:
                mode = child.lstat().st_mode
            except OSError as exc:
                raise SdkError("file_not_found", f"cannot inspect package path: {child}") from exc
            relative = child.relative_to(root).as_posix()
            if stat.S_ISLNK(mode):
                raise SdkError(
                    "package_path_unsafe", f"package symlink is forbidden: {relative}"
                )
            if stat.S_ISDIR(mode):
                visit(child)
            elif stat.S_ISREG(mode):
                found.add(relative)
            else:
                raise SdkError(
                    "package_path_unsafe", f"special package file is forbidden: {relative}"
                )

    visit(root)
    return found


def _validate_v2_file_inventory(
    manifest: dict[str, Any], tokenizer: dict[str, Any], container: dict[str, Any]
) -> dict[str, tuple[str, int, str]]:
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise SdkError("package_invalid", "files must inventory every payload")
    files: dict[str, tuple[str, int, str]] = {}
    valid_roles = {
        "tokenizer",
        "tensor_container",
        "tool_index",
        "training_receipt",
        "head_codebook",
        "resource_receipt",
        "auxiliary",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"path", "sha256", "nbytes", "role"}:
            raise SdkError("package_invalid", f"files[{index}] fields do not match v2")
        rel = row.get("path")
        # Validate the lexical path without requiring the payload to exist when
        # the caller explicitly requests a manifest-only diagnostic load.
        if (
            not isinstance(rel, str)
            or not rel
            or "\\" in rel
            or "\x00" in rel
            or rel.startswith("/")
            or any(part in {"", ".", ".."} for part in rel.split("/"))
        ):
            raise SdkError("package_path_unsafe", f"unsafe files path: {rel}")
        if rel == "mei-model.json":
            raise SdkError(
                "package_invalid",
                "mei-model.json is the sole self-hash exception and must not appear in files",
            )
        digest = _require_sha(row.get("sha256"), label=f"files[{rel}].sha256")
        nbytes = row.get("nbytes")
        role = row.get("role")
        if (
            not isinstance(nbytes, int)
            or isinstance(nbytes, bool)
            or nbytes < 0
            or role not in valid_roles
        ):
            raise SdkError("package_invalid", f"invalid files entry for {rel}")
        if rel in files:
            raise SdkError("package_invalid", f"duplicate files path {rel}")
        files[rel] = (digest, nbytes, str(role))

    required = [
        (tokenizer.get("file"), tokenizer.get("sha256"), None, "tokenizer"),
        (
            tokenizer.get("vocab_file"),
            tokenizer.get("vocab_sha256"),
            None,
            "tokenizer",
        ),
        (
            container.get("file"),
            container.get("sha256"),
            container.get("payload_bytes"),
            "tensor_container",
        ),
    ]
    for rel, digest, nbytes, role in required:
        if rel is None and digest is None:
            continue
        if not isinstance(rel, str) or not isinstance(digest, str):
            raise SdkError("package_invalid", "file/hash reference pair is incomplete")
        listed = files.get(rel)
        if listed is None:
            raise SdkError(
                "package_invalid", f"referenced payload {rel} is missing from files"
            )
        listed_hash, listed_nbytes, listed_role = listed
        if (
            listed_hash != digest
            or listed_role != role
            or (nbytes is not None and listed_nbytes != nbytes)
        ):
            raise SdkError("package_invalid", f"files entry disagrees with {rel} reference")
    return files


def _require_sha(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if _SHA256_RE.fullmatch(digest) is None:
        raise SdkError("package_invalid", f"{label} must be lowercase sha256")
    return digest


def _validate_v2_manifest(root: Path, manifest: dict[str, Any], *, verify_hashes: bool) -> bool:
    if manifest.get("product") != "mei-1.0-51m" or manifest.get("runtime_min") != "mei-runtime-abi-2":
        raise SdkError("package_invalid", "v2 package product/runtime_min mismatch")
    contracts = manifest.get("contracts") or {}
    expected_contracts = {
        "weight_contract_sha256": _CANONICAL_WEIGHT_CONTRACT_SHA256,
        "runtime_profile_sha256": _CANONICAL_RUNTIME_CONTRACT_SHA256,
        "training_aux_sha256": _CANONICAL_TRAINING_AUX_SHA256,
    }
    if contracts != expected_contracts:
        raise SdkError("package_invalid", "v2 contracts do not match frozen 51M identity")
    architecture = manifest.get("architecture") or {}
    if architecture != _CANONICAL_ARCHITECTURE:
        raise SdkError("package_invalid", "v2 package architecture topology mismatch")
    runtime_profile = manifest.get("runtime_profile") or {}
    if runtime_profile != _CANONICAL_RUNTIME_PROFILE:
        raise SdkError("package_invalid", "v2 package runtime profile mismatch")
    tokenizer = manifest.get("tokenizer") or {}
    for key, expected in {
        "id": "zh-24k-v1",
        "pad_id": 0,
        "eos_id": 1,
        "bos_id": 2,
        "unk_id": 3,
    }.items():
        if tokenizer.get(key) != expected:
            raise SdkError("package_invalid", f"tokenizer.{key} mismatch")
    container = manifest.get("tensor_container") or {}
    if container.get("format") != "mei-cq-tensor-v2":
        raise SdkError("package_invalid", "tensor_container.format must be mei-cq-tensor-v2")
    if container.get("quant_math_id") != "mei-cq-v2-g128-wht-codebook":
        raise SdkError("package_invalid", "v2 quant_math_id mismatch")
    payload_bytes = int(container.get("payload_bytes") or 0)
    if payload_bytes <= 0:
        raise SdkError("package_invalid", "tensor_container.payload_bytes must be positive")
    entries = container.get("directory")
    if not isinstance(entries, list) or not entries:
        raise SdkError("package_invalid", "tensor_container.directory must be non-empty")
    names: set[str] = set()
    ranges: list[tuple[int, int, str]] = []
    roles: dict[str, list[str]] = {name: [] for name in ALL_HEADS}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise SdkError("package_invalid", f"tensor directory entry {index} is not an object")
        name = str(entry.get("name") or "")
        if not name or name in names:
            raise SdkError("duplicate_tensor", f"duplicate or empty tensor name: {name}")
        names.add(name)
        role = str(entry.get("role") or "")
        if role not in roles:
            raise SdkError("package_invalid", f"unsupported tensor role: {role}")
        roles[role].append(name)
        shape = entry.get("shape")
        if not isinstance(shape, list) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape
        ):
            raise SdkError("package_invalid", f"invalid tensor shape: {name}")
        if math.prod(shape) != int(entry.get("n_params") or 0):
            raise SdkError("package_invalid", f"tensor n_params mismatch: {name}")
        if entry.get("dtype") in {"cq2", "cq4"}:
            if entry.get("group_size") != 128 or entry.get("transform") != "wht":
                raise SdkError("package_invalid", f"CQ tensor contract mismatch: {name}")
            expected_codebook = (
                "gaussian-lloyd-q2-v1" if entry.get("dtype") == "cq2" else "gaussian-lloyd-q4-v1"
            )
            if entry.get("codebook") != expected_codebook:
                raise SdkError("package_invalid", f"CQ tensor codebook mismatch: {name}")
        elif entry.get("dtype") in {"f16", "f32", "i8"}:
            if entry.get("transform") != "none" or entry.get("codebook") != "none":
                raise SdkError("package_invalid", f"safe tensor contract mismatch: {name}")
            if any(key in entry for key in ("group_size", "scales", "bit_map")):
                raise SdkError("package_invalid", f"safe tensor has CQ metadata: {name}")
        else:
            raise SdkError("package_invalid", f"unsupported tensor dtype: {name}")
        for range_name in ("data", "scales", "bit_map"):
            if range_name not in entry:
                continue
            raw_range = entry[range_name]
            if not isinstance(raw_range, dict):
                raise SdkError("package_invalid", f"invalid {range_name} range: {name}")
            offset = raw_range.get("offset")
            nbytes = raw_range.get("nbytes")
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or not isinstance(nbytes, int)
                or isinstance(nbytes, bool)
                or offset < 0
                or nbytes < 0
                or offset + nbytes > payload_bytes
            ):
                raise SdkError("package_range_invalid", f"out-of-bounds {range_name} range: {name}")
            if nbytes:
                ranges.append((offset, offset + nbytes, f"{name}.{range_name}"))
    files = _validate_v2_file_inventory(manifest, tokenizer, container)
    ranges.sort()
    for left, right in zip(ranges, ranges[1:]):
        if right[0] < left[1]:
            raise SdkError("package_range_invalid", f"overlapping tensor ranges: {left[2]} and {right[2]}")
    heads_raw = manifest.get("heads") or {}
    for role in ALL_HEADS:
        item = heads_raw.get(role)
        if role in OPTIONAL_HEADS and item is None:
            if roles[role]:
                raise SdkError("package_invalid", f"heads.{role} hides present tensors")
            continue
        if not isinstance(item, dict):
            raise SdkError("package_invalid", f"heads.{role} is required")
        prefixes = item.get("tensor_prefixes") or []
        ready = item.get("status") == "ready"
        if ready != (item.get("present") is True and item.get("trained") is True):
            raise SdkError(
                "package_invalid",
                f"heads.{role} ready status must match present=true/trained=true",
            )
        if item.get("present") and not prefixes:
            raise SdkError("package_invalid", f"heads.{role} has no tensor prefixes")
        if not item.get("present") and roles[role]:
            raise SdkError("package_invalid", f"heads.{role} hides present tensors")
        matched: set[str] = set()
        for prefix in prefixes:
            if not isinstance(prefix, str) or not prefix or prefix.endswith("."):
                raise SdkError("package_invalid", f"heads.{role} has invalid tensor prefix")
            current = {
                name for name in roles[role] if name == prefix or name.startswith(prefix + ".")
            }
            if not current:
                raise SdkError("package_invalid", f"heads.{role} tensor prefix missing: {prefix}")
            matched.update(current)
        if item.get("present") and matched != set(roles[role]):
            uncovered = sorted(set(roles[role]) - matched)
            raise SdkError(
                "package_invalid", f"heads.{role} has uncovered tensor: {uncovered[0]}"
            )
        if ready:
            receipt = _require_sha(
                item.get("training_receipt_sha256"),
                label=f"heads.{role}.training_receipt_sha256",
            )
            if receipt not in (manifest.get("training_receipts") or []):
                raise SdkError(
                    "package_invalid",
                    f"heads.{role} training receipt is absent from training_receipts",
                )
            if not any(
                file_role == "training_receipt" and digest == receipt
                for digest, _nbytes, file_role in files.values()
            ):
                raise SdkError(
                    "package_invalid",
                    f"heads.{role} training receipt has no inventoried payload",
                )
        if ready and role in _READY_AUX_HEAD_CONTRACTS:
            if role == "mw_disposition":
                if item.get("label_codebook_sha256") != MW_LABEL_CODEBOOK_SHA256:
                    raise SdkError(
                        "package_invalid",
                        "heads.mw_disposition label codebook does not match canonical reason codes",
                    )
                if sum(
                    file_role == "head_codebook" and digest == MW_LABEL_CODEBOOK_SHA256
                    for digest, _nbytes, file_role in files.values()
                ) != 1:
                    raise SdkError(
                        "package_invalid",
                        "heads.mw_disposition canonical codebook has no inventoried payload",
                    )
            if prefixes != [f"heads.{role}"]:
                raise SdkError(
                    "package_invalid", f"heads.{role} must use canonical tensor prefix"
                )
            expected = _READY_AUX_HEAD_CONTRACTS[role]
            actual = {
                str(entry["name"]): entry
                for entry in entries
                if entry.get("role") == role
            }
            if set(actual) != set(expected):
                raise SdkError(
                    "package_invalid", f"ready {role} tensor contract is incomplete"
                )
            for tensor_name, shape in expected.items():
                entry = actual[tensor_name]
                if (
                    tuple(entry.get("shape") or ()) != shape
                    or entry.get("dtype")
                    != ("cq2" if role == "narration_adapter" else "f16")
                    or entry.get("transform")
                    != ("wht" if role == "narration_adapter" else "none")
                    or entry.get("codebook")
                    != (
                        "gaussian-lloyd-q2-v1"
                        if role == "narration_adapter"
                        else "none"
                    )
                ):
                    raise SdkError(
                        "package_invalid", f"ready {role} tensor contract mismatch: {tensor_name}"
                    )
    capabilities = manifest.get("capabilities") or {}
    required_capabilities = {
        "retrieval",
        "full_call",
        "mw_disposition",
        "confidence",
        "multi_step",
    }
    allowed_capabilities = required_capabilities | {"narration"}
    if not required_capabilities.issubset(capabilities) or not set(capabilities).issubset(
        allowed_capabilities
    ):
        raise SdkError("package_invalid", "v2 capabilities are incomplete")
    if not all(isinstance(value, bool) for value in capabilities.values()):
        raise SdkError("package_invalid", "v2 capabilities must be boolean")
    receipts = manifest.get("training_receipts") or []
    if len(receipts) != len(set(receipts)):
        raise SdkError("package_invalid", "training_receipts must be unique")
    for receipt in receipts:
        digest = _require_sha(receipt, label="training_receipts[]")
        if sum(
            file_role == "training_receipt" and file_digest == digest
            for file_digest, _nbytes, file_role in files.values()
        ) != 1:
            raise SdkError(
                "package_invalid",
                "every training_receipts entry must have one inventoried payload",
            )
    resources = manifest.get("resources") or {}
    for key in ("package_bytes", "rust_session_peak_bytes", "wasm_heap_peak_bytes"):
        if not isinstance(resources.get(key), int) or isinstance(resources.get(key), bool) or resources[key] < 0:
            raise SdkError("package_invalid", f"resources.{key} must be non-negative integer")
    measurement_spec = resources.get("measurement_receipt")
    if measurement_spec is not None:
        if not isinstance(measurement_spec, dict):
            raise SdkError("package_invalid", "resources.measurement_receipt must be an object")
        receipt_file = measurement_spec.get("file")
        receipt_sha = measurement_spec.get("sha256")
        if sum(
            path == receipt_file and digest == receipt_sha and role == "resource_receipt"
            for path, (digest, _nbytes, role) in files.items()
        ) != 1:
            raise SdkError(
                "package_invalid",
                "resource measurement receipt is not uniquely inventoried",
            )
    index_files = [
        path for path, (_digest, _nbytes, role) in files.items() if role == "tool_index"
    ]
    if len(index_files) > 1 or (
        manifest.get("release_class") in {"candidate", "release"}
        and len(index_files) != 1
    ):
        raise SdkError(
            "package_invalid",
            "candidate/release packages require exactly one portable tool index",
        )
    if not verify_hashes:
        return False
    actual_files = _collect_payload_files(root)
    if "mei-model.json" not in actual_files:
        raise SdkError("file_not_found", "missing mei-model.json")
    actual_files.remove("mei-model.json")
    declared_files = set(files)
    if actual_files != declared_files:
        undeclared = sorted(actual_files - declared_files)
        missing = sorted(declared_files - actual_files)
        raise SdkError(
            "package_invalid",
            f"payload inventory mismatch undeclared={undeclared} missing={missing}",
        )
    measured_payload_bytes = 0
    for rel, (expected, expected_nbytes, _role) in files.items():
        _reject_symlink_components(root, rel)
        path = _safe_file(root, rel, label=f"files[{rel}].path")
        if not path.is_file():
            raise SdkError("file_not_found", f"missing package file: {rel}")
        actual_nbytes = path.stat().st_size
        if actual_nbytes != expected_nbytes:
            raise SdkError(
                "package_range_invalid",
                f"{rel} declares {expected_nbytes} bytes, file has {actual_nbytes}",
            )
        digest = _sha256_file(path)
        if digest != expected:
            raise SdkError("package_hash_mismatch", f"{rel} sha256 mismatch")
        measured_payload_bytes += actual_nbytes
    manifest_path = root / "mei-model.json"
    measured_package_bytes = measured_payload_bytes + manifest_path.stat().st_size
    if resources["package_bytes"] != measured_package_bytes:
        raise SdkError(
            "package_range_invalid",
            "resources.package_bytes declares "
            f"{resources['package_bytes']}, measured {measured_package_bytes}",
        )

    container_path = _safe_file(
        root, container.get("file"), label="tensor_container.file"
    )
    if container_path.stat().st_size != payload_bytes:
        raise SdkError("package_invalid", "tensor container size differs from payload_bytes")
    from .cq2 import TensorContainer

    parsed = TensorContainer.load(container_path)
    manifest_entries = {str(entry["name"]): entry for entry in entries}
    if set(parsed.entries) != set(manifest_entries):
        raise SdkError("package_invalid", "container and manifest tensor directories differ")
    for name, parsed_entry in parsed.entries.items():
        declared = manifest_entries[name]
        comparisons = {
            "shape": (list(parsed_entry.shape), list(declared.get("shape") or [])),
            "n_params": (parsed_entry.n_params, int(declared.get("n_params") or 0)),
            "dtype": (parsed_entry.dtype, str(declared.get("dtype") or "")),
            "role": (parsed_entry.role, declared.get("role")),
            "transform": (parsed_entry.transform, declared.get("transform")),
            "codebook": (parsed_entry.codebook, declared.get("codebook")),
            "group_size": (
                parsed_entry.group_size if parsed_entry.dtype in {"cq2", "cq4"} else None,
                declared.get("group_size"),
            ),
            "data": (
                {"offset": parsed_entry.data_offset, "nbytes": parsed_entry.data_nbytes},
                declared.get("data"),
            ),
            "scales": (
                {"offset": parsed_entry.scales_offset, "nbytes": parsed_entry.scales_nbytes}
                if parsed_entry.scales_nbytes
                else None,
                declared.get("scales"),
            ),
            "bit_map": (
                {"offset": parsed_entry.bit_map_offset, "nbytes": parsed_entry.bit_map_nbytes}
                if parsed_entry.bit_map_nbytes
                else None,
                declared.get("bit_map"),
            ),
        }
        for field_name, (actual, expected_value) in comparisons.items():
            if actual != expected_value:
                raise SdkError(
                    "package_invalid", f"container manifest mismatch: {name}.{field_name}"
                )
    if index_files:
        from .shared import ToolIndex

        try:
            index = ToolIndex.load(root / index_files[0])
        except (OSError, TypeError, ValueError) as exc:
            raise SdkError("package_invalid", f"invalid portable tool index: {exc}") from exc
        contrastive_names = sorted(
            name
            for name, entry in parsed.entries.items()
            if entry.role == "contrastive"
        )
        head_digest = hashlib.sha256()
        for name in contrastive_names:
            entry = parsed.entries[name]
            head_digest.update(name.encode("utf-8"))
            for offset, nbytes in (
                (entry.data_offset, entry.data_nbytes),
                (entry.scales_offset, entry.scales_nbytes),
                (entry.bit_map_offset, entry.bit_map_nbytes),
            ):
                if nbytes:
                    head_digest.update(parsed.blob[offset : offset + nbytes])
        expected_head_hash = head_digest.hexdigest()
        expected_tokenizer_hash = str(
            tokenizer.get("sha256") or tokenizer.get("vocab_sha256") or ""
        )
        if (
            index.model_hash != str(container.get("sha256") or "")
            or index.head_hash != expected_head_hash
            or index.tokenizer_hash != expected_tokenizer_hash
        ):
            raise SdkError(
                "package_hash_mismatch",
                "tool index model/head/tokenizer fingerprint mismatch",
            )
    return True


def _verify_resource_measurement_receipt(
    root: Path, manifest: dict[str, Any]
) -> bool:
    global _RESOURCE_RECEIPT_SCHEMA
    spec = (manifest.get("resources") or {}).get("measurement_receipt")
    if spec is None:
        return False
    path = _safe_file(root, spec.get("file"), label="resources.measurement_receipt.file")
    if _sha256_file(path) != spec.get("sha256"):
        raise SdkError("package_hash_mismatch", "resource receipt sha256 mismatch")
    try:
        receipt = loads_strict(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SdkError("invalid_json", f"resource receipt is not valid JSON: {exc}") from exc
    if not isinstance(receipt, dict):
        raise SdkError("package_invalid", "resource receipt root must be an object")
    if _RESOURCE_RECEIPT_SCHEMA is None:
        try:
            loaded = loads_strict(
                (SPEC_DIR / "resource-measurement-receipt-v1.schema.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError) as exc:
            raise SdkError("package_invalid", f"cannot load resource receipt schema: {exc}") from exc
        if not isinstance(loaded, dict):
            raise SdkError("package_invalid", "resource receipt schema root must be an object")
        _RESOURCE_RECEIPT_SCHEMA = loaded
    try:
        validate_instance(receipt, _RESOURCE_RECEIPT_SCHEMA)
    except ContractSchemaError as exc:
        raise SdkError("package_invalid", f"resource receipt JSON Schema: {exc}") from exc
    if (
        receipt.get("package_id") != manifest.get("package_id")
        or receipt.get("tensor_container_sha256")
        != (manifest.get("tensor_container") or {}).get("sha256")
        or receipt.get("runtime_abi") != "mei-runtime-abi-2"
    ):
        raise SdkError("package_invalid", "resource measurement receipt identity mismatch")
    measurements = receipt.get("measurements") or {}
    resources = manifest.get("resources") or {}
    for key in ("package_bytes", "rust_session_peak_bytes", "wasm_heap_peak_bytes"):
        if measurements.get(key) != resources.get(key):
            raise SdkError(
                "package_invalid", f"resource measurement receipt disagrees on {key}"
            )
    return True


def load_package(package_dir: str | Path, *, verify_hashes: bool = True) -> ModelPackage:
    root = Path(package_dir).resolve()
    manifest_path = root / "mei-model.json"
    if manifest_path.is_symlink():
        raise SdkError(
            "package_path_unsafe", "mei-model.json must be a regular non-symlink file"
        )
    if not manifest_path.is_file() or not stat.S_ISREG(manifest_path.lstat().st_mode):
        raise SdkError("file_not_found", f"missing mei-model.json: {manifest_path}")
    try:
        manifest = loads_strict(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise SdkError("invalid_json", str(exc)) from exc
    if not isinstance(manifest, dict):
        raise SdkError("package_invalid", "package manifest root must be an object")
    package_format = str(manifest.get("package_format") or "")
    if package_format not in {"mei-model-package-v1", "mei-model-package-v2"}:
        raise SdkError("package_invalid", "unsupported package_format")
    product = str(manifest.get("product") or "")
    if product != "mei-1.0-51m":
        raise SdkError("package_invalid", "product must be mei-1.0-51m")
    if package_format == "mei-model-package-v2":
        _validate_against_v2_schema(manifest)
    heads_raw = manifest.get("heads")
    if not isinstance(heads_raw, dict):
        raise SdkError("package_invalid", "heads object is required")
    heads = HeadReport(
        lm=_head(heads_raw, "lm"),
        contrastive=_head(heads_raw, "contrastive"),
        mw_disposition=_head(heads_raw, "mw_disposition"),
        confidence=_head(heads_raw, "confidence"),
        narration_adapter=_optional_head(heads_raw, "narration_adapter"),
    )
    warnings: list[str] = []
    for name in heads.missing():
        warnings.append(f"head {name} is missing or untrained")
    verified = False
    resource_measurement_verified = False
    if package_format == "mei-model-package-v2":
        verified = _validate_v2_manifest(root, manifest, verify_hashes=verify_hashes)
        if verify_hashes:
            resource_measurement_verified = _verify_resource_measurement_receipt(
                root, manifest
            )
    elif verify_hashes:
        for key in ("tokenizer", "weights"):
            spec = manifest.get(key) or {}
            rel = spec.get("file")
            expected = str(spec.get("sha256") or "")
            path = _safe_file(root, rel, label=f"{key}.file") if rel else None
            if path is None or not path.is_file():
                raise SdkError("file_not_found", f"missing {key} file: {rel}")
            digest = _sha256_file(path)
            if digest != expected.lower():
                raise SdkError(
                    "package_hash_mismatch",
                    f"{key} sha256 mismatch: expected {expected}, got {digest}",
                )
        verified = True
        tok = manifest.get("tokenizer") or {}
        vocab_rel = tok.get("vocab_file")
        vocab_sha = str(tok.get("vocab_sha256") or "")
        if vocab_rel:
            vpath = _safe_file(root, vocab_rel, label="tokenizer.vocab_file")
            if not vpath.is_file():
                raise SdkError("file_not_found", f"missing tokenizer vocab file: {vocab_rel}")
            if vocab_sha and _sha256_file(vpath) != vocab_sha.lower():
                raise SdkError(
                    "package_hash_mismatch",
                    f"tokenizer vocab sha256 mismatch: expected {vocab_sha}",
                )
    package = ModelPackage(
        path=root,
        manifest=manifest,
        heads=heads,
        verified_hashes=verified,
        resource_measurement_verified=resource_measurement_verified,
        warnings=warnings,
        compatibility=(
            {
                "mode": "native_v2",
                "source_format": package_format,
                "capability_complete": False,
            }
            if package_format == "mei-model-package-v2"
            else {
                "mode": "read_only_degraded_adapter",
                "source_format": package_format,
                "capability_complete": False,
            }
        ),
    )
    if package_format == "mei-model-package-v2":
        complete = (
            not heads.missing()
            and package.tensor_identity_complete()
            and package.verified_hashes
            and all(bool(value) for value in (manifest.get("capabilities") or {}).values())
        )
        package.compatibility["capability_complete"] = complete
        if not package.tensor_identity_complete():
            package.warnings.append("LM tensor directory does not match the 51,463,797 identity or contains MTP")
    return package
