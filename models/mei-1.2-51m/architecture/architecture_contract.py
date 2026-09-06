"""Stable compatibility contracts for the mei-1.0-51m architecture.

The historical ``architecture_sha256`` hashes source bytes and is useful as
provenance, but it is not a checkpoint compatibility contract.  This module
keeps checkpoint geometry, deployed runtime policy, and training-only
auxiliaries independently versioned and hashable without importing MLX.
"""

from __future__ import annotations

import hashlib
import json
import ast
import struct
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SPEC_PATH = HERE / "spec/model.json"
LEGACY_ALIAS_PATH = HERE / "spec/legacy-architecture-aliases.json"
CONTRACT_SCHEMA = "mei-architecture-contract-v1"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value: object) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def load_spec(path: Path | None = None) -> dict[str, Any]:
    return _load(path or SPEC_PATH)


def ordered_weight_tensors(spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Expand the deterministic 400-tensor deployed backbone topology."""
    doc = deepcopy(spec or load_spec())
    arch = doc["architecture"]
    d_model = int(arch["d_model"])
    n_layers = int(arch["n_layers"])
    n_heads = int(arch["n_heads"])
    n_kv_heads = int(arch["n_kv_heads"])
    head_dim = d_model // n_heads
    kv_dim = n_kv_heads * head_dim
    vocab_size = int(arch["vocab_size"])
    engram_orders = tuple(int(value) for value in arch["engram_orders"])
    engram_heads = max(1, d_model // (len(engram_orders) * 128))
    engram_sub_dim = max(1, d_model // (len(engram_orders) * engram_heads))
    engram_tables = len(engram_orders) * engram_heads
    engram_width = engram_tables * engram_sub_dim
    lanes = int(arch["mhc_lanes"])
    lane_width = lanes * d_model
    conf_probes = int((doc.get("training_aux") or {}).get("confidence_probes") or 8)

    tensors: list[dict[str, Any]] = []

    def add(name: str, *shape: int) -> None:
        tensors.append({"name": name, "shape": list(shape)})

    add("embed.weight", vocab_size, d_model)
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
    for site in range(len(arch["engram_layers"])):
        prefix = f"engrams.{site}"
        add(f"{prefix}.tables", engram_tables, int(arch["engram_slots"]), engram_sub_dim)
        add(f"{prefix}.key_proj.weight", d_model, engram_width)
        add(f"{prefix}.value_proj.weight", d_model, engram_width)
        add(f"{prefix}.taps", int(arch["engram_conv_taps"]), d_model)
    add("mhc_phi_pre", n_layers, lane_width, lanes)
    add("mhc_phi_post", n_layers, lane_width, lanes)
    add("mhc_phi_res", n_layers, lane_width, lanes * lanes)
    add("mhc_b_pre", n_layers, lanes)
    add("mhc_b_post", n_layers, lanes)
    add("mhc_b_res", n_layers, lanes, lanes)
    add("mhc_a_pre", n_layers)
    add("mhc_a_post", n_layers)
    add("mhc_a_res", n_layers)
    if bool(arch.get("confidence_head", True)):
        add("conf_probes", conf_probes, d_model)
        add("conf_proj.weight", 1, conf_probes * d_model)
        add("conf_proj.bias", 1)
    if not bool(arch.get("tie_embeddings", True)):
        add("lm_head.weight", vocab_size, d_model)
    return tensors


def _n_params(tensors: list[dict[str, Any]]) -> int:
    total = 0
    for row in tensors:
        size = 1
        for dim in row["shape"]:
            size *= int(dim)
        total += size
    return total


def npz_weight_geometry(path: Path) -> list[dict[str, Any]]:
    """Read only NPY headers from a named NPZ; tensor payloads stay untouched."""
    rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            if not member.filename.endswith(".npy"):
                continue
            with archive.open(member) as handle:
                if handle.read(6) != b"\x93NUMPY":
                    raise ValueError(f"invalid NPY member: {member.filename}")
                major, _minor = struct.unpack("BB", handle.read(2))
                if major == 1:
                    header_len = struct.unpack("<H", handle.read(2))[0]
                elif major in {2, 3}:
                    header_len = struct.unpack("<I", handle.read(4))[0]
                else:
                    raise ValueError(f"unsupported NPY version {major}")
                header = ast.literal_eval(handle.read(header_len).decode("latin1").strip())
                rows.append(
                    {
                        "name": member.filename[: -len(".npy")],
                        "shape": list(header["shape"]),
                    }
                )
    return rows


def validate_npz_weight_geometry(path: Path, spec: dict[str, Any] | None = None) -> dict[str, Any]:
    expected = ordered_weight_tensors(spec)
    actual = npz_weight_geometry(path)
    return {
        "ok": actual == expected,
        "expected_tensors": len(expected),
        "actual_tensors": len(actual),
        "expected_params": _n_params(expected),
        "actual_params": _n_params(actual),
    }


def weight_contract(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    doc = deepcopy(spec or load_spec())
    arch = doc["architecture"]
    tensors = ordered_weight_tensors(doc)
    contract = {
        "schema": CONTRACT_SCHEMA,
        "kind": "weight_geometry",
        "architecture_id": doc["architecture_id"],
        "tokenizer": deepcopy(doc["tokenizer"]),
        "tensor_order": tensors,
        "trainable_params": _n_params(tensors),
        "topology": {
            "tie_embeddings": bool(arch["tie_embeddings"]),
            "mlp": arch["mlp"],
            "rope_theta": float(arch["rope_theta"]),
            "engram_layers": list(arch["engram_layers"]),
            "engram_orders": list(arch["engram_orders"]),
            "sinkhorn_iters": int(arch["sinkhorn_iters"]),
            "fixed_non_parameters": list(doc.get("fixed_non_parameters") or []),
        },
    }
    expected = int((doc.get("expected_trainable_params") or {}).get("target") or 0)
    if expected and contract["trainable_params"] != expected:
        raise ValueError(
            f"weight tensor contract has {contract['trainable_params']} params, expected {expected}"
        )
    return contract


def runtime_profile_contract(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    doc = deepcopy(spec or load_spec())
    profile = deepcopy(doc.get("runtime_profile") or {})
    return {
        "schema": CONTRACT_SCHEMA,
        "kind": "runtime_profile",
        "architecture_id": doc["architecture_id"],
        "max_context_tokens": int(profile.get("max_context_tokens") or 2048),
        "default_profile": str(profile.get("default_profile") or "standard"),
        "stable_prefix_profiles": {
            str(name): int(value)
            for name, value in sorted(
                (profile.get("stable_prefix_profiles") or {"compact": 1024, "standard": 1536}).items()
            )
        },
        "ordinary_window_policy": str(
            profile.get("ordinary_window_policy") or "dynamic_remainder"
        ),
        "default_output_max_tokens": int(profile.get("default_output_max_tokens") or 128),
        "tool_batch_size": int(profile.get("tool_batch_size") or 5),
        "context_packer_id": str(
            profile.get("context_packer_id") or "mei-tool-context-packer-v1"
        ),
        "retrieval_batch_policy_id": str(
            profile.get("retrieval_batch_policy_id")
            or "mei-retrieval-fixed-five-batches-v1"
        ),
        "prompt_framing_id": str(
            profile.get("prompt_framing_id") or "mei-tool-prompt-framing-v1"
        ),
        "assistant_suffix": str(
            profile.get("assistant_suffix")
            or "<|im_end|>\n<|im_start|>assistant\n"
        ),
        "kv_cache_dtype": str(profile.get("kv_cache_dtype") or "int8"),
        "activation_dtype": str(profile.get("activation_dtype") or "int8"),
    }


def training_aux_contract(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    doc = deepcopy(spec or load_spec())
    aux = deepcopy(doc.get("training_aux") or {})
    return {
        "schema": CONTRACT_SCHEMA,
        "kind": "training_aux",
        "architecture_id": doc["architecture_id"],
        "mtp": deepcopy(
            aux.get("mtp")
            or {
                "mode": "training_only_ablation",
                "enabled_by_default": False,
                "export": False,
            }
        ),
        "confidence_probes": int(aux.get("confidence_probes") or 8),
    }


def contract_bundle(spec: dict[str, Any] | None = None) -> dict[str, Any]:
    doc = deepcopy(spec or load_spec())
    weight = weight_contract(doc)
    runtime = runtime_profile_contract(doc)
    training = training_aux_contract(doc)
    return {
        "architecture_id": doc["architecture_id"],
        "weight_contract": weight,
        "weight_contract_sha256": _digest(weight),
        "runtime_profile": runtime,
        "runtime_profile_sha256": _digest(runtime),
        "training_aux": training,
        "training_aux_sha256": _digest(training),
    }


def legacy_aliases(path: Path | None = None) -> dict[str, Any]:
    source = path or LEGACY_ALIAS_PATH
    return _load(source) if source.is_file() else {"aliases": []}


def resolve_legacy_weight_contract(
    architecture_sha256: str, *, aliases: dict[str, Any] | None = None
) -> str | None:
    for row in (aliases or legacy_aliases()).get("aliases") or []:
        if row.get("legacy_architecture_sha256") != architecture_sha256:
            continue
        if row.get("compatibility") != "weight_compatible":
            return None
        return str(row.get("weight_contract_sha256") or "") or None
    return None


def main() -> int:
    print(json.dumps(contract_bundle(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
