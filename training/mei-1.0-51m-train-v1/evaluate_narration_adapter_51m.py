#!/usr/bin/env python3
"""Evaluate the packed CQ2 narration sidecar by actual greedy generation.

This is deliberately separate from narration SFT loss.  It loads the final
``mei-model-package-v2`` package, generates every frozen eval example through
the same bounded MLX oracle used for portable numerical comparison, and
records both learned-adapter quality and the fail-closed deterministic result.

The learned score is evidence, not a reason to keep tuning indefinitely.  A
completed run therefore has ``terminal_status=passed`` even when adapter
quality is degraded, while deterministic fallback must remain exact for every
row or the stage fails.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from _repo import CURRENT_PATH, ROOT


DEFAULT_RELEASE = (
    ROOT
    / "notebook/sft/mei-1.0-51m/releases"
    / "mei-1.0-51m-narration-sft-agent300m-v3"
)
MAX_NEW = 48
SOURCE_FILES = (
    "training/mei-1.0-51m-train-v1/evaluate_narration_adapter_51m.py",
    "sdk/python/mei_sdk/runtime_51m.py",
    "sdk/python/mei_sdk/package.py",
    "runtime/_shared/narration.py",
    "architecture/mei-1.0-51m-arch-v1/heads.py",
)
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        raise RuntimeError(f"refusing to overwrite a different artifact: {path}")
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    temporary.replace(path)


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    write_once(path, canonical_bytes(value) + b"\n")


def _verify_package_receipt(path: Path, package_dir: Path) -> dict[str, Any]:
    receipt = load_json(path)
    if (
        receipt.get("schema") != "mei-productization-downstream-package-receipt-v1"
        or receipt.get("terminal_status") != "passed"
        or Path(str(receipt.get("package_path") or "")).resolve()
        != package_dir.resolve()
        or receipt.get("current_unchanged") is not True
        or receipt.get("current_sha256") != sha_file(CURRENT_PATH)
    ):
        raise RuntimeError("downstream package receipt is not reusable")
    for raw, digest in (receipt.get("output_hashes") or {}).items():
        artifact = Path(raw).resolve()
        if not artifact.is_relative_to(package_dir.resolve()):
            raise RuntimeError(f"package receipt output escaped package: {artifact}")
        if not artifact.is_file() or sha_file(artifact) != digest:
            raise RuntimeError(f"package receipt output drifted: {artifact}")
    return receipt


def _verify_release(release_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = release_dir / "manifest.json"
    isolation_path = release_dir / "isolation-receipt.json"
    manifest = load_json(manifest_path)
    isolation = load_json(isolation_path)
    if (
        manifest.get("schema") != "mei-narration-sft-data-release-v2"
        or (manifest.get("adapter") or {}).get("rank") != 16
        or (manifest.get("adapter") or {}).get("parameter_count") != 392_192
        or isolation.get("status") != "passed"
        or isolation.get("terminal_only") is not True
        or isolation.get("verified_result_only") is not True
        or isolation.get("executor_access") is not False
    ):
        raise RuntimeError("narration data release contract is incomplete")
    isolation_spec = manifest.get("isolation_receipt") or {}
    if (
        isolation_spec.get("file") != isolation_path.name
        or isolation_spec.get("sha256") != sha_file(isolation_path)
    ):
        raise RuntimeError("narration isolation receipt hash drifted")
    eval_spec = (manifest.get("files") or {}).get("narration.eval.jsonl") or {}
    eval_path = release_dir / "narration.eval.jsonl"
    if (
        not eval_path.is_file()
        or eval_spec.get("sha256") != sha_file(eval_path)
        or not isinstance(eval_spec.get("rows"), int)
    ):
        raise RuntimeError("narration eval split hash drifted")
    rows = load_jsonl(eval_path)
    if len(rows) != int(eval_spec["rows"]) or len(rows) != 600:
        raise RuntimeError(f"narration eval must contain exactly 600 rows, got {len(rows)}")
    return manifest, rows


def _verify_package_manifest(package_dir: Path) -> dict[str, Any]:
    manifest = load_json(package_dir / "mei-model.json")
    narration = (manifest.get("heads") or {}).get("narration_adapter") or {}
    directory = (manifest.get("tensor_container") or {}).get("directory") or []
    tensors = {
        str(row.get("name")): row
        for row in directory
        if isinstance(row, dict) and row.get("role") == "narration_adapter"
    }
    expected = {
        "heads.narration_adapter.down.weight": ([16, 512], 8_192),
        "heads.narration_adapter.up.weight": ([24_000, 16], 384_000),
    }
    if (
        manifest.get("package_format") != "mei-model-package-v2"
        or (manifest.get("capabilities") or {}).get("narration") is not True
        or narration.get("present") is not True
        or narration.get("trained") is not True
        or narration.get("status") != "ready"
        or set(tensors) != set(expected)
    ):
        raise RuntimeError("packed narration capability is incomplete")
    for name, (shape, parameters) in expected.items():
        row = tensors[name]
        if (
            row.get("shape") != shape
            or row.get("n_params") != parameters
            or row.get("dtype") != "cq2"
            or row.get("group_size") != 128
            or row.get("transform") != "wht"
        ):
            raise RuntimeError(f"packed narration tensor contract drifted: {name}")
    return manifest


def extract_numbers(text: str) -> list[str]:
    return NUMBER_PATTERN.findall(text)


def required_facts_match(text: str, required_facts: Iterable[Any]) -> bool:
    return all(str(fact) in text for fact in required_facts)


def polarity_matches(text: str, polarity: str, required_facts: Iterable[Any]) -> bool:
    required = list(required_facts)
    if polarity.startswith("failure"):
        return "失败" in text
    if polarity == "cancelled":
        return "取消" in text
    if polarity == "no_change":
        return "无需调整" in text
    if polarity == "partial":
        return "部分" in text and "失败" in text
    if polarity == "success_off":
        return "关闭" in text
    if polarity == "success_on":
        return "启动" in text or "开始" in text
    if polarity == "success_lock":
        return "锁定" in text
    if polarity == "success_multi":
        return "\n" in text and required_facts_match(text, required)
    return bool(text) and required_facts_match(text, required)


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        raise RuntimeError("cannot summarize an empty narration evaluation")
    total = len(cases)

    def count(key: str) -> int:
        return sum(int(bool(row[key])) for row in cases)

    fact_rows = [row for row in cases if row["required_facts"]]
    numeric_rows = [row for row in cases if row["target_numbers"]]
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cases:
        families[str(row["semantic_family"])].append(row)
    per_family = {}
    for family in sorted(families, key=lambda value: value.encode("utf-8")):
        rows = families[family]
        size = len(rows)
        per_family[family] = {
            "rows": size,
            "adapter_exact": sum(int(row["adapter_exact"]) for row in rows),
            "adapter_exact_rate": sum(int(row["adapter_exact"]) for row in rows) / size,
            "required_facts_match": sum(int(row["required_facts_match"]) for row in rows),
            "polarity_match": sum(int(row["polarity_match"]) for row in rows),
            "fallback_used": sum(int(row["fallback_used"]) for row in rows),
            "max_new_reached": sum(int(row["max_new_reached"]) for row in rows),
        }
    exact = count("adapter_exact")
    fallback_exact = count("delivered_exact")
    return {
        "rows": total,
        "semantic_family_count": len(families),
        "adapter_exact": exact,
        "adapter_exact_rate": exact / total,
        "adapter_fallback": total - exact,
        "adapter_fallback_rate": (total - exact) / total,
        "adapter_nonempty": count("adapter_nonempty"),
        "adapter_nonempty_rate": count("adapter_nonempty") / total,
        "required_fact_rows": len(fact_rows),
        "required_facts_match": sum(int(row["required_facts_match"]) for row in fact_rows),
        "required_facts_match_rate": (
            sum(int(row["required_facts_match"]) for row in fact_rows) / len(fact_rows)
            if fact_rows
            else 1.0
        ),
        "numeric_rows": len(numeric_rows),
        "numeric_exact": sum(int(row["numeric_exact"]) for row in numeric_rows),
        "numeric_exact_rate": (
            sum(int(row["numeric_exact"]) for row in numeric_rows) / len(numeric_rows)
            if numeric_rows
            else 1.0
        ),
        "polarity_match": count("polarity_match"),
        "polarity_match_rate": count("polarity_match") / total,
        "bounded": count("bounded"),
        "max_new_reached": count("max_new_reached"),
        "deterministic_fallback_exact": fallback_exact,
        "deterministic_fallback_exact_rate": fallback_exact / total,
        "delivered_grounded": fallback_exact,
        "delivered_grounded_rate": fallback_exact / total,
        "learned_adapter_quality": "validated" if exact == total else "degraded",
        "mechanism_status": "passed" if fallback_exact == total else "failed",
        "per_family": per_family,
    }


def _build_fingerprint(
    *,
    package_dir: Path,
    package_receipt_path: Path,
    release_dir: Path,
    release_manifest: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    source_hashes = {relative: sha_file(ROOT / relative) for relative in SOURCE_FILES}
    packages = {}
    for name in ("mlx", "numpy", "sentencepiece"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing"
    inputs = {
        "package_dir": str(package_dir),
        "package_receipt_sha256": sha_file(package_receipt_path),
        "package_manifest_sha256": sha_file(package_dir / "mei-model.json"),
        "release_dir": str(release_dir),
        "release_manifest_sha256": sha_file(release_dir / "manifest.json"),
        "eval_sha256": release_manifest["files"]["narration.eval.jsonl"]["sha256"],
        "isolation_receipt_sha256": sha_file(release_dir / "isolation-receipt.json"),
        "max_new": MAX_NEW,
        "backend": "mlx-reference",
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "source_hashes": source_hashes,
    }
    return sha_bytes(canonical_bytes(inputs)), inputs


def _preflight_rows(rows: list[dict[str, Any]]) -> list[str]:
    sdk_path = str(ROOT / "sdk/python")
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    from mei_sdk.shared import NarrationProvider, verified_result_view

    provider = NarrationProvider()
    seen = set()
    deterministic_targets = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("sample_id") or "")
        views = row.get("verified_result_views")
        target = str(row.get("target") or "").strip()
        if not sample_id or sample_id in seen or not isinstance(views, list) or not views or not target:
            raise RuntimeError(f"invalid narration eval row at index {index}")
        seen.add(sample_id)
        if any((view.get("provenance") or {}).get("verified") is not True for view in views):
            raise RuntimeError(f"unverified narration eval row: {sample_id}")
        deterministic = "\n".join(
            provider.narrate(verified_result_view(view)) for view in views
        )
        if deterministic != target:
            raise RuntimeError(f"deterministic fallback drifted for {sample_id}")
        deterministic_targets.append(deterministic)
    return deterministic_targets


def _evaluate_one(runtime: Any, row: dict[str, Any], deterministic: str) -> dict[str, Any]:
    generated = runtime.generate_narration(str(row["prompt"]), max_new=MAX_NEW)
    text = str(generated.get("text") or "").strip()
    target = str(row["target"]).strip()
    required_facts = [str(value) for value in (row.get("required_facts") or [])]
    target_numbers = extract_numbers(target)
    generated_numbers = extract_numbers(text)
    exact = text == target
    bounded = generated.get("bounded") is True
    n_out = int(generated.get("n_out") or 0)
    if not bounded or n_out < 0 or n_out > MAX_NEW:
        raise RuntimeError(f"unbounded narration generation: {row['sample_id']}")
    delivered = text if exact else deterministic
    return {
        "sample_id": str(row["sample_id"]),
        "semantic_family": str(row.get("semantic_family") or ""),
        "polarity": str(row.get("polarity") or ""),
        "prompt_sha256": sha_bytes(str(row["prompt"]).encode("utf-8")),
        "target": target,
        "generated": text,
        "required_facts": required_facts,
        "target_numbers": target_numbers,
        "generated_numbers": generated_numbers,
        "adapter_exact": exact,
        "adapter_nonempty": bool(text),
        "required_facts_match": required_facts_match(text, required_facts),
        "numeric_exact": Counter(generated_numbers) == Counter(target_numbers),
        "polarity_match": polarity_matches(
            text, str(row.get("polarity") or ""), required_facts
        ),
        "bounded": bounded,
        "n_out": n_out,
        "max_new_reached": n_out == MAX_NEW,
        "fallback_used": not exact,
        "delivered": delivered,
        "delivered_exact": delivered == target,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = args.package.resolve()
    package_receipt_path = args.package_receipt.resolve()
    release_dir = args.release.resolve()
    out_dir = args.out_dir.resolve()
    package_receipt = _verify_package_receipt(package_receipt_path, package_dir)
    package_manifest = _verify_package_manifest(package_dir)
    release_manifest, rows = _verify_release(release_dir)
    deterministic_targets = _preflight_rows(rows)
    fingerprint, fingerprint_inputs = _build_fingerprint(
        package_dir=package_dir,
        package_receipt_path=package_receipt_path,
        release_dir=release_dir,
        release_manifest=release_manifest,
    )
    if args.dry_run:
        return {
            "ok": True,
            "dry_run": True,
            "planned_rows": len(rows),
            "stage_fingerprint_sha256": fingerprint,
            "package_id": package_manifest.get("package_id"),
            "release_id": release_manifest.get("release_id"),
        }

    receipt_path = out_dir / "receipt.json"
    report_path = out_dir / "narration-generation-eval.json"
    cases_path = out_dir / "narration-generation-cases.jsonl"
    if receipt_path.is_file():
        receipt = load_json(receipt_path)
        if (
            receipt.get("terminal_status") == "passed"
            and receipt.get("stage_fingerprint_sha256") == fingerprint
        ):
            for raw, digest in (receipt.get("output_hashes") or {}).items():
                artifact = Path(raw)
                if not artifact.is_file() or sha_file(artifact) != digest:
                    raise RuntimeError(f"narration evaluation output drifted: {artifact}")
            return receipt
        raise RuntimeError("narration evaluation receipt exists with a different fingerprint")

    sdk_path = str(ROOT / "sdk/python")
    if sdk_path not in sys.path:
        sys.path.insert(0, sdk_path)
    from mei_sdk.package import load_package
    from mei_sdk.runtime_51m import load_51m_runtime
    from mlx_memory_policy_51m import (
        configure_mlx_memory,
        mlx_memory_snapshot,
        release_mlx_memory,
    )

    mx, memory_policy = configure_mlx_memory(reset_peak=True)
    package = load_package(package_dir)
    runtime, load_report = load_51m_runtime(package, backend="mlx-reference")
    if runtime.narration_adapter is None or load_report.get("narration_adapter") is not True:
        raise RuntimeError("packed narration adapter did not load into MLX runtime")

    attempt_dir = out_dir / "attempts" / fingerprint
    case_dir = attempt_dir / "cases"
    evaluated: list[dict[str, Any]] = []
    for index, (row, deterministic) in enumerate(zip(rows, deterministic_targets, strict=True)):
        case_path = case_dir / f"{index:06d}.json"
        if case_path.is_file():
            case = load_json(case_path)
            if (
                case.get("sample_id") != row.get("sample_id")
                or case.get("prompt_sha256")
                != sha_bytes(str(row["prompt"]).encode("utf-8"))
            ):
                raise RuntimeError(f"narration resume case drifted: {case_path}")
        else:
            case = _evaluate_one(runtime, row, deterministic)
            write_json_once(case_path, case)
        evaluated.append(case)
        # The locked evaluation policy clears allocator cache per generation;
        # otherwise hundreds of variable-length prefills can make Metal cache
        # growth look like a model-memory requirement.
        release_mlx_memory(mx)
        if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(rows):
            print(
                json.dumps(
                    {
                        "narration_eval": index + 1,
                        "rows": len(rows),
                        "adapter_exact": sum(
                            int(item["adapter_exact"]) for item in evaluated
                        ),
                        "fallback": sum(int(item["fallback_used"]) for item in evaluated),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    metrics = summarize_cases(evaluated)
    if metrics["mechanism_status"] != "passed" or metrics["bounded"] != len(rows):
        raise RuntimeError("narration deterministic fallback or bounded runtime gate failed")
    cases_payload = b"".join(canonical_bytes(case) + b"\n" for case in evaluated)
    write_once(cases_path, cases_payload)
    report = {
        "schema": "mei-narration-generation-eval-v1",
        "product": "mei-1.0-51m",
        "package_id": package_manifest["package_id"],
        "release_id": release_manifest["release_id"],
        "stage_fingerprint_sha256": fingerprint,
        "terminal_status": "passed",
        "process_complete": True,
        "quality_status": metrics["learned_adapter_quality"],
        "not_a_score_claim": True,
        "packed_cq2_evaluated": True,
        "backbone_frozen": True,
        "rank": 16,
        "parameter_count": 392_192,
        "max_new": MAX_NEW,
        "acceptance_contract": "exact-deterministic-template-else-fallback",
        "metrics": metrics,
        "runtime_load": load_report,
        "mlx_memory_policy": {
            **memory_policy,
            "clear_cache_per_generation": True,
            "final": mlx_memory_snapshot(mx),
        },
        "inputs": fingerprint_inputs,
    }
    write_json_once(report_path, report)
    output_hashes = {
        str(cases_path): sha_file(cases_path),
        str(report_path): sha_file(report_path),
    }
    receipt = {
        "schema": "mei-productization-downstream-stage-receipt-v1",
        "stage_id": "narration_generation_eval",
        "terminal_status": "passed",
        "process_complete": True,
        "quality_status": metrics["learned_adapter_quality"],
        "product": "mei-1.0-51m",
        "package_id": package_manifest["package_id"],
        "package_receipt": str(package_receipt_path),
        "package_receipt_sha256": sha_file(package_receipt_path),
        "stage_fingerprint_sha256": fingerprint,
        "metrics": metrics,
        "output_hashes": output_hashes,
        "current_sha256": sha_file(CURRENT_PATH),
        "current_unchanged": True,
    }
    write_json_once(receipt_path, receipt)
    return receipt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--package-receipt", type=Path, required=True)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    print(json.dumps(run(parse_args(argv)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
