#!/usr/bin/env python3
"""Fail-closed corpus quality receipts for natural, synthetic, and SFT data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


class QualityError(RuntimeError):
    pass


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_once(path: Path, value: Any) -> None:
    encoded = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == encoded:
            return
        raise FileExistsError(f"refusing to overwrite different quality receipt: {path}")
    path.write_bytes(encoded)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise QualityError(f"JSON root must be object: {path}")
    return value


def iter_text(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise QualityError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise QualityError(f"{path}:{line_number}: row must be object")
            text = value.get("text")
            if not isinstance(text, str):
                for key in ("query", "prompt", "content"):
                    if isinstance(value.get(key), str):
                        text = value[key]
                        break
            if not isinstance(text, str) or not text.strip():
                raise QualityError(f"{path}:{line_number}: no auditable text")
            yield re.sub(r"\s+", " ", text).strip()


def template_signature(text: str) -> str:
    value = text.lower()
    value = re.sub(r"https?://\S+", "<url>", value)
    value = re.sub(r"\b[0-9a-f]{16,}\b", "<hash>", value)
    value = re.sub(r"\b\d+(?:\.\d+)?\b", "<num>", value)
    value = re.sub(r"[_-]?\b[a-z]*\d+[a-z0-9_-]*\b", "<id>", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def load_markers(path: Path | None) -> list[str]:
    if path is None:
        return []
    markers = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            marker = line.strip()
            if marker:
                markers.append(marker)
    return markers


def verify_human_review(path: Path) -> dict[str, Any]:
    review = load_json(path)
    required = {
        "status": "passed",
        "semantic_consistency": "passed",
    }
    errors = [
        f"{key}={review.get(key)!r}"
        for key, expected in required.items()
        if review.get(key) != expected
    ]
    if not str(review.get("reviewer") or "").strip():
        errors.append("reviewer missing")
    if int(review.get("sample_size") or 0) <= 0:
        errors.append("sample_size must be positive")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "status": "passed" if not errors else "blocked",
        "errors": errors,
    }


def audit_synthetic(
    inputs: list[Path],
    human_review: Path,
    *,
    eval_markers: Path | None,
    min_template_ratio: float,
) -> dict[str, Any]:
    if not 0 < min_template_ratio <= 1:
        raise QualityError("min template ratio must be in (0, 1]")
    texts = [text for path in inputs for text in iter_text(path)]
    if not texts:
        raise QualityError("synthetic audit input is empty")
    exact = Counter(texts)
    templates = Counter(template_signature(text) for text in texts)
    markers = load_markers(eval_markers)
    leakage = [
        {"row": index, "markers": [marker for marker in markers if marker in text]}
        for index, text in enumerate(texts)
        if any(marker in text for marker in markers)
    ]
    review = verify_human_review(human_review)
    template_ratio = len(templates) / len(texts)
    reasons = []
    if template_ratio < min_template_ratio:
        reasons.append("template_diversity_degraded")
    if leakage:
        reasons.append("eval_leakage")
    if review["status"] != "passed":
        reasons.append("human_semantic_review_blocked")
    return {
        "schema": "mei-51m-corpus-quality-receipt-v1",
        "kind": "synthetic",
        "status": "passed" if not reasons else "blocked",
        "corpus_reuse_eligible": not reasons,
        "inputs": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in inputs
        ],
        "rows": len(texts),
        "exact_unique_ratio": len(exact) / len(texts),
        "template_unique_ratio": template_ratio,
        "minimum_template_unique_ratio": min_template_ratio,
        "largest_template_count": max(templates.values()),
        "eval_leakage_rows": len(leakage),
        "eval_leakage_examples": leakage[:10],
        "human_review": review,
        "reasons": reasons,
        "note": "hash/dedup success proves integrity, not naturalness or semantics",
    }


def _sibling_module(path: Path, module_name: str) -> Any:
    import functools
    import importlib.util

    @functools.lru_cache(maxsize=8)
    def load(target: Path):
        spec = importlib.util.spec_from_file_location(module_name, target)
        if spec is None or spec.loader is None:
            raise QualityError(f"cannot load module: {target}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load(path.resolve())


def _policy() -> dict[str, Any]:
    path = Path(__file__).parent / "policy/source-policy-v1.json"
    if not path.is_file():
        raise QualityError(f"quality policy not found: {path}")
    return load_json(path)


def _registry_entry(source_id: str) -> dict[str, Any]:
    module = _sibling_module(
        Path(__file__).parent.parent / "sources/registry.py",
        "mei_51m_source_registry",
    )
    try:
        return module.entry_for(source_id)
    except module.RegistryError as error:
        raise QualityError(str(error)) from error


def _policy_errors(manifest: dict[str, Any]) -> list[str]:
    """Band/license/clearance/structure policy checks for v2 manifests."""
    errors: list[str] = []
    entry = _registry_entry(manifest["source_id"])
    policy = _policy()
    role_policy = policy.get("roles", {}).get(entry["role"])
    if not role_policy:
        return [f"no quality policy for role {entry['role']!r}"]
    if entry.get("band") not in role_policy.get("allowed_bands", []):
        errors.append(
            f"band {entry.get('band')!r} not allowed for role {entry['role']!r}"
        )
    if role_policy.get("clearance_required") and manifest.get(
        "clearance_receipt"
    ) is None:
        errors.append("clearance receipt required by policy")
    check = role_policy.get("structure_check")
    if check:
        stats = manifest.get("structure_check") or {}
        if stats.get("invalid_ratio", 1.0) > check["max_invalid_ratio"]:
            errors.append("structure check exceeds policy threshold")
    return errors


def audit_source(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    errors = []
    schema = manifest.get("schema")
    if schema == "mei-51m-admitted-natural-source-v1":
        pass
    elif schema == "mei-51m-admitted-natural-source-v2":
        if manifest.get("provenance_version") != 2:
            errors.append("provenance_version must be 2")
        if not manifest.get("source_id"):
            errors.append("source_id missing")
        if manifest.get("dedup_mode") not in ("text", "record"):
            errors.append("dedup_mode must be text or record")
        if manifest.get("registry_entry_sha256") is None:
            errors.append("registry_entry_sha256 missing")
        if not manifest.get("tokenizer_id"):
            errors.append("tokenizer_id missing")
        clearance = manifest.get("clearance_receipt")
        if clearance is not None:
            receipt_path = Path(str(clearance.get("path") or ""))
            if not receipt_path.is_file():
                errors.append("clearance receipt file missing")
            elif sha256_file(receipt_path) != clearance.get("sha256"):
                errors.append("clearance receipt hash mismatch")
        if manifest.get("source_id"):
            try:
                errors.extend(_policy_errors(manifest))
            except QualityError as error:
                errors.append(f"policy lookup failed: {error}")
    else:
        errors.append("schema mismatch")
    if manifest.get("license_reviewed") is not True:
        errors.append("license not reviewed")
    if not manifest.get("license_id"):
        errors.append("license id missing")
    for name, expected in (manifest.get("artifacts") or {}).items():
        path = manifest_path.parent / name
        if not path.is_file():
            errors.append(f"missing artifact: {name}")
        elif sha256_file(path) != expected:
            errors.append(f"artifact hash mismatch: {name}")
    return {
        "schema": "mei-51m-corpus-quality-receipt-v1",
        "kind": "natural_source",
        "status": "passed" if not errors else "blocked",
        "corpus_reuse_eligible": not errors,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "errors": errors,
    }


def audit_structured(
    manifest_path: Path, *, sample: int = 100
) -> dict[str, Any]:
    """Second gate for record-mode admissions: token layout + re-hash samples."""
    base = audit_source(manifest_path)
    errors = list(base["errors"])
    manifest = load_json(manifest_path)
    rows = []
    documents_path = manifest_path.parent / "documents.jsonl"
    if documents_path.is_file():
        for line_number, line in enumerate(
            documents_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                errors.append(f"documents.jsonl:{line_number}: {error}")
    if manifest.get("dedup_mode") == "record":
        offsets = [int(row.get("token_offset") or 0) for row in rows]
        tokens = [int(row.get("tokens") or 0) for row in rows]
        running = 0
        layout_broken = False
        for offset, count in zip(offsets, tokens):
            if offset != running:
                layout_broken = True
                break
            running += count
        if layout_broken:
            errors.append("token_offset layout broken")
        tokens_path = manifest_path.parent / "tokens.bin"
        if tokens_path.is_file() and tokens_path.stat().st_size != 2 * running:
            errors.append("tokens.bin size mismatch")
        sampled = [row for row in rows[:sample] if row.get("record_key")]
        if sampled:
            digest_by_key: dict[str, str] = {}
            module = _sibling_module(
                Path(__file__).parent.parent / "sources/structured.py",
                "mei_51m_source_structured",
            )
            wanted = {str(row["record_key"]) for row in sampled}
            record_elements = ()
            key_field = None
            try:
                entry = _registry_entry(manifest["source_id"])
                admission = entry.get("admit") or {}
                record_elements = tuple(
                    part
                    for part in (admission.get("record_element") or "").split("|")
                    if part
                )
                key_field = admission.get("record_key")
            except QualityError:
                pass
            for row in sampled:
                source_path = Path(str(row["source_path"]))
                if not source_path.is_file():
                    errors.append(f"source file missing: {source_path}")
                    continue
                try:
                    for record in module.iter_records(
                        source_path,
                        record_elements=record_elements,
                        key_field=key_field,
                    ):
                        if not record["valid"] or record["key"] not in wanted:
                            continue
                        digest_by_key[str(record["key"])] = hashlib.sha256(
                            record["canonical_bytes"]
                        ).hexdigest()
                except module.StructuredError as error:
                    errors.append(f"re-hash failed: {source_path}: {error}")
            for row in sampled:
                key = str(row["record_key"])
                if key not in digest_by_key:
                    errors.append(f"record not re-found in source: {key}")
                elif digest_by_key[key] != row.get("record_sha256"):
                    errors.append(f"record_sha256 mismatch: {key}")
    return {
        "schema": "mei-51m-corpus-quality-receipt-v1",
        "kind": "structured_source",
        "status": "passed" if not errors else "blocked",
        "corpus_reuse_eligible": not errors,
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "errors": errors,
    }


def audit_sft(release: Path) -> dict[str, Any]:
    manifest_path = release / "release-manifest.json"
    if not manifest_path.is_file():
        raise QualityError(f"missing release manifest: {manifest_path}")
    manifest = load_json(manifest_path)
    gates_path = release / "governance/gates.json"
    audits_path = release / "governance/audits.json"
    errors = []
    for path in (gates_path, audits_path):
        if not path.is_file():
            errors.append(f"missing governance artifact: {path.name}")
    if manifest.get("process_complete") is not True:
        errors.append("release process incomplete")
    if manifest.get("training_started") is not False:
        errors.append("corpus release mutated training state")
    if manifest.get("current_mutated") is not False:
        errors.append("CURRENT.json mutation asserted")
    return {
        "schema": "mei-51m-corpus-quality-receipt-v1",
        "kind": "sft_release",
        "status": "passed" if not errors else "blocked",
        "corpus_reuse_eligible": not errors,
        "release": str(release.resolve()),
        "release_manifest_sha256": sha256_file(manifest_path),
        "governance": {
            "gates_sha256": sha256_file(gates_path) if gates_path.is_file() else None,
            "audits_sha256": sha256_file(audits_path) if audits_path.is_file() else None,
        },
        "errors": errors,
        "model_quality": "pending_training",
    }


def decide_reuse(receipts: list[Path], source_id: str) -> dict[str, Any]:
    rows = [
        {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "receipt": load_json(path),
        }
        for path in receipts
    ]
    blockers = [
        str(row["path"])
        for row in rows
        if row["receipt"].get("status") != "passed"
        or row["receipt"].get("corpus_reuse_eligible") is not True
    ]
    return {
        "schema": "mei-51m-corpus-reuse-decision-v1",
        "source_id": source_id,
        "action": "reuse" if not blockers else "retire",
        "corpus_reuse_eligible": not blockers,
        "receipts": [
            {"path": row["path"], "sha256": row["sha256"]} for row in rows
        ],
        "blockers": blockers,
    }


def compare(left: Path, right: Path) -> dict[str, Any]:
    a, b = load_json(left), load_json(right)
    keys = sorted(set(a) | set(b))
    return {
        "schema": "mei-51m-corpus-quality-comparison-v1",
        "left": {"path": str(left.resolve()), "sha256": sha256_file(left)},
        "right": {"path": str(right.resolve()), "sha256": sha256_file(right)},
        "changed": [key for key in keys if a.get(key) != b.get(key)],
        "left_status": a.get("status"),
        "right_status": b.get("status"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("audit-source")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("audit-structured")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--sample", type=int, default=100)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("audit-synthetic")
    command.add_argument("--input", action="append", type=Path, required=True)
    command.add_argument("--human-review", type=Path, required=True)
    command.add_argument("--eval-markers", type=Path)
    command.add_argument("--min-template-ratio", type=float, default=0.1)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("audit-sft")
    command.add_argument("--release", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("decide-reuse")
    command.add_argument("--receipt", action="append", type=Path, required=True)
    command.add_argument("--source-id", required=True)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("compare")
    command.add_argument("--left", type=Path, required=True)
    command.add_argument("--right", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit-source":
        result = audit_source(args.manifest)
    elif args.command == "audit-structured":
        result = audit_structured(args.manifest, sample=args.sample)
    elif args.command == "audit-synthetic":
        result = audit_synthetic(
            args.input,
            args.human_review,
            eval_markers=args.eval_markers,
            min_template_ratio=args.min_template_ratio,
        )
    elif args.command == "audit-sft":
        result = audit_sft(args.release)
    elif args.command == "decide-reuse":
        result = decide_reuse(args.receipt, args.source_id)
    else:
        result = compare(args.left, args.right)
    write_once(args.out, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status", "passed") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
