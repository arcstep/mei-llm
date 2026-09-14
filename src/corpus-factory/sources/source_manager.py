#!/usr/bin/env python3
"""Natural-corpus inventory, mix planning, authorized download, and admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from array import array
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[3]
HQ_REPO = "epfml/FineWeb2-HQ"
HQ_SUBSET = "cmn_Hani"
DOWNLOAD_AUTHORIZATION = f"{HQ_REPO}:{HQ_SUBSET}"
# Legacy roles for reading/validating v1 artifacts only; new logic reads
# the versioned registry (corpus-factory/sources/source_registry.json).
SOURCE_ROLES_LEGACY = ("wiki", "fineweb2_hq")
TEXT_KEYS = ("text", "content", "document", "body")


def _sibling_module(filename: str, module_name: str) -> Any:
    # Sibling modules (registry.py, collector.py) live next to this file and are
    # loaded by path — mirrors the tests' loader, avoids sys.path pollution.
    import functools
    import importlib.util

    @functools.lru_cache(maxsize=8)
    def load(path: Path):
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise SourceError(f"cannot load sourcing module: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load((Path(__file__).parent / filename).resolve())


def load_registry() -> dict[str, Any]:
    module = _sibling_module("registry.py", "mei_51m_source_registry")
    try:
        return module.load_registry()
    except module.RegistryError as error:
        raise SourceError(str(error)) from error


def entry_for(source_id: str) -> dict[str, Any]:
    module = _sibling_module("registry.py", "mei_51m_source_registry")
    try:
        return module.entry_for(source_id)
    except module.RegistryError as error:
        raise SourceError(str(error)) from error


def collector() -> Any:
    return _sibling_module("collector.py", "mei_51m_source_collector")


class SourceError(RuntimeError):
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


def write_once_json(path: Path, value: Any) -> None:
    encoded = canonical(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == encoded:
            return
        raise FileExistsError(f"refusing to overwrite different artifact: {path}")
    path.write_bytes(encoded)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _text_from_object(value: Any) -> str | None:
    if isinstance(value, str):
        return normalize_text(value) or None
    if not isinstance(value, dict):
        return None
    for key in TEXT_KEYS:
        text = value.get(key)
        if isinstance(text, str) and normalize_text(text):
            return normalize_text(text)
    return None


def iter_documents(path: Path) -> Iterator[str]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SourceError(f"{path}:{line_number}: invalid JSON: {error}") from error
                text = _text_from_object(value)
                if text:
                    yield text
        return
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        rows = value if isinstance(value, list) else [value]
        for row in rows:
            text = _text_from_object(row)
            if text:
                yield text
        return
    if path.suffix.lower() in {".txt", ".md"}:
        text = normalize_text(path.read_text(encoding="utf-8", errors="replace"))
        if text:
            yield text
        return
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as parquet
        except ImportError as error:
            raise SourceError("reading parquet requires pyarrow") from error
        source = parquet.ParquetFile(path)
        column = next((key for key in TEXT_KEYS if key in source.schema_arrow.names), None)
        if not column:
            raise SourceError(f"{path}: no supported text column")
        for batch in source.iter_batches(batch_size=256, columns=[column]):
            for value in batch.column(0).to_pylist():
                text = _text_from_object(value)
                if text:
                    yield text
        return
    raise SourceError(f"unsupported source format: {path}")


def inventory(paths: Iterable[Path], role: str) -> dict[str, Any]:
    if role not in SOURCE_ROLES_LEGACY:
        raise SourceError(f"unsupported source role: {role}")
    files = []
    for path in sorted({item.resolve() for item in paths}, key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise SourceError("source inventory cannot be empty")
    return {
        "schema": "mei-51m-natural-source-inventory-v1",
        "source_role": role,
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
    }


def parse_role_values(values: list[str], *, name: str) -> dict[str, int]:
    valid_roles = set(load_registry()["roles"]) | set(SOURCE_ROLES_LEGACY)
    result: dict[str, int] = {}
    for value in values:
        role, separator, raw = value.partition("=")
        if not separator or role not in valid_roles:
            raise SourceError(f"{name} must use role=tokens for known roles")
        amount = int(raw)
        if amount < 0:
            raise SourceError(f"{name} cannot be negative: {value}")
        result[role] = amount
    return result


def parse_fraction_values(values: list[str], *, name: str) -> dict[str, float]:
    valid_roles = set(load_registry()["roles"]) | set(SOURCE_ROLES_LEGACY)
    result: dict[str, float] = {}
    for value in values:
        role, separator, raw = value.partition("=")
        if not separator or role not in valid_roles:
            raise SourceError(f"{name} must use role=fraction for known roles")
        amount = float(raw)
        if not 0 < amount <= 1:
            raise SourceError(f"{name} fraction must be in (0, 1]: {value}")
        result[role] = amount
    return result


def plan_mix(
    target_tokens: int,
    capacities: dict[str, int],
    consumed: dict[str, int],
    *,
    hq_fraction: float | None = None,
    fractions: dict[str, float] | None = None,
    quotas: dict[str, int] | None = None,
    candidate_id: str | None = None,
    supersedes: str | None = None,
    reasons: list[str] | None = None,
) -> dict[str, Any]:
    if target_tokens <= 0:
        raise SourceError("target increment must be positive")
    if sum(1 for value in (hq_fraction, fractions, quotas) if value is not None) != 1:
        raise SourceError("plan_mix requires exactly one of --hq-fraction, --fraction, --quota")
    deprecated_input = None
    rounding = None
    if hq_fraction is not None:
        if not 0 <= hq_fraction <= 1:
            raise SourceError("hq fraction must be between 0 and 1")
        # Deprecated alias: the legacy pool keys map onto the wiki_zh role.
        capacities = {
            ("wiki_zh" if role == "wiki" else role): amount
            for role, amount in capacities.items()
        }
        consumed = {
            ("wiki_zh" if role == "wiki" else role): amount
            for role, amount in consumed.items()
        }
        hq_tokens = round(target_tokens * hq_fraction)
        quotas = {"fineweb2_hq": hq_tokens, "wiki_zh": target_tokens - hq_tokens}
        deprecated_input = "hq_fraction"
        rounding = "legacy_hq_fraction"
    elif fractions is not None:
        total = sum(fractions.values())
        if not 0.999 <= total <= 1.001:
            raise SourceError(f"fractions must sum to 1, got {total}")
        order = list(fractions)
        assigned = {role: round(target_tokens * value) for role, value in fractions.items()}
        # floor_last_role: the last listed role absorbs the rounding remainder.
        assigned[order[-1]] = target_tokens - sum(
            assigned[role] for role in order[:-1]
        )
        quotas = assigned
        rounding = "floor_last_role"
    else:
        quotas = dict(quotas)
        rounding = "explicit_quotas"
    valid_roles = set(load_registry()["roles"]) | set(SOURCE_ROLES_LEGACY)
    unknown = sorted(set(quotas) - valid_roles)
    if unknown:
        raise SourceError(f"unknown roles in plan: {unknown}")
    if any(amount < 0 for amount in quotas.values()) or sum(quotas.values()) != target_tokens:
        raise SourceError("nonnegative quotas must sum to target increment")
    if any(amount < 0 for amount in capacities.values()) or any(amount < 0 for amount in consumed.values()):
        raise SourceError("capacity and consumed tokens must be nonnegative")
    if any(consumed.get(role, 0) > capacities.get(role, 0) for role in quotas):
        raise SourceError("consumed tokens exceed physical capacity")
    remaining = {
        role: max(0, capacities.get(role, 0) - consumed.get(role, 0))
        for role in quotas
    }
    shortages = {
        role: quotas[role] - remaining[role]
        for role in quotas
        if quotas[role] > remaining[role]
    }
    return {
        "schema": "mei-51m-cpt-natural-mix-candidate-v2",
        "candidate_id": candidate_id,
        "status": "blocked_insufficient_pool" if shortages else "passed",
        "target_increment_tokens": target_tokens,
        "quotas": quotas,
        "capacity_tokens": capacities,
        "consumed_tokens": consumed,
        "remaining_tokens": remaining,
        "shortages": shortages,
        "allow_repeat": False,
        "selection": "unseen_first",
        "synthetic_fraction": None,
        "validation_scope": "quota_capacity_only",
        "training_adoption_eligible": False,
        "policy": {
            "rounding": rounding,
            "fractions": fractions,
            "needs_reason": (fractions is not None or quotas is not None)
            and not reasons,
        },
        "supersedes": supersedes,
        "reasons": reasons or [],
        "deprecated_input": deprecated_input,
    }


def tokenizer_pointer(path: Path | None = None) -> dict[str, Any]:
    pointer_path = path or (
        ROOT / "models/mei-1.2-51m/tokenizer/TOKENIZER.json"
    )
    if not pointer_path.is_file():
        raise SourceError(
            f"tokenizer pointer missing ({pointer_path}); tokenizer not frozen"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    if pointer.get("schema") != "mei-51m-tokenizer-pointer-v1":
        raise SourceError("tokenizer pointer schema mismatch")
    if pointer.get("status") != "frozen":
        raise SourceError(f"tokenizer not frozen: {pointer.get('status')}")
    return pointer


def load_tokenizer(manifest_path: Path | None = None) -> Any:
    if manifest_path is not None:
        manifest_path = manifest_path.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "mei-51m-tokenizer-release-v1":
            raise SourceError("explicit tokenizer manifest schema mismatch")
        if manifest.get("status") != "frozen":
            raise SourceError(f"explicit tokenizer is not frozen: {manifest.get('status')}")
        tokenizer_id = str(manifest.get("tokenizer_id") or "")
        model_rel = manifest.get("model_file")
        if not tokenizer_id or not model_rel:
            raise SourceError("explicit tokenizer manifest lacks tokenizer_id/model_file")
        model_path = (manifest_path.parent / model_rel).resolve()
        if not model_path.is_relative_to(manifest_path.parent.resolve()):
            raise SourceError("explicit tokenizer model escapes release directory")
        architecture = ROOT / "src/architecture/mei-1.2-51m"
        sys.path.insert(0, str(architecture))
        from tokenizer import ZhTokenizerV2

        tokenizer = ZhTokenizerV2(
            tokenizer_id=tokenizer_id,
            vocab_size=int(manifest.get("vocab_size") or 0),
            manifest_path=manifest_path,
            model_path=model_path,
        )
        if tokenizer.model_sha256 != manifest.get("model_sha256"):
            raise SourceError("tokenizer hash mismatch vs explicit manifest")
        tokenizer.tokenizer_id = tokenizer_id
        tokenizer.release_manifest = manifest_path
        return tokenizer
    pointer = tokenizer_pointer()
    architecture = ROOT / "src/architecture/mei-1.2-51m"
    sys.path.insert(0, str(architecture))
    from tokenizer import ZhTokenizerV1, ZhTokenizerV2

    tokenizer_id = pointer["tokenizer_id"]
    if tokenizer_id == "zh-24k-v1":
        tokenizer = ZhTokenizerV1()
    else:
        manifest_name = str(pointer.get("manifest") or "")
        manifest_path = (
            (ROOT / "models/mei-1.2-51m/tokenizer" / manifest_name)
            if manifest_name
            else None
        )
        tokenizer = ZhTokenizerV2(
            tokenizer_id=tokenizer_id,
            vocab_size=int(pointer.get("vocab_size") or 0),
            manifest_path=manifest_path,
        )
    if tokenizer.model_sha256 != pointer.get("model_sha256"):
        raise SourceError("tokenizer hash mismatch vs pointer")
    tokenizer.tokenizer_id = tokenizer_id
    return tokenizer


def load_seen_hashes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    if not path.is_file():
        return set()  # fresh ledger: first admit of a new pool
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            digest = str(
                row.get("normalized_sha256")
                or row.get("record_sha256")
                or row.get("sha256")
                or ""
            )
            if digest:
                result.add(digest)
    return result


def append_seen_ledger(path: Path | None, digests: Iterable[str]) -> None:
    """Accumulate admitted digests into the pool ledger (append-only)."""
    if path is None or not digests:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for digest in digests:
            handle.write(
                json.dumps(
                    {"normalized_sha256": digest}, ensure_ascii=False, sort_keys=True
                )
                + "\n"
            )


def admit(
    inputs: list[Path],
    out: Path,
    *,
    role: str,
    license_id: str,
    license_reviewed: bool,
    seen_ledger: Path | None,
    source_id: str | None = None,
    source_url: str | None = None,
    subset: str | None = None,
    dataset_version: str | None = None,
    acquired_at: str | None = None,
    clearance_receipt: Path | None = None,
    mode: str = "text",
    max_invalid_ratio: float | None = None,
    expected_tokenizer_id: str | None = None,
    tokenizer_manifest: Path | None = None,
) -> dict[str, Any]:
    if role not in SOURCE_ROLES_LEGACY and role not in load_registry()["roles"]:
        raise SourceError(f"unsupported source role: {role}")
    if expected_tokenizer_id is not None:
        frozen_id = (
            json.loads(tokenizer_manifest.read_text(encoding="utf-8")).get("tokenizer_id")
            if tokenizer_manifest is not None
            else tokenizer_pointer().get("tokenizer_id")
        )
        if expected_tokenizer_id != frozen_id:
            raise SourceError(
                f"expected tokenizer {expected_tokenizer_id!r}, "
                f"frozen pointer is {frozen_id!r}"
            )
    if not license_reviewed or not license_id.strip():
        raise SourceError("license id and explicit --license-reviewed are required")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite admitted source: {out}")
    provenance_v2 = source_id is not None
    registry_entry_sha256 = None
    clearance: dict[str, Any] | None = None
    if provenance_v2:
        entry = entry_for(source_id)
        if entry["role"] != role:
            raise SourceError(
                f"source_id {source_id!r} is registered as {entry['role']!r}, "
                f"not {role!r}"
            )
        registry_entry_sha256 = hashlib.sha256(
            canonical(entry)
        ).hexdigest()
        if clearance_receipt is not None:
            receipt_path = clearance_receipt.resolve()
            if not receipt_path.is_file():
                raise SourceError(f"missing clearance receipt: {receipt_path}")
            clearance = {
                "path": str(receipt_path),
                "sha256": sha256_file(receipt_path),
            }
    if mode not in ("text", "structured"):
        raise SourceError(f"unsupported admit mode: {mode}")
    structure_threshold = 0.001
    record_elements: tuple[str, ...] = ()
    if mode == "structured":
        if not provenance_v2:
            raise SourceError("structured admit requires --source-id (provenance v2)")
        admission = entry.get("admit") or {}
        structure_threshold = admission.get("max_invalid_ratio", 0.001)
        if max_invalid_ratio is not None and max_invalid_ratio > structure_threshold:
            raise SourceError(
                "--max-invalid-ratio may only tighten the registry threshold "
                f"({structure_threshold})"
            )
        if max_invalid_ratio is not None:
            structure_threshold = max_invalid_ratio
        record_elements = tuple(
            part for part in (admission.get("record_element") or "").split("|") if part
        )
    tokenizer = load_tokenizer(tokenizer_manifest)
    seen = load_seen_hashes(seen_ledger)
    accepted: list[tuple[str, str, list[int], dict[str, Any]]] = []
    duplicates = 0
    structure_stats: dict[str, Any] | None = None
    if mode == "structured":
        module = _sibling_module("structured.py", "mei_51m_source_structured")
        total_records = 0
        invalid_records = 0
        for path in inputs:
            if path.suffix.lower() == ".xml" and not record_elements:
                raise SourceError(
                    f"xml structured source needs admit.record_element: {path}"
                )
            try:
                records = module.iter_records(
                    path.resolve(),
                    record_elements=record_elements,
                    key_field=admission.get("record_key"),
                )
            except module.StructuredError as error:
                raise SourceError(str(error)) from error
            for record in records:
                total_records += 1
                if not record["valid"]:
                    invalid_records += 1
                    continue
                digest = hashlib.sha256(record["canonical_bytes"]).hexdigest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                token_ids = tokenizer.encode_document(record["text"])
                accepted.append(
                    (
                        str(path.resolve()),
                        digest,
                        token_ids,
                        {"record_key": record["key"]},
                    )
                )
        invalid_ratio = invalid_records / total_records if total_records else 1.0
        if invalid_ratio > structure_threshold:
            raise SourceError(
                f"structure check failed: {invalid_records}/{total_records} invalid "
                f"(ratio {invalid_ratio:.6f} > threshold {structure_threshold})"
            )
        structure_stats = {
            "records": total_records,
            "invalid": invalid_records,
            "invalid_ratio": invalid_ratio,
            "threshold": structure_threshold,
            "validated": "all",
        }
    else:
        for path in inputs:
            for text in iter_documents(path.resolve()):
                digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)
                accepted.append(
                    (str(path.resolve()), digest, tokenizer.encode_document(text), {})
                )
    if not accepted:
        raise SourceError("no unseen documents were admitted")

    if provenance_v2 and not acquired_at:
        acquired_at = datetime.now(timezone.utc).isoformat()
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent.resolve()))
    try:
        token_path = temporary / "tokens.bin"
        index_path = temporary / "documents.jsonl"
        token_offset = 0
        with token_path.open("wb") as token_handle, index_path.open(
            "w", encoding="utf-8"
        ) as index_handle:
            for source_path, digest, token_ids, row_extra in accepted:
                values = array("H", token_ids)
                if sys.byteorder != "little":
                    values.byteswap()
                values.tofile(token_handle)
                row: dict[str, Any] = {
                    "source_path": source_path,
                    "source_role": role,
                    "token_offset": token_offset,
                    "tokens": len(token_ids),
                    "license_id": license_id,
                    "license_reviewed": True,
                }
                if mode == "structured":
                    row["record_sha256"] = digest
                else:
                    row["normalized_sha256"] = digest
                if provenance_v2:
                    row.update(
                        {
                            "source_id": source_id,
                            "source_url": source_url,
                            "subset": subset,
                            "dataset_version": dataset_version,
                            "acquired_at": acquired_at,
                            "record_key": None,
                        }
                    )
                    row.update(row_extra)
                index_handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                token_offset += len(token_ids)
        if provenance_v2:
            manifest = {
                "schema": "mei-51m-admitted-natural-source-v2",
                "provenance_version": 2,
                "source_role": role,
                "source_id": source_id,
                "source_url": source_url,
                "subset": subset,
                "dataset_version": dataset_version,
                "acquired_at": acquired_at,
                "dedup_mode": "record" if mode == "structured" else "text",
                "registry_entry_sha256": registry_entry_sha256,
                "tokenizer_id": getattr(tokenizer, "tokenizer_id", None),
                "tokenizer_model_sha256": tokenizer.model_sha256,
                "license_id": license_id,
                "license_reviewed": True,
                "clearance_receipt": clearance,
                "structure_check": structure_stats,
                "documents": len(accepted),
                "tokens": token_offset,
                "duplicates_skipped": duplicates,
                "artifacts": {
                    "documents.jsonl": sha256_file(index_path),
                    "tokens.bin": sha256_file(token_path),
                },
            }
        else:
            manifest = {
                "schema": "mei-51m-admitted-natural-source-v1",
                "source_role": role,
                "license_id": license_id,
                "license_reviewed": True,
                "tokenizer_model_sha256": tokenizer.model_sha256,
                "documents": len(accepted),
                "tokens": token_offset,
                "duplicates_skipped": duplicates,
                "artifacts": {
                    "documents.jsonl": sha256_file(index_path),
                    "tokens.bin": sha256_file(token_path),
                },
            }
        write_once_json(temporary / "manifest.json", manifest)
        temporary.replace(out)
        # Append AFTER the artifacts are durable: a crash here can only cause
        # duplicate documents on a retry, never silently skipped documents.
        append_seen_ledger(seen_ledger, (digest for _, digest, _, _ in accepted))
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def freeze_pool(
    inputs: list[Path],
    out: Path,
    release_id: str,
    *,
    supersedes: str | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite pool release: {out}")
    entries = []
    for directory in inputs:
        manifest_path = directory.resolve() / "manifest.json"
        if not manifest_path.is_file():
            raise SourceError(f"missing admitted manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        revocations = _sibling_module("../quality/revocations.py", "mei_pool_revocations")
        revoked = revocations.manifest_revocation_errors(manifest_path, manifest)
        if revoked:
            raise SourceError("; ".join(revoked))
        entry = {
            "path": str(directory.resolve()),
            "manifest_sha256": sha256_file(manifest_path),
            "source_role": manifest["source_role"],
            "documents": manifest["documents"],
            "tokens": manifest["tokens"],
        }
        if manifest.get("schema") == "mei-51m-admitted-natural-source-v2":
            entry.update(
                {
                    "source_id": manifest["source_id"],
                    "registry_entry_sha256": manifest["registry_entry_sha256"],
                    "tokenizer_id": manifest.get("tokenizer_id"),
                    "tokenizer_model_sha256": manifest["tokenizer_model_sha256"],
                }
            )
        entries.append(entry)
    if not entries:
        raise SourceError("pool release needs at least one admitted source")
    tokenizer_ids = {row.get("tokenizer_id") for row in entries}
    if len(tokenizer_ids) != 1:
        raise SourceError(
            f"pool release must bind exactly one tokenizer generation: {sorted(map(str, tokenizer_ids))}"
        )
    roles = sorted({row["source_role"] for row in entries})
    value = {
        "schema": "mei-51m-natural-pool-release-v2",
        "release_id": release_id,
        "status": "frozen",
        "allow_repeat": False,
        "selection": "unseen_first",
        "sources": entries,
        "tokens_by_role": {
            role: sum(row["tokens"] for row in entries if row["source_role"] == role)
            for role in roles
        },
        "tokenizer_id": tokenizer_ids.pop(),
        "supersedes": supersedes,
        "reason": reason,
    }
    out.mkdir(parents=True)
    write_once_json(out / "RELEASE.json", value)
    return value


def download_hq(filenames: list[str], out: Path, authorization: str) -> dict[str, Any]:
    if authorization != DOWNLOAD_AUTHORIZATION:
        raise SourceError(
            "download refused; pass --authorize-download "
            f"{DOWNLOAD_AUTHORIZATION!r} for this exact batch"
        )
    if not filenames:
        raise SourceError("at least one explicit --filename is required")
    for filename in filenames:
        parts = Path(filename).parts
        if HQ_SUBSET not in parts or ".." in parts or Path(filename).is_absolute():
            raise SourceError(f"filename must belong to {HQ_SUBSET}: {filename}")
    entry = entry_for("fineweb2-hq-cmn-hani")
    try:
        batch = collector().resolve_batch(entry, filenames=filenames)
        return collector().download_batch(
            batch, out, entry=entry, authorization=batch["batch_token"]
        )
    except collector().CollectorError as error:
        raise SourceError(str(error)) from error


def download(
    source_id: str,
    out: Path | None,
    *,
    subset: str | None,
    filenames: list[str] | None,
    authorization: str | None,
    dry_run: bool,
    cache_dir: Path | None,
) -> dict[str, Any]:
    entry = entry_for(source_id)
    try:
        batch = collector().resolve_batch(
            entry,
            subset=subset,
            filenames=filenames or None,
            cache_dir=cache_dir,
        )
    except collector().CollectorError as error:
        raise SourceError(str(error)) from error
    if dry_run:
        total_bytes = sum(
            row["bytes"] for row in batch["files"] if isinstance(row.get("bytes"), int)
        )
        return {
            "schema": "mei-51m-download-plan-v1",
            "source_id": source_id,
            "status": "dry_run",
            "batch_token": batch["batch_token"],
            "file_count": len(batch["files"]),
            "total_bytes": total_bytes if total_bytes else None,
            "files": [
                {key: value for key, value in row.items() if key in ("filename", "url", "bytes")}
                for row in batch["files"]
            ],
        }
    if not authorization:
        raise SourceError("--authorize-download is required unless --dry-run")
    if out is None:
        raise SourceError("--out is required unless --dry-run")
    try:
        return collector().download_batch(
            batch, out.resolve(), entry=entry, authorization=authorization,
            cache_dir=cache_dir,
        )
    except collector().CollectorError as error:
        raise SourceError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("inventory")
    command.add_argument("--role", choices=SOURCE_ROLES_LEGACY, required=True)
    command.add_argument("--input", action="append", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("plan-mix")
    command.add_argument("--target-tokens", type=int, required=True)
    command.add_argument("--capacity", action="append", default=[])
    command.add_argument("--consumed", action="append", default=[])
    command.add_argument("--hq-fraction", type=float)
    command.add_argument("--fraction", action="append", default=[])
    command.add_argument("--quota", action="append", default=[])
    command.add_argument("--candidate-id")
    command.add_argument("--supersedes")
    command.add_argument("--reason", action="append", default=[])
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("download-hq")
    command.add_argument("--filename", action="append", required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--authorize-download", required=True)

    command = sub.add_parser("download")
    command.add_argument("--source-id", required=True)
    command.add_argument("--subset", help="glob pattern narrowing hf_api files")
    command.add_argument("--filename", action="append", default=[])
    command.add_argument("--cache-dir", type=Path)
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--authorize-download")
    command.add_argument("--out", type=Path)

    command = sub.add_parser("admit")
    command.add_argument("--role", required=True)
    command.add_argument("--input", action="append", type=Path, required=True)
    command.add_argument("--license-id", required=True)
    command.add_argument("--license-reviewed", action="store_true")
    command.add_argument("--seen-ledger", type=Path)
    command.add_argument("--source-id", help="registry source_id; enables provenance v2")
    command.add_argument("--source-url")
    command.add_argument("--subset")
    command.add_argument("--dataset-version")
    command.add_argument("--acquired-at")
    command.add_argument("--clearance-receipt", type=Path)
    command.add_argument("--mode", choices=("text", "structured"), default="text")
    command.add_argument("--max-invalid-ratio", type=float)
    command.add_argument("--expected-tokenizer-id")
    command.add_argument("--tokenizer-manifest", type=Path)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("freeze-pool")
    command.add_argument("--input", action="append", type=Path, required=True)
    command.add_argument("--release-id", required=True)
    command.add_argument("--supersedes")
    command.add_argument("--reason")
    command.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        result = inventory(args.input, args.role)
        write_once_json(args.out, result)
    elif args.command == "plan-mix":
        result = plan_mix(
            args.target_tokens,
            parse_role_values(args.capacity, name="--capacity"),
            parse_role_values(args.consumed, name="--consumed"),
            hq_fraction=args.hq_fraction,
            fractions=parse_fraction_values(args.fraction, name="--fraction") or None,
            quotas=parse_role_values(args.quota, name="--quota") or None,
            candidate_id=args.candidate_id,
            supersedes=args.supersedes,
            reasons=args.reason or None,
        )
        write_once_json(args.out, result)
    elif args.command == "download-hq":
        result = download_hq(
            args.filename, args.out.resolve(), args.authorize_download
        )
        write_once_json(args.out.resolve() / "DOWNLOAD.json", result)
    elif args.command == "download":
        result = download(
            args.source_id,
            args.out,
            subset=args.subset,
            filenames=args.filename,
            authorization=args.authorize_download,
            dry_run=args.dry_run,
            cache_dir=args.cache_dir,
        )
    elif args.command == "admit":
        result = admit(
            args.input,
            args.out.resolve(),
            role=args.role,
            license_id=args.license_id,
            license_reviewed=args.license_reviewed,
            seen_ledger=args.seen_ledger,
            source_id=args.source_id,
            source_url=args.source_url,
            subset=args.subset,
            dataset_version=args.dataset_version,
            acquired_at=args.acquired_at,
            clearance_receipt=args.clearance_receipt,
            mode=args.mode,
            max_invalid_ratio=args.max_invalid_ratio,
            expected_tokenizer_id=args.expected_tokenizer_id,
            tokenizer_manifest=args.tokenizer_manifest,
        )
    else:
        result = freeze_pool(
            args.input,
            args.out.resolve(),
            args.release_id,
            supersedes=args.supersedes,
            reason=args.reason,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") != "blocked_insufficient_pool" else 2


if __name__ == "__main__":
    raise SystemExit(main())
