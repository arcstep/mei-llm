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
from pathlib import Path
from typing import Any, Iterable, Iterator


ROOT = Path(__file__).resolve().parents[2]
HQ_REPO = "epfml/FineWeb2-HQ"
HQ_SUBSET = "cmn_Hani"
DOWNLOAD_AUTHORIZATION = f"{HQ_REPO}:{HQ_SUBSET}"
SOURCE_ROLES = ("wiki", "fineweb2_hq")
TEXT_KEYS = ("text", "content", "document", "body")


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
    if role not in SOURCE_ROLES:
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
    result: dict[str, int] = {}
    for value in values:
        role, separator, raw = value.partition("=")
        if not separator or role not in SOURCE_ROLES:
            raise SourceError(f"{name} must use role=tokens for {SOURCE_ROLES}")
        amount = int(raw)
        if amount < 0:
            raise SourceError(f"{name} cannot be negative: {value}")
        result[role] = amount
    return result


def plan_mix(
    target_tokens: int,
    capacities: dict[str, int],
    consumed: dict[str, int],
    *,
    hq_fraction: float,
) -> dict[str, Any]:
    if target_tokens <= 0:
        raise SourceError("target increment must be positive")
    if not 0 <= hq_fraction <= 1:
        raise SourceError("hq fraction must be between 0 and 1")
    quotas = {
        "fineweb2_hq": round(target_tokens * hq_fraction),
        "wiki": target_tokens - round(target_tokens * hq_fraction),
    }
    remaining = {
        role: max(0, capacities.get(role, 0) - consumed.get(role, 0))
        for role in SOURCE_ROLES
    }
    shortages = {
        role: quotas[role] - remaining[role]
        for role in SOURCE_ROLES
        if quotas[role] > remaining[role]
    }
    return {
        "schema": "mei-51m-cpt-natural-mix-candidate-v1",
        "status": "blocked_insufficient_pool" if shortages else "passed",
        "target_increment_tokens": target_tokens,
        "quotas": quotas,
        "capacity_tokens": capacities,
        "consumed_tokens": consumed,
        "remaining_tokens": remaining,
        "shortages": shortages,
        "allow_repeat": False,
        "selection": "unseen_first",
        "synthetic_fraction": 0.0,
    }


def load_tokenizer() -> Any:
    architecture = ROOT / "models/mei-1.0-51m/architecture"
    sys.path.insert(0, str(architecture))
    from tokenizer import ZhTokenizerV1

    return ZhTokenizerV1()


def load_seen_hashes(path: Path | None) -> set[str]:
    if path is None:
        return set()
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            digest = str(row.get("normalized_sha256") or row.get("sha256") or "")
            if digest:
                result.add(digest)
    return result


def admit(
    inputs: list[Path],
    out: Path,
    *,
    role: str,
    license_id: str,
    license_reviewed: bool,
    seen_ledger: Path | None,
) -> dict[str, Any]:
    if role not in SOURCE_ROLES:
        raise SourceError(f"unsupported source role: {role}")
    if not license_reviewed or not license_id.strip():
        raise SourceError("license id and explicit --license-reviewed are required")
    if out.exists():
        raise FileExistsError(f"refusing to overwrite admitted source: {out}")
    tokenizer = load_tokenizer()
    seen = load_seen_hashes(seen_ledger)
    accepted: list[tuple[str, str, list[int]]] = []
    duplicates = 0
    for path in inputs:
        for text in iter_documents(path.resolve()):
            digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            accepted.append((str(path.resolve()), digest, tokenizer.encode_document(text)))
    if not accepted:
        raise SourceError("no unseen documents were admitted")

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent.resolve()))
    try:
        token_path = temporary / "tokens.bin"
        index_path = temporary / "documents.jsonl"
        token_offset = 0
        with token_path.open("wb") as token_handle, index_path.open(
            "w", encoding="utf-8"
        ) as index_handle:
            for source_path, digest, token_ids in accepted:
                values = array("H", token_ids)
                if sys.byteorder != "little":
                    values.byteswap()
                values.tofile(token_handle)
                index_handle.write(
                    json.dumps(
                        {
                            "source_path": source_path,
                            "source_role": role,
                            "normalized_sha256": digest,
                            "token_offset": token_offset,
                            "tokens": len(token_ids),
                            "license_id": license_id,
                            "license_reviewed": True,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                token_offset += len(token_ids)
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
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def freeze_pool(inputs: list[Path], out: Path, release_id: str) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite pool release: {out}")
    entries = []
    for directory in inputs:
        manifest_path = directory.resolve() / "manifest.json"
        if not manifest_path.is_file():
            raise SourceError(f"missing admitted manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "path": str(directory.resolve()),
                "manifest_sha256": sha256_file(manifest_path),
                "source_role": manifest["source_role"],
                "documents": manifest["documents"],
                "tokens": manifest["tokens"],
            }
        )
    if not entries:
        raise SourceError("pool release needs at least one admitted source")
    value = {
        "schema": "mei-51m-natural-pool-release-v1",
        "release_id": release_id,
        "status": "frozen",
        "allow_repeat": False,
        "selection": "unseen_first",
        "sources": entries,
        "tokens_by_role": {
            role: sum(row["tokens"] for row in entries if row["source_role"] == role)
            for role in SOURCE_ROLES
        },
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
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise SourceError("authorized download requires huggingface_hub") from error
    downloaded = []
    for filename in filenames:
        path = Path(
            hf_hub_download(
                repo_id=HQ_REPO,
                repo_type="dataset",
                filename=filename,
                local_dir=out,
            )
        )
        downloaded.append(
            {"filename": filename, "path": str(path), "sha256": sha256_file(path)}
        )
    return {
        "schema": "mei-51m-authorized-download-receipt-v1",
        "repository": HQ_REPO,
        "subset": HQ_SUBSET,
        "files": downloaded,
        "distribution_clearance_asserted": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("inventory")
    command.add_argument("--role", choices=SOURCE_ROLES, required=True)
    command.add_argument("--input", action="append", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("plan-mix")
    command.add_argument("--target-tokens", type=int, required=True)
    command.add_argument("--capacity", action="append", default=[])
    command.add_argument("--consumed", action="append", default=[])
    command.add_argument("--hq-fraction", type=float, default=0.65)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("download-hq")
    command.add_argument("--filename", action="append", required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--authorize-download", required=True)

    command = sub.add_parser("admit")
    command.add_argument("--role", choices=SOURCE_ROLES, required=True)
    command.add_argument("--input", action="append", type=Path, required=True)
    command.add_argument("--license-id", required=True)
    command.add_argument("--license-reviewed", action="store_true")
    command.add_argument("--seen-ledger", type=Path)
    command.add_argument("--out", type=Path, required=True)

    command = sub.add_parser("freeze-pool")
    command.add_argument("--input", action="append", type=Path, required=True)
    command.add_argument("--release-id", required=True)
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
        )
        write_once_json(args.out, result)
    elif args.command == "download-hq":
        result = download_hq(
            args.filename, args.out.resolve(), args.authorize_download
        )
        write_once_json(args.out.resolve() / "DOWNLOAD.json", result)
    elif args.command == "admit":
        result = admit(
            args.input,
            args.out.resolve(),
            role=args.role,
            license_id=args.license_id,
            license_reviewed=args.license_reviewed,
            seen_ledger=args.seen_ledger,
        )
    else:
        result = freeze_pool(args.input, args.out.resolve(), args.release_id)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") != "blocked_insufficient_pool" else 2


if __name__ == "__main__":
    raise SystemExit(main())
