from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import SdkError
from .version import load_json

REQUIRED_HEADS = ("lm", "contrastive", "mw_disposition", "confidence")


@dataclass
class HeadStatus:
    present: bool
    trained: bool
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {"present": self.present, "trained": self.trained, "status": self.status}


@dataclass
class HeadReport:
    lm: HeadStatus
    contrastive: HeadStatus
    mw_disposition: HeadStatus
    confidence: HeadStatus

    def as_dict(self) -> dict[str, Any]:
        return {
            "lm": self.lm.as_dict(),
            "contrastive": self.contrastive.as_dict(),
            "mw_disposition": self.mw_disposition.as_dict(),
            "confidence": self.confidence.as_dict(),
        }

    def missing(self) -> list[str]:
        out: list[str] = []
        for name in REQUIRED_HEADS:
            head: HeadStatus = getattr(self, name)
            if head.status in {"missing", "untrained"} or not head.present or not head.trained:
                out.append(name)
        return out


@dataclass
class ModelPackage:
    path: Path
    manifest: dict[str, Any]
    heads: HeadReport
    verified_hashes: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def package_id(self) -> str:
        return str(self.manifest["package_id"])

    def packed_inference_ready(self) -> bool:
        weights = self.manifest.get("weights") or {}
        fmt = str(weights.get("format") or "")
        scheme = str((weights.get("quantization") or {}).get("scheme") or "")
        return fmt == "mei-q4-packed-v1" and scheme in {"q4", "cq2"}

    def capabilities(self) -> dict[str, Any]:
        missing = self.heads.missing()
        return {
            "package_id": self.package_id,
            "release_class": self.manifest.get("release_class"),
            "inference": self.packed_inference_ready(),
            "protocol": True,
            "heads": self.heads.as_dict(),
            "missing_or_untrained_heads": missing,
            "hash_verified": self.verified_hashes,
        }


def _head(raw: dict[str, Any], name: str) -> HeadStatus:
    if name not in raw:
        raise SdkError("package_invalid", f"heads.{name} must be listed explicitly")
    item = raw[name]
    return HeadStatus(
        present=bool(item.get("present")),
        trained=bool(item.get("trained")),
        status=str(item.get("status") or "missing"),
    )


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_package(package_dir: str | Path, *, verify_hashes: bool = True) -> ModelPackage:
    root = Path(package_dir).resolve()
    manifest_path = root / "mei-model.json"
    if not manifest_path.is_file():
        raise SdkError("file_not_found", f"missing mei-model.json: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SdkError("invalid_json", str(exc)) from exc
    schema = load_json("model-package.schema.json")
    if manifest.get("package_format") != schema.get("properties", {}).get("package_format", {}).get("const"):
        raise SdkError("package_invalid", "package_format must be mei-model-package-v1")
    product = str(manifest.get("product") or "")
    if product not in {"mei-1.0-58m", "mei-1.0-51m"}:
        raise SdkError("package_invalid", "product must be mei-1.0-58m or mei-1.0-51m")
    heads_raw = manifest.get("heads")
    if not isinstance(heads_raw, dict):
        raise SdkError("package_invalid", "heads object is required")
    heads = HeadReport(
        lm=_head(heads_raw, "lm"),
        contrastive=_head(heads_raw, "contrastive"),
        mw_disposition=_head(heads_raw, "mw_disposition"),
        confidence=_head(heads_raw, "confidence"),
    )
    warnings: list[str] = []
    for name in heads.missing():
        warnings.append(f"head {name} is missing or untrained")
    verified = False
    if verify_hashes:
        for key in ("tokenizer", "weights"):
            spec = manifest.get(key) or {}
            rel = spec.get("file")
            expected = str(spec.get("sha256") or "")
            path = (root / rel).resolve() if rel else None
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
            vpath = (root / vocab_rel).resolve()
            if not vpath.is_file():
                raise SdkError("file_not_found", f"missing tokenizer vocab file: {vocab_rel}")
            if vocab_sha and _sha256_file(vpath) != vocab_sha.lower():
                raise SdkError(
                    "package_hash_mismatch",
                    f"tokenizer vocab sha256 mismatch: expected {vocab_sha}",
                )
    return ModelPackage(
        path=root,
        manifest=manifest,
        heads=heads,
        verified_hashes=verified,
        warnings=warnings,
    )
