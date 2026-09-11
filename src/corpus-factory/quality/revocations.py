from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def revoked_hashes(root: Path = ROOT) -> dict[str, str]:
    revoked = {}
    for path in sorted((root / "corpus/pools").glob("*/quality/REVOKED*.json")):
        receipt = json.loads(path.read_text())
        if receipt.get("status") != "do_not_adopt":
            continue
        for value in receipt.get("artifacts", {}).values():
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"invalid revocation hash: {path}")
            revoked[value] = str(path)
    return revoked


def manifest_revocation_errors(path: Path, manifest: dict, root: Path = ROOT) -> list[str]:
    revoked = revoked_hashes(root)
    checked = {str(path): hashlib.sha256(path.read_bytes()).hexdigest(), **manifest.get("artifacts", {})}
    return [f"revoked artifact {name}: {revoked[value]}" for name, value in checked.items() if value in revoked]
