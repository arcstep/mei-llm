from __future__ import annotations

import json
from pathlib import Path

SDK_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = SDK_ROOT / "spec"


def load_json(name: str) -> dict:
    return json.loads((SPEC_DIR / name).read_text(encoding="utf-8"))


def sdk_versions() -> dict:
    raw = load_json("versions.json")
    return {
        "sdk_semver": raw["sdk_semver"],
        "wire_version": raw["wire_version"],
        "model_package_version": raw["model_package_version"],
        "runtime_abi_version": raw["runtime_abi_version"],
        "protocol_id": raw["protocol_id"],
        "serializer_id": raw["serializer_id"],
        "release_class": raw["release_class"],
        "product": raw["product"],
    }
