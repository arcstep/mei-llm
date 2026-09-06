#!/usr/bin/env python3
"""Source registry for natural-corpus sourcing v2.

Governance-by-registry: what may be downloaded, under which band/license it may
be admitted, and which admit mode applies, all come from the versioned
``source_registry.json`` data file — never from hardcoded tables in code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).with_name("source_registry.json")
SCHEMA_ID = "mei-51m-source-registry-v1"
BANDS = ("A", "B", "C")
MANIFEST_SOURCE_TYPES = ("static_manifest", "hf_api", "direct_urls")
DEDUP_MODES = ("text", "record")


class RegistryError(RuntimeError):
    pass


def validate_registry(value: Any) -> list[str]:
    reasons: list[str] = []
    if not isinstance(value, dict):
        return ["registry must be a JSON object"]
    if value.get("schema") != SCHEMA_ID:
        reasons.append(f"schema must be {SCHEMA_ID}")
    roles = value.get("roles")
    if not isinstance(roles, dict) or not roles:
        reasons.append("roles must be a non-empty object")
    else:
        for name, spec in roles.items():
            if not isinstance(spec, dict):
                reasons.append(f"role {name!r} spec must be an object")
                continue
            if spec.get("band_policy") not in BANDS:
                reasons.append(f"role {name!r}: band_policy must be one of {BANDS}")
            if spec.get("dedup_mode") not in DEDUP_MODES:
                reasons.append(
                    f"role {name!r}: dedup_mode must be one of {DEDUP_MODES}"
                )
    sources = value.get("sources")
    if not isinstance(sources, list) or not sources:
        reasons.append("sources must be a non-empty list")
    else:
        seen_ids: set[str] = set()
        for index, source in enumerate(sources):
            label = f"sources[{index}]"
            if not isinstance(source, dict):
                reasons.append(f"{label} must be an object")
                continue
            source_id = source.get("source_id")
            if not isinstance(source_id, str) or not source_id.strip():
                reasons.append(f"{label}: source_id must be a non-empty string")
            elif source_id in seen_ids:
                reasons.append(f"duplicate source_id: {source_id}")
            else:
                seen_ids.add(source_id)
            if isinstance(roles, dict) and source.get("role") not in roles:
                reasons.append(f"{label}: role must be a registered role")
            if source.get("band") not in BANDS:
                reasons.append(f"{label}: band must be one of {BANDS}")
            manifest = source.get("manifest_source")
            if not isinstance(manifest, dict) or manifest.get("type") not in (
                MANIFEST_SOURCE_TYPES
            ):
                reasons.append(
                    f"{label}: manifest_source.type must be one of "
                    f"{MANIFEST_SOURCE_TYPES}"
                )
            else:
                kind = manifest["type"]
                if kind == "static_manifest" and not manifest.get("manifest"):
                    reasons.append(
                        f"{label}: static_manifest requires a 'manifest' path"
                    )
                if kind == "direct_urls" and not isinstance(
                    source.get("urls"), list
                ):
                    reasons.append(
                        f"{label}: direct_urls requires an inline 'urls' list"
                    )
            license_ = source.get("license")
            if not isinstance(license_, dict) or not license_.get("license_id"):
                reasons.append(f"{label}: license.license_id must be non-empty")
            admit = source.get("admit")
            if isinstance(admit, dict) and admit.get("dedup_mode") not in (
                None,
                *DEDUP_MODES,
            ):
                reasons.append(
                    f"{label}: admit.dedup_mode must be one of {DEDUP_MODES}"
                )
    return reasons


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RegistryError(f"registry not found: {path}") from error
    except json.JSONDecodeError as error:
        raise RegistryError(f"registry is not valid JSON: {path}: {error}") from error
    reasons = validate_registry(value)
    if reasons:
        raise RegistryError(f"registry invalid ({path}): {'; '.join(reasons)}")
    return value


def roles(registry: dict[str, Any] | None = None) -> list[str]:
    table = registry if registry is not None else load_registry()
    return list(table["roles"])


def entry_for(
    source_id: str, registry: dict[str, Any] | None = None
) -> dict[str, Any]:
    table = registry if registry is not None else load_registry()
    for source in table["sources"]:
        if source.get("source_id") == source_id:
            return source
    raise RegistryError(f"unknown source_id: {source_id}")
