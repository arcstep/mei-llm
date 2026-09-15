from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "CURRENT.json").is_file():
            return candidate
    raise RuntimeError("cannot locate mei-llm root")


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


@dataclass(frozen=True)
class Registry:
    root: Path

    @classmethod
    def open(cls, root: Path | None = None) -> "Registry":
        return cls((root or repository_root()).resolve())

    def cycles(self) -> dict:
        return _load(self.root / ".internal" / "registry" / "cycles.json")

    def models(self) -> dict:
        return _load(self.root / ".internal" / "registry" / "models.json")

    def artifacts(self) -> dict:
        return _load(self.root / ".internal" / "registry" / "artifacts.json")

    def migration(self) -> dict:
        return _load(
            self.root
            / ".internal"
            / "registry"
            / "migrations"
            / "2026-09-four-domain.json"
        )

    def model_factory_legacy_paths(self) -> dict:
        return _load(
            self.root
            / "src/model-factory"
            / "contracts"
            / "LEGACY_PATH_MAP.json"
        )

    def cycle(self, cycle_id: str) -> dict:
        match = next((item for item in self.cycles()["cycles"] if item["cycle_id"] == cycle_id), None)
        if match is None:
            raise KeyError(f"unknown cycle: {cycle_id}")
        if "path" not in match:
            return match
        return _load(self.root / match["path"])

    def resolve(self, value: str) -> Path:
        """URI 查 artifacts 表，其余路径解析统一委托
        model-factory/common/_repo.resolve_repo_path（唯一实现）。"""
        artifacts = self.artifacts()["entries"]
        match = next((item for item in artifacts if item["uri"] == value), None)
        if match is not None:
            value = match["path"]

        import sys

        factory = str(self.root / "src/model-factory")
        added = factory not in sys.path
        if added:
            sys.path.insert(0, factory)
        try:
            from common.paths import resolve_repo_path

            return resolve_repo_path(value).resolve()
        finally:
            if added:
                sys.path.remove(factory)

    def current_sha256(self) -> str:
        return hashlib.sha256((self.root / "CURRENT.json").read_bytes()).hexdigest()
