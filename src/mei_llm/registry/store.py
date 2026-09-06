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
        artifacts = self.artifacts()["entries"]
        match = next((item for item in artifacts if item["uri"] == value), None)
        if match is not None:
            return (self.root / match["path"]).resolve()

        candidate = Path(value)
        if candidate.is_absolute():
            if candidate.exists():
                return candidate
            for historical_root in (self.root, self.root.parent, self.root / "mei-llm"):
                try:
                    value = candidate.relative_to(historical_root).as_posix()
                    break
                except ValueError:
                    continue
            else:
                return candidate

        source_migration = self.model_factory_legacy_paths()
        if value in source_migration.get("intermediate_hidden_exact", {}):
            return (
                self.root / source_migration["intermediate_hidden_exact"][value]
            ).resolve()
        if value.rstrip("/") == source_migration["old_root"].rstrip("/"):
            return (self.root / source_migration["new_root"]).resolve()
        for alias_root in source_migration.get("alias_roots", []):
            alias_prefix = alias_root.rstrip("/") + "/"
            if value.startswith(alias_prefix):
                value = (
                    source_migration["old_root"].rstrip("/")
                    + "/"
                    + value[len(alias_prefix) :]
                )
                break
        if value in source_migration["exact"]:
            return (self.root / source_migration["exact"][value]).resolve()
        migration = self.migration()
        old_run_prefix = "training/runs/mei-1.0-51m/"
        if value.startswith(old_run_prefix):
            suffix = value[len(old_run_prefix) :]
            candidates = [
                *self.root.glob(f"cycles/mei-*/exp-*/runs/{suffix}"),
                self.root / "cycles/mei-1.1-51m/comparisons/runs" / suffix,
            ]
            matches = sorted(path for path in candidates if path.exists())
            if len(matches) == 1:
                return matches[0].resolve()
            if len(matches) > 1:
                raise RuntimeError(f"ambiguous migrated run path: {value}")
        if value in migration["exact"]:
            return (self.root / migration["exact"][value]).resolve()
        for old, new in sorted(
            migration["exact"].items(), key=lambda item: -len(item[0])
        ):
            prefix = old.rstrip("/") + "/"
            if value.startswith(prefix):
                return (self.root / new / value[len(prefix) :]).resolve()
        for old, new in sorted(migration["prefix"].items(), key=lambda item: -len(item[0])):
            if value.startswith(old):
                return (self.root / new / value[len(old):]).resolve()
        return (self.root / value).resolve()

    def current_sha256(self) -> str:
        return hashlib.sha256((self.root / "CURRENT.json").read_bytes()).hexdigest()
