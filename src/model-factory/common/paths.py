"""Locate the mei-llm root and expose the single supported 51M identity."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in (cur, *cur.parents):
        if (candidate / "CURRENT.json").is_file():
            return candidate
    raise RuntimeError("cannot locate mei-llm root (need CURRENT.json)")


ROOT = find_root()
CURRENT_PATH = ROOT / "CURRENT.json"
# 四分离：源码 src/、语料 corpus/、过程 cycles/、成果 models/（mei-1.2-51m 现役，
# mei-1.1-51m 为旧链归档名）。产物路径一律经本文件常量，禁止散落硬编码。
SRC_ROOT = ROOT / "src"
CYCLES_ROOT = ROOT / "cycles"
LEGACY_CYCLES_ROOT = CYCLES_ROOT / "mei-1.1-51m"
CORPUS_ROOT = ROOT / "corpus"
_tokenizer_dir_env = os.environ.get("MEI_TOKENIZER_DIR")
TOKENIZER_DIR = (
    Path(_tokenizer_dir_env).resolve()
    if _tokenizer_dir_env
    else ROOT / "models/mei-1.2-51m/tokenizer"
)
TOKENIZER_ZH_V1 = ROOT / "models/mei-1.1-51m/tokenizer/zh-24k-v1.model"


def frozen_tokenizer_path() -> Path:
    """当前冻结词表模型路径（TOKENIZER.json 指针；默认 zh-24k-v1）。"""
    pointer = TOKENIZER_DIR / "TOKENIZER.json"
    if pointer.is_file():
        data = json.loads(pointer.read_text(encoding="utf-8"))
        if data.get("status") in ("frozen", "frozen_in_training_use") and data.get("tokenizer_id"):
            return TOKENIZER_DIR / f"{data['tokenizer_id']}.model"
    return TOKENIZER_ZH_V1
TOKENIZER_MANIFEST = TOKENIZER_DIR / "tokenizer-zh-24k-v3-manifest.json"

_ARCHITECTURE_ID_RE = re.compile(r"^mei-1\.0-51m-arch-v\d+$")
ARCHITECTURE_ID = os.environ.get("MEI_ARCHITECTURE_ID", "mei-1.0-51m-arch-v1")
if "/" in ARCHITECTURE_ID or "\\" in ARCHITECTURE_ID or not _ARCHITECTURE_ID_RE.fullmatch(ARCHITECTURE_ID):
    raise RuntimeError(f"unsupported architecture identity: {ARCHITECTURE_ID}")
_architecture_dir_env = os.environ.get("MEI_ARCHITECTURE_DIR")
ARCHITECTURE_DIR = (
    Path(_architecture_dir_env).resolve()
    if _architecture_dir_env
    else SRC_ROOT / "architecture/mei-1.2-51m"
)
if not (ARCHITECTURE_DIR / "spec/model.json").is_file():
    raise RuntimeError(f"missing architecture spec: {ARCHITECTURE_DIR / 'spec/model.json'}")
ARCHITECTURE_SPEC = ARCHITECTURE_DIR / "spec"
ARCHITECTURE_CONTRACT = ARCHITECTURE_DIR / "architecture_contract.py"

MODEL_FACTORY_DIR = SRC_ROOT / "model-factory"
RECIPES_DIR = MODEL_FACTORY_DIR / "recipes"
# Compatibility name used by older pipeline code. It now points at the visible
# model-factory package root, not at a hidden flat scripts directory.
TRAINING_DIR = MODEL_FACTORY_DIR
RUNTIME_SHARED = SRC_ROOT / "platform/_shared/runtime"
TRAIN_RUNS = LEGACY_CYCLES_ROOT / "exp-00300m/runs"
CORPUS_LM_V1 = LEGACY_CYCLES_ROOT / "exp-00300m/corpus/cpt-delta/lm-v1"
CORPUS_LM_V2 = LEGACY_CYCLES_ROOT / "_legacy/corpus/planned-exp-001000m-lm-v2"
CORPUS_ZH_PRETRAIN = CORPUS_LM_V1 / "language/zh-pretrain-v0"
BASE_SCRATCH300M = LEGACY_CYCLES_ROOT / "exp-00300m/models/base/mei-1.0-51m-base-scratch300m-v1"
BASE_CPT1B = LEGACY_CYCLES_ROOT / "exp-001000m/models/base/mei-1.0-51m-base-cpt1b-v1"
CPT_1B_RUN = TRAIN_RUNS / "pretrain-1b-cpt-from-scratch300m"
EVAL_SHARED_ROOT = LEGACY_CYCLES_ROOT / "_legacy/notebook/evaluation/shared"


def cycle_lineage(cycle_id: str) -> str:
    """registry 里的血缘：zh-v2-rebuild 走 exp-XXXm 正式序列，legacy 走归档区。"""
    registry = ROOT / ".internal/registry/cycles.json"
    try:
        data = json.loads(registry.read_text(encoding="utf-8"))
        rows = data.get("cycles") if isinstance(data, dict) else data
        for row in rows or []:
            if str(row.get("cycle_id")) == cycle_id:
                return str(row.get("lineage") or "legacy")
    except (OSError, ValueError):
        pass
    return "zh-v2-rebuild" if re.search(r"-v[0-9]+$", cycle_id) else "legacy"


def cycle_artifacts(cycle_id: str) -> Path:
    if not re.fullmatch(r"exp-[0-9]{6}m(-v[0-9]+)?", cycle_id):
        raise ValueError(f"invalid cycle id: {cycle_id}")
    base_id = re.sub(r"-v[0-9]+$", "", cycle_id)  # exp-000300m → exp-00300m
    match = re.fullmatch(r"exp-([0-9]{6})m", base_id)
    millions = int(match.group(1)) if match else 0
    normalized = f"exp-{millions:05d}m"
    if cycle_lineage(cycle_id) == "zh-v2-rebuild":
        # 新链正式序列：cycles/mei-1.2-51m/exp-XXXm（大版本入产品名，cycle 直属根）
        return CYCLES_ROOT / "mei-1.2-51m" / normalized
    return LEGACY_CYCLES_ROOT / normalized


def cycle_runs(cycle_id: str) -> Path:
    return cycle_artifacts(cycle_id) / "runs"


def phase_binding_identity() -> dict[str, str] | None:
    path_value = os.environ.get("MEI_PHASE_BINDING")
    names = {
        "binding_sha256": os.environ.get("MEI_PHASE_BINDING_SHA256"),
        "binding_id": os.environ.get("MEI_PHASE_BINDING_ID"),
        "cycle_id": os.environ.get("MEI_PHASE_CYCLE_ID"),
        "phase": os.environ.get("MEI_PHASE"),
        "pipeline_id": os.environ.get("MEI_PHASE_PIPELINE_ID"),
    }
    if path_value is None:
        if any(value is not None for value in names.values()):
            raise RuntimeError("partial MEI phase binding environment")
        return None
    if any(not value for value in names.values()):
        raise RuntimeError("incomplete MEI phase binding environment")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise RuntimeError(f"phase binding is missing: {path}")
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha != names["binding_sha256"]:
        raise RuntimeError("phase binding changed after control-plane verification")
    binding = json.loads(path.read_text(encoding="utf-8"))
    for key in ("binding_id", "cycle_id", "phase", "pipeline_id"):
        if binding.get(key) != names[key]:
            raise RuntimeError(f"phase binding environment disagrees on {key}")
    return {key: str(value) for key, value in names.items()}


def relocate_corpus_path(value: str | Path) -> Path:
    """Apply the explicit corpus folder move map without editing old receipts."""
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        return path
    if not relative.startswith("corpus/pools/"):
        return path
    manifest = ROOT / ".internal/registry/migrations/2026-09-corpus-pools.json"
    if not manifest.is_file():
        return path
    moves = json.loads(manifest.read_text(encoding="utf-8"))["exact"]
    for old, new in moves.items():
        if relative == old or relative.startswith(old + "/"):
            target = ROOT / new / relative[len(old):].lstrip("/")
            if not target.resolve().is_relative_to((ROOT / "corpus/pools").resolve()):
                raise ValueError("corpus relocation escapes pools")
            return target
    return path


def resolve_repo_path(value: str | Path) -> Path:
    """Resolve a current path or an immutable pre-migration receipt path."""

    candidate = Path(value)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
        for historical_root in (ROOT, ROOT.parent, ROOT / "mei-llm"):
            try:
                candidate = candidate.relative_to(historical_root)
                break
            except ValueError:
                continue
        else:
            return candidate
    direct = relocate_corpus_path(ROOT / candidate)
    if direct.exists():
        return direct

    raw = candidate.as_posix()
    # 信封拍平前的旧内层根前缀：剥掉后继续按前缀表路由
    if raw.startswith("mei-llm/"):
        raw = raw[len("mei-llm/"):]
        candidate = Path(raw)
        # 剥掉旧内层根后先直接命中一次：这类路径的目标多半已在现布局原地，
        # 不必再经迁移表。缺这一步会让 61 个历史 receipt 的绝对路径永远解析失败。
        stripped = relocate_corpus_path(ROOT / candidate)
        if stripped.exists():
            return stripped
    # 旧扁平训练管线 → 现 factory 布局（LEGACY_PATH_MAP，唯一实现）
    legacy_map_path = ROOT / "src/model-factory/contracts/LEGACY_PATH_MAP.json"
    if legacy_map_path.is_file():
        legacy_map = json.loads(legacy_map_path.read_text(encoding="utf-8"))
        if raw in legacy_map.get("intermediate_hidden_exact", {}):
            return ROOT / str(legacy_map["intermediate_hidden_exact"][raw])
        if raw.rstrip("/") == str(legacy_map.get("old_root") or "").rstrip("/"):
            return ROOT / str(legacy_map["new_root"])
        for alias_root in legacy_map.get("alias_roots") or []:
            alias_prefix = str(alias_root).rstrip("/") + "/"
            if raw.startswith(alias_prefix):
                raw = str(legacy_map["old_root"]).rstrip("/") + "/" + raw[len(alias_prefix):]
                break
        if raw in legacy_map.get("exact", {}):
            return ROOT / str(legacy_map["exact"][raw])

    old_run_prefix = "training/runs/mei-1.0-51m/"
    if raw.startswith(old_run_prefix):
        suffix = raw[len(old_run_prefix) :]
        candidates = [
            *CYCLES_ROOT.glob(f"mei-*/exp-*/runs/{suffix}"),
            LEGACY_CYCLES_ROOT / "comparisons/runs" / suffix,
        ]
        matches = sorted(path for path in candidates if path.exists())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(f"ambiguous migrated run path: {raw}")

    migration_path = ROOT / ".internal/registry/migrations/2026-09-four-domain.json"
    if migration_path.is_file():
        migration = json.loads(migration_path.read_text(encoding="utf-8"))
        exact = migration.get("exact") or {}
        if raw in exact:
            return relocate_corpus_path(ROOT / str(exact[raw]))
        for old, new in sorted(exact.items(), key=lambda item: -len(item[0])):
            prefix = old.rstrip("/") + "/"
            if raw.startswith(prefix):
                return relocate_corpus_path(ROOT / str(new) / raw[len(prefix) :])
        prefixes = migration.get("prefix") or {}
        for old, new in sorted(prefixes.items(), key=lambda item: -len(item[0])):
            if raw.startswith(old):
                return relocate_corpus_path(ROOT / str(new) / raw[len(old) :])
    return direct


def architecture_sha256() -> str:
    """Hash source bytes for provenance only; never use as weight compatibility."""
    digest = hashlib.sha256()
    for rel in (
        "architecture.py",
        "config.py",
        "architecture_contract.py",
        "spec/model.json",
        "spec/legacy-architecture-aliases.json",
    ):
        path = ARCHITECTURE_DIR / rel
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def architecture_contracts() -> dict:
    """Return the three normalized architecture contracts without importing MLX."""
    text = str(ARCHITECTURE_DIR)
    added = text not in sys.path
    if added:
        sys.path.insert(0, text)
    try:
        from architecture_contract import contract_bundle

        return contract_bundle()
    finally:
        if added:
            sys.path.remove(text)


def weight_contract_sha256() -> str:
    return str(architecture_contracts()["weight_contract_sha256"])


def runtime_profile_sha256() -> str:
    return str(architecture_contracts()["runtime_profile_sha256"])


def training_aux_sha256() -> str:
    return str(architecture_contracts()["training_aux_sha256"])


def legacy_weight_contract_sha256(legacy_architecture_sha256: str) -> str | None:
    text = str(ARCHITECTURE_DIR)
    added = text not in sys.path
    if added:
        sys.path.insert(0, text)
    try:
        from architecture_contract import resolve_legacy_weight_contract

        return resolve_legacy_weight_contract(str(legacy_architecture_sha256 or ""))
    finally:
        if added:
            sys.path.remove(text)


def load_current() -> dict:
    if not CURRENT_PATH.is_file():
        return {}
    return json.loads(CURRENT_PATH.read_text(encoding="utf-8"))


def ensure_formal_on_path() -> None:
    for path in reversed((ROOT / "src", ARCHITECTURE_DIR, MODEL_FACTORY_DIR, RUNTIME_SHARED)):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
