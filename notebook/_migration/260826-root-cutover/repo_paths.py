#!/usr/bin/env python3
"""Canonical paths after the 2026-08-26 five-root cutover.

Published assets live at repo root: tokenizer/ corpus/ base/ sft/.
Process, eval, scripts, and archive live under notebook/.

Do not recreate corpora/ or tasks/ at repo root.
"""

from __future__ import annotations

import json
from pathlib import Path


def find_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    if cur.is_file():
        cur = cur.parent
    for p in [cur, *cur.parents]:
        if (p / "CURRENT.json").is_file() and (p / "notebook").is_dir():
            return p
        if (p / "DESIGN.md").is_file() and (p / "tokenizer").is_dir() and (p / "notebook").is_dir():
            return p
    raise RuntimeError("cannot locate mei-llm root (need CURRENT.json or tokenizer/ + notebook/)")


ROOT = find_root()
NOTEBOOK = ROOT / "notebook"
TOOLING = NOTEBOOK / "_tooling"
SCRIPTS_ROOT = TOOLING / "scripts"
MODEL_MEI_51M = TOOLING / "model" / "mei-1.0-51m"
SKILLS_ROOT = TOOLING / "skills"
REQUIREMENTS_ROOT = TOOLING / "requirements"

TOKENIZER_DIR = ROOT / "tokenizer" / "zh-24k-v1"
TOKENIZER_ZH_V1 = TOKENIZER_DIR / "zh-24k-v1.model"
TOKENIZER_MANIFEST = TOKENIZER_DIR / "tokenizer-v1-manifest.json"

PRETRAIN_V1 = NOTEBOOK / "base" / "pretrain-v1"
SPEC_NEEDLE_ZH = PRETRAIN_V1 / "spec"
SPEC_MEI_51M = SPEC_NEEDLE_ZH
SFT_MEI_51M = NOTEBOOK / "sft" / "mei-1.0-51m"
SFT_TRAIN = SFT_MEI_51M / "train"
SFT_RECIPES = SFT_MEI_51M / "recipes"
CKPT_ARCHIVE = NOTEBOOK / "archive" / "base" / "mei-1.0-51m-checkpoints"
ARCHIVE_SFT = NOTEBOOK / "archive" / "sft" / "mei-1.0-51m"
ARCHIVE_LEGACY = NOTEBOOK / "archive" / "legacy-products"
ARCHIVE_CORPUS = NOTEBOOK / "archive" / "corpus"
EVAL_FIXTURES = NOTEBOOK / "evaluation" / "jobs" / "mei-1.0-51m"

LM_V1 = NOTEBOOK / "corpus" / "lm-v1"
LANGUAGE_ACCEPTED = LM_V1 / "language" / "outbox" / "accepted"
STRUCTURE_ACCEPTED = LM_V1 / "structure" / "outbox" / "accepted"
ASSEMBLE_WORK = LM_V1 / "assemble" / "work"
COLLOQUIAL = LM_V1 / "colloquial"
COLLOQUIAL_JOB = COLLOQUIAL / "jobs" / "260826-01-synthesize"
COLLOQUIAL_LANES = COLLOQUIAL_JOB / "work" / "lanes"
COLLOQUIAL_DRAFT = COLLOQUIAL / "outbox" / "draft" / "colloquial-v1"
VOCAB_WORK = NOTEBOOK / "corpus" / "tokenizer-v1" / "work" / "zh-vocab-v0"

CORPUS_ZH_VOCAB = VOCAB_WORK
CORPUS_ZH_PRETRAIN = LANGUAGE_ACCEPTED / "zh-pretrain-v0"
CORPUS_ZH_PRETRAIN_V1 = LANGUAGE_ACCEPTED / "zh-pretrain-v1"
CORPUS_ZH_PRETRAIN_V2 = ARCHIVE_CORPUS / "zh-pretrain-v2"
CORPUS_ZH_PRETRAIN_V3 = STRUCTURE_ACCEPTED / "zh-pretrain-v3"
CORPUS_ZH_PRETRAIN_V4 = ASSEMBLE_WORK / "zh-pretrain-v4"
CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_V1 = ARCHIVE_CORPUS / "zh-pretrain-colloquial-synth-v1"
CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1 = COLLOQUIAL_LANES / "qwen-plus"
CORPUS_SHARED_TRACES = NOTEBOOK / "sft" / "shared-tool-traces-v0"

CORPUS_ID_PATHS: dict[str, Path] = {
    "zh-vocab-v0": VOCAB_WORK,
    "zh-pretrain-v0": CORPUS_ZH_PRETRAIN,
    "zh-pretrain-v1": CORPUS_ZH_PRETRAIN_V1,
    "zh-pretrain-v2": CORPUS_ZH_PRETRAIN_V2,
    "zh-pretrain-v3": CORPUS_ZH_PRETRAIN_V3,
    "zh-pretrain-v4": CORPUS_ZH_PRETRAIN_V4,
    "zh-pretrain-colloquial-synth-v1": CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_V1,
    "zh-pretrain-colloquial-synth-v2": ARCHIVE_CORPUS / "zh-pretrain-colloquial-synth-v2",
    "zh-pretrain-colloquial-synth-smoke": ARCHIVE_CORPUS / "zh-pretrain-colloquial-synth-smoke",
    "zh-pretrain-colloquial-synth-qwen-smoke": ARCHIVE_CORPUS / "zh-pretrain-colloquial-synth-qwen-smoke",
    "zh-pretrain-colloquial-synth-ollama-bakeoff": ARCHIVE_CORPUS / "zh-pretrain-colloquial-synth-ollama-bakeoff",
    "zh-pretrain-colloquial-synth-qwen-v1": COLLOQUIAL_LANES / "qwen-plus",
    "zh-pretrain-colloquial-synth-dsflash-v1": COLLOQUIAL_LANES / "dsflash",
    "zh-pretrain-colloquial-synth-qwen36plus-v1": COLLOQUIAL_LANES / "qwen36",
    "zh-pretrain-colloquial-synth-qwen37plus-v1": COLLOQUIAL_LANES / "qwen37",
    "zh-pretrain-colloquial-synth-glm52-v1": COLLOQUIAL_LANES / "glm52",
    "zh-pretrain-colloquial-synth-kimi-k3-v1": COLLOQUIAL_LANES / "kimi",
    "zh-pretrain-colloquial-synth-pooled-v1": COLLOQUIAL_DRAFT,
    "_probe": ARCHIVE_CORPUS / "_probe",
    "sft-style-v0": NOTEBOOK / "sft" / "style-v0",
    "shared-tool-traces-v0": CORPUS_SHARED_TRACES,
}


class _MappedRoot:
    """Resolve historical corpora/<id> names onto split notebook locations."""

    def __init__(self, mapping: dict[str, Path], fallback: Path):
        self._mapping = mapping
        self._fallback = fallback

    def __truediv__(self, other: object) -> Path:
        key = str(other)
        if key in self._mapping:
            return self._mapping[key]
        return self._fallback / key

    def glob(self, pattern: str) -> list[Path]:
        import fnmatch

        out: list[Path] = []
        seen: set[Path] = set()
        for key, path in self._mapping.items():
            if fnmatch.fnmatch(key, pattern) and path not in seen:
                seen.add(path)
                out.append(path)
        return sorted(out)


CORPORA_ROOT = _MappedRoot(CORPUS_ID_PATHS, ARCHIVE_CORPUS)

ISOLATION_SCOPES = ("legacy", "cpt-v2", "sft-v2", "all")
_EVAL_SKIP_NAME_PARTS = (".review.", ".candidates.", ".canary.", "universe-pool")
_LEGACY_MARKERS = (
    "INVALID_FOR_PUBLISH.json",
    "INVALID_FOR_V2_PUBLISH.json",
    "LEGACY_DIAGNOSTIC_ONLY.json",
)
EVAL_ROOT = NOTEBOOK / "evaluation"
EVAL_BANKS_ROOT = EVAL_ROOT / "banks"
EVAL_SHARED_ROOT = EVAL_ROOT / "shared"
EXPERIMENTS_RUNS = NOTEBOOK / "archive" / "runs"
PRODUCT_INDEX = PRETRAIN_V1 / "product.json"
TASK_INDEX = PRODUCT_INDEX

TASK_MEI_EXPERT = "mei-expert-qwen35-0p8b"
TASK_NEEDLE_ZH = "needle-zh"
TASK_MEI_1_0_51M = "mei-1.0-51m"
TASK_ALIASES = {
    "mei-1.0-51m": "needle-zh",
    "needle-zh": "needle-zh",
}


class _SplitTask:
    """Resolve historical tasks/needle-zh/<sub> onto split locations."""

    def __truediv__(self, other: object) -> Path:
        key = str(other)
        mapping = {
            "model": MODEL_MEI_51M,
            "spec": SPEC_NEEDLE_ZH,
            "checkpoints": CKPT_ARCHIVE,
            "train": SFT_TRAIN,
            "recipes": SFT_RECIPES,
            "eval": EVAL_FIXTURES,
            "README.md": PRETRAIN_V1 / "README.md",
            "DESIGN.md": PRETRAIN_V1 / "DESIGN.md",
        }
        if key in mapping:
            return mapping[key]
        return PRETRAIN_V1 / key


class _TasksRoot:
    def __truediv__(self, other: object) -> Path | _SplitTask:
        name = str(other)
        if name in {"needle-zh", "mei-1.0-51m"}:
            return _SplitTask()
        if name == TASK_MEI_EXPERT:
            return ARCHIVE_LEGACY / TASK_MEI_EXPERT
        return ARCHIVE_LEGACY / name

    def glob(self, pattern: str) -> list[Path]:
        out: list[Path] = []
        seen: set[Path] = set()
        bases = [SFT_MEI_51M, ARCHIVE_SFT, ARCHIVE_LEGACY]
        inner = pattern[2:] if pattern.startswith("*/") else pattern
        for base in bases:
            if not base.exists():
                continue
            if pattern.startswith("*/"):
                for child in [base, *sorted(p for p in base.glob("*") if p.is_dir())]:
                    for p in child.glob(inner):
                        if p not in seen:
                            seen.add(p)
                            out.append(p)
            else:
                for p in base.glob(pattern):
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
        return sorted(out)


TASKS_ROOT = _TasksRoot()

JOBS_ROOT = COLLOQUIAL
JOBS_INDEX = NOTEBOOK / "_migration" / "260826-jobs-overlay" / "index.json"
CORPORA_INDEX = ROOT / "CURRENT.json"


def resolve_task_id(task_id: str) -> str:
    return TASK_ALIASES.get(task_id, task_id)


BANK_MEI_EXPERT = EVAL_BANKS_ROOT / "mei-expert-v0/eval-bank-v0.pending.jsonl"
HELDOUT_MEI_EXPERT = EVAL_BANKS_ROOT / "mei-expert-v0/heldout-task-ids.v0.json"
SEED_MEI_EXPERT = ARCHIVE_LEGACY / TASK_MEI_EXPERT / "train/seed/sft-smoke-v0.jsonl"
MLX_DIR_MEI_EXPERT = ARCHIVE_LEGACY / TASK_MEI_EXPERT / "mlx"
MLX_EXPORT_MEI_EXPERT = MLX_DIR_MEI_EXPERT / "exports/sft-smoke-v0"
MLX_ADAPTER_MEI_EXPERT = MLX_DIR_MEI_EXPERT / "adapters/qwen35-0.8b-sft-smoke"

BANK_NEEDLE_TOOLCALL = EVAL_BANKS_ROOT / "needle-toolcall-v0/eval-bank-v0.jsonl"
BANK_NEEDLE_VRM_AGENT = EVAL_BANKS_ROOT / "needle-vrm-agent-v0/eval-bank-v0.jsonl"
BANK_NEEDLE_VRM_AGENT_V2 = EVAL_BANKS_ROOT / "needle-vrm-agent-v0/eval-bank-v2.jsonl"
BANK_NEEDLE_VRM_AGENT_V2_RECIPE = EVAL_BANKS_ROOT / "needle-vrm-agent-v0/holdout-v2.recipe.json"
BANK_NEEDLE_VRM_AGENT_V2_LOCK = EVAL_BANKS_ROOT / "needle-vrm-agent-v0/holdout-v2.lock.json"
BANK_NEEDLE_VRM_AGENT_EN = EVAL_BANKS_ROOT / "needle-vrm-agent-en-v0/eval-bank-v0.jsonl"
BANK_NEEDLE_VRM_MW = EVAL_BANKS_ROOT / "needle-vrm-mw-v0/eval-bank-v0.jsonl"
BANK_NEEDLE_VRM_MW_RECIPE = EVAL_BANKS_ROOT / "needle-vrm-mw-v0/holdout-mw-v0.recipe.json"
BANK_NEEDLE_VRM_MW_LOCK = EVAL_BANKS_ROOT / "needle-vrm-mw-v0/holdout-mw-v0.lock.json"
PACK_NEEDLE_MW_SFT_2K = SFT_TRAIN / "packs/mw-sft-v0-2k.jsonl"
RECIPE_NEEDLE_MW_SFT = SFT_RECIPES / "mw-sft-v0.recipe.json"
SCHEMA_MW_GOVERNANCE = EVAL_SHARED_ROOT / "mw-governance-v0.json"
SEED_NEEDLE_ZH = ARCHIVE_SFT / "seed/sft-phase1-v0.jsonl"
BANK_NEEDLE_PRETRAIN_PROBES = EVAL_BANKS_ROOT / "needle-pretrain-probes-v0/probes-v0.jsonl"
BANK_MEI_TOOL_SCHEMA = EVAL_BANKS_ROOT / "mei-tool-schema-v1/eval-bank-v0.jsonl"
BANK_MEI_TOOL_SCHEMA_LOCK = EVAL_BANKS_ROOT / "mei-tool-schema-v1/holdout-schema-v1.lock.json"
PACK_MEI_TOOL_SFT_2K = ARCHIVE_SFT / "packs/mei-tool-sft-v1-2k.jsonl"
PACK_MEI_TOOL_SFT_10K = ARCHIVE_SFT / "packs/mei-tool-sft-v1-10k.jsonl"
REGISTRY_MEI_51M_CPT300M = CKPT_ARCHIVE / "registry/mei-1.0-51m-base-cpt300m-v1.json"


def load_task_index() -> dict:
    if not PRODUCT_INDEX.is_file():
        return {"version": 1, "tasks": []}
    return json.loads(PRODUCT_INDEX.read_text(encoding="utf-8"))


def task_record(task_id: str) -> dict:
    want = resolve_task_id(task_id)
    for row in load_task_index().get("tasks") or []:
        aliases = set(row.get("aliases") or [])
        if row.get("id") in {task_id, want}:
            return row
        if row.get("canonical_id") in {task_id, want}:
            return row
        if task_id in aliases or want in aliases:
            return row
        if row.get("alias_of") and resolve_task_id(str(row.get("id"))) == want:
            return task_record(str(row["alias_of"]))
    raise KeyError(f"unknown task_id={task_id}")


def repo_file(rel: str | Path) -> Path:
    return ROOT / rel


def resolve_rel(rel: str | Path, *, root: Path | None = None) -> Path:
    """Resolve a repo-relative path and refuse to escape the repo root."""
    base = (root or ROOT).resolve()
    path = Path(rel)
    resolved = path.resolve() if path.is_absolute() else (base / path).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"path escapes repo root: {rel}")
    return resolved


def job_topic_dir(topic: str, *, root: Path | None = None) -> Path:
    if root is not None:
        return root / "jobs" / topic
    if topic in {"colloquial-cpt", "colloquial"}:
        return COLLOQUIAL
    return NOTEBOOK / "jobs" / topic


def job_work(topic: str, job_id: str, *, root: Path | None = None) -> Path:
    return job_topic_dir(topic, root=root) / "jobs" / job_id / "work"


def publish_corpus(corpus_id: str, *, root: Path | None = None) -> Path:
    if root is not None:
        return root / "corpora" / corpus_id
    return CORPUS_ID_PATHS.get(corpus_id, ARCHIVE_CORPUS / corpus_id)


def load_jobs_index(*, root: Path | None = None) -> dict:
    if root is not None:
        path = root / "jobs" / "index.json"
    else:
        path = JOBS_INDEX
    if not path.is_file():
        return {"version": 1, "topics": {}, "jobs": [], "sft_packs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpora_index(*, root: Path | None = None) -> dict:
    path = (root or ROOT) / "CURRENT.json"
    if not path.is_file():
        return {"version": 1, "tokenizer": None, "corpus": None, "base": None, "sft": None}
    return json.loads(path.read_text(encoding="utf-8"))


def _marker_hit(path: Path, names: tuple[str, ...]) -> bool:
    cur = path if path.is_dir() else path.parent
    for _ in range(6):
        for name in names:
            if (cur / name).is_file():
                return True
        if cur == ROOT or cur.parent == cur:
            break
        cur = cur.parent
    stem_marker = path.with_name(path.name.replace(".jsonl", "") + ".LEGACY_DIAGNOSTIC_ONLY.json")
    if stem_marker.is_file():
        return True
    pack_marker = path.with_name("mei-tool-route-sft-v1.INVALID_FOR_V2_PUBLISH.json")
    if path.parent.name == "packs" and pack_marker.is_file() and path.name.startswith("mei-tool-route-sft-v1"):
        return True
    return False


def all_eval_jsonl(scope: str = "all") -> list[Path]:
    if not EVAL_BANKS_ROOT.is_dir():
        return []
    skip_noise = (".review.", ".candidates.", ".canary.")

    def iter_banks(*, include_invalid_for_publish: bool) -> list[Path]:
        out = []
        for p in sorted(EVAL_BANKS_ROOT.rglob("*.jsonl")):
            if not p.is_file() or any(s in p.name for s in skip_noise):
                continue
            if not include_invalid_for_publish and (p.parent / "INVALID_FOR_PUBLISH.json").is_file():
                continue
            out.append(p)
        return out

    if scope in {"legacy", "all"}:
        return iter_banks(include_invalid_for_publish=False)
    rows = []
    for p in iter_banks(include_invalid_for_publish=False):
        if p.name.startswith("universe-pool"):
            continue
        if scope == "sft-v2" and not (
            "mei-toolcall-v2" in str(p) or "mei-51m-toolcall" in str(p)
        ):
            continue
        rows.append(p)
    return rows


def _iter_sft_jsonl() -> list[Path]:
    paths: list[Path] = []
    for base in (SFT_TRAIN, ARCHIVE_SFT, ARCHIVE_LEGACY):
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.jsonl")):
            if "/packs/" in p.as_posix() or "/seed/" in p.as_posix():
                paths.append(p)
    return paths


def all_train_seeds(scope: str = "all") -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    def add(p: Path) -> None:
        if p not in seen:
            paths.append(p)
            seen.add(p)

    if scope == "sft-v2":
        for p in _iter_sft_jsonl():
            if "packs" not in p.as_posix() or "v2" not in p.name:
                continue
            if ".review-sample." in p.name or ".candidates." in p.name:
                continue
            add(p)
        return paths
    if scope == "cpt-v2":
        return []
    for row in load_task_index().get("tasks") or []:
        rel = row.get("train_seed")
        if rel:
            add(ROOT / rel)
        for extra in row.get("train_seeds") or []:
            add(ROOT / extra)
    for p in _iter_sft_jsonl():
        if ".review-sample." in p.name or ".candidates." in p.name:
            continue
        if p.name.startswith("mei-tool-sft-v1-"):
            continue
        if scope in {"cpt-v2", "sft-v2"} and (
            p.name.startswith("mei-tool-route-sft-v1") or _marker_hit(p, _LEGACY_MARKERS)
        ):
            continue
        add(p)
    return paths


def cpt_corpus_dirs(scope: str = "cpt-v2") -> list[Path]:
    dirs = [CORPUS_ZH_PRETRAIN]
    if scope in {"cpt-v2", "all"}:
        for d in (
            CORPUS_ZH_PRETRAIN_V1,
            CORPUS_ZH_PRETRAIN_V2,
            CORPUS_ZH_PRETRAIN_V3,
            CORPUS_ZH_PRETRAIN_V4,
            COLLOQUIAL_DRAFT,
        ):
            if d.is_dir() and d not in dirs:
                dirs.append(d)
        if COLLOQUIAL_LANES.is_dir():
            for d in sorted(COLLOQUIAL_LANES.iterdir()):
                if d.is_dir() and d not in dirs:
                    dirs.append(d)
        for key, d in CORPUS_ID_PATHS.items():
            if key.startswith("zh-pretrain-colloquial-synth") and d.is_dir() and d not in dirs:
                dirs.append(d)
    return dirs
