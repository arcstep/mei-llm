#!/usr/bin/env python3
"""Canonical paths after the formal-root cutover.

Published assets live at repo root: tokenizer/ corpus/ architecture/ training/
runtime/ base/ sft/. Notebook is the peripheral lab (corpus jobs, eval, archive).

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

PUBLISHED_LM_V1 = ROOT / "corpus" / "lm-v1"
PUBLISHED_LM_V2 = ROOT / "corpus" / "lm-v2"
LM_V2 = NOTEBOOK / "corpus" / "lm-v2"
ARCHITECTURE_V1 = ROOT / "architecture" / "mei-1.0-51m-arch-v1"
TRAINING_V1 = ROOT / "training" / "mei-1.0-51m-train-v1"
TRAIN_RUNS = ROOT / "training" / "runs"
RUNTIME_SHARED = ROOT / "runtime" / "_shared"
RUNTIME_ROUTE_V1 = ROOT / "runtime" / "mei-1.0-51m-route-v1"
RUNTIME_NEEDLE2_V2 = ROOT / "runtime" / "mei-1.0-51m-needle2-v2"

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
LANGUAGE_WORK = LM_V1 / "language" / "work"
LANGUAGE_WORK_V0 = LANGUAGE_WORK / "zh-pretrain-v0"
LANGUAGE_WORK_V1 = LANGUAGE_WORK / "zh-pretrain-v1"
LANGUAGE_WORK_HQ = LANGUAGE_WORK / "hq"
STRUCTURE_WORK = LM_V1 / "structure" / "work"
STRUCTURE_WORK_V3 = STRUCTURE_WORK / "zh-pretrain-v3"
ASSEMBLE_V4 = ASSEMBLE_WORK / "zh-pretrain-v4"
APPROVED_COLLOQUIAL = ASSEMBLE_V4 / "approved-colloquial.json"
PUBLISHED_COLLOQUIAL_POOLED = (
    PUBLISHED_LM_V1 / "colloquial" / "zh-pretrain-colloquial-synth-pooled-v1"
)
VOCAB_WORK = NOTEBOOK / "corpus" / "tokenizer-v1" / "work" / "zh-vocab-v0"

CORPUS_ZH_VOCAB = VOCAB_WORK
CORPUS_ZH_PRETRAIN = PUBLISHED_LM_V1 / "language" / "zh-pretrain-v0"
CORPUS_ZH_PRETRAIN_V1 = PUBLISHED_LM_V1 / "language" / "zh-pretrain-v1"
CORPUS_ZH_PRETRAIN_HQ = PUBLISHED_LM_V1 / "language" / "hq"
CORPUS_ZH_PRETRAIN_V2 = ARCHIVE_CORPUS / "zh-pretrain-v2"
CORPUS_ZH_PRETRAIN_V3 = PUBLISHED_LM_V1 / "structure" / "zh-pretrain-v3"
CORPUS_ZH_PRETRAIN_V4 = PUBLISHED_LM_V1
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
    "zh-pretrain-colloquial-synth-pooled-v1": PUBLISHED_COLLOQUIAL_POOLED,
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

ISOLATION_SCOPES = ("legacy", "cpt-v2", "sft-v2", "sft-v2-lock-v2", "all")
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
    """Resolve historical notebook/base/pretrain-v1/<sub> onto split locations."""

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
NOTEBOOK_JOBS = NOTEBOOK / "jobs"
JOBS_INDEX_LIVE = NOTEBOOK_JOBS / "index.json"
JOBS_INDEX_OVERLAY = NOTEBOOK / "_migration" / "260826-jobs-overlay" / "index.json"
JOBS_INDEX = JOBS_INDEX_LIVE
CORPORA_INDEX = ROOT / "CURRENT.json"
TOPIC_TOOLCALL_SFT = "toolcall-sft"
JOB_RETRIEVAL_V2 = "260827-01-retrieval-v2"
JOB_FULLCALL_V2 = "260827-02-fullcall-v2"
JOB_MW_DISPOSITION_V1 = "260827-03-mw-disposition-v1"
FLEET_SFT_SYNTH = SFT_RECIPES / "sft-synth-fleet-v1.json"
CODEBOOK_MW_DISPOSITION_V1 = SFT_RECIPES / "mw-disposition-codebook-v1.json"
STAIRCASE_SFT_SYNTH = SFT_RECIPES / "sft-synth-staircase-v1.json"
BANK_MEI_RETRIEVAL_V2 = EVAL_BANKS_ROOT / "mei-retrieval-v2/eval-bank-smoke.jsonl"
BANK_MEI_MW_DISPOSITION_V2 = EVAL_BANKS_ROOT / "mei-mw-disposition-v2/eval-bank-smoke.jsonl"
BANK_MEI_PARK_TOOLCALL_V1 = EVAL_BANKS_ROOT / "mei-park-toolcall-v1/eval-bank-v1.jsonl"
PACK_MEI_RETRIEVAL_V2_SMOKE = SFT_TRAIN / "packs/mei-retrieval-v2-smoke.jsonl"
PACK_MEI_TOOLCALL_V2_ORACLE_SMOKE = SFT_TRAIN / "packs/mei-toolcall-v2-oracle-smoke.jsonl"
PACK_MEI_MW_DISPOSITION_V2_SMOKE = SFT_TRAIN / "packs/mei-mw-disposition-v2-smoke.jsonl"
PACK_MEI_RETRIEVAL_V2_2K = SFT_TRAIN / "packs/mei-retrieval-v2-2k.candidates.jsonl"
PACK_MEI_TOOLCALL_V2_ORACLE_2K = SFT_TRAIN / "packs/mei-toolcall-v2-oracle-2k.candidates.jsonl"
PACK_MEI_MW_DISPOSITION_V2_2K = SFT_TRAIN / "packs/mei-mw-disposition-v2-2k.candidates.jsonl"
PACK_MEI_RETRIEVAL_V2_2K_PAID = SFT_TRAIN / "packs/mei-retrieval-v2-2k.paid.candidates.jsonl"
PACK_MEI_TOOLCALL_V2_ORACLE_2K_PAID = SFT_TRAIN / "packs/mei-toolcall-v2-oracle-2k.paid.candidates.jsonl"
PACK_MEI_MW_DISPOSITION_V2_2K_PAID = SFT_TRAIN / "packs/mei-mw-disposition-v2-2k.paid.candidates.jsonl"
PACK_MEI_RETRIEVAL_V2_10K_PAID = SFT_TRAIN / "packs/mei-retrieval-v2-10k.paid.candidates.jsonl"
PACK_MEI_TOOLCALL_V2_ORACLE_10K_PAID = SFT_TRAIN / "packs/mei-toolcall-v2-oracle-10k.paid.candidates.jsonl"
PACK_MEI_MW_DISPOSITION_V2_10K_PAID = SFT_TRAIN / "packs/mei-mw-disposition-v2-10k.paid.candidates.jsonl"
PACK_MEI_RETRIEVAL_V2_10K_CLEAN = SFT_TRAIN / "packs/mei-retrieval-v2-10k.clean.candidates.jsonl"
PACK_MEI_TOOLCALL_V2_ORACLE_10K_CLEAN = SFT_TRAIN / "packs/mei-toolcall-v2-oracle-10k.clean.candidates.jsonl"
PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN = SFT_TRAIN / "packs/mei-mw-disposition-v2-10k.clean.candidates.jsonl"
BANK_MEI_RETRIEVAL_V2_DEV = EVAL_BANKS_ROOT / "mei-retrieval-v2/eval-bank-dev.jsonl"
BANK_MEI_RETRIEVAL_V2_TEST = EVAL_BANKS_ROOT / "mei-retrieval-v2/eval-bank-test.jsonl"
BANK_MEI_TOOLCALL_V2_DEV = EVAL_BANKS_ROOT / "mei-toolcall-v2/eval-bank-dev.jsonl"
BANK_MEI_TOOLCALL_V2_TEST = EVAL_BANKS_ROOT / "mei-toolcall-v2/eval-bank-test.jsonl"
BANK_MEI_MW_DISPOSITION_V2_DEV = EVAL_BANKS_ROOT / "mei-mw-disposition-v2/eval-bank-dev.jsonl"
BANK_MEI_MW_DISPOSITION_V2_TEST = EVAL_BANKS_ROOT / "mei-mw-disposition-v2/eval-bank-test.jsonl"
SFT_V2_BASELINE_CONTRACT = SFT_RECIPES / "sft-v2-baseline-contract-v1.json"
SFT_V2_BASELINE_MODELS = SFT_RECIPES / "sft-v2-baseline-models-v1.json"
SFT_V2_EVAL_LOCK_DIR = EVAL_BANKS_ROOT / "sft-v2-eval-lock-v1"
SFT_V2_BASELINE_CONTRACT_V2 = SFT_RECIPES / "sft-v2-baseline-contract-v2.json"
SFT_V2_BASELINE_MODELS_V2 = SFT_RECIPES / "sft-v2-baseline-models-v2.json"
MW_REASON_DEFINITIONS_V1 = SFT_RECIPES / "mw-reason-definitions-v1.json"
SFT_V2_EVAL_LOCK_DIR_V2 = EVAL_BANKS_ROOT / "sft-v2-eval-lock-v2"
BANK_MEI_RETRIEVAL_V2_FAIR_DEV = SFT_V2_EVAL_LOCK_DIR_V2 / "eval-retrieval-dev.jsonl"
BANK_MEI_RETRIEVAL_V2_FAIR_TEST = SFT_V2_EVAL_LOCK_DIR_V2 / "eval-retrieval-test.jsonl"
BANK_MEI_TOOLCALL_V2_FAIR_DEV = SFT_V2_EVAL_LOCK_DIR_V2 / "eval-fullcall-dev.jsonl"
BANK_MEI_TOOLCALL_V2_FAIR_TEST = SFT_V2_EVAL_LOCK_DIR_V2 / "eval-fullcall-test.jsonl"
BANK_MEI_MW_V2_FAIR_DEV = SFT_V2_EVAL_LOCK_DIR_V2 / "eval-mw-dev.jsonl"
BANK_MEI_MW_V2_FAIR_TEST = SFT_V2_EVAL_LOCK_DIR_V2 / "eval-mw-test.jsonl"
PACK_MEI_RETRIEVAL_V2_10K_CLEAN_V2 = SFT_TRAIN / "packs/mei-retrieval-v2-10k.clean.v2.candidates.jsonl"
PACK_MEI_TOOLCALL_V2_ORACLE_10K_CLEAN_V2 = SFT_TRAIN / "packs/mei-toolcall-v2-oracle-10k.clean.v2.candidates.jsonl"
PACK_MEI_MW_DISPOSITION_V2_10K_CLEAN_V2 = SFT_TRAIN / "packs/mei-mw-disposition-v2-10k.clean.v2.candidates.jsonl"
EXTERNAL_BASELINES = NOTEBOOK / "archive" / "external-baselines"


def sft_v2_candidate_pack_name(stem: str, limit: int) -> str:
    """Template pack filename. Paid packs keep the `.paid.candidates.jsonl` suffix."""
    if 0 < limit <= 24:
        return f"{stem}-smoke.jsonl"
    tier = "10k" if limit >= 10000 else "2k"
    return f"{stem}-{tier}.candidates.jsonl"


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


def is_mei_llm_root(path: Path | None = None) -> bool:
    base = (path or ROOT).resolve()
    return (base / "CURRENT.json").is_file() and (base / "notebook").is_dir()


def jobs_home(root: Path | None = None) -> Path:
    """Canonical jobs container.

    Repo root (CURRENT.json + notebook/) always uses notebook/jobs, even when
    callers pass an explicit --root pointing at the mei-llm checkout. Scratch
    roots used in tests keep root/jobs.
    """
    base = (root or ROOT).resolve()
    if is_mei_llm_root(base):
        return base / "notebook" / "jobs"
    return base / "jobs"


def jobs_rel_prefix(root: Path | None = None) -> str:
    home = jobs_home(root).resolve()
    base = (root or ROOT).resolve()
    try:
        return home.relative_to(base).as_posix()
    except ValueError:
        return "jobs"


def job_topic_dir(topic: str, *, root: Path | None = None) -> Path:
    base = (root or ROOT).resolve()
    if topic in {"colloquial-cpt", "colloquial"} and is_mei_llm_root(base):
        return COLLOQUIAL if base == ROOT.resolve() else jobs_home(base) / topic
    return jobs_home(root) / topic


def job_work(topic: str, job_id: str, *, root: Path | None = None) -> Path:
    return job_topic_dir(topic, root=root) / "jobs" / job_id / "work"


def publish_corpus(corpus_id: str, *, root: Path | None = None) -> Path:
    if root is not None:
        return root / "corpus" / corpus_id
    return CORPUS_ID_PATHS.get(corpus_id, ARCHIVE_CORPUS / corpus_id)


def load_jobs_index(*, root: Path | None = None) -> dict:
    home = jobs_home(root)
    path = home / "index.json"
    if not path.is_file() and is_mei_llm_root(root or ROOT):
        path = JOBS_INDEX_OVERLAY
    if not path.is_file():
        return {"version": 1, "topics": {}, "jobs": [], "sft_packs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpora_index(*, root: Path | None = None) -> dict:
    path = (root or ROOT) / "CURRENT.json"
    if not path.is_file():
        return {
            "version": 1,
            "tokenizer": None,
            "corpus": None,
            "architecture": None,
            "training": None,
            "base": None,
            "sft": None,
            "runtime": None,
        }
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
            "mei-toolcall-v2" in str(p)
            or "mei-51m-toolcall" in str(p)
            or "mei-retrieval-v2" in str(p)
            or "mei-mw-disposition-v2" in str(p)
            or "mei-park-toolcall-v1" in str(p)
        ):
            continue
        if scope == "sft-v2-lock-v2" and "sft-v2-eval-lock-v2" not in str(p):
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
        allow = ("mei-retrieval-v2", "mei-toolcall-v2-oracle", "mei-mw-disposition-v2")
        for p in _iter_sft_jsonl():
            if "packs" not in p.as_posix() or "v2" not in p.name:
                continue
            if ".review-sample." in p.name:
                continue
            if not any(tag in p.name for tag in allow):
                continue
            add(p)
        return paths
    if scope == "sft-v2-lock-v2":
        allow = (
            "mei-retrieval-v2-10k.clean.v2",
            "mei-toolcall-v2-oracle-10k.clean.v2",
            "mei-mw-disposition-v2-10k.clean.v2",
        )
        for p in _iter_sft_jsonl():
            if "packs" not in p.as_posix():
                continue
            if any(tag in p.name for tag in allow):
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
    """Live CPT leak-scan dirs: published wiki v0 + structure v3 + colloquial process.

    Do not scan the whole `corpus/lm-v1` mix tree (re-walks v0/v1/hq).
    Do not scan archived dirty v2 / CWT2.
    """
    dirs: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if not path.is_dir():
            return
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        dirs.append(path)

    add(LANGUAGE_WORK_V0)
    if scope in {"cpt-v2", "all"}:
        add(LANGUAGE_WORK_V1)
        add(LANGUAGE_WORK_HQ)
        add(STRUCTURE_WORK_V3)
        add(COLLOQUIAL_DRAFT)
        add(PUBLISHED_COLLOQUIAL_POOLED)
        if COLLOQUIAL_LANES.is_dir():
            for d in sorted(COLLOQUIAL_LANES.iterdir()):
                add(d)
        for key, d in CORPUS_ID_PATHS.items():
            if key.startswith("zh-pretrain-colloquial-synth"):
                add(d)
    return dirs
