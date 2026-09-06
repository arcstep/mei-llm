#!/usr/bin/env python3
"""Shared loaders, budgeter, hashing and split primitives for the
``mei-1.0-51m-exp-000600m-sft-zh-rebuild-v1`` corpus rebuild.

This package is a new, offline, write-once generator suite for the six SFT
capabilities that ``corpus-factory/generators/factory_51m.py`` (factory-v3)
does not implement at all (retrieval/no-match, confidence, narration) or only
implements at human-reviewed pilot scale bound to the 300M base (full_call,
multi_step, mw_disposition). It intentionally reuses factory-v1's validator
and hashing primitives (``compiler_v1.py``) for consistency, but is a
separate module because factory-v3's worklist/campaign/human-review pipeline
shape is not suited to bulk template generation across six families.

Nothing here calls a model provider, mutates CURRENT.json, or writes into an
existing immutable release. All counts (tool universe size, family counts)
are read from the live registry files at run time -- never hard-coded.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[3]
PRODUCT = "mei-1.0-51m"
PARAMS = 51_463_797
CYCLE_ID = "exp-000600m"
GENERATOR_ID = "mei-51m-sft-zh-rebuild-v1"
GENERATOR_VERSION = "rebuild-zh-v1.0"
NEAR_DUPLICATE_THRESHOLD = 0.92

# ---------------------------------------------------------------------------
# compiler_v1 (factory-v1) reuse: same import pattern factory_51m.py uses.
# ---------------------------------------------------------------------------
V1_PATH = Path(__file__).resolve().parents[1] / "compiler_v1.py"
V1_SPEC = importlib.util.spec_from_file_location("mei_corpus_factory_v1_for_rebuild_zh_v1", V1_PATH)
if V1_SPEC is None or V1_SPEC.loader is None:
    raise RuntimeError(f"cannot load v1 corpus compiler: {V1_PATH}")
V1 = importlib.util.module_from_spec(V1_SPEC)
sys.modules[V1_SPEC.name] = V1
V1_SPEC.loader.exec_module(V1)

validate_instance = V1.validate_instance
merkle_root = V1.merkle_root


class RebuildError(RuntimeError):
    """A fail-closed contract violation in the rebuild-v1 generator suite."""


# ---------------------------------------------------------------------------
# Canonical live registry paths (verified current: 147 deploy/14 families,
# 211 training/22 families -- but this module never hard-codes those counts,
# it counts the loaded JSON at run time).
# ---------------------------------------------------------------------------
_V4_300M_RELEASE_DIR = (
    ROOT
    / "artifacts/mei-1.2-51m/legacy/mei-1.0-51m/exp-00300m/corpus/sft-suite"
    / "historical-notebook-releases/releases/mei-1.0-51m-tool-sft-v4-300m-v4"
)
DEPLOY_TOOLS_PATH = _V4_300M_RELEASE_DIR / "tool-universe.json"
TRAINING_TOOLS_PATH = _V4_300M_RELEASE_DIR / "training-tool-universe.json"
MW_CODEBOOK_PATH = _V4_300M_RELEASE_DIR / "mw-disposition-codebook-v1.json"
MW_DEFINITIONS_PATH = _V4_300M_RELEASE_DIR / "mw-reason-definitions-v2-20class.json"

EVAL_V7_BANK_DIR = ROOT / "artifacts/mei-1.0-51m/legacy/_legacy/notebook/evaluation/banks/mei-51m-longitudinal-eval-v7"

BASE_RELEASE_PATH = (
    ROOT
    / "models/mei-1.2-51m/releases/exp-000600m/base"
    / "mei-1.0-51m-base-cpt600m-clean-source-v3-v1/RELEASE.json"
)
BASE_WEIGHTS_SHA256 = "6d55a61773cdd0a6713c43fa7565f8c2fa64414c3e4a1beadfb6507e0d91752a"
CURRENT_JSON_PATH = ROOT / "CURRENT.json"

TOKENIZER_DIR = ROOT / "models/mei-1.2-51m/architecture"

RELEASE_ROOT = ROOT / "artifacts/mei-1.2-51m/legacy/mei-1.0-51m/exp-00600m/corpus/sft-suite"
EVAL_LOCK_ROOT = ROOT / "artifacts/mei-1.2-51m/legacy/mei-1.0-51m/exp-00600m/corpus/eval-lock"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_json(path: Path) -> Any:
    if not path.is_file():
        raise RebuildError(f"missing required registry file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolRegistry:
    path: Path
    sha256: str
    fingerprint: str
    tools: tuple[dict[str, Any], ...]
    by_name: dict[str, dict[str, Any]]
    families: dict[str, list[str]]

    @property
    def n_tools(self) -> int:
        return len(self.tools)

    @property
    def family_names(self) -> list[str]:
        return sorted(self.families)


def _index_tools(raw_tools: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    by_name: dict[str, dict[str, Any]] = {}
    families: dict[str, list[str]] = {}
    for tool in raw_tools:
        name = str(tool.get("name") or tool.get("tool_id"))
        if not name:
            raise RebuildError(f"tool entry missing name/tool_id: {tool!r}")
        by_name[name] = tool
        fam = str(tool.get("family") or "unknown")
        families.setdefault(fam, []).append(name)
    return by_name, families


def load_deploy_tools() -> ToolRegistry:
    raw = load_json(DEPLOY_TOOLS_PATH)
    tools = tuple(raw["tools"])
    by_name, families = _index_tools(list(tools))
    return ToolRegistry(
        path=DEPLOY_TOOLS_PATH,
        sha256=sha256_file(DEPLOY_TOOLS_PATH),
        fingerprint=str(raw.get("fingerprint") or ""),
        tools=tools,
        by_name=by_name,
        families=families,
    )


def load_training_tools() -> ToolRegistry:
    raw = load_json(TRAINING_TOOLS_PATH)
    tools = tuple(raw["tools"])
    by_name, families = _index_tools(list(tools))
    return ToolRegistry(
        path=TRAINING_TOOLS_PATH,
        sha256=sha256_file(TRAINING_TOOLS_PATH),
        fingerprint=str(raw.get("fingerprint") or ""),
        tools=tools,
        by_name=by_name,
        families=families,
    )


def load_mw_codebook() -> dict[str, Any]:
    return load_json(MW_CODEBOOK_PATH)


def load_mw_definitions() -> dict[str, Any]:
    return load_json(MW_DEFINITIONS_PATH)


def mw_classes() -> list[dict[str, Any]]:
    book = load_mw_codebook()
    classes = list(book["classes"])
    if len(classes) != 20:
        raise RebuildError(f"expected 20 MW classes, registry has {len(classes)}")
    return classes


def mw_reason_codes() -> list[str]:
    return [str(c["reason_code"]) for c in mw_classes()]


def mw_neighbors() -> dict[str, list[str]]:
    defs = load_mw_definitions()
    codes = defs.get("codes") or []
    out: dict[str, list[str]] = {}
    for item in codes:
        out[str(item["reason_code"])] = [str(n) for n in item.get("neighbors", [])]
    return out


def mw_definition_by_code() -> dict[str, dict[str, Any]]:
    defs = load_mw_definitions()
    codes = defs.get("codes") or []
    return {str(item["reason_code"]): item for item in codes}


# ---------------------------------------------------------------------------
# Tokenizer (real zh-24k-v1 SentencePiece model; falls back to a documented
# conservative char-based estimate only if the model file is unavailable in
# the current environment -- generation must still be reproducible offline).
# ---------------------------------------------------------------------------
class _CharFallbackTokenizer:
    """Conservative offline fallback: ~1 token per CJK char, ~0.3 token per
    other char, used ONLY if sentencepiece/the frozen model cannot load.
    Always over-estimates real SentencePiece token counts so budget gates
    stay conservative rather than silently permissive."""

    def count(self, text: str) -> int:
        total = 0
        for ch in text:
            if unicodedata.category(ch).startswith("L") and ord(ch) > 0x2E80:
                total += 1
            else:
                total += 1 if ch.strip() else 0
                total += 0  # whitespace/punct folded into neighboring pieces
        # Add a 15% safety margin over the raw char count for ASCII/JSON runs.
        ascii_run = sum(1 for ch in text if ord(ch) < 128)
        return int(total + ascii_run * 0.35) + 1


_tokenizer_singleton: Any = None
_tokenizer_is_fallback = False


def get_tokenizer():
    global _tokenizer_singleton, _tokenizer_is_fallback
    if _tokenizer_singleton is not None:
        return _tokenizer_singleton
    text = str(TOKENIZER_DIR)
    added = text not in sys.path
    if added:
        sys.path.insert(0, text)
    try:
        from tokenizer import ZhTokenizerV1  # type: ignore

        _tokenizer_singleton = ZhTokenizerV1()
        _tokenizer_is_fallback = False
    except Exception:
        _tokenizer_singleton = _CharFallbackTokenizer()
        _tokenizer_is_fallback = True
    return _tokenizer_singleton


def tokenizer_is_fallback() -> bool:
    get_tokenizer()
    return _tokenizer_is_fallback


def count_tokens(text: str) -> int:
    tok = get_tokenizer()
    if hasattr(tok, "count"):
        return int(tok.count(text))
    return int(len(tok.encode(text)))


# ---------------------------------------------------------------------------
# 2048-token joint budget contract (adaptive-tool-context.md).
#
# The exact runtime fixed task prompt text is owned by model-factory's
# serializer and is not duplicated here; this generator reserves a
# conservative, documented token allowance for it (FIXED_PROMPT_RESERVE)
# rather than guessing its literal contents, and always measures schema
# projection / query / context / evidence / history / tool-results with the
# real zh-24k-v1 tokenizer against the remainder of the 2048 budget.
# ---------------------------------------------------------------------------
JOINT_BUDGET = 2048
DEFAULT_OUTPUT_RESERVE = 128
FIXED_PROMPT_RESERVE = 200  # conservative estimate for wire/task boilerplate
PROMPT_CAP = JOINT_BUDGET - DEFAULT_OUTPUT_RESERVE  # 1920, matches contract
COMPACT_CAP = 1024
STANDARD_CAP = 1536
GENERATOR_INPUT_CAP = PROMPT_CAP - FIXED_PROMPT_RESERVE  # 1720
BATCH_SIZE = 5


def project_tool(tool: Mapping[str, Any], *, profile: str) -> dict[str, Any]:
    """Compact/standard schema projection. Never weakens name, parameter
    structure, validation constraints, or execution schema -- only trims
    natural-language description length, per adaptive-tool-context.md."""

    params = tool.get("parameters") or {}
    props = dict(params.get("properties") or {})
    desc_cap = 24 if profile == "compact" else 60
    projected_props: dict[str, Any] = {}
    for name, spec in props.items():
        spec2 = dict(spec)
        if "description" in spec2 and profile == "compact":
            spec2["description"] = str(spec2["description"])[:desc_cap]
        projected_props[name] = spec2
    tool_desc = str(tool.get("description") or "")
    if profile == "compact" and len(tool_desc) > 40:
        tool_desc = tool_desc[:40] + "…"
    return {
        "name": tool.get("name") or tool.get("tool_id"),
        "description": tool_desc,
        "parameters": {
            "type": params.get("type", "object"),
            "properties": projected_props,
            "required": list(params.get("required") or []),
            "additionalProperties": params.get("additionalProperties", False),
        },
    }


@dataclass
class BudgetResult:
    profile: str
    prompt_tokens: int
    cap: int
    fits: bool
    truncated_history: bool = False
    truncated_tool_results: bool = False
    truncated_query: bool = False
    context_unrepresentable: bool = False


def _trim_head_tail(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max_chars - 1
    head = keep * 2 // 3
    tail = keep - head
    return text[:head] + "…" + text[len(text) - tail :]


def fit_batch_to_budget(
    *,
    tools_batch: Sequence[Mapping[str, Any]],
    query: str,
    context: Mapping[str, Any] | None,
    evidence: Sequence[Any] | None,
    history: Sequence[Mapping[str, Any]] | None,
    tool_results: Sequence[Mapping[str, Any]] | None,
    profile: str = "standard",
) -> tuple[dict[str, Any], BudgetResult]:
    """Apply the ordered joint-budget trimming procedure and return the
    (possibly trimmed) payload plus a BudgetResult receipt."""

    if len(tools_batch) > BATCH_SIZE:
        raise RebuildError("tools_batch exceeds fixed-five batch size")
    cap = COMPACT_CAP if profile == "compact" else min(STANDARD_CAP, GENERATOR_INPUT_CAP)
    cap = min(cap, GENERATOR_INPUT_CAP)

    projected = [project_tool(t, profile=profile) for t in tools_batch]
    history_list = list(history or [])
    results_list = list(tool_results or [])
    query_text = query
    truncated_history = False
    truncated_results = False
    truncated_query = False

    def payload() -> dict[str, Any]:
        return {
            "tools": projected,
            "query": query_text,
            "context": dict(context or {}),
            "evidence": list(evidence or []),
            "history": history_list,
            "tool_results": results_list,
        }

    def tokens() -> int:
        return count_tokens(canonical_json(payload()))

    # 1) tools/query/context/evidence/history/results are already in minimal
    #    structural form (schema constraints preserved). 2) drop oldest
    #    history first.
    while tokens() > cap and history_list:
        history_list.pop(0)
        truncated_history = True

    # 3) compress older tool-result payloads (keep latest verified result
    #    call_id/status/provenance intact).
    idx = 0
    while tokens() > cap and idx < len(results_list) - 1:
        r = dict(results_list[idx])
        if "payload" in r:
            r["payload"] = {"_truncated": True}
            results_list[idx] = r
            truncated_results = True
        idx += 1

    # 4) trim query head/tail-preserving as a last resort.
    guard = 0
    while tokens() > cap and len(query_text) > 8 and guard < 200:
        query_text = _trim_head_tail(query_text, max(8, len(query_text) - 16))
        truncated_query = True
        guard += 1

    final_tokens = tokens()
    unrepresentable = final_tokens > cap and len(projected) <= 1
    return payload(), BudgetResult(
        profile=profile,
        prompt_tokens=final_tokens,
        cap=cap,
        fits=final_tokens <= cap,
        truncated_history=truncated_history,
        truncated_tool_results=truncated_results,
        truncated_query=truncated_query,
        context_unrepresentable=unrepresentable,
    )


# ---------------------------------------------------------------------------
# Split assignment (family/group-aware isolation): a cf_group always maps to
# the same split so paraphrases/negatives sharing a group never leak across
# train/valid/dev/test.
# ---------------------------------------------------------------------------
SPLIT_NAMES = ("train", "valid", "dev", "test")
DEFAULT_SPLIT_WEIGHTS = (0.70, 0.12, 0.09, 0.09)


def assign_split(cf_group: str, weights: tuple[float, ...] = DEFAULT_SPLIT_WEIGHTS) -> str:
    digest = hashlib.sha256(cf_group.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    cumulative = 0.0
    for name, weight in zip(SPLIT_NAMES, weights):
        cumulative += weight
        if bucket <= cumulative:
            return name
    return SPLIT_NAMES[-1]


def stratified_sample(rows: Sequence[Mapping[str, Any]], key: str, n: int) -> list[Mapping[str, Any]]:
    """Round-robin across distinct values of `key` so a truncated sample stays
    diverse across scenario/kind groups instead of favoring whichever group a
    generator happened to emit first."""

    groups: dict[Any, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row.get(key), []).append(row)
    order = list(groups.keys())
    out: list[Mapping[str, Any]] = []
    idx = 0
    while len(out) < n and any(groups.values()):
        key_i = order[idx % len(order)]
        bucket = groups[key_i]
        if bucket:
            out.append(bucket.pop(0))
        idx += 1
        if idx > n * len(order) + len(order):
            break
    return out[:n]


def cf_group(*parts: str) -> str:
    return "cfg:" + sha256_text("|".join(parts))[:24]


def case_id(family: str, *parts: str) -> str:
    return f"{family}:" + sha256_text("|".join((family, *parts)))[:20]


# ---------------------------------------------------------------------------
# Dedup: exact + near-duplicate (char trigram Jaccard).
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def char_trigrams(text: str) -> set[str]:
    norm = normalize_text(text)
    if len(norm) < 3:
        return {norm} if norm else set()
    return {norm[i : i + 3] for i in range(len(norm) - 2)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class DedupIndex:
    exact: set[str] = field(default_factory=set)
    trigram_by_id: dict[str, set[str]] = field(default_factory=dict)

    def check_and_add(self, row_id: str, visible_text: str) -> tuple[bool, str | None]:
        """Return (is_duplicate, duplicate_of_id_or_none). Adds row if unique."""
        norm = normalize_text(visible_text)
        exact_key = sha256_text(norm)
        if exact_key in self.exact:
            return True, "exact"
        trigrams = char_trigrams(visible_text)
        for other_id, other_trigrams in self.trigram_by_id.items():
            if jaccard(trigrams, other_trigrams) >= NEAR_DUPLICATE_THRESHOLD:
                return True, other_id
        self.exact.add(exact_key)
        self.trigram_by_id[row_id] = trigrams
        return False, None


# ---------------------------------------------------------------------------
# Leakage scan: model-visible text must never contain reason codes, class
# names, gold-route markers, or holdout markers.
# ---------------------------------------------------------------------------
_LEAKAGE_MARKERS = (
    "gold_name",
    "gold_route",
    "holdout",
    "reason_code",
    "class_id",
    "answer_key",
    "ground_truth",
)


def scan_leakage(visible_text: str, forbidden_tokens: Iterable[str] = ()) -> list[str]:
    hits: list[str] = []
    lowered = visible_text.lower()
    for marker in _LEAKAGE_MARKERS:
        if marker in lowered:
            hits.append(marker)
    for tok in forbidden_tokens:
        if tok and tok in visible_text:
            hits.append(f"forbidden_token:{tok}")
    return hits


# ---------------------------------------------------------------------------
# Deterministic host simulator: verified, non-empty ToolResult generation for
# full-call / agent trajectories. Never fabricates a value the schema does
# not allow; payload values are seeded deterministically from case_id so
# reruns are byte-identical.
# ---------------------------------------------------------------------------
def _seeded_value(schema: Mapping[str, Any], seed: str) -> Any:
    seed_int = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16)
    if "const" in schema:
        return schema["const"]
    if "enum" in schema and schema["enum"]:
        return schema["enum"][seed_int % len(schema["enum"])]
    t = schema.get("type")
    if t == "boolean":
        return bool(seed_int % 2)
    if t == "integer":
        lo = int(schema.get("minimum", 0))
        hi = int(schema.get("maximum", lo + 100))
        step = int(schema.get("multipleOf", 1)) or 1
        span = max(1, (hi - lo) // step)
        return lo + (seed_int % (span + 1)) * step
    if t == "number":
        lo = float(schema.get("minimum", 0))
        hi = float(schema.get("maximum", lo + 100))
        frac = (seed_int % 1000) / 1000.0
        return round(lo + frac * (hi - lo), 2)
    if t == "string":
        min_len = int(schema.get("minLength", 1))
        base = f"值{seed_int % 9999}"
        while len(base) < min_len:
            base += "补"
        max_len = schema.get("maxLength")
        if max_len is not None:
            base = base[: int(max_len)]
        return base
    if t == "array":
        items_schema = schema.get("items") or {"type": "string"}
        min_items = int(schema.get("minItems", 1))
        count = max(min_items, 1)
        return [_seeded_value(items_schema, f"{seed}:{i}") for i in range(count)]
    return None


seeded_value = _seeded_value  # public alias for cross-module reuse


def simulate_tool_result(tool: Mapping[str, Any], arguments: Mapping[str, Any], *, call_id: str) -> dict[str, Any]:
    """A frozen, deterministic host simulator. Produces a non-empty, verified
    ToolResult grounded only in the tool's own schema + supplied arguments --
    never fabricates facts absent from either."""

    params = (tool.get("parameters") or {}).get("properties") or {}
    payload: dict[str, Any] = {}
    for name, spec in params.items():
        if name in arguments:
            payload[f"confirmed_{name}"] = arguments[name]
        else:
            payload[f"observed_{name}"] = _seeded_value(spec, f"{call_id}:{name}")
    payload["tool"] = tool.get("name") or tool.get("tool_id")
    result = {
        "status": "ok",
        "payload": payload,
        "verified": True,
        "provenance": "mei-51m-deterministic-host-simulator-rebuild-v1",
        "call_id": call_id,
    }
    result["result_sha256"] = sha256_text(canonical_json(result))
    return result


def simulate_tool_error(tool: Mapping[str, Any], *, call_id: str, error_code: str) -> dict[str, Any]:
    result = {
        "status": "error",
        "error_code": error_code,
        "payload": {"tool": tool.get("name") or tool.get("tool_id")},
        "verified": True,
        "provenance": "mei-51m-deterministic-host-simulator-rebuild-v1",
        "call_id": call_id,
    }
    result["result_sha256"] = sha256_text(canonical_json(result))
    return result
