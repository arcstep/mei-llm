"""Deterministic tool batching and joint context budgeting for runtime wire v2.

This module is intentionally free of MLX/Rust binding details.  Python is the
executable semantic oracle and the same constants/fixtures are consumed by the
portable runtime.  Full registered schemas never pass through this module for
validation: only the model-visible projection is budgeted here.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    from .canonical_json import dumps_canonical
except ImportError:
    from canonical_json import dumps_canonical

CONTEXT_PACKER_ID = "mei-tool-context-packer-v1"
PROJECTION_SERIALIZER_ID = "mei-schema-projection-v1"
RETRIEVAL_BATCH_POLICY_ID = "mei-retrieval-fixed-five-batches-v1"
RETRIEVAL_CALIBRATION_ID = "mei-retrieval-platt-v1"

MAX_CONTEXT_TOKENS = 2048
DEFAULT_OUTPUT_RESERVE = 128
MAX_PROMPT_TOKENS = MAX_CONTEXT_TOKENS - DEFAULT_OUTPUT_RESERVE
TOOL_BATCH_SIZE = 5

PROFILE_STABLE_CAPS = {
    "compact": 1024,
    "standard": 1536,
}

# These defaults are deliberately permissive until a package supplies its
# validation-fitted calibrator and thresholds.  They retain the historical
# top-five behaviour without pretending that an uncalibrated score is a
# trustworthy no-match detector.
DEFAULT_DISCARD_THRESHOLD = 0.0
DEFAULT_EXPAND_THRESHOLD = 0.0

_DROP_ANNOTATIONS = frozenset(
    {
        "$comment",
        "comment",
        "default",
        "deprecated",
        "examples",
        "example",
        "readOnly",
        "title",
        "writeOnly",
    }
)


def _canonical(value: Any) -> str:
    return dumps_canonical(value)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _encode(tokenizer: Any, text: str, *, add_bos: bool = False) -> list[int]:
    """Call the repository tokenizer without coupling to one concrete class."""

    try:
        return list(tokenizer.encode(text, add_bos=add_bos, add_eos=False))
    except TypeError:
        return list(tokenizer.encode(text))


def token_count(tokenizer: Any, text: str, *, add_bos: bool = False) -> int:
    return len(_encode(tokenizer, text, add_bos=add_bos))


def _decode(tokenizer: Any, ids: Sequence[int]) -> str:
    return str(tokenizer.decode([int(value) for value in ids]))


def platt_relevance(score: float, *, scale: float = 1.0, bias: float = 0.0) -> float:
    """Map a cosine score to calibrated relevance with a stable sigmoid."""

    z = float(scale) * float(score) + float(bias)
    if not math.isfinite(z):
        raise ValueError("retrieval calibration input must be finite")
    if z >= 0:
        exp = math.exp(-z)
        return 1.0 / (1.0 + exp)
    exp = math.exp(z)
    return exp / (1.0 + exp)


def fit_platt_calibrator(
    scored_labels: Iterable[tuple[float, int]],
    *,
    max_iterations: int = 100,
    l2: float = 1e-6,
) -> dict[str, Any]:
    """Fit a deterministic two-parameter Platt calibrator.

    Positive and negative examples receive equal aggregate weight so a large
    catalog cannot make the all-negative solution appear calibrated.
    """

    pairs = [(float(score), int(label)) for score, label in scored_labels]
    if not pairs or any(label not in {0, 1} or not math.isfinite(score) for score, label in pairs):
        raise ValueError("Platt calibration requires finite binary-labelled scores")
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Platt calibration requires both positive and negative examples")
    positive_weight = len(pairs) / (2.0 * positives)
    negative_weight = len(pairs) / (2.0 * negatives)
    scale = 1.0
    bias = math.log((positives + 1.0) / (negatives + 1.0))
    converged = False
    for iteration in range(max(1, int(max_iterations))):
        grad_scale = 0.0
        grad_bias = 0.0
        h_ss = float(l2)
        h_sb = 0.0
        h_bb = float(l2)
        loss = 0.0
        for score, label in pairs:
            weight = positive_weight if label else negative_weight
            probability = platt_relevance(score, scale=scale, bias=bias)
            probability = min(1.0 - 1e-12, max(1e-12, probability))
            error = probability - label
            curvature = weight * probability * (1.0 - probability)
            grad_scale += weight * error * score
            grad_bias += weight * error
            h_ss += curvature * score * score
            h_sb += curvature * score
            h_bb += curvature
            loss -= weight * (
                label * math.log(probability) + (1 - label) * math.log(1.0 - probability)
            )
        grad_scale += l2 * scale
        grad_bias += l2 * bias
        determinant = h_ss * h_bb - h_sb * h_sb
        if determinant <= 1e-18 or not math.isfinite(determinant):
            break
        step_scale = (h_bb * grad_scale - h_sb * grad_bias) / determinant
        step_bias = (-h_sb * grad_scale + h_ss * grad_bias) / determinant
        next_scale = max(1e-6, scale - step_scale)
        next_bias = bias - step_bias
        if not math.isfinite(next_scale) or not math.isfinite(next_bias):
            break
        scale, bias = next_scale, next_bias
        if max(abs(step_scale), abs(step_bias)) < 1e-9:
            converged = True
            break
    probabilities = [platt_relevance(score, scale=scale, bias=bias) for score, _ in pairs]
    brier = sum((probability - label) ** 2 for probability, (_, label) in zip(probabilities, pairs)) / len(pairs)
    return {
        "calibration_id": RETRIEVAL_CALIBRATION_ID,
        "scale": scale,
        "bias": bias,
        "converged": converged,
        "iterations": iteration + 1,
        "rows": len(pairs),
        "positive_rows": positives,
        "negative_rows": negatives,
        "brier": brier,
    }


def select_retrieval_thresholds(
    examples: Sequence[dict[str, Any]],
    *,
    max_gold_recall_drop: float = 0.005,
    max_no_match_false_selection: float = 0.05,
    min_rank_gt5_gold_retention: float = 0.99,
) -> dict[str, Any]:
    """Select discard/expand thresholds from locked validation examples only."""

    if not examples:
        raise ValueError("retrieval threshold selection requires validation examples")
    catalog_sizes = sorted({int(row.get("catalog_size") or 0) for row in examples})
    positive = [row for row in examples if row.get("gold_index") is not None]
    no_match = [row for row in examples if row.get("gold_index") is None]
    if not positive or not no_match:
        raise ValueError("threshold validation requires positive and no-match examples")

    values = {0.0, 1.0}
    for row in examples:
        values.update(float(value) for value in row.get("relevances") or [])
    grid = sorted(value for value in values if 0.0 <= value <= 1.0)

    def recall(threshold: float) -> float:
        hits = 0
        for row in positive:
            index = int(row["gold_index"])
            relevances = list(row.get("relevances") or [])
            hits += int(0 <= index < len(relevances) and float(relevances[index]) >= threshold)
        return hits / len(positive)

    def false_selection(threshold: float) -> float:
        return sum(
            any(float(value) >= threshold for value in row.get("relevances") or [])
            for row in no_match
        ) / len(no_match)

    baseline_recall = recall(0.0)
    valid_discard = [
        threshold
        for threshold in grid
        if baseline_recall - recall(threshold) <= max_gold_recall_drop
        and false_selection(threshold) <= max_no_match_false_selection
    ]
    discard_validated = bool(valid_discard)
    if valid_discard:
        discard = max(valid_discard)
    else:
        recall_only = [
            threshold
            for threshold in grid
            if baseline_recall - recall(threshold) <= max_gold_recall_drop
        ]
        discard = max(recall_only or [0.0])

    rank_gt5 = [row for row in positive if int(row["gold_index"]) >= TOOL_BATCH_SIZE]

    def tail_retention(threshold: float) -> float:
        if not rank_gt5:
            return 0.0
        return sum(
            float(row["relevances"][int(row["gold_index"])]) >= threshold
            for row in rank_gt5
        ) / len(rank_gt5)

    valid_expand = [
        threshold
        for threshold in grid
        if threshold >= discard and tail_retention(threshold) >= min_rank_gt5_gold_retention
    ]
    expand_validated = bool(valid_expand) and bool(rank_gt5)
    expand = max(valid_expand) if expand_validated else discard
    coverage_validated = catalog_sizes == [10, 20, 50]
    validated = discard_validated and expand_validated and coverage_validated
    return {
        "discard_threshold": discard,
        "expand_threshold": expand,
        "validated": validated,
        "fallback_scan_all_eligible": not validated,
        "metrics": {
            "baseline_gold_recall": baseline_recall,
            "eligible_gold_recall": recall(discard),
            "gold_recall_drop": baseline_recall - recall(discard),
            "no_match_false_selection_rate": false_selection(discard),
            "rank_gt5_gold_rows": len(rank_gt5),
            "rank_gt5_gold_retention": tail_retention(expand),
            "catalog_sizes": catalog_sizes,
            "discard_validated": discard_validated,
            "expand_validated": expand_validated,
            "catalog_coverage_validated": coverage_validated,
        },
    }


@dataclass(frozen=True)
class RankedCandidate:
    tool_id: str
    schema: dict[str, Any]
    raw_score: float
    relevance: float
    rank: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "rank": self.rank,
            "raw_score": self.raw_score,
            "retrieval_relevance": self.relevance,
        }


@dataclass(frozen=True)
class CandidateBatchPlan:
    candidates: tuple[RankedCandidate, ...]
    batches: tuple[tuple[RankedCandidate, ...], ...]
    discarded: tuple[RankedCandidate, ...]
    non_expandable: tuple[RankedCandidate, ...]
    unscanned: tuple[RankedCandidate, ...]
    limited: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_id": RETRIEVAL_BATCH_POLICY_ID,
            "candidates": [row.as_dict() for row in self.candidates],
            "batches": [
                [row.as_dict() for row in batch]
                for batch in self.batches
            ],
            "discarded": [row.as_dict() for row in self.discarded],
            "non_expandable": [row.as_dict() for row in self.non_expandable],
            "unscanned": [row.as_dict() for row in self.unscanned],
            "limited": self.limited,
        }


def plan_candidate_batches(
    ranked: Iterable[RankedCandidate],
    *,
    discard_threshold: float = DEFAULT_DISCARD_THRESHOLD,
    expand_threshold: float = DEFAULT_EXPAND_THRESHOLD,
    max_candidate_batches: int | None = None,
    batch_size: int = TOOL_BATCH_SIZE,
) -> CandidateBatchPlan:
    """Plan a fixed-five first batch and threshold-gated continuation batches.

    First-batch invariant（AGENTS.md「工具上下文与候选扫描不变量」）：
    retrieval 按 rank 稳定批次处理，每批至多 5；不得因阈值把首批砍到 1-2——
    **首批恒为 rank 前 5**（仅当目录本身不足 5 个时短少）。

    ``discard`` 的职责是 all-or-nothing 的可用性闸门：没有任何候选过 discard
    时返回空批次（调用方走 retrieval_no_match 终端）；只要有一个过闸，
    首批就无条件取 rank 前 5。``expand`` 仅控制 rank>5 的候选进入后续批次。
    """

    discard = float(discard_threshold)
    expand = float(expand_threshold)
    if not (0.0 <= discard <= 1.0 and 0.0 <= expand <= 1.0):
        raise ValueError("retrieval thresholds must be in [0,1]")
    if expand < discard:
        raise ValueError("retrieval_expand_threshold must be >= retrieval_discard_threshold")
    if int(batch_size) != TOOL_BATCH_SIZE:
        raise ValueError("runtime v2 tool batches must contain at most five tools")
    if max_candidate_batches is not None and int(max_candidate_batches) < 1:
        raise ValueError("max_candidate_batches must be >= 1 or null")

    ordered = tuple(sorted(ranked, key=lambda row: (-row.raw_score, row.tool_id)))
    discarded = tuple(row for row in ordered if row.relevance < discard)
    # discard 只做 all-or-nothing 可用性闸门：全灭 → 空批次（no_match 终端）；
    # 有任一过闸 → 首批恒为 rank 前 5（不得砍到 1-2，目录不足 5 个时为实际数量）
    eligible = tuple(row for row in ordered if row.relevance >= discard)
    if not eligible:
        first: tuple[RankedCandidate, ...] = ()
        expandable: tuple[RankedCandidate, ...] = ()
        non_expandable: tuple[RankedCandidate, ...] = ()
    else:
        first = ordered[:TOOL_BATCH_SIZE]
        # 续批候选与旧语义一致：rank>5 且过 discard 才有资格；其中过 expand 的才可扫描
        tail_eligible = tuple(
            row for row in ordered[TOOL_BATCH_SIZE:] if row.relevance >= discard
        )
        expandable = tuple(row for row in tail_eligible if row.relevance >= expand)
        non_expandable = tuple(row for row in tail_eligible if row.relevance < expand)

    all_batches: list[tuple[RankedCandidate, ...]] = []
    if first:
        all_batches.append(first)
    for offset in range(0, len(expandable), TOOL_BATCH_SIZE):
        all_batches.append(expandable[offset : offset + TOOL_BATCH_SIZE])

    if max_candidate_batches is None:
        used = tuple(all_batches)
        unscanned: tuple[RankedCandidate, ...] = ()
    else:
        limit = int(max_candidate_batches)
        used = tuple(all_batches[:limit])
        unscanned = tuple(row for batch in all_batches[limit:] for row in batch)
    return CandidateBatchPlan(
        # candidates = 模型可见的主候选集 = 恒定的首批（rank 前 5）；
        # 后续批次在 batches/non_expandable 中单独呈现
        candidates=first,
        batches=used,
        discarded=discarded,
        non_expandable=non_expandable,
        unscanned=unscanned,
        limited=bool(unscanned),
    )


def compact_tool(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(tool.get("name") or ""),
        "description": str(tool.get("description") or ""),
        "parameters": copy.deepcopy(
            tool.get("parameters") or {"type": "object", "properties": {}}
        ),
    }


def _strip_annotations(value: Any, *, property_map: bool = False) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, child in value.items():
            # Keys inside a JSON Schema `properties` map are user-defined
            # parameter names.  A parameter named `title`, `default`, or
            # `examples` is structural and must never be mistaken for the
            # annotation keyword with the same spelling.
            if not property_map and (key in _DROP_ANNOTATIONS or key == "description"):
                continue
            output[str(key)] = _strip_annotations(
                child,
                property_map=(not property_map and key == "properties" and isinstance(child, dict)),
            )
        return output
    if isinstance(value, list):
        return [_strip_annotations(child) for child in value]
    return copy.deepcopy(value)


def structural_tool_projection(tool: dict[str, Any]) -> dict[str, Any]:
    """Return the non-truncatable execution semantics visible to the model."""

    return {
        "name": str(tool.get("name") or ""),
        "description": "",
        "parameters": _strip_annotations(
            tool.get("parameters") or {"type": "object", "properties": {}}
        ),
    }


@dataclass(frozen=True)
class _DescriptionField:
    tool_index: int
    path: tuple[str | int, ...]
    text: str
    weight: float


def _description_fields(
    value: Any,
    *,
    tool_index: int,
    path: tuple[str | int, ...],
    rank_weight: float,
    required: bool = False,
) -> list[_DescriptionField]:
    fields: list[_DescriptionField] = []
    if isinstance(value, dict):
        text = value.get("description")
        if isinstance(text, str) and text:
            priority = 2.5 if required else 1.0
            fields.append(
                _DescriptionField(tool_index, path + ("description",), text, rank_weight * priority)
            )
        required_names = {str(item) for item in value.get("required", []) if isinstance(item, str)}
        for key, child in value.items():
            # Annotation subtrees are removed from the structural projection,
            # so descriptions nested inside them must not become allocation
            # targets.  Otherwise materialization tries to write through a
            # path (for example title.description) that no longer exists.
            if key == "description" or key in _DROP_ANNOTATIONS:
                continue
            child_required = bool(key == "properties")
            if key == "properties" and isinstance(child, dict):
                for prop_name, prop in child.items():
                    fields.extend(
                        _description_fields(
                            prop,
                            tool_index=tool_index,
                            path=path + (key, prop_name),
                            rank_weight=rank_weight,
                            required=str(prop_name) in required_names,
                        )
                    )
                continue
            fields.extend(
                _description_fields(
                    child,
                    tool_index=tool_index,
                    path=path + (key,),
                    rank_weight=rank_weight,
                    required=required if child_required else False,
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            fields.extend(
                _description_fields(
                    child,
                    tool_index=tool_index,
                    path=path + (index,),
                    rank_weight=rank_weight,
                    required=required,
                )
            )
    return fields


def _set_path(target: Any, path: Sequence[str | int], value: Any) -> None:
    cursor = target
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def _clip_prefix(tokenizer: Any, text: str, tokens: int, *, marker: str = "…") -> str:
    ids = _encode(tokenizer, text)
    if len(ids) <= tokens:
        return text
    if tokens <= 0:
        return ""
    marker_ids = _encode(tokenizer, marker)
    if len(marker_ids) >= tokens:
        return _decode(tokenizer, ids[:tokens])
    return _decode(tokenizer, ids[: tokens - len(marker_ids)]) + marker


def clip_head_tail(tokenizer: Any, text: str, tokens: int, *, marker: str = "…已裁剪…") -> str:
    """Tokenizer-safe last-resort query clipping which preserves both ends."""

    ids = _encode(tokenizer, str(text))
    if len(ids) <= tokens:
        return str(text)
    if tokens <= 0:
        return ""
    marker_ids = _encode(tokenizer, marker)
    if len(marker_ids) >= tokens:
        return _decode(tokenizer, ids[:tokens])
    remaining = tokens - len(marker_ids)
    left = (remaining + 1) // 2
    right = remaining - left
    tail = _decode(tokenizer, ids[-right:]) if right else ""
    return _decode(tokenizer, ids[:left]) + marker + tail


def render_projected_tools(tools: Sequence[dict[str, Any]]) -> str:
    return "<tools>" + _canonical(list(tools)) + "</tools>"


def minimum_tool_projection_tokens(
    tokenizer: Any, tool: dict[str, Any], *, stable_overhead: str = ""
) -> int:
    """Measure the immutable one-tool structure used for profile eligibility."""

    rendered = stable_overhead + render_projected_tools([structural_tool_projection(tool)])
    return token_count(tokenizer, rendered, add_bos=True)


@dataclass(frozen=True)
class ToolProjection:
    tools: tuple[dict[str, Any], ...]
    full_tools: tuple[dict[str, Any], ...]
    dropped_tool_ids: tuple[str, ...]
    rendered: str
    tokens: int
    cap: int
    profile: str
    compression_level: str
    omitted_description_tokens: int
    projection_sha256: str
    context_unrepresentable: bool

    def as_budget_dict(self) -> dict[str, Any]:
        return {
            "packer_id": CONTEXT_PACKER_ID,
            "projection_serializer_id": PROJECTION_SERIALIZER_ID,
            "profile": self.profile,
            "cap": self.cap,
            "used": self.tokens,
            "compression_level": self.compression_level,
            "omitted_description_tokens": self.omitted_description_tokens,
            "dropped_tools": list(self.dropped_tool_ids),
            "context_unrepresentable": self.context_unrepresentable,
        }


def project_tool_batch(
    tokenizer: Any,
    tools: Sequence[dict[str, Any]],
    *,
    relevances: Sequence[float] | None = None,
    profile: str = "standard",
    stable_overhead: str = "",
    stable_cap: int | None = None,
) -> ToolProjection:
    """Fit up to five model-visible tool schemas without weakening constraints."""

    if len(tools) > TOOL_BATCH_SIZE:
        raise ValueError("a projected tool batch may contain at most five tools")
    if profile not in PROFILE_STABLE_CAPS:
        raise ValueError(f"unknown runtime profile: {profile}")
    cap = int(stable_cap if stable_cap is not None else PROFILE_STABLE_CAPS[profile])
    scores = list(relevances or [1.0] * len(tools))
    if len(scores) != len(tools):
        raise ValueError("one retrieval relevance is required per tool")
    full = [compact_tool(tool) for tool in tools]

    def render(rows: Sequence[dict[str, Any]]) -> str:
        return stable_overhead + render_projected_tools(rows)

    full_rendered = render(full)
    full_tokens = token_count(tokenizer, full_rendered, add_bos=True)
    if full_tokens <= cap:
        return ToolProjection(
            tools=tuple(full),
            full_tools=tuple(full),
            dropped_tool_ids=(),
            rendered=full_rendered,
            tokens=full_tokens,
            cap=cap,
            profile=profile,
            compression_level="none",
            omitted_description_tokens=0,
            projection_sha256=_sha256(full),
            context_unrepresentable=False,
        )

    kept_full = list(full)
    skeletons = [structural_tool_projection(tool) for tool in kept_full]
    dropped: list[str] = []
    while skeletons and token_count(tokenizer, render(skeletons), add_bos=True) > cap:
        dropped.append(str(kept_full[-1].get("name") or ""))
        kept_full.pop()
        skeletons.pop()

    if not skeletons:
        empty_rendered = render([])
        return ToolProjection(
            tools=(),
            full_tools=(),
            dropped_tool_ids=tuple(dropped),
            rendered=empty_rendered,
            tokens=token_count(tokenizer, empty_rendered, add_bos=True),
            cap=cap,
            profile=profile,
            compression_level="unrepresentable",
            omitted_description_tokens=sum(
                token_count(tokenizer, str(tool.get("description") or "")) for tool in full
            ),
            projection_sha256=_sha256([]),
            context_unrepresentable=True,
        )

    # Start with the immutable structural skeleton, then distribute description
    # tokens in deterministic weighted increments.  The final shrink loop uses
    # exact tokenizer counts, so JSON punctuation overhead can never overflow.
    projected = copy.deepcopy(skeletons)
    fields: list[_DescriptionField] = []
    for index, tool in enumerate(kept_full):
        rank_weight = (len(kept_full) - index) * max(0.05, float(scores[index]))
        root_description = str(tool.get("description") or "")
        if root_description:
            fields.append(_DescriptionField(index, ("description",), root_description, rank_weight * 2.0))
        fields.extend(
            _description_fields(
                tool.get("parameters") or {},
                tool_index=index,
                path=("parameters",),
                rank_weight=rank_weight,
            )
        )
    full_lengths = [token_count(tokenizer, field.text) for field in fields]
    allocations = [0] * len(fields)
    base_tokens = token_count(tokenizer, render(projected), add_bos=True)
    token_budget = max(0, cap - base_tokens)
    # Leave a small deterministic allowance for description keys/JSON quoting.
    allocatable = max(0, token_budget - 2 * len(fields))
    for _ in range(allocatable):
        choices = [
            (field.weight / (allocations[index] + 1), -index, index)
            for index, field in enumerate(fields)
            if allocations[index] < full_lengths[index]
        ]
        if not choices:
            break
        _, _, winner = max(choices)
        allocations[winner] += 1

    def materialize() -> tuple[list[dict[str, Any]], str, int]:
        rows = copy.deepcopy(skeletons)
        for field, amount in zip(fields, allocations):
            if amount <= 0:
                continue
            _set_path(rows[field.tool_index], field.path, _clip_prefix(tokenizer, field.text, amount))
        rendered = render(rows)
        return rows, rendered, token_count(tokenizer, rendered, add_bos=True)

    projected, projected_rendered, projected_tokens = materialize()
    while projected_tokens > cap and any(allocations):
        choices = [
            (fields[index].weight / allocations[index], index)
            for index in range(len(fields))
            if allocations[index] > 0
        ]
        _, loser = min(choices)
        allocations[loser] -= 1
        projected, projected_rendered, projected_tokens = materialize()

    omitted = sum(full_lengths) - sum(allocations)
    level = "structural" if not any(allocations) else "descriptions"
    if dropped:
        level = "reduced_batch"
    return ToolProjection(
        tools=tuple(projected),
        full_tools=tuple(kept_full),
        dropped_tool_ids=tuple(dropped),
        rendered=projected_rendered,
        tokens=projected_tokens,
        cap=cap,
        profile=profile,
        compression_level=level,
        omitted_description_tokens=omitted,
        projection_sha256=_sha256(projected),
        context_unrepresentable=False,
    )


def validate_prompt_budget(
    tokenizer: Any,
    prompt: str,
    *,
    output_reserve: int = DEFAULT_OUTPUT_RESERVE,
    max_context: int = MAX_CONTEXT_TOKENS,
) -> int:
    reserve = int(output_reserve)
    if reserve < 0 or reserve > int(max_context):
        raise ValueError("invalid output reserve")
    tokens = token_count(tokenizer, prompt, add_bos=True)
    if tokens + reserve > int(max_context):
        raise ValueError("internal_context_budget_invariant")
    return tokens
