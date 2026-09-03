"""Imports the repository's architecture-neutral runtime semantic modules."""

from __future__ import annotations

import sys
from pathlib import Path

from .version import SDK_ROOT

SHARED_ROOT = SDK_ROOT / "runtime"
if str(SHARED_ROOT) not in sys.path:
    sys.path.insert(0, str(SHARED_ROOT))

from byte_grammar import (  # noqa: E402,F401
    GRAMMAR_ID,
    allowed_token_ids,
    compile_byte_grammar_cached,
    is_accept_bytes,
    is_legal_byte_prefix,
    parse_call_text,
    token_to_bytes,
)
from context_budget import (  # noqa: E402,F401
    CONTEXT_PACKER_ID,
    DEFAULT_DISCARD_THRESHOLD,
    DEFAULT_EXPAND_THRESHOLD,
    MAX_PROMPT_TOKENS,
    PROFILE_STABLE_CAPS,
    PROJECTION_SERIALIZER_ID,
    RETRIEVAL_BATCH_POLICY_ID,
    RETRIEVAL_CALIBRATION_ID,
    TOOL_BATCH_SIZE,
    CandidateBatchPlan,
    RankedCandidate,
    ToolProjection,
    clip_head_tail,
    fit_platt_calibrator,
    minimum_tool_projection_tokens,
    plan_candidate_batches,
    platt_relevance,
    project_tool_batch,
    select_retrieval_thresholds,
    token_count,
    validate_prompt_budget,
)
from kv_manager import (  # noqa: E402,F401
    DEFAULT_OUTPUT_RESERVE,
    MAX_CONTEXT,
    ORDINARY_CAP,
    SINK_CAP,
    BoundedKVManager,
    KVManager,
)
from narration import (  # noqa: E402,F401
    NARRATION_ADAPTER_ID,
    NarrationProvider,
    narration_prompt_v2,
    verified_result_view,
)
from provenance import (  # noqa: E402,F401
    ESCALATE_LOW,
    EXECUTE_HIGH,
    VALIDATION_ORDER,
    validate_generated_call,
    verified_tool_result,
)
from schema_subset import (  # noqa: E402,F401
    SCHEMA_SUBSET_ID,
    UnsupportedSchemaError,
    validate_arguments,
    validate_tool,
    validate_tools,
)
from tool_index import ToolIndex, catalog_fingerprint, index_fingerprint  # noqa: E402,F401

__all__ = [
    "BoundedKVManager",
    "CONTEXT_PACKER_ID",
    "CandidateBatchPlan",
    "DEFAULT_OUTPUT_RESERVE",
    "DEFAULT_DISCARD_THRESHOLD",
    "DEFAULT_EXPAND_THRESHOLD",
    "ESCALATE_LOW",
    "EXECUTE_HIGH",
    "GRAMMAR_ID",
    "KVManager",
    "MAX_CONTEXT",
    "MAX_PROMPT_TOKENS",
    "NARRATION_ADAPTER_ID",
    "ORDINARY_CAP",
    "PROFILE_STABLE_CAPS",
    "PROJECTION_SERIALIZER_ID",
    "RETRIEVAL_BATCH_POLICY_ID",
    "RETRIEVAL_CALIBRATION_ID",
    "RankedCandidate",
    "SCHEMA_SUBSET_ID",
    "SHARED_ROOT",
    "SINK_CAP",
    "TOOL_BATCH_SIZE",
    "ToolIndex",
    "ToolProjection",
    "UnsupportedSchemaError",
    "VALIDATION_ORDER",
    "allowed_token_ids",
    "catalog_fingerprint",
    "clip_head_tail",
    "fit_platt_calibrator",
    "compile_byte_grammar_cached",
    "index_fingerprint",
    "is_accept_bytes",
    "is_legal_byte_prefix",
    "narration_prompt_v2",
    "minimum_tool_projection_tokens",
    "parse_call_text",
    "plan_candidate_batches",
    "platt_relevance",
    "project_tool_batch",
    "select_retrieval_thresholds",
    "token_count",
    "token_to_bytes",
    "validate_arguments",
    "validate_generated_call",
    "validate_tool",
    "validate_tools",
    "verified_tool_result",
    "validate_prompt_budget",
]
