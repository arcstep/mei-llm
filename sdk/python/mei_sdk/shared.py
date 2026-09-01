"""Imports the repository's architecture-neutral runtime semantic modules."""

from __future__ import annotations

import sys
from pathlib import Path

from .version import SDK_ROOT

SHARED_ROOT = SDK_ROOT.parent / "runtime" / "_shared"
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
    "DEFAULT_OUTPUT_RESERVE",
    "ESCALATE_LOW",
    "EXECUTE_HIGH",
    "GRAMMAR_ID",
    "KVManager",
    "MAX_CONTEXT",
    "NARRATION_ADAPTER_ID",
    "ORDINARY_CAP",
    "SCHEMA_SUBSET_ID",
    "SHARED_ROOT",
    "SINK_CAP",
    "ToolIndex",
    "UnsupportedSchemaError",
    "VALIDATION_ORDER",
    "allowed_token_ids",
    "catalog_fingerprint",
    "compile_byte_grammar_cached",
    "index_fingerprint",
    "is_accept_bytes",
    "is_legal_byte_prefix",
    "narration_prompt_v2",
    "parse_call_text",
    "token_to_bytes",
    "validate_arguments",
    "validate_generated_call",
    "validate_tool",
    "validate_tools",
    "verified_tool_result",
]
