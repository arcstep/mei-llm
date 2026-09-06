#!/usr/bin/env python3
"""Runtime-neutral producers for immutable mei-51m longitudinal scorecards."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import evaluation.tool_use.longitudinal_eval_metrics_51m as metrics
import contracts.sft_v4_contract_51m as contract
import training.tool_use.sft_v3_training_51m as training


EVALUATOR_ID = "mei-51m-longitudinal-runtime-evaluator-v5"


def _ensure_sdk_path() -> None:
    path = str(contract.ROOT / "src/platform/python-sdk")
    if path not in sys.path:
        sys.path.insert(0, path)


def catalog_from_lock(lock_dir: Path) -> list[dict[str, Any]]:
    lock = metrics.verify_lock(lock_dir)
    document = contract.load_json(lock_dir / "tool-universe.json")
    tools = [contract.compact_tool(tool) for tool in document.get("tools") or []]
    if len(tools) != 147 or len({str(tool["name"]) for tool in tools}) != 147:
        raise RuntimeError("portable deploy catalog must contain 147 unique tools")
    expected = (lock.get("artifacts") or {}).get("tool-universe.json") or {}
    if contract.sha_file(lock_dir / "tool-universe.json") != expected.get("sha256"):
        raise RuntimeError("eval tool universe hash drift")
    return tools


def _lexical_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.casefold())
        if token
    }


def lexical_top5(query: str, catalog: Sequence[dict[str, Any]]) -> list[str]:
    query_tokens = _lexical_tokens(query)
    scored: list[tuple[float, str]] = []
    for tool in catalog:
        name = str(tool.get("name") or "")
        body = name + " " + str(tool.get("description") or "")
        tool_tokens = _lexical_tokens(body)
        union = query_tokens | tool_tokens
        score = len(query_tokens & tool_tokens) / len(union) if union else 0.0
        scored.append((score, name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored[:5]]


def balanced_fullcall_sample(
    rows: Sequence[dict[str, Any]], *, per_kind_per_tool: int = 1
) -> list[dict[str, Any]]:
    """Select the same bounded tool×kind control slice for every Base."""

    if per_kind_per_tool <= 0:
        raise ValueError("per_kind_per_tool must be positive")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("candidate_tool") or row.get("gold_name") or ""),
                str(row.get("kind") or ""),
            )
        ].append(row)
    tools = sorted({key[0] for key in groups})
    if len(tools) != 147 or any(
        len(groups.get((tool, kind), [])) < per_kind_per_tool
        for tool in tools
        for kind in ("execute", "refuse")
    ):
        raise RuntimeError("balanced full-call sample requires every tool and both labels")
    return [
        groups[(tool, kind)][offset]
        for tool in tools
        for kind in ("execute", "refuse")
        for offset in range(per_kind_per_tool)
    ]


def evaluate_retrieval(
    runtime: Any,
    rows: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    *,
    progress_every: int = 100,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        learned = runtime.search_top_k(str(row.get("query") or ""), catalog, k=5)
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "ranked_tools": [str(tool.get("name") or "") for tool in learned],
                "lexical_ranked_tools": lexical_top5(str(row.get("query") or ""), catalog),
            }
        )
        if index == 0 or (index + 1) % progress_every == 0:
            print(
                contract.canonical_bytes(
                    {"retrieval_eval": index + 1, "total": len(rows)}
                ).decode(),
                flush=True,
            )
    report = metrics.retrieval_metrics(rows, predictions)
    report.update(
        {
            "evaluator_id": EVALUATOR_ID,
            "elapsed_seconds": time.perf_counter() - started,
            "retrieval_encoding_id": contract.RETRIEVAL_ENCODING_ID,
            "retrieval_max_tokens": contract.RETRIEVAL_MAX_TOKENS,
        }
    )
    return report, predictions


def _validated_turn(validated: dict[str, Any]) -> dict[str, Any]:
    calls = validated.get("function_calls") or []
    if validated.get("ok") is True and validated.get("refuse") is True:
        return {"kind": "refuse"}
    if validated.get("ok") is True and len(calls) == 1:
        return {"kind": "call", "call": calls[0]}
    return {
        "kind": "error",
        "error": str(validated.get("error") or "runtime_validation_failed"),
    }


def _parsed_turn(parsed: dict[str, Any]) -> dict[str, Any]:
    """Project grammar/schema output without conflating later runtime gates."""

    calls = parsed.get("function_calls") or []
    if parsed.get("ok") is True and parsed.get("refuse") is True:
        return {"kind": "refuse"}
    if parsed.get("ok") is True and len(calls) == 1:
        return {"kind": "call", "call": calls[0]}
    return {
        "kind": "error",
        "error": str(parsed.get("error") or "model_output_invalid"),
    }


def _error_counts(
    predictions: Sequence[dict[str, Any]], field: str
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for prediction in predictions:
        value = prediction.get(field)
        if value:
            counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _budget_turn(exc: training.ToolSchemaBudgetExceeded) -> dict[str, Any]:
    return {
        "kind": "error",
        "error": exc.code,
        "details": exc.as_error(),
    }


def evaluate_fullcall(
    runtime: Any,
    rows: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    *,
    retrieval_mode: str,
    limit: int | None = None,
    progress_every: int = 25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _ensure_sdk_path()
    from mei_sdk.protocol import normalize_request
    from mei_sdk.shared import parse_call_text, validate_generated_call

    selected_rows = rows[:limit] if limit is not None else rows
    tools_by_name = {str(tool["name"]): tool for tool in catalog}
    predictions: list[dict[str, Any]] = []
    total_output_tokens = 0
    total_decode_ms = 0.0
    started = time.perf_counter()
    for index, row in enumerate(selected_rows):
        if retrieval_mode == "oracle_top5":
            visible = [
                tools_by_name[str(item.get("name") if isinstance(item, dict) else item)]
                for item in row.get("oracle_top5") or []
                if str(item.get("name") if isinstance(item, dict) else item) in tools_by_name
            ]
        elif retrieval_mode == "learned_top5":
            visible = runtime.search_top_k(str(row.get("query") or ""), catalog, k=5)
        elif retrieval_mode == "no_retrieval":
            visible = list(catalog[:5])
        else:
            raise ValueError(f"unknown full-call retrieval mode: {retrieval_mode}")
        if len(visible) != 5:
            raise RuntimeError(f"full-call eval row lacks five schemas: {row.get('sample_id')}")
        prompt_row = dict(row)
        prompt_row["retrieved_tools"] = [str(tool["name"]) for tool in visible]
        deterministic_error: dict[str, Any] | None = None
        try:
            prompt, _answer, _stats = training.encode_fullcall_row(
                runtime.tokenizer, prompt_row, tools_by_name
            )
        except training.ToolSchemaBudgetExceeded as exc:
            deterministic_error = exc.as_error()
            decoded = {"text": "", "n_out": 0, "decode_ms": 0.0}
            model_turn = _budget_turn(exc)
            pipeline_turn = _budget_turn(exc)
            pipeline_validation_error = exc.code
        else:
            decoded = runtime.greedy(
                prompt, tools=visible, max_new=128, decode_mode="constrained"
            )
            decoded_text = str(decoded.get("text") or "")
            parsed = parse_call_text(decoded_text, visible)
            model_turn = _parsed_turn(parsed)
            request = normalize_request(training.deployment_request_v3(row))
            validated = validate_generated_call(
                decoded_text,
                tools=visible,
                request=request,
                confidence=None,
                enforce_confidence=False,
            )
            pipeline_turn = _validated_turn(validated)
            pipeline_validation_error = validated.get("error")
        predictions.append(
            {
                "sample_id": row["sample_id"],
                # ``result`` is the model-task result used by the frozen
                # full-call metric contract. Runtime eligibility remains a
                # separate, fail-closed diagnostic and never changes it.
                "result": model_turn,
                "pipeline_result": pipeline_turn,
                "pipeline_validation_error": pipeline_validation_error,
                "retrieval_mode": retrieval_mode,
                "retrieved_tools": [str(tool["name"]) for tool in visible],
                "deterministic_error": deterministic_error,
                "decode_text_sha256": contract.sha_bytes(
                    str(decoded.get("text") or "").encode("utf-8")
                ),
                "n_out": int(decoded.get("n_out") or 0),
                "decode_ms": float(decoded.get("decode_ms") or 0.0),
            }
        )
        total_output_tokens += int(decoded.get("n_out") or 0)
        total_decode_ms += float(decoded.get("decode_ms") or 0.0)
        if index == 0 or (index + 1) % progress_every == 0:
            print(
                contract.canonical_bytes(
                    {
                        "fullcall_eval": index + 1,
                        "total": len(selected_rows),
                        "mode": retrieval_mode,
                    }
                ).decode(),
                flush=True,
            )
    gold = selected_rows
    report = metrics.fullcall_metrics(gold, predictions, catalog)
    pipeline_predictions = [
        {**prediction, "result": prediction["pipeline_result"]}
        for prediction in predictions
    ]
    pipeline_report = metrics.fullcall_metrics(gold, pipeline_predictions, catalog)
    report.update(
        {
            "evaluator_id": EVALUATOR_ID,
            "retrieval_mode": retrieval_mode,
            "elapsed_seconds": time.perf_counter() - started,
            "decode_tokens_per_second": total_output_tokens
            / max(total_decode_ms / 1000.0, 1e-9),
            "prompt_framing_id": contract.PROMPT_FRAMING_ID,
            "serializer_id": contract.SERIALIZER_ID,
            "deterministic_gates_applied": True,
            "quality_scope": "model-output-after-grammar-and-schema",
            "model_scoring_contract": "exact-call-or-correct-refusal-v3",
            "pipeline_metrics": pipeline_report,
            "pipeline_scoring_contract": "strict-wire-v2-runtime-gates-v1",
            "pipeline_validation_error_counts": _error_counts(
                predictions, "pipeline_validation_error"
            ),
            "tool_schema_budget_error_count": sum(
                prediction.get("deterministic_error") is not None
                for prediction in predictions
            ),
            "tool_schema_budget_error_rate": sum(
                prediction.get("deterministic_error") is not None
                for prediction in predictions
            )
            / max(len(predictions), 1),
        }
    )
    return report, predictions


def evaluate_mw_disposition(
    runtime: Any,
    rows: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    *,
    retrieval_mode: str,
    limit: int | None = None,
    progress_every: int = 100,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import mlx.core as mx

    head = getattr(runtime, "mw_disposition_head", None)
    if head is None:
        raise RuntimeError("MW disposition head is unavailable")
    selected_rows = rows[:limit] if limit is not None else rows
    tools_by_name = {str(tool["name"]): tool for tool in catalog}
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(selected_rows):
        if retrieval_mode == "oracle_top5":
            # Frozen MW-v7 rows use ``retrieved_tools`` as the canonical
            # five-schema oracle view. Adopted historical rows may also
            # retain ``oracle_top5``; clean supplemental rows intentionally
            # do not. Keep evaluation aligned with ``mw_training_view``.
            visible = [
                tools_by_name[str(item.get("name") if isinstance(item, dict) else item)]
                for item in row.get("oracle_top5") or row.get("retrieved_tools") or []
                if str(item.get("name") if isinstance(item, dict) else item) in tools_by_name
            ]
        elif retrieval_mode == "learned_top5":
            visible = runtime.search_top_k(str(row.get("query") or ""), catalog, k=5)
        else:
            raise ValueError(f"unknown MW retrieval mode: {retrieval_mode}")
        rendered = training.render_mw_prompt_parts(row, visible)
        try:
            ids, _stats = training.encode_stable_ring_prompt(
                runtime.tokenizer,
                rendered,
                sample_id=str(row.get("sample_id") or "") or None,
            )
        except training.ToolSchemaBudgetExceeded as exc:
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "result": _budget_turn(exc),
                    "prediction_error": exc.code,
                    "deterministic_error": exc.as_error(),
                    "retrieval_mode": retrieval_mode,
                }
            )
        else:
            cells = runtime.model(mx.array([ids], dtype=mx.int32), return_cells=True)["cells"]
            logits = head(cells).astype(mx.float32)
            mx.eval(logits)
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "predicted_class_id": int(mx.argmax(logits[0]).item()),
                    "deterministic_error": None,
                    "retrieval_mode": retrieval_mode,
                }
            )
        if index == 0 or (index + 1) % progress_every == 0:
            print(
                contract.canonical_bytes(
                    {"mw_eval": index + 1, "total": len(selected_rows), "mode": retrieval_mode}
                ).decode(),
                flush=True,
            )
    report = metrics.mw_metrics(selected_rows, predictions)
    report.update(
        {
            "evaluator_id": EVALUATOR_ID,
            "retrieval_mode": retrieval_mode,
            "elapsed_seconds": time.perf_counter() - started,
            "semantic_boundary": "independent-20class-sidecar-not-mw-deviation-gate",
            "prompt_id": training.MW_PROMPT_ID,
            "tool_schema_budget_error_count": sum(
                prediction.get("deterministic_error") is not None
                for prediction in predictions
            ),
            "tool_schema_budget_error_rate": sum(
                prediction.get("deterministic_error") is not None
                for prediction in predictions
            )
            / max(len(predictions), 1),
        }
    )
    return report, predictions


def evaluate_frozen_mw_batches(
    runtime: Any,
    rows: list[dict[str, Any]],
    *,
    limit: int | None = None,
    progress_every: int = 100,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate the independent 20-class head on frozen budgeted prompts."""

    import mlx.core as mx

    head = getattr(runtime, "mw_disposition_head", None)
    if head is None:
        raise RuntimeError("MW disposition head is unavailable")
    eligible = [row for row in rows if row.get("mw_eligible") is True]
    selected = eligible[:limit] if limit is not None else eligible
    if not selected:
        raise RuntimeError("frozen MW evaluation has no eligible rows")
    gold_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(selected):
        ids = runtime.tokenizer.encode(
            str(row.get("prompt") or ""), add_bos=True, add_eos=False
        )
        if len(ids) != int(row.get("prompt_tokens") or -1):
            raise RuntimeError(f"frozen MW prompt token drift: {row.get('view_id')}")
        if len(ids) + 128 > 2048:
            raise RuntimeError(f"frozen MW prompt exceeds joint budget: {row.get('view_id')}")
        cells = runtime.model(mx.array([ids], dtype=mx.int32), return_cells=True)["cells"]
        logits = head(cells).astype(mx.float32)
        mx.eval(logits)
        sample_id = str(row.get("view_id") or "")
        label = int(row["effective_reason_class_id"])
        gold_rows.append(
            {
                "sample_id": sample_id,
                "reason_class_id": label,
                "retrieval_mode": row.get("retrieval_mode"),
                "runtime_profile": row.get("runtime_profile"),
            }
        )
        predictions.append(
            {
                "sample_id": sample_id,
                "predicted_class_id": int(mx.argmax(logits[0]).item()),
                "deterministic_error": None,
                "retrieval_mode": row.get("retrieval_mode"),
                "runtime_profile": row.get("runtime_profile"),
                "batch_index": row.get("batch_index"),
            }
        )
        if index == 0 or (index + 1) % progress_every == 0:
            print(
                contract.canonical_bytes(
                    {"mw_frozen_eval": index + 1, "total": len(selected)}
                ).decode(),
                flush=True,
            )
    report = metrics.mw_metrics(gold_rows, predictions)
    report.update(
        {
            "evaluator_id": EVALUATOR_ID,
            "retrieval_mode": "frozen_oracle_learned_profiles",
            "elapsed_seconds": time.perf_counter() - started,
            "semantic_boundary": "independent-20class-sidecar-not-mw-deviation-gate",
            "prompt_id": training.MW_PROMPT_ID,
            "tool_schema_budget_error_count": 0,
            "tool_schema_budget_error_rate": 0.0,
            "frozen_prompt_rows": len(selected),
            "live_retrieval_calls": 0,
        }
    )
    return report, predictions


def _evaluation_call_id(
    trajectory_id: str, step: int, call: Mapping[str, Any]
) -> str:
    nonce = hashlib.sha256(trajectory_id.encode("utf-8")).hexdigest()[:8]
    digest = hashlib.sha256(
        contract.canonical_bytes([trajectory_id, step, dict(call)])
    ).hexdigest()[:12]
    return f"call-s{nonce}-{step}-{digest}"


def _trajectory_fixture_result(
    rows: Sequence[dict[str, Any]], result_index: int, call_id: str
) -> dict[str, Any] | None:
    for row in reversed(rows):
        results = list(row.get("prior_tool_results") or row.get("tool_results") or [])
        if result_index >= len(results):
            continue
        result = dict(results[result_index])
        result["call_id"] = call_id
        provenance = dict(result.get("provenance") or {})
        provenance.update(
            {
                "source": "frozen-eval-host-simulator:mei-agent-host-simulator-v1",
                "verified": True,
            }
        )
        result["provenance"] = provenance
        result["wire_version"] = contract.WIRE_ID
        return result
    return None


def evaluate_multistep(
    runtime: Any,
    rows: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    *,
    retrieval_mode: str = "learned_top5",
    trajectory_limit: int | None = None,
    progress_every: int = 25,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run the frozen Agent bank as call→result→continue trajectories.

    A later step receives a fixture result only when every preceding generated
    call was exact. This prevents gold continuation context from hiding an
    earlier model failure while preserving deterministic offline execution.
    """

    _ensure_sdk_path()
    from mei_sdk.protocol import normalize_request
    from mei_sdk.shared import parse_call_text, validate_generated_call

    tools_by_name = {str(tool["name"]): tool for tool in catalog}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trajectory_id = str(row.get("trajectory_id") or row.get("cf_group") or "")
        if not trajectory_id:
            raise RuntimeError("multistep row lacks trajectory identity")
        grouped[trajectory_id].append(row)
    trajectory_ids = sorted(grouped)
    if trajectory_limit is not None:
        trajectory_ids = trajectory_ids[:trajectory_limit]
    predictions: list[dict[str, Any]] = []
    total_output_tokens = 0
    total_decode_ms = 0.0
    completed = 0
    started = time.perf_counter()
    for trajectory_id in trajectory_ids:
        trajectory = sorted(
            grouped[trajectory_id], key=lambda row: int(row.get("trajectory_step") or 0)
        )
        actual_calls: list[dict[str, Any]] = []
        actual_results: list[dict[str, Any]] = []
        prefix_exact = True
        for row in trajectory:
            if not prefix_exact:
                predictions.append(
                    {
                        "sample_id": row["sample_id"],
                        "result": {"kind": "error", "error": "prior_step_inexact"},
                        "pipeline_result": {
                            "kind": "error",
                            "error": "prior_step_inexact",
                        },
                        "pipeline_validation_error": "prior_step_inexact",
                        "retrieval_mode": retrieval_mode,
                        "retrieved_tools": [],
                    }
                )
                continue
            if retrieval_mode == "oracle_top5":
                names = [
                    str(item.get("name") if isinstance(item, dict) else item)
                    for item in row.get("oracle_top5") or row.get("retrieved_tools") or []
                ][:5]
                visible = [tools_by_name[name] for name in names if name in tools_by_name]
            elif retrieval_mode == "learned_top5":
                visible = runtime.search_top_k(str(row.get("query") or ""), catalog, k=5)
            else:
                raise ValueError(f"unknown multistep retrieval mode: {retrieval_mode}")
            if len(visible) != 5:
                raise RuntimeError(f"multistep row lacks five schemas: {row.get('sample_id')}")
            prompt_row = dict(row)
            prompt_row["retrieved_tools"] = [str(tool["name"]) for tool in visible]
            prompt_row["prior_calls"] = list(actual_calls)
            prompt_row["prior_tool_results"] = list(actual_results)
            prompt_row["tool_results"] = list(actual_results)
            deterministic_error: dict[str, Any] | None = None
            try:
                prompt, _answer, _stats = training.encode_fullcall_row(
                    runtime.tokenizer, prompt_row, tools_by_name
                )
            except training.ToolSchemaBudgetExceeded as exc:
                deterministic_error = exc.as_error()
                decoded = {"text": "", "n_out": 0, "decode_ms": 0.0}
                turn = _budget_turn(exc)
                pipeline_turn = _budget_turn(exc)
                pipeline_validation_error = exc.code
                raw_text = ""
                prefix_exact = False
            else:
                decoded = runtime.greedy(
                    prompt, tools=visible, max_new=128, decode_mode="constrained"
                )
                request = normalize_request(training.deployment_request_v3(prompt_row))
                request["_verified_result_map"] = {
                    str(result["call_id"]): dict(result) for result in actual_results
                }
                raw_text = str(decoded.get("text") or "")
                parsed = parse_call_text(raw_text, visible)
                turn = _parsed_turn(parsed)
                validated = validate_generated_call(
                    raw_text,
                    tools=visible,
                    request=request,
                    confidence=None,
                    enforce_confidence=False,
                )
                pipeline_turn = _validated_turn(validated)
                pipeline_validation_error = validated.get("error")
                if (
                    turn["kind"] == "refuse"
                    and actual_results
                    and all(result.get("status") == "ok" for result in actual_results)
                ):
                    try:
                        if json.loads(raw_text) == [] and parsed.get("error") is None:
                            turn = {"kind": "respond"}
                            if validated.get("error") is None:
                                pipeline_turn = {"kind": "respond"}
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass
            expected = (row.get("answers") or [{}])[0] if row.get("kind") == "execute" else None
            if turn["kind"] == "call" and isinstance(turn.get("call"), dict):
                call = dict(turn["call"])
                call_id = _evaluation_call_id(
                    trajectory_id, len(actual_calls) + 1, call
                )
                call["call_id"] = call_id
                turn["call"] = call
                exact = bool(
                    isinstance(expected, dict)
                    and call.get("name") == expected.get("name")
                    and call.get("arguments") == expected.get("arguments")
                )
                if exact:
                    result = _trajectory_fixture_result(
                        trajectory, len(actual_results), call_id
                    )
                    if result is None:
                        raise RuntimeError(
                            f"trajectory fixture lacks result for {row.get('sample_id')}"
                        )
                    actual_calls.append(call)
                    actual_results.append(result)
                else:
                    prefix_exact = False
            elif row.get("kind") == "execute":
                prefix_exact = False
            predictions.append(
                {
                    "sample_id": row["sample_id"],
                    "result": turn,
                    "pipeline_result": pipeline_turn,
                    "pipeline_validation_error": pipeline_validation_error,
                    "retrieval_mode": retrieval_mode,
                    "retrieved_tools": [str(tool["name"]) for tool in visible],
                    "deterministic_error": deterministic_error,
                    "decode_text_sha256": contract.sha_bytes(raw_text.encode("utf-8")),
                    "n_out": int(decoded.get("n_out") or 0),
                    "decode_ms": float(decoded.get("decode_ms") or 0.0),
                }
            )
            total_output_tokens += int(decoded.get("n_out") or 0)
            total_decode_ms += float(decoded.get("decode_ms") or 0.0)
        completed += 1
        if completed == 1 or completed % progress_every == 0:
            print(
                contract.canonical_bytes(
                    {
                        "multistep_eval": completed,
                        "total_trajectories": len(trajectory_ids),
                        "mode": retrieval_mode,
                    }
                ).decode(),
                flush=True,
            )
    selected_gold = [
        row
        for trajectory_id in trajectory_ids
        for row in sorted(
            grouped[trajectory_id], key=lambda value: int(value.get("trajectory_step") or 0)
        )
    ]
    report = metrics.multistep_metrics(selected_gold, predictions)
    report.update(
        {
            "evaluator_id": EVALUATOR_ID,
            "retrieval_mode": retrieval_mode,
            "elapsed_seconds": time.perf_counter() - started,
            "decode_tokens_per_second": total_output_tokens
            / max(total_decode_ms / 1000.0, 1e-9),
            "closed_loop_fixture_policy": "continue_only_after_exact_prior_call-v1",
            "quality_scope": "model-closed-loop-after-grammar-and-schema",
            "model_scoring_contract": "exact-prefix-call-result-continue-v2",
            "pipeline_scoring_contract": "strict-wire-v2-runtime-gates-diagnostic-v1",
            "pipeline_validation_error_counts": _error_counts(
                predictions, "pipeline_validation_error"
            ),
            "tool_schema_budget_error_count": sum(
                prediction.get("deterministic_error") is not None
                for prediction in predictions
            ),
            "tool_schema_budget_error_rate": sum(
                prediction.get("deterministic_error") is not None
                for prediction in predictions
            )
            / max(len(predictions), 1),
        }
    )
    return report, predictions


def evaluate_confidence(
    runtime: Any,
    outcomes: list[dict[str, Any]],
    calibration: Mapping[str, float],
    *,
    candidates: list[dict[str, Any]] | None = None,
    minimum_class_rows: int = 100,
    progress_every: int = 100,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import mlx.core as mx

    if getattr(runtime, "conf_v2", None) is None:
        raise RuntimeError("confidence head is unavailable")
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, sample in enumerate(outcomes):
        head_eligible = sample.get("head_eligible") is not False
        if head_eligible:
            cells = runtime.model(
                mx.array([sample["prompt_ids"]], dtype=mx.int32), return_cells=True
            )["cells"]
            logit = runtime.conf_v2(cells).reshape(())
            mx.eval(logit)
            raw_score = training.combined_confidence_score(
                float(logit.item()),
                float(sample["logprob_sum"]),
                int(sample["output_tokens"]),
            )
            score = training.apply_platt(raw_score, calibration)
        else:
            # Deterministic validation already rejected the request.  The
            # confidence head is not invoked and cannot override that result.
            raw_score = 0.0
            score = 0.0
        predictions.append(
            {
                "sample_id": sample["sample_id"],
                "source_sample_id": sample.get("source_sample_id"),
                "label": int(sample["label"]),
                "score": score,
                "raw_score": raw_score,
                "expected_kind": sample.get("expected_kind"),
                "candidate_tool": sample.get("candidate_tool"),
                "head_invoked": head_eligible,
                "deterministic_error": sample.get("deterministic_error"),
            }
        )
        if index == 0 or (index + 1) % progress_every == 0:
            print(
                contract.canonical_bytes(
                    {"confidence_eval": index + 1, "total": len(outcomes)}
                ).decode(),
                flush=True,
            )
    if candidates is None:
        pipeline_report = metrics.confidence_metrics(
            predictions, minimum_class_rows=minimum_class_rows
        )
    else:
        pipeline_report = metrics.confidence_outcome_metrics(
            candidates,
            predictions,
            minimum_class_rows=minimum_class_rows,
        )
    eligible_predictions = [row for row in predictions if row["head_invoked"]]
    if candidates is None:
        report = metrics.confidence_metrics(
            eligible_predictions, minimum_class_rows=minimum_class_rows
        )
    else:
        eligible_ids = {str(row["sample_id"]) for row in eligible_predictions}
        eligible_candidates = [
            row for row in candidates if str(row["sample_id"]) in eligible_ids
        ]
        report = metrics.confidence_outcome_metrics(
            eligible_candidates,
            eligible_predictions,
            minimum_class_rows=minimum_class_rows,
        )
    report.update(
        {
            "evaluator_id": EVALUATOR_ID,
            "elapsed_seconds": time.perf_counter() - started,
            "score_contract": training.CONFIDENCE_SCORE_ID,
            "calibration_kind": "platt-on-combined-score-v1",
            "quality_scope": "head-eligible-r1-model-output-outcomes",
            "head_eligible_n": len(eligible_predictions),
            "deterministic_bypass_n": len(predictions) - len(eligible_predictions),
            "deterministic_bypass_rate": (
                len(predictions) - len(eligible_predictions)
            )
            / max(len(predictions), 1),
            "pipeline_metrics": pipeline_report,
            "deterministic_bypass_contract": "validator-before-confidence-v1",
        }
    )
    return report, predictions


def evaluate_narration(
    runtime: Any,
    rows: list[dict[str, Any]],
    *,
    limit: int | None = None,
    progress_every: int = 50,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _ensure_sdk_path()
    from mei_sdk.shared import NarrationProvider, verified_result_view

    if getattr(runtime, "narration_adapter", None) is None:
        raise RuntimeError("narration adapter is unavailable")
    selected_rows = rows[:limit] if limit is not None else rows
    provider = NarrationProvider()
    predictions: list[dict[str, Any]] = []
    adapter_exact = 0
    started = time.perf_counter()
    for index, row in enumerate(selected_rows):
        views = list(row.get("verified_result_views") or [])
        deterministic = "\n".join(
            provider.narrate(verified_result_view(view)) for view in views
        )
        generated = runtime.generate_narration(str(row.get("prompt") or ""), max_new=48)
        adapter_text = str(generated.get("text") or "").strip()
        accepted = adapter_text == deterministic
        adapter_exact += int(accepted)
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "text": adapter_text if accepted else deterministic,
                "fallback_text": deterministic,
                "adapter_text_sha256": contract.sha_bytes(adapter_text.encode("utf-8")),
                "adapter_accepted": accepted,
                "fallback_used": not accepted,
            }
        )
        if index == 0 or (index + 1) % progress_every == 0:
            print(
                contract.canonical_bytes(
                    {"narration_eval": index + 1, "total": len(selected_rows)}
                ).decode(),
                flush=True,
            )
    report = metrics.narration_metrics(selected_rows, predictions)
    report.update(
        {
            "evaluator_id": EVALUATOR_ID,
            "elapsed_seconds": time.perf_counter() - started,
            "adapter_exact_acceptance": adapter_exact / max(len(selected_rows), 1),
            "adapter_exact_accepted_n": adapter_exact,
            "deterministic_fallback_enforced": True,
        }
    )
    return report, predictions


def threshold_status(
    metric_report: Mapping[str, Any], floors: Mapping[str, float]
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for name, floor in floors.items():
        if name.endswith("_max"):
            metric_name = name[: -len("_max")]
            checks[name] = float(metric_report.get(metric_name, float("inf"))) <= float(floor)
        else:
            checks[name] = float(metric_report.get(name, float("-inf"))) >= float(floor)
    return {"passed": all(checks.values()), "checks": checks}
