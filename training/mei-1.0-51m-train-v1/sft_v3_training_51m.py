#!/usr/bin/env python3
"""Quality-first SFT-v4 training primitives for mei-1.0-51m.

This module is deliberately a new downstream source file so a live CPT run
does not observe source drift.  MLX is imported only inside training
functions; prompt, sampler and receipt contracts remain testable without
claiming Metal resources.
"""

from __future__ import annotations

import json
import math
import random
import types
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sft_v4_contract_51m as contract


TRAINING_ID = "mei-sft-v4-training-implementation-v1-quality-schema"
RETRIEVAL_OBJECTIVE_ID = "mei-retrieval-infonce-all-negatives-v1"
RETRIEVAL_SAMPLER_ID = "mei-retrieval-bank-tool-uniform-source-cycle-v3"
FULLCALL_SAMPLER_ID = "mei-fullcall-kind-tool-uniform-v1"
BANKED_FULLCALL_SAMPLER_ID = "mei-fullcall-weighted-bank-kind-tool-full-coverage-v3"
AGENT_SAMPLER_ID = "mei-agent-trajectory-step-uniform-v1"
MW_SAMPLER_ID = "mei-mw-class-uniform-oracle-learned-v1"
MW_PROMPT_ID = "mei-mw-disposition-prompt-v3-structured"
CONFIDENCE_SCORE_ID = "mei-confidence-combined-score-v2"
CONFIDENCE_LABEL_ID = "actual-r1-model-exact-call-or-correct-refusal-v3"


class ToolSchemaBudgetExceeded(RuntimeError):
    """Structured fail-closed outcome for an oversized stable tool prefix.

    Training banks must normally be frozen below the limit.  Learned retrieval
    can nevertheless assemble a different top-5 combination at evaluation or
    inference time.  That is a deterministic runtime outcome, not an evaluator
    crash and not a prediction made by MW or confidence heads.
    """

    code = "tool_schema_budget_exceeded"

    def __init__(
        self,
        *,
        stable_prefix_tokens: int,
        stable_prefix_max: int,
        sample_id: str | None = None,
    ) -> None:
        self.sample_id = sample_id
        self.stable_prefix_tokens = int(stable_prefix_tokens)
        self.stable_prefix_max = int(stable_prefix_max)
        identity = f" {sample_id}" if sample_id else ""
        super().__init__(
            f"{self.code}:{identity} {self.stable_prefix_tokens}"
        )

    def as_error(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "sample_id": self.sample_id,
            "stable_prefix_tokens": self.stable_prefix_tokens,
            "stable_prefix_max": self.stable_prefix_max,
        }

# 50% structural, 15% schema, 35% linguistic.  The two linguistic sources
# receive equal slots so the smaller provenance-bound natural subset cannot be
# drowned by the deterministic offline program.
SFT_V4_BANK_WEIGHTS = {
    "structural": 20,
    "schema": 6,
    "linguistic_natural": 7,
    "linguistic_offline": 7,
}


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _visible_context(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("context")
    if isinstance(raw, dict):
        context = dict(raw)
    else:
        context = {"locale": "zh-CN"}
    facts = str(row.get("system_facts") or row.get("scene") or "").strip()
    if facts:
        context.setdefault("facts", [facts])
    normalized_facts: list[dict[str, Any]] = []
    for index, fact in enumerate(context.get("facts") or []):
        if isinstance(fact, dict):
            normalized_facts.append(dict(fact))
            continue
        text = str(fact).strip()
        if not text:
            raise RuntimeError("request context contains an empty legacy fact")
        identity = contract.sha_bytes(
            contract.canonical_bytes(
                [row.get("sample_id"), row.get("query"), index, text]
            )
        )[:16]
        normalized_facts.append(
            {
                "id": f"fact-{identity}",
                "subject": "request-context",
                "predicate": "frozen-fixture-fact",
                "value": text,
                "source": "frozen-corpus-context",
                "verified": True,
            }
        )
    if "facts" in context or normalized_facts:
        context["facts"] = normalized_facts
    return context


def _history_lines(row: dict[str, Any]) -> list[str]:
    output: list[str] = []
    for turn in row.get("history") or []:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role") or "user")
        content = str(turn.get("content") or turn.get("text") or "").strip()
        if content:
            output.append(f"{role}：{content}")

    calls = list(row.get("prior_calls") or [])
    results = list(row.get("tool_results") or row.get("prior_tool_results") or [])
    if len(calls) != len(results):
        if calls or results:
            raise RuntimeError("Agent prompt requires an exact call/result prefix")
        return output
    trusted = row.get("agent_trace") or {}
    if results and not (
        isinstance(trusted, dict)
        and trusted.get("trusted_offline_fixture") is True
        and trusted.get("simulator_id") == "mei-agent-host-simulator-v1"
    ):
        raise RuntimeError("Agent prompt refuses untrusted tool-result fixtures")
    for call, result in zip(calls, results):
        if not isinstance(call, dict) or not isinstance(result, dict):
            raise RuntimeError("Agent call/result prefix must contain objects")
        call_id = str(call.get("call_id") or "")
        provenance = result.get("provenance") or {}
        if (
            not call_id
            or result.get("call_id") != call_id
            or result.get("status") != "ok"
            or not isinstance(provenance, dict)
            or provenance.get("verified") is not True
        ):
            raise RuntimeError("Agent prompt refuses an invalid ToolResultV2 prefix")
        output.append(
            "assistant："
            + _json(
                {
                    "call_id": call_id,
                    "name": str(call.get("name") or ""),
                    "arguments": call.get("arguments") or {},
                }
            )
        )
        output.append("tool：" + _json(result))
    return output


def render_fullcall_prompt_parts(
    row: dict[str, Any], selected_tools: Sequence[dict[str, Any]]
) -> dict[str, str]:
    """Render the exact v3 prompt without consulting any offline gold field."""

    if len(selected_tools) != 5:
        raise RuntimeError("SFT-v3 full-call prompts require exactly five schemas")
    sink = (
        contract.TASK_CONTRACT
        + "\n<tools>"
        + _json([contract.compact_tool(tool) for tool in selected_tools])
        + "</tools>"
    )
    permissions = row.get("permissions")
    if permissions is None:
        permissions = {}
    if not isinstance(permissions, dict):
        raise RuntimeError("SFT-v3 permissions must be an object")
    if row.get("slot_provenance"):
        raise RuntimeError("gold-derived slot provenance is forbidden in SFT-v3 prompts")
    ordinary = [
        "<context>" + _json(_visible_context(row)) + "</context>",
        "<evidence>" + _json(row.get("evidence") or []) + "</evidence>",
        "<permissions>" + _json(permissions) + "</permissions>",
        "<state>" + _json(row.get("state") or {}) + "</state>",
        "<mw>" + _json(row.get("mw") or row.get("mw_disposition") or {}) + "</mw>",
    ]
    ordinary.extend(_history_lines(row))
    ordinary.append("user：" + str(row.get("query") or "").strip())
    ordinary_text = "\n".join(ordinary)
    return {
        "sink": sink,
        "ordinary": ordinary_text,
        "prompt": sink + "\n" + ordinary_text + contract.ASSISTANT_SUFFIX,
    }


def selected_tools_for_row(
    row: dict[str, Any], tools_by_name: Mapping[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    names = list(row.get("retrieved_tools") or [])[:5]
    selected = [tools_by_name[str(name)] for name in names if str(name) in tools_by_name]
    if len(selected) != 5 or len({str(tool["name"]) for tool in selected}) != 5:
        raise RuntimeError(f"row {row.get('sample_id')} does not resolve to five tools")
    return selected


def encode_fullcall_row(
    tokenizer: Any,
    row: dict[str, Any],
    tools_by_name: Mapping[str, dict[str, Any]],
    *,
    stable_prefix_max: int = contract.STABLE_PREFIX_TOKENS_MAX,
    rolling_window: int = contract.ROLLING_WINDOW_TOKENS,
    max_answer: int = 128,
) -> tuple[list[int], list[int], dict[str, Any]]:
    selected = selected_tools_for_row(row, tools_by_name)
    rendered = render_fullcall_prompt_parts(row, selected)
    full_ids = tokenizer.encode(rendered["prompt"], add_bos=True, add_eos=False)
    stable_ids = tokenizer.encode(
        rendered["sink"] + "\n", add_bos=True, add_eos=False
    )
    if len(stable_ids) > stable_prefix_max:
        raise ToolSchemaBudgetExceeded(
            sample_id=str(row.get("sample_id") or "") or None,
            stable_prefix_tokens=len(stable_ids),
            stable_prefix_max=stable_prefix_max,
        )
    if full_ids[: len(stable_ids)] != stable_ids:
        raise RuntimeError("tokenizer framing does not preserve the stable sink prefix")
    if len(full_ids) > len(stable_ids) + rolling_window:
        prompt = stable_ids + full_ids[-rolling_window:]
    else:
        prompt = full_ids
    answers = row.get("answers")
    if answers is None:
        name = row.get("gold_name")
        answers = [] if not name else [{"name": name, "arguments": row.get("gold_args") or {}}]
    target = contract.serialize_tool_target(answers)
    if row.get("target_text") not in (None, target):
        raise RuntimeError(f"target serializer drift: {row.get('sample_id')}")
    answer = (
        tokenizer.encode(target, add_bos=False, add_eos=False)
        + tokenizer.encode(contract.TURN_END, add_bos=False, add_eos=False)
        + [tokenizer.eos_id]
    )
    if len(answer) > max_answer:
        raise RuntimeError(f"tool target exceeds {max_answer} tokens: {row.get('sample_id')}")
    return prompt, answer, {
        "sample_id": row.get("sample_id"),
        "stable_prefix_tokens": len(stable_ids),
        "ordinary_tokens_retained": len(prompt) - len(stable_ids),
        "prompt_tokens": len(prompt),
        "answer_tokens": len(answer),
        "prompt_framing_id": contract.PROMPT_FRAMING_ID,
        "serializer_id": contract.SERIALIZER_ID,
    }


def fullcall_epoch_order(rows: Sequence[dict[str, Any]], epoch: int) -> list[int]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[(str(row.get("kind") or "refuse"), str(row.get("candidate_tool") or row.get("gold_name") or ""))].append(index)
    kinds = [kind for kind in ("execute", "refuse") if any(key[0] == kind for key in groups)]
    tools = sorted({key[1] for key in groups})
    if kinds != ["execute", "refuse"] or not tools:
        raise RuntimeError("full-call sampler requires execute/refuse rows grouped by tool")
    shuffled: dict[tuple[str, str], list[int]] = {}
    for key, values in groups.items():
        selected = list(values)
        random.Random(3109 + epoch * 101 + int(contract.sha_bytes(_json(key).encode())[:8], 16)).shuffle(selected)
        shuffled[key] = selected
    maximum = max(len(values) for values in shuffled.values())
    order: list[int] = []
    for offset in range(maximum):
        for tool in tools:
            for kind in kinds:
                values = shuffled.get((kind, tool)) or []
                if offset < len(values):
                    order.append(values[offset])
    if len(order) != len(rows) or len(set(order)) != len(rows):
        raise RuntimeError("full-call sampler failed exact once-per-epoch coverage")
    return order


def _weighted_bank_cycle(weights: Mapping[str, int]) -> list[str]:
    if not weights or any(int(value) <= 0 for value in weights.values()):
        raise ValueError("bank weights must be positive")
    names = sorted(weights)
    used = {name: 0 for name in names}
    output: list[str] = []
    for _ in range(sum(int(weights[name]) for name in names)):
        name = min(
            names,
            key=lambda candidate: (
                used[candidate] / int(weights[candidate]),
                candidate,
            ),
        )
        output.append(name)
        used[name] += 1
    return output


def fullcall_training_schedule(
    rows: Sequence[dict[str, Any]],
    steps: int,
    *,
    bank_weights: Mapping[str, int] = SFT_V4_BANK_WEIGHTS,
) -> list[int]:
    """Apply the declared bank mix while covering every frozen row at least once."""

    if steps < len(rows):
        raise RuntimeError("full-call budget cannot expose every frozen row")
    by_bank: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_bank[str(row.get("_training_bank") or "")].append(index)
    if set(by_bank) != set(bank_weights) or "" in by_bank:
        raise RuntimeError("full-call bank partition does not match v4 weights")

    cycle = _weighted_bank_cycle(bank_weights)
    planned_banks = [cycle[index % len(cycle)] for index in range(steps)]
    planned_counts = {
        bank: planned_banks.count(bank) for bank in sorted(bank_weights)
    }
    insufficient = {
        bank: {"planned": planned_counts[bank], "rows": len(indices)}
        for bank, indices in by_bank.items()
        if planned_counts[bank] < len(indices)
    }
    if insufficient:
        raise RuntimeError(
            "full-call weighted budget cannot expose every bank row: "
            + _json(insufficient)
        )

    bank_streams: dict[str, list[int]] = {}
    for bank, indices in by_bank.items():
        local_rows = [rows[index] for index in indices]
        stream: list[int] = []
        epoch = 0
        while len(stream) < planned_counts[bank]:
            stream.extend(
                indices[index]
                for index in fullcall_epoch_order(local_rows, epoch)
            )
            epoch += 1
        bank_streams[bank] = stream[: planned_counts[bank]]
    cursors = {bank: 0 for bank in bank_streams}
    output: list[int] = []
    for bank in planned_banks:
        output.append(bank_streams[bank][cursors[bank]])
        cursors[bank] += 1
    if set(output) != set(range(len(rows))):
        raise RuntimeError("full-call weighted schedule omitted a frozen row")
    return output


def agent_epoch_order(rows: Sequence[dict[str, Any]], epoch: int) -> list[int]:
    trajectories: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        trajectories[str(row.get("trajectory_id") or row.get("cf_group") or "")].append(index)
    if "" in trajectories:
        raise RuntimeError("Agent sampler requires trajectory IDs")
    names = sorted(trajectories)
    random.Random(5101 + epoch).shuffle(names)
    for values in trajectories.values():
        values.sort(key=lambda index: int(rows[index].get("trajectory_step") or 0))
    order: list[int] = []
    depth = max(len(values) for values in trajectories.values())
    for step_offset in range(depth):
        for name in names:
            values = trajectories[name]
            if step_offset < len(values):
                order.append(values[step_offset])
    if len(order) != len(rows) or len(set(order)) != len(rows):
        raise RuntimeError("Agent sampler failed exact once-per-epoch coverage")
    return order


def retrieval_training_schedule(
    rows: Sequence[dict[str, Any]], steps: int, batch_size: int = 8
) -> list[list[int]]:
    """Build a deterministic, resume-safe tool-uniform retrieval schedule.

    The former sampler used a global epoch offset for every tool.  Because a
    tool is not necessarily selected in every global epoch, that coupling can
    permanently skip one of its rows even when the nominal number of
    selections is sufficient.  Independent per-tool counters preserve the
    tool-uniform batch order while cycling through every structural and
    admitted-natural row.  Materializing the schedule from step zero makes a
    resumed run identical to a fresh run at the same step.
    """

    if steps <= 0 or batch_size <= 0:
        raise ValueError("retrieval schedule requires positive steps and batch size")
    by_tool: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_tool[str(row.get("gold_tool") or "")].append(index)
    tool_names = sorted(by_tool)
    if not tool_names or "" in by_tool or batch_size > len(tool_names):
        raise RuntimeError(
            "retrieval schedule requires a non-empty complete tool partition"
        )
    banks = {str(row.get("_training_bank") or "") for row in rows}
    if banks != {""}:
        if banks != set(SFT_V4_BANK_WEIGHTS):
            raise RuntimeError("retrieval bank partition does not match v4 weights")
        by_bank_tool: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, row in enumerate(rows):
            by_bank_tool[str(row["_training_bank"])][str(row["gold_tool"])].append(
                index
            )
        cycle = _weighted_bank_cycle(SFT_V4_BANK_WEIGHTS)
        bank_selections: dict[str, int] = defaultdict(int)
        row_selections: dict[tuple[str, str], int] = defaultdict(int)
        schedule: list[list[int]] = []
        global_slot = 0
        for _step in range(steps):
            batch: list[int] = []
            selected_tools: set[str] = set()
            for _slot in range(batch_size):
                bank = cycle[global_slot % len(cycle)]
                global_slot += 1
                names = sorted(by_bank_tool[bank])
                if not names:
                    raise RuntimeError(f"retrieval bank is empty: {bank}")
                for _attempt in range(len(names)):
                    selection = bank_selections[bank]
                    epoch = selection // len(names)
                    shuffled = list(names)
                    random.Random(
                        7103
                        + epoch
                        + int(contract.sha_bytes(bank.encode())[:8], 16)
                    ).shuffle(shuffled)
                    name = shuffled[selection % len(shuffled)]
                    bank_selections[bank] += 1
                    if name not in selected_tools:
                        break
                else:
                    raise RuntimeError("retrieval bank cannot provide a unique batch tool")
                values = by_bank_tool[bank][name]
                cursor = row_selections[(bank, name)]
                batch.append(values[cursor % len(values)])
                row_selections[(bank, name)] += 1
                selected_tools.add(name)
            schedule.append(batch)
        return schedule

    consumed: dict[str, int] = defaultdict(int)
    schedule: list[list[int]] = []
    for step in range(steps):
        cycle = step * batch_size
        epoch = cycle // len(tool_names)
        names = list(tool_names)
        random.Random(7103 + epoch).shuffle(names)
        offset = cycle % len(names)
        selected_names = [
            names[(offset + index) % len(names)] for index in range(batch_size)
        ]
        batch: list[int] = []
        for name in selected_names:
            values = by_tool[name]
            batch.append(values[consumed[name] % len(values)])
            consumed[name] += 1
        if len(batch) != batch_size or len(set(selected_names)) != batch_size:
            raise RuntimeError("retrieval schedule produced an invalid batch")
        schedule.append(batch)
    return schedule


def _cells(runtime: Any, text: str, max_tokens: int) -> list[Any]:
    import mlx.core as mx

    ids = runtime.tokenizer.encode(text, add_bos=True, add_eos=False)[:max_tokens]
    output = runtime.model(mx.array([ids], dtype=mx.int32), return_cells=True)
    mx.eval(*output["cells"])
    return [mx.stop_gradient(cell).astype(mx.float16) for cell in output["cells"]]


def train_retrieval_v3(
    runtime: Any,
    rows: list[dict[str, Any]],
    tools_by_name: Mapping[str, dict[str, Any]],
    *,
    steps: int,
    lr: float = 1e-3,
    batch_size: int = 8,
    temperature: float = 0.07,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_every_steps: int = 100,
    stage_id: str,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from checkpoint import load_train_state, save_train_state
    from heads import ContrastiveHead

    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        gold = str(row.get("gold_tool") or "")
        negatives = [str(value) for value in row.get("hard_negatives") or []]
        if gold not in tools_by_name or len(negatives) != 4 or any(
            value not in tools_by_name or value == gold for value in negatives
        ):
            raise RuntimeError(f"invalid retrieval row: {row.get('sample_id')}")
        by_tool[gold].append(row)
    tool_names = sorted(by_tool)
    if not tool_names or batch_size > len(tool_names):
        raise RuntimeError(
            "retrieval training requires a non-empty complete tool partition"
        )
    schedule = retrieval_training_schedule(rows, steps, batch_size)
    bank_exposures = {
        bank: sum(
            1
            for batch in schedule
            for index in batch
            if str(rows[index].get("_training_bank") or "") == bank
        )
        for bank in sorted(
            {str(row.get("_training_bank") or "") for row in rows} - {""}
        )
    }
    head = ContrastiveHead(runtime.model.cfg.d_model, runtime.model.cfg.n_layers, dim=128, probes=4)
    mx.eval(head.parameters())
    optimizer = optim.Adam(learning_rate=lr)
    state_path = checkpoint_dir / f"retrieval-{stage_id}-head-state.npz" if checkpoint_dir else None
    data_fingerprint = contract.sha_bytes(
        contract.canonical_bytes([row.get("sample_id") for row in rows])
    )
    expected_meta = {
        "kind": "mei-sft-v3-retrieval-head-state",
        "implementation": TRAINING_ID,
        "objective": RETRIEVAL_OBJECTIVE_ID,
        "sampler": RETRIEVAL_SAMPLER_ID,
        "retrieval_encoding_id": contract.RETRIEVAL_ENCODING_ID,
        "max_tokens": contract.RETRIEVAL_MAX_TOKENS,
        "temperature": temperature,
        "all_row_negatives": 4,
        "in_batch_positives": True,
        "tool_uniform": True,
        "data_fingerprint": data_fingerprint,
        "target_steps": steps,
        "stage_id": stage_id,
    }
    start_step = 0
    last_loss: float | None = None
    if resume and state_path and state_path.is_file():
        meta = load_train_state(state_path, head, optimizer, mode="strict", expected_meta=expected_meta)
        start_step = int(meta.get("steps_completed") or 0)
        last_loss = meta.get("last_loss")
    tool_cells: dict[str, list[Any]] = {}

    def cached_tool(name: str) -> list[Any]:
        if name not in tool_cells:
            tool_cells[name] = _cells(
                runtime,
                contract.retrieval_tool_text(tools_by_name[name]),
                contract.RETRIEVAL_MAX_TOKENS,
            )
        return tool_cells[name]

    for step in range(start_step, steps):
        batch_rows = [rows[index] for index in schedule[step]]
        query_cells = [
            _cells(runtime, str(row.get("query") or ""), contract.RETRIEVAL_MAX_TOKENS)
            for row in batch_rows
        ]
        candidate_names: list[str] = []
        for row in batch_rows:
            for name in [str(row["gold_tool"]), *[str(v) for v in row["hard_negatives"]]]:
                if name not in candidate_names:
                    candidate_names.append(name)
        labels = [candidate_names.index(str(row["gold_tool"])) for row in batch_rows]
        candidate_cells = [cached_tool(name) for name in candidate_names]

        def loss_fn(candidate_head: Any) -> Any:
            queries = mx.concatenate([candidate_head(cells) for cells in query_cells], axis=0).astype(mx.float32)
            candidates = mx.concatenate([candidate_head(cells) for cells in candidate_cells], axis=0).astype(mx.float32)
            logits = (queries @ candidates.T) / float(temperature)
            log_probabilities = nn.log_softmax(logits, axis=-1)
            losses = mx.stack([-log_probabilities[index, label] for index, label in enumerate(labels)])
            return mx.mean(losses)

        loss, gradients = mx.value_and_grad(loss_fn)(head)
        optimizer.update(head, gradients)
        mx.eval(head.parameters(), loss)
        last_loss = float(loss.item())
        if state_path and ((step + 1) % max(1, checkpoint_every_steps) == 0 or step + 1 == steps):
            save_train_state(
                state_path,
                head,
                optimizer,
                {**expected_meta, "steps_completed": step + 1, "last_loss": last_loss},
            )
        if step == 0 or (step + 1) % 50 == 0:
            print(_json({"retrieval_stage": stage_id, "step": step + 1, "loss": last_loss}), flush=True)
    runtime.contrastive = head
    return {
        "implementation": TRAINING_ID,
        "stage_id": stage_id,
        "steps": steps,
        "resumed_from_step": start_step,
        "last_loss": last_loss,
        "rows": len(rows),
        "tools": len(by_tool),
        "cached_tools": len(tool_cells),
        "batch_size": batch_size,
        "temperature": temperature,
        "objective": RETRIEVAL_OBJECTIVE_ID,
        "sampler": RETRIEVAL_SAMPLER_ID,
        "all_four_hard_negatives": True,
        "in_batch_positives": True,
        "lm_frozen": True,
        "data_fingerprint": data_fingerprint,
        "bank_exposures": bank_exposures,
        "unique_source_rows_exposed": len(
            {index for batch in schedule for index in batch}
        ),
    }


def configure_retrieval_encoding_v3(runtime: Any) -> None:
    """Bind the trained 384-token encoding to Python index build and queries."""

    def embed_text(instance: Any, text: str) -> Any:
        import mlx.core as mx

        ids = instance.tokenizer.encode(text, add_bos=True, add_eos=False)[
            : contract.RETRIEVAL_MAX_TOKENS
        ]
        output = instance.model(mx.array([ids], dtype=mx.int32), return_cells=True)
        cells = output.get("cells")
        if instance.contrastive is None or cells is None:
            hidden = output["hidden"][:, -1, :].astype(mx.float32)
            norm = mx.sqrt(mx.sum(hidden * hidden, axis=-1, keepdims=True) + 1e-8)
            result = hidden / norm
            mx.eval(result)
            return result[0]
        vector = instance.contrastive(cells)
        mx.eval(vector)
        return vector[0]

    runtime.embed_text = types.MethodType(embed_text, runtime)
    runtime._retrieval_encoding_id = contract.RETRIEVAL_ENCODING_ID
    runtime._retrieval_max_tokens = contract.RETRIEVAL_MAX_TOKENS


def finalize_tool_index_v3(
    runtime: Any,
    catalog: list[dict[str, Any]],
    path: Path,
    *,
    model_sha256: str,
    head_sha256: str,
    tokenizer_sha256: str,
) -> dict[str, Any]:
    from mei_sdk.shared import ToolIndex
    import tool_index as tool_index_module

    configure_retrieval_encoding_v3(runtime)
    runtime.index = ToolIndex(
        model_hash=model_sha256,
        head_hash=head_sha256,
        tokenizer_hash=tokenizer_sha256,
    )
    runtime._catalog_fp = ""
    original_render = tool_index_module.render_tool_text
    tool_index_module.render_tool_text = contract.retrieval_tool_text
    try:
        runtime.build_index(catalog)
    finally:
        tool_index_module.render_tool_text = original_render
    path.parent.mkdir(parents=True, exist_ok=True)
    runtime.index.save(path)
    loaded = ToolIndex.load(path, expected_fingerprint=runtime.index.fingerprint)
    return {
        "path": str(path),
        "sha256": contract.sha_file(path),
        "fingerprint": loaded.fingerprint,
        "catalog_sha256": loaded.catalog_hash,
        "n_tools": len(loaded.records),
        "dimension": len(next(iter(loaded.records.values())).embedding),
        "dtype": "float16",
        "normalized": True,
        "stable_tool_id_ties": True,
        "retrieval_encoding_id": contract.RETRIEVAL_ENCODING_ID,
        "retrieval_max_tokens": contract.RETRIEVAL_MAX_TOKENS,
        "tool_text_serializer": "sft_v3_contract_51m.retrieval_tool_text",
    }


def train_lm_sft_v3(
    runtime: Any,
    rows: list[dict[str, Any]],
    tools_by_name: Mapping[str, dict[str, Any]],
    *,
    steps: int,
    lr: float,
    group_map: Mapping[str, Sequence[int]] | None,
    activation_ste: bool,
    sampler: str,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_every_steps: int = 100,
    stage_id: str,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import architecture as architecture_module
    from checkpoint import load_train_state, save_train_state
    from cq2_qat_51m import QUANT_MATH_ID, quantize_tree
    from train_common import clip_grads

    pairs = [encode_fullcall_row(runtime.tokenizer, row, tools_by_name) for row in rows]
    if not pairs:
        raise RuntimeError("SFT-v3 LM stage has no encoded rows")
    fixed_schedule: list[int] | None = None
    if sampler == FULLCALL_SAMPLER_ID:
        order_fn = fullcall_epoch_order
    elif sampler == BANKED_FULLCALL_SAMPLER_ID:
        fixed_schedule = fullcall_training_schedule(rows, steps)
        order_fn = fullcall_epoch_order
    elif sampler == AGENT_SAMPLER_ID:
        order_fn = agent_epoch_order
    else:
        raise ValueError(f"unsupported SFT-v3 sampler: {sampler}")
    data_fingerprint = contract.sha_bytes(
        contract.canonical_bytes([row.get("sample_id") for row in rows])
    )
    model = runtime.model
    optimizer = optim.Adam(learning_rate=lr)
    state_path = checkpoint_dir / f"{stage_id}-state.npz" if checkpoint_dir else None
    expected_meta = {
        "kind": "mei-sft-v3-lm-state",
        "implementation": TRAINING_ID,
        "stage_id": stage_id,
        "sampler": sampler,
        "prompt_framing_id": contract.PROMPT_FRAMING_ID,
        "serializer_id": contract.SERIALIZER_ID,
        "data_fingerprint": data_fingerprint,
        "n_pairs": len(pairs),
        "target_steps": steps,
        "quant_math_id": QUANT_MATH_ID if group_map else None,
        "weight_qat_ste": bool(group_map),
        "activation_kv_int8_ste": bool(activation_ste and group_map),
    }
    start_step = 0
    last_loss: float | None = None
    if resume and state_path and state_path.is_file():
        meta = load_train_state(state_path, model, optimizer, mode="strict", expected_meta=expected_meta)
        start_step = int(meta.get("steps_completed") or 0)
        last_loss = meta.get("last_loss")
    architecture_module.QAT_ACTIVATION_STE = bool(activation_ste and group_map)
    used: set[int] = set()
    cached_order: list[int] = []
    cached_epoch = -1
    try:
        for step in range(start_step, steps):
            epoch = step // len(pairs)
            if fixed_schedule is not None:
                pair_index = fixed_schedule[step]
            else:
                if epoch != cached_epoch:
                    cached_order = order_fn(rows, epoch)
                    cached_epoch = epoch
                pair_index = cached_order[step % len(cached_order)]
            used.add(pair_index)
            prompt, answer, _stats = pairs[pair_index]
            ids = prompt + answer

            def loss_fn(parameters: Any) -> Any:
                model.update(quantize_tree(parameters, group_map, ste=True) if group_map else parameters)
                logits = model(mx.array([ids], dtype=mx.int32))["logits"].astype(mx.float32)
                log_probabilities = nn.log_softmax(logits, axis=-1)
                losses = [
                    -log_probabilities[0, len(prompt) - 1 + index, int(token_id)]
                    for index, token_id in enumerate(answer)
                ]
                return mx.mean(mx.stack(losses))

            parameters = model.parameters()
            loss, gradients = mx.value_and_grad(loss_fn)(parameters)
            gradients, gradient_norm = clip_grads(gradients, max_norm=1.0)
            model.update(parameters)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), loss)
            last_loss = float(loss.item())
            if state_path and ((step + 1) % max(1, checkpoint_every_steps) == 0 or step + 1 == steps):
                save_train_state(
                    state_path,
                    model,
                    optimizer,
                    {
                        **expected_meta,
                        "steps_completed": step + 1,
                        "last_loss": last_loss,
                        "epoch": epoch,
                    },
                )
            if step == 0 or (step + 1) % 100 == 0:
                print(
                    _json(
                        {
                            "stage": stage_id,
                            "step": step + 1,
                            "loss": last_loss,
                            "grad_norm": float(gradient_norm.item()),
                            "qat": bool(group_map),
                        }
                    ),
                    flush=True,
                )
    finally:
        architecture_module.QAT_ACTIVATION_STE = False
    prompt_stats = [stats for _, _, stats in pairs]
    bank_exposures = {
        bank: sum(
            1
            for index in (fixed_schedule or [])
            if str(rows[index].get("_training_bank") or "") == bank
        )
        for bank in sorted(
            {str(row.get("_training_bank") or "") for row in rows} - {""}
        )
    }
    return {
        "implementation": TRAINING_ID,
        "stage_id": stage_id,
        "steps": steps,
        "resumed_from_step": start_step,
        "last_loss": last_loss,
        "rows": len(rows),
        "unique_rows_executed_this_process": len(used),
        "unique_rows_scheduled": len(set(fixed_schedule or used)),
        "sampler": sampler,
        "prompt_framing_id": contract.PROMPT_FRAMING_ID,
        "serializer_id": contract.SERIALIZER_ID,
        "max_prompt_tokens": max(value["prompt_tokens"] for value in prompt_stats),
        "max_answer_tokens": max(value["answer_tokens"] for value in prompt_stats),
        "weight_qat_ste": bool(group_map),
        "activation_kv_int8_ste": bool(activation_ste and group_map),
        "quant_math_id": QUANT_MATH_ID if group_map else None,
        "data_fingerprint": data_fingerprint,
        "bank_exposures": bank_exposures,
        "bank_weighted_full_coverage": fixed_schedule is not None,
    }


def render_mw_prompt_parts(
    row: dict[str, Any], selected_tools: Sequence[dict[str, Any]]
) -> dict[str, str]:
    if len(selected_tools) != 5:
        raise RuntimeError("MW-v3 prompts require exactly five visible schemas")
    task = (
        "任务：按结构化请求证据与五工具schema计算20类MW处置；"
        "不生成调用或解释。"
    )
    sink = (
        task
        + "\n<tools>"
        + _json([contract.compact_tool(tool) for tool in selected_tools])
        + "</tools>"
    )
    request_view = {
        "query": str(row.get("query") or "").strip(),
        "context": _visible_context(row),
        "evidence": list(row.get("evidence") or []),
        "history": list(row.get("history") or []),
        "tool_results": list(
            row.get("tool_results") or row.get("prior_tool_results") or []
        ),
        "permissions": dict(row.get("permissions") or {}),
        "state": dict(row.get("state") or {}),
    }
    ordinary_text = "<request>" + _json(request_view) + "</request>"
    return {
        "sink": sink,
        "ordinary": ordinary_text,
        "prompt": sink + "\n" + ordinary_text,
        "prompt_id": MW_PROMPT_ID,
    }


def encode_stable_ring_prompt(
    tokenizer: Any,
    rendered: dict[str, str],
    *,
    stable_prefix_max: int = contract.STABLE_PREFIX_TOKENS_MAX,
    rolling_window: int = contract.ROLLING_WINDOW_TOKENS,
    sample_id: str | None = None,
) -> tuple[list[int], dict[str, int]]:
    full_ids = tokenizer.encode(rendered["prompt"], add_bos=True, add_eos=False)
    stable_ids = tokenizer.encode(rendered["sink"] + "\n", add_bos=True, add_eos=False)
    if len(stable_ids) > stable_prefix_max:
        raise ToolSchemaBudgetExceeded(
            sample_id=sample_id,
            stable_prefix_tokens=len(stable_ids),
            stable_prefix_max=stable_prefix_max,
        )
    if full_ids[: len(stable_ids)] != stable_ids:
        raise RuntimeError("tokenizer framing does not preserve the stable MW sink")
    ids = (
        stable_ids + full_ids[-rolling_window:]
        if len(full_ids) > len(stable_ids) + rolling_window
        else full_ids
    )
    return ids, {
        "stable_prefix_tokens": len(stable_ids),
        "ordinary_tokens_retained": len(ids) - len(stable_ids),
        "prompt_tokens": len(ids),
    }


def mw_training_view(
    row: dict[str, Any],
    tools_by_name: Mapping[str, dict[str, Any]],
    learned_names: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], int, str]:
    original_label = int(row["reason_class_id"])
    original_reason = str(row.get("reason_code") or "")
    if learned_names is None:
        names = [str(value) for value in row.get("retrieved_tools") or []][:5]
        mode = "oracle_top5"
    else:
        names = [str(value) for value in learned_names][:5]
        mode = "learned_top5"
    selected = [tools_by_name[name] for name in names if name in tools_by_name]
    if len(selected) != 5 or len({str(tool["name"]) for tool in selected}) != 5:
        raise RuntimeError(f"MW row {row.get('sample_id')} lacks five visible schemas")
    candidate = str(row.get("candidate_tool") or row.get("gold_name") or "")
    if mode == "learned_top5" and original_label == 0 and candidate and candidate not in names:
        return selected, 10, "capability_insufficient"
    return selected, original_label, original_reason


def train_mw_disposition_v3(
    runtime: Any,
    rows: list[dict[str, Any]],
    tools_by_name: Mapping[str, dict[str, Any]],
    catalog: list[dict[str, Any]],
    *,
    steps: int,
    lr: float = 1e-3,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_every_steps: int = 100,
) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from checkpoint import load_train_state, save_train_state
    from heads import MW_DISPOSITION_CLASSES, MWDispositionHead

    by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_class[int(row["reason_class_id"])].append(row)
    if set(by_class) != set(range(MW_DISPOSITION_CLASSES)):
        raise RuntimeError("MW-v3 requires all 20 disposition classes")
    head = MWDispositionHead(runtime.model.cfg.d_model)
    mx.eval(head.parameters())
    optimizer = optim.Adam(learning_rate=lr)
    data_fingerprint = contract.sha_bytes(
        contract.canonical_bytes([row.get("sample_id") for row in rows])
    )
    state_path = checkpoint_dir / "mw-disposition-v3-state.npz" if checkpoint_dir else None
    expected_meta = {
        "kind": "mei-sft-v3-mw-disposition-state",
        "implementation": TRAINING_ID,
        "sampler": MW_SAMPLER_ID,
        "prompt_id": MW_PROMPT_ID,
        "n_classes": MW_DISPOSITION_CLASSES,
        "data_fingerprint": data_fingerprint,
        "target_steps": steps,
    }
    start_step = 0
    last_loss: float | None = None
    if resume and state_path and state_path.is_file():
        meta = load_train_state(state_path, head, optimizer, mode="strict", expected_meta=expected_meta)
        start_step = int(meta.get("steps_completed") or 0)
        last_loss = meta.get("last_loss")
    observed: dict[int, int] = {index: 0 for index in range(20)}
    learned_views = oracle_views = learned_miss_relabels = 0
    for step in range(start_step, steps):
        requested_class = step % 20
        occurrence = step // 20
        candidates = list(by_class[requested_class])
        epoch = occurrence // len(candidates)
        random.Random(1701 + requested_class * 101 + epoch).shuffle(candidates)
        row = candidates[occurrence % len(candidates)]
        use_learned = bool(occurrence % 2)
        learned_names = None
        if use_learned:
            visible = runtime.search_top_k(str(row.get("query") or ""), catalog, k=5)
            learned_names = [str(tool.get("name") or "") for tool in visible]
        selected, label, reason = mw_training_view(row, tools_by_name, learned_names)
        learned_views += int(use_learned)
        oracle_views += int(not use_learned)
        learned_miss_relabels += int(use_learned and requested_class == 0 and label == 10)
        observed[label] += 1
        rendered = render_mw_prompt_parts(row, selected)
        ids, _stats = encode_stable_ring_prompt(runtime.tokenizer, rendered)
        frozen = runtime.model(mx.array([ids], dtype=mx.int32), return_cells=True)["cells"]
        mx.eval(*frozen)
        cells = [mx.stop_gradient(value).astype(mx.float16) for value in frozen]

        def loss_fn(candidate_head: Any) -> Any:
            logits = candidate_head(cells).astype(mx.float32)
            return -nn.log_softmax(logits, axis=-1)[0, label]

        loss, gradients = mx.value_and_grad(loss_fn)(head)
        optimizer.update(head, gradients)
        mx.eval(head.parameters(), loss)
        last_loss = float(loss.item())
        if state_path and ((step + 1) % max(1, checkpoint_every_steps) == 0 or step + 1 == steps):
            save_train_state(
                state_path,
                head,
                optimizer,
                {**expected_meta, "steps_completed": step + 1, "last_loss": last_loss},
            )
        if step == 0 or (step + 1) % 100 == 0:
            print(
                _json(
                    {
                        "mw_step": step + 1,
                        "loss": last_loss,
                        "requested_class": requested_class,
                        "effective_class": label,
                        "reason": reason,
                    }
                ),
                flush=True,
            )
    runtime.mw_disposition_head = head
    return {
        "implementation": TRAINING_ID,
        "steps": steps,
        "resumed_from_step": start_step,
        "last_loss": last_loss,
        "n_rows": len(rows),
        "n_classes": 20,
        "sampler": MW_SAMPLER_ID,
        "prompt_id": MW_PROMPT_ID,
        "source_class_counts": {str(key): len(by_class[key]) for key in sorted(by_class)},
        "effective_training_class_counts": {str(key): value for key, value in observed.items()},
        "oracle_top5_views": oracle_views,
        "learned_top5_views": learned_views,
        "learned_ready_relabelled_capability_insufficient": learned_miss_relabels,
        "semantic_boundary": "independent-20class-sidecar-not-mw-deviation-gate",
        "lm_frozen": True,
        "data_fingerprint": data_fingerprint,
    }


def confidence_candidate_sample(
    rows: Sequence[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            str(row.get("expected_kind") or row.get("kind") or ""),
            str(row.get("family") or "unknown"),
            str(row.get("candidate_tool") or ""),
        )
        groups[key].append(row)
    for key, values in groups.items():
        values.sort(
            key=lambda row: contract.sha_bytes(
                contract.canonical_bytes([key, row.get("sample_id")])
            )
        )
    keys = sorted(groups)
    output: list[dict[str, Any]] = []
    offset = 0
    while len(output) < min(limit, len(rows)):
        progressed = False
        for key in keys:
            values = groups[key]
            if offset < len(values):
                output.append(values[offset])
                progressed = True
                if len(output) >= min(limit, len(rows)):
                    break
        if not progressed:
            break
        offset += 1
    if len({str(row.get("sample_id")) for row in output}) != len(output):
        raise RuntimeError("confidence sampler produced duplicate sample IDs")
    return output


def deployment_request_v3(row: dict[str, Any]) -> dict[str, Any]:
    permissions = row.get("permissions")
    if permissions is None:
        permissions = {}
    if not isinstance(permissions, dict) or row.get("slot_provenance"):
        raise RuntimeError("confidence harvest refuses non-v3 provenance/permission input")
    return {
        "wire_version": contract.WIRE_ID,
        "query": str(row.get("query") or ""),
        "context": _visible_context(row),
        "evidence": list(row.get("evidence") or []),
        "permissions": permissions,
        "history": list(row.get("history") or []),
        "tool_results": list(row.get("tool_results") or []),
        "state": dict(row.get("state") or {}),
    }


def _calls_equal(left: Sequence[dict[str, Any]], right: Sequence[dict[str, Any]]) -> bool:
    return contract.canonical_bytes(list(left or [])) == contract.canonical_bytes(list(right or []))


def harvest_confidence_outcomes_v3(
    runtime: Any,
    rows: list[dict[str, Any]],
    tools_by_name: Mapping[str, dict[str, Any]],
    catalog: list[dict[str, Any]],
    *,
    limit: int,
    minimum_class_rows: int = 100,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from mei_sdk.protocol import normalize_request
    from mei_sdk.shared import parse_call_text, validate_generated_call

    selected_rows = confidence_candidate_sample(rows, limit)
    outcomes: list[dict[str, Any]] = []
    for index, source in enumerate(selected_rows):
        query = str(source.get("query") or "")
        visible = runtime.search_top_k(query, catalog, k=5)
        names = [str(tool.get("name") or "") for tool in visible]
        prompt_row = dict(source)
        prompt_row["retrieved_tools"] = names
        expected_call = source.get("expected_call")
        prompt_row["answers"] = [expected_call] if isinstance(expected_call, dict) else []
        expected_kind = str(source.get("expected_kind") or "")
        if expected_kind not in {"call", "refuse"}:
            raise RuntimeError(f"unknown confidence expected kind: {expected_kind}")
        common = {
            "sample_id": str(source.get("sample_id") or ""),
            "source_sample_id": source.get("source_sample_id"),
            "family": source.get("family") or "unknown",
            "candidate_tool": source.get("candidate_tool"),
            "expected_kind": expected_kind,
            "retrieved_tools": names,
            "retrieval_hit": bool(
                expected_kind != "call"
                or str((expected_call or {}).get("name") or "") in names
            ),
            "label_contract": CONFIDENCE_LABEL_ID,
        }
        try:
            prompt, _answer, prompt_stats = encode_fullcall_row(
                runtime.tokenizer, prompt_row, tools_by_name
            )
        except ToolSchemaBudgetExceeded as exc:
            # The deterministic runtime gate executes before confidence.  Keep
            # exact candidate coverage, label the model outcome incorrect,
            # and explicitly exclude it from head training/calibration.
            outcomes.append(
                {
                    **common,
                    "prompt_ids": [],
                    "prompt_stats": {
                        "stable_prefix_tokens": exc.stable_prefix_tokens,
                        "stable_prefix_max": exc.stable_prefix_max,
                    },
                    "label": 0,
                    "predicted_refuse": False,
                    "validated_ok": False,
                    "validation_error": exc.code,
                    "model_parse_error": exc.code,
                    "pipeline_validation_error": exc.code,
                    "deterministic_error": exc.as_error(),
                    "head_eligible": False,
                    "decode_text_sha256": contract.sha_bytes(b""),
                    "logprob_sum": 0.0,
                    "output_tokens": 0,
                }
            )
        else:
            decoded = runtime.greedy(
                prompt, tools=visible, max_new=128, decode_mode="constrained"
            )
            decoded_text = str(decoded.get("text") or "")
            parsed = parse_call_text(decoded_text, visible)
            request = normalize_request(deployment_request_v3(source))
            validated = validate_generated_call(
                decoded_text,
                tools=visible,
                request=request,
                confidence=None,
                enforce_confidence=False,
            )
            if expected_kind == "call":
                expected = [expected_call] if isinstance(expected_call, dict) else []
                correct = bool(
                    parsed.get("ok") is True
                    and parsed.get("refuse") is not True
                    and _calls_equal(parsed.get("function_calls") or [], expected)
                )
            else:
                correct = bool(
                    parsed.get("ok") is True and parsed.get("refuse") is True
                )
            outcomes.append(
                {
                    **common,
                    "prompt_ids": prompt,
                    "prompt_stats": prompt_stats,
                    "label": int(correct),
                    "predicted_refuse": parsed.get("refuse") is True,
                    "validated_ok": parsed.get("ok") is True,
                    "validation_error": parsed.get("error"),
                    "model_parse_error": parsed.get("error"),
                    "pipeline_validation_error": validated.get("error"),
                    "deterministic_error": None,
                    "head_eligible": True,
                    "decode_text_sha256": contract.sha_bytes(
                        decoded_text.encode("utf-8")
                    ),
                    "logprob_sum": float(decoded.get("logprob_sum") or 0.0),
                    "output_tokens": int(decoded.get("n_out") or 0),
                }
            )
        if index == 0 or (index + 1) % 100 == 0:
            print(
                _json(
                    {
                        "confidence_harvest": index + 1,
                        "positive": sum(int(row["label"]) for row in outcomes),
                        "negative": sum(int(not row["label"]) for row in outcomes),
                    }
                ),
                flush=True,
            )
    positives = sum(int(row["label"]) for row in outcomes)
    negatives = len(outcomes) - positives
    head_eligible = [row for row in outcomes if row.get("head_eligible") is not False]
    eligible_positives = sum(int(row["label"]) for row in head_eligible)
    eligible_negatives = len(head_eligible) - eligible_positives
    if positives < minimum_class_rows or negatives < minimum_class_rows:
        raise RuntimeError(
            "confidence outcome class coverage failed: "
            f"positive={positives} negative={negatives} floor={minimum_class_rows}"
        )
    if (
        eligible_positives < minimum_class_rows
        or eligible_negatives < minimum_class_rows
    ):
        raise RuntimeError(
            "confidence head-eligible class coverage failed: "
            f"positive={eligible_positives} negative={eligible_negatives} "
            f"floor={minimum_class_rows}"
        )
    sample_fingerprint = contract.sha_bytes(
        contract.canonical_bytes([row["sample_id"] for row in outcomes])
    )
    receipt = {
        "schema": "mei-confidence-outcome-harvest-receipt-v3",
        "status": "passed",
        "rows": len(outcomes),
        "positive": positives,
        "negative": negatives,
        "head_eligible_rows": len(head_eligible),
        "head_eligible_positive": eligible_positives,
        "head_eligible_negative": eligible_negatives,
        "deterministic_bypass_rows": len(outcomes) - len(head_eligible),
        "deterministic_bypass_contract": "validator-before-confidence-v1",
        "minimum_class_rows": minimum_class_rows,
        "sample_fingerprint": sample_fingerprint,
        "sampling_contract": "kind_then_family_then_tool_stratified_v1",
        "label_contract": CONFIDENCE_LABEL_ID,
        "pipeline_validation_error_counts": {
            error: sum(row.get("pipeline_validation_error") == error for row in outcomes)
            for error in sorted(
                {
                    str(row["pipeline_validation_error"])
                    for row in outcomes
                    if row.get("pipeline_validation_error")
                }
            )
        },
        "correct_refusal_is_positive": True,
        "score_contract": CONFIDENCE_SCORE_ID,
    }
    return outcomes, receipt


def _fit_platt(scores: Sequence[float], labels: Sequence[int]) -> dict[str, float]:
    if set(labels) != {0, 1}:
        raise RuntimeError("Platt calibration requires both outcome classes")
    features = [
        math.log(min(max(float(score), 1e-6), 1.0 - 1e-6) / (1.0 - min(max(float(score), 1e-6), 1.0 - 1e-6)))
        for score in scores
    ]
    scale = 1.0
    bias = 0.0
    for _ in range(1000):
        grad_scale = 0.0
        grad_bias = 0.0
        for feature, label in zip(features, labels):
            value = scale * feature + bias
            probability = (
                1.0 / (1.0 + math.exp(-value))
                if value >= 0
                else math.exp(value) / (1.0 + math.exp(value))
            )
            error = probability - label
            grad_scale += error * feature
            grad_bias += error
        scale -= 0.02 * grad_scale / len(labels)
        bias -= 0.02 * grad_bias / len(labels)
        scale = min(max(scale, 0.05), 20.0)
        bias = min(max(bias, -20.0), 20.0)
    return {"scale": scale, "bias": bias}


def apply_platt(score: float, calibration: Mapping[str, float]) -> float:
    bounded = min(max(float(score), 1e-6), 1.0 - 1e-6)
    feature = math.log(bounded / (1.0 - bounded))
    value = float(calibration["scale"]) * feature + float(calibration["bias"])
    return (
        1.0 / (1.0 + math.exp(-value))
        if value >= 0
        else math.exp(value) / (1.0 + math.exp(value))
    )


def train_confidence_v3(
    runtime: Any,
    train_outcomes: list[dict[str, Any]],
    valid_outcomes: list[dict[str, Any]],
    *,
    steps: int,
    lr: float = 1e-3,
    minimum_class_rows: int = 100,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    checkpoint_every_steps: int = 100,
) -> tuple[dict[str, Any], dict[str, float]]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from checkpoint import load_train_state, save_train_state
    from heads import ConfidenceV2Head
    from longitudinal_eval_metrics_51m import confidence_metrics

    eligible_sets = {
        name: [row for row in rows if row.get("head_eligible") is not False]
        for name, rows in (("train", train_outcomes), ("valid", valid_outcomes))
    }
    for name, rows in eligible_sets.items():
        positives = sum(int(row["label"]) for row in rows)
        negatives = len(rows) - positives
        if positives < minimum_class_rows or negatives < minimum_class_rows:
            raise RuntimeError(
                "confidence head-eligible "
                f"{name} coverage below floor: {positives}/{negatives}"
            )
    train_eligible = eligible_sets["train"]
    valid_eligible = eligible_sets["valid"]
    by_label = {
        label: [row for row in train_eligible if int(row["label"]) == label]
        for label in (0, 1)
    }
    head = ConfidenceV2Head(runtime.model.cfg.d_model)
    mx.eval(head.parameters())
    optimizer = optim.Adam(learning_rate=lr)
    data_fingerprint = contract.sha_bytes(
        contract.canonical_bytes(
            [
                [row["sample_id"], int(row["label"]), True]
                for row in train_eligible
            ]
        )
    )
    state_path = checkpoint_dir / "confidence-v3-state.npz" if checkpoint_dir else None
    expected_meta = {
        "kind": "mei-sft-v3-confidence-state",
        "implementation": TRAINING_ID,
        "label_contract": CONFIDENCE_LABEL_ID,
        "score_contract": CONFIDENCE_SCORE_ID,
        "label_uniform": True,
        "data_fingerprint": data_fingerprint,
        "target_steps": steps,
    }
    start_step = 0
    last_loss: float | None = None
    if resume and state_path and state_path.is_file():
        meta = load_train_state(state_path, head, optimizer, mode="strict", expected_meta=expected_meta)
        start_step = int(meta.get("steps_completed") or 0)
        last_loss = meta.get("last_loss")
    for step in range(start_step, steps):
        label = step % 2
        occurrence = step // 2
        choices = list(by_label[label])
        epoch = occurrence // len(choices)
        random.Random(2303 + label * 101 + epoch).shuffle(choices)
        sample = choices[occurrence % len(choices)]
        frozen = runtime.model(
            mx.array([sample["prompt_ids"]], dtype=mx.int32), return_cells=True
        )["cells"]
        mx.eval(*frozen)
        cells = [mx.stop_gradient(value).astype(mx.float16) for value in frozen]

        def loss_fn(candidate_head: Any) -> Any:
            logit = candidate_head(cells).reshape(())
            target = mx.array(float(label), dtype=mx.float32)
            return nn.losses.binary_cross_entropy(logit, target, with_logits=True, reduction="mean")

        loss, gradients = mx.value_and_grad(loss_fn)(head)
        optimizer.update(head, gradients)
        mx.eval(head.parameters(), loss)
        last_loss = float(loss.item())
        if state_path and ((step + 1) % max(1, checkpoint_every_steps) == 0 or step + 1 == steps):
            save_train_state(
                state_path,
                head,
                optimizer,
                {**expected_meta, "steps_completed": step + 1, "last_loss": last_loss},
            )
    raw_scores: list[float] = []
    labels: list[int] = []
    for sample in valid_eligible:
        frozen = runtime.model(
            mx.array([sample["prompt_ids"]], dtype=mx.int32), return_cells=True
        )["cells"]
        logit = head([mx.stop_gradient(value) for value in frozen]).reshape(())
        mx.eval(logit)
        raw_scores.append(
            combined_confidence_score(
                float(logit.item()),
                float(sample["logprob_sum"]),
                int(sample["output_tokens"]),
            )
        )
        labels.append(int(sample["label"]))
    calibration = _fit_platt(raw_scores, labels)
    calibrated_rows = [
        {"label": label, "score": apply_platt(score, calibration)}
        for score, label in zip(raw_scores, labels)
    ]
    head_metrics = confidence_metrics(
        calibrated_rows, minimum_class_rows=minimum_class_rows
    )
    calibrated_by_id = {
        str(sample["sample_id"]): row
        for sample, row in zip(valid_eligible, calibrated_rows)
    }
    pipeline_rows = [
        calibrated_by_id.get(
            str(sample["sample_id"]),
            {"label": int(sample["label"]), "score": 0.0},
        )
        for sample in valid_outcomes
    ]
    pipeline_metrics = confidence_metrics(
        pipeline_rows, minimum_class_rows=minimum_class_rows
    )
    runtime.conf_v2 = head
    return (
        {
            "implementation": TRAINING_ID,
            "steps": steps,
            "resumed_from_step": start_step,
            "last_loss": last_loss,
            "n_train": len(train_outcomes),
            "n_valid": len(valid_outcomes),
            "n_train_head_eligible": len(train_eligible),
            "n_valid_head_eligible": len(valid_eligible),
            "train_deterministic_bypass": len(train_outcomes) - len(train_eligible),
            "valid_deterministic_bypass": len(valid_outcomes) - len(valid_eligible),
            "train_positive": len(by_label[1]),
            "train_negative": len(by_label[0]),
            "label_uniform": True,
            "correct_refusal_is_positive": True,
            "score_contract": CONFIDENCE_SCORE_ID,
            "label_contract": CONFIDENCE_LABEL_ID,
            "calibration": {"kind": "platt-on-combined-score-v1", **calibration},
            # The preregistered head-quality gate is deliberately not inflated
            # by easy deterministic rejections. The invocation-aware aggregate
            # is retained separately; runtime safety remains in gate receipts.
            "valid_metrics": head_metrics,
            "pipeline_valid_metrics": pipeline_metrics,
            "deterministic_bypass_contract": "validator-before-confidence-v1",
            "lm_frozen": True,
            "data_fingerprint": data_fingerprint,
        },
        calibration,
    )


def combined_confidence_score(
    head_logit: float, logprob_sum: float, output_tokens: int
) -> float:
    value = float(head_logit)
    p_head = (
        1.0 / (1.0 + math.exp(-value))
        if value >= 0
        else math.exp(value) / (1.0 + math.exp(value))
    )
    p_decode = math.exp(min(0.0, float(logprob_sum) / max(int(output_tokens), 1)))
    return min(p_head, p_decode)


def verify_release_contract(release_dir: Path) -> dict[str, Any]:
    manifest = contract.load_json(release_dir / "manifest.json")
    allowed = {
        "mei-sft-data-release-v3": "mei-sft-data-contract-v3",
        "mei-sft-data-release-v4": "mei-sft-data-contract-v4",
    }
    if (
        manifest.get("schema") not in allowed
        or manifest.get("status") != "frozen"
        or manifest.get("contract_id") != allowed.get(str(manifest.get("schema")))
    ):
        raise RuntimeError("not a supported frozen SFT release")
    for name, spec in (
        manifest.get("outputs") or manifest.get("artifacts") or {}
    ).items():
        path = release_dir / name
        if not path.is_file() or contract.sha_file(path) != spec.get("sha256"):
            raise RuntimeError(f"SFT artifact drift: {path}")
    isolation = contract.load_json(release_dir / "isolation-receipt.json")
    if isolation.get("status") != "passed":
        raise RuntimeError("SFT isolation receipt did not pass")
    return manifest
