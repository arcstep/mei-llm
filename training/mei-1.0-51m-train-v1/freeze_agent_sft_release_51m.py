#!/usr/bin/env python3
"""Freeze deterministic multi-step Agent SFT data on top of the 300M release.

The parent data release is immutable.  This script creates a new release whose
full-call split is the parent split plus deployment-shaped call/result
continuations.  Each trajectory keeps one counterfactual group in one split,
uses real ToolResultV2 objects, and terminates with an explicit ``respond``
target (the model action is ``[]``; the runtime distinguishes it from refusal
only after accepting successful results in the same Session).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable

from _repo import ROOT


PARENT_ID = "mei-1.0-51m-tool-sft-v2-300m-v1"
RELEASE_ID = "mei-1.0-51m-tool-sft-v2-agent300m-v1"
SIMULATOR_ID = "mei-agent-host-simulator-v1"
RELEASE_ROOT = ROOT / "notebook/sft/mei-1.0-51m/releases"
PARENT_DIR = RELEASE_ROOT / PARENT_ID
UNIVERSE_PATH = (
    ROOT / "notebook/evaluation/banks/sft-v2-eval-lock-v3-20class/tool-universe-v1.json"
)
CALL_ID = re.compile(r"^call-s[0-9a-f]{8}-[1-8]-[0-9a-f]{12}$")


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def call_id(trajectory_id: str, step: int, call: dict[str, Any]) -> str:
    nonce = sha_bytes(trajectory_id.encode("utf-8"))[:8]
    digest = sha_bytes(
        f"{trajectory_id}:{step}:".encode("utf-8") + canonical(call)
    )[:12]
    return f"call-s{nonce}-{step}-{digest}"


def result_for(
    *, call_id_value: str, payload: Any, source_suffix: str
) -> dict[str, Any]:
    return {
        "wire_version": "mei-runtime-wire-v2",
        "call_id": call_id_value,
        "status": "ok",
        "payload": payload,
        "provenance": {
            "source": f"deterministic-host-simulator:{SIMULATOR_ID}:{source_suffix}",
            "verified": True,
        },
    }


def _travel(index: int) -> tuple[str, list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    cities = [("北京", "上海"), ("成都", "广州"), ("西安", "杭州"), ("武汉", "深圳")]
    source, dest = cities[index % len(cities)]
    date = f"{2026 + index // 336:04d}-{1 + (index // 28) % 12:02d}-{1 + index % 28:02d}"
    query = f"请预订{date}从{source}到{dest}的航班，拿到订座编码后再申请机场轮椅。"
    pnr = f"PNR-{index:05d}"
    return query, [
        ("book_flight", {"from_city": source, "to_city": dest, "date": date}, {"pnr": pnr}),
        ("request_wheelchair", {"pnr": pnr}, {"pnr": pnr, "wheelchair": "confirmed"}),
    ]


def _finance(index: int) -> tuple[str, list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    titles = ["客户差旅", "会议餐费", "项目耗材", "市内交通"]
    title = titles[index % len(titles)]
    amount = float(80 + index)
    query = f"创建一张“{title}”报销单，金额{amount:.0f}元，创建成功后直接批准它。"
    expense_id = f"EXP-{index:05d}"
    return query, [
        ("create_expense", {"title": title, "amount": amount}, {"expense_id": expense_id}),
        ("approve_expense", {"expense_id": expense_id}, {"expense_id": expense_id, "approved": True}),
    ]


def _card(index: int) -> tuple[str, list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    label = f"{['供应商试用', '差旅备用', '订阅测试', '采购临时卡'][index % 4]}-{index:03d}"
    query = f"开通标签为“{label}”的虚拟卡，拿到卡号后立即关闭这张卡。"
    card_id = f"VC-{index:05d}"
    return query, [
        ("open_virtual_card", {"label": label}, {"card_id": card_id}),
        ("close_virtual_card", {"card_id": card_id}, {"card_id": card_id, "closed": True}),
    ]


def _medical(index: int) -> tuple[str, list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    dept = ["内科", "眼科", "皮肤科", "骨科"][index % 4]
    date = f"{2026 + index // 336:04d}-{1 + (index // 28) % 12:02d}-{1 + index % 28:02d}"
    query = f"预约{date}的{dept}门诊，取得预约号以后再取消该预约。"
    appointment_id = f"APT-{index:05d}"
    return query, [
        ("book_clinic", {"dept": dept, "date": date}, {"appointment_id": appointment_id}),
        ("cancel_clinic", {"appointment_id": appointment_id}, {"appointment_id": appointment_id, "cancelled": True}),
    ]


def _logistics(index: int) -> tuple[str, list[tuple[str, dict[str, Any], dict[str, Any]]]]:
    sku = f"{['样品A', '零件B', '试剂C', '资料D'][index % 4]}-{index:03d}"
    dest = ["上海", "成都", "广州", "武汉"][(index // 4) % 4]
    query = f"为{sku}创建发往{dest}的运单，然后打印面单并查询一次运输状态。"
    waybill = f"WB-{index:06d}"
    return query, [
        ("create_shipment", {"sku": sku, "dest": dest}, {"waybill": waybill}),
        ("print_label", {"waybill": waybill}, {"waybill": waybill, "label": f"label://{waybill}"}),
        ("track_shipment", {"waybill": waybill}, {"waybill": waybill, "state": "created"}),
    ]


TEMPLATES: list[
    tuple[str, Callable[[int], tuple[str, list[tuple[str, dict[str, Any], dict[str, Any]]]]], list[str]]
] = [
    ("agent_travel", _travel, ["book_flight", "request_wheelchair", "upgrade_seat", "cancel_flight", "book_hotel"]),
    ("agent_finance", _finance, ["create_expense", "approve_expense", "pay_invoice", "get_balance", "list_transactions"]),
    ("agent_virtual_card", _card, ["open_virtual_card", "close_virtual_card", "freeze_card", "set_payment_limit", "get_balance"]),
    ("agent_medical", _medical, ["book_clinic", "cancel_clinic", "get_queue_number", "book_physical", "get_lab_result"]),
    ("agent_logistics", _logistics, ["create_shipment", "print_label", "track_shipment", "cancel_shipment", "schedule_pickup"]),
]


def trajectory_rows(
    *, family: str, generator: Callable, index: int, split: str, retrieved_tools: list[str]
) -> list[dict[str, Any]]:
    query, operations = generator(index)
    trajectory_id = f"AGT-{family}-{split}-{index:05d}"
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for step, (name, arguments, payload) in enumerate(operations, start=1):
        raw_call = {"name": name, "arguments": arguments}
        identifier = call_id(trajectory_id, step, raw_call)
        call = {**raw_call, "call_id": identifier}
        base = {
            "sample_id": f"{trajectory_id}-S{step}",
            "case_id": trajectory_id,
            "cf_group": trajectory_id,
            "trajectory_id": trajectory_id,
            "trajectory_step": step,
            "trajectory_length": len(operations) + 1,
            "task": "fullcall",
            "query": query,
            "family": family,
            "kind": "execute",
            "split": split,
            "retrieved_tools": retrieved_tools,
            "catalog_tools": retrieved_tools,
            "prior_calls": list(calls),
            "tool_results": list(results),
            "prior_tool_results": list(results),
            "answers": [raw_call],
            "gold_name": name,
            "gold_args": arguments,
            "history": [],
            "permissions": [],
            "slot_provenance": [],
            "agent_trace": {
                "simulator_id": SIMULATOR_ID,
                "trusted_offline_fixture": True,
                "target_kind": "call",
            },
            "serializer": "mei-tool-call-serializer-v2",
            "source_role": "deterministic-host-simulator",
            "generator_version": "mei-agent-continuation-sft-v1",
            "status": "accepted",
        }
        rows.append(base)
        calls.append(call)
        results.append(
            result_for(
                call_id_value=identifier,
                payload=payload,
                source_suffix=name,
            )
        )
    rows.append(
        {
            **{key: value for key, value in rows[-1].items() if key not in {"answers", "gold_name", "gold_args"}},
            "sample_id": f"{trajectory_id}-RESPOND",
            "trajectory_step": len(operations) + 1,
            "kind": "respond",
            "prior_calls": list(calls),
            "tool_results": list(results),
            "prior_tool_results": list(results),
            "answers": [],
            "gold_name": None,
            "gold_args": {},
            "agent_trace": {
                "simulator_id": SIMULATOR_ID,
                "trusted_offline_fixture": True,
                "target_kind": "respond",
            },
        }
    )
    return rows


def build_rows() -> dict[str, list[dict[str, Any]]]:
    rows = {"train": [], "valid": [], "eval": []}
    ranges = {
        "train": range(0, 160),
        "valid": range(160, 180),
        "eval": range(180, 200),
    }
    for split, indices in ranges.items():
        for family, generator, tools in TEMPLATES:
            for index in indices:
                rows[split].extend(
                    trajectory_rows(
                        family=family,
                        generator=generator,
                        index=index,
                        split=split,
                        retrieved_tools=tools,
                    )
                )
    return rows


def validate_rows(rows: dict[str, list[dict[str, Any]]], tool_names: set[str]) -> dict[str, Any]:
    query_sets: dict[str, set[str]] = {}
    group_splits: dict[str, set[str]] = {}
    target_counts: dict[str, int] = {"call": 0, "respond": 0}
    nonempty_result_rows = 0
    for split, split_rows in rows.items():
        query_sets[split] = {str(row["query"]) for row in split_rows}
        for row in split_rows:
            group_splits.setdefault(str(row["cf_group"]), set()).add(split)
            trace = row.get("agent_trace") or {}
            target = str(trace.get("target_kind") or "")
            if target not in target_counts:
                raise RuntimeError(f"unknown agent target kind: {target}")
            target_counts[target] += 1
            calls = row.get("prior_calls") or []
            results = row.get("tool_results") or []
            if len(calls) != len(results):
                raise RuntimeError("agent call/result prefix length mismatch")
            for call, result in zip(calls, results):
                identifier = str(call.get("call_id") or "")
                if not CALL_ID.fullmatch(identifier) or result.get("call_id") != identifier:
                    raise RuntimeError("agent call/result id mismatch")
                if call.get("name") not in tool_names:
                    raise RuntimeError(f"agent fixture references unknown tool {call.get('name')}")
                if result.get("status") != "ok" or (result.get("provenance") or {}).get("verified") is not True:
                    raise RuntimeError("agent fixture result is not verified success")
            if results:
                nonempty_result_rows += 1
            if target == "respond" and (row.get("answers") != [] or not results):
                raise RuntimeError("respond row must be [] after successful ToolResult")
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise RuntimeError("agent trajectory crosses split")
    for left, right in (("train", "valid"), ("train", "eval"), ("valid", "eval")):
        if query_sets[left] & query_sets[right]:
            raise RuntimeError(f"agent query leakage between {left} and {right}")
    return {
        "ok": True,
        "trajectory_group_cross_split": 0,
        "query_overlap": 0,
        "target_counts": target_counts,
        "nonempty_tool_result_rows": nonempty_result_rows,
        "families": sorted({str(row["family"]) for row in rows["train"]}),
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    parent = args.release_root / args.parent_id
    release = args.release_root / args.release_id
    if release.exists():
        raise RuntimeError("refusing to overwrite frozen Agent SFT release")
    parent_manifest_path = parent / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest.get("release_id") != args.parent_id:
        raise RuntimeError("parent SFT release identity mismatch")
    universe = json.loads(args.universe.read_text(encoding="utf-8"))
    tool_names = {str(tool["name"]) for tool in universe.get("tools") or []}
    rows = build_rows()
    isolation = validate_rows(rows, tool_names)

    old_eval_queries: set[str] = set()
    eval_dir = ROOT / "notebook/evaluation/banks/sft-v2-eval-lock-v3-20class"
    for path in eval_dir.glob("eval-*.jsonl"):
        old_eval_queries.update(str(row.get("query") or "") for row in load_jsonl(path))
    train_queries = {str(row["query"]) for row in rows["train"] + rows["valid"]}
    overlap = sorted((old_eval_queries & train_queries) - {""})
    if overlap:
        raise RuntimeError(f"Agent train query overlaps frozen eval: {overlap[0]}")
    isolation["parent_eval_query_overlap"] = 0

    args.release_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mei-agent-sft-freeze-", dir=args.release_root.parent) as temp_name:
        temp = Path(temp_name) / "release"
        shutil.copytree(parent, temp)
        dump_jsonl(temp / "agent-continuation.train.jsonl", rows["train"])
        dump_jsonl(temp / "agent-continuation.valid.jsonl", rows["valid"])
        dump_jsonl(temp / "agent-continuation.eval.jsonl", rows["eval"])
        combined_train = load_jsonl(parent / "full-call.train.jsonl") + rows["train"]
        combined_valid = load_jsonl(parent / "full-call.valid.jsonl") + rows["valid"]
        dump_jsonl(temp / "full-call.train.jsonl", combined_train)
        dump_jsonl(temp / "full-call.valid.jsonl", combined_valid)

        simulator = {
            "schema": "mei-deterministic-host-simulator-v1",
            "simulator_id": SIMULATOR_ID,
            "wire_version": "mei-runtime-wire-v2",
            "semantics": "pure deterministic call arguments to ToolResultV2 payload fixtures",
            "families": [family for family, _, _ in TEMPLATES],
            "result_statuses": ["ok"],
            "network": False,
            "side_effects": False,
        }
        (temp / "host-simulator-v1.json").write_bytes(canonical(simulator))
        receipt = {
            "schema": "mei-agent-sft-isolation-receipt-v1",
            "release_id": args.release_id,
            "parent_release_id": args.parent_id,
            "status": "passed",
            **isolation,
            "runtime_contract": {
                "one_call_per_step": True,
                "tool_result_v2": True,
                "trusted_call_history": True,
                "terminal_kinds": ["respond", "refuse", "error"],
                "default_max_steps": 4,
                "hard_max_steps": 8,
            },
        }
        (temp / "agent-isolation-receipt.json").write_bytes(canonical(receipt))

        manifest = dict(parent_manifest)
        manifest["release_id"] = args.release_id
        manifest["parent_release"] = {
            "release_id": args.parent_id,
            "manifest_sha256": sha_file(parent_manifest_path),
        }
        outputs = dict(manifest["outputs"])
        for name in ("full-call.train.jsonl", "full-call.valid.jsonl"):
            outputs[name] = {
                "sha256": sha_file(temp / name),
                "rows": len(combined_train if name.endswith("train.jsonl") else combined_valid),
            }
        manifest["outputs"] = outputs
        manifest["agent_continuation"] = {
            "schema": "mei-agent-continuation-sft-v1",
            "simulator_id": SIMULATOR_ID,
            "simulator_file": "host-simulator-v1.json",
            "simulator_sha256": sha_file(temp / "host-simulator-v1.json"),
            "isolation_receipt": "agent-isolation-receipt.json",
            "isolation_receipt_sha256": sha_file(temp / "agent-isolation-receipt.json"),
            "files": {
                f"agent-continuation.{split}.jsonl": {
                    "rows": len(rows[split]),
                    "sha256": sha_file(temp / f"agent-continuation.{split}.jsonl"),
                }
                for split in ("train", "valid", "eval")
            },
            "target_kinds": ["call", "respond"],
            "nonempty_tool_result_rows": isolation["nonempty_tool_result_rows"],
            "family_aware": True,
            "group_aware": True,
        }
        manifest["isolation"] = {
            **dict(manifest.get("isolation") or {}),
            "agent_query_overlap": 0,
            "agent_trajectory_group_cross_split": 0,
        }
        (temp / "manifest.json").write_bytes(canonical(manifest))
        temp.replace(release)
    return {
        "ok": True,
        "release_id": args.release_id,
        "release_dir": str(release),
        "parent_release_id": args.parent_id,
        "manifest_sha256": sha_file(release / "manifest.json"),
        "agent_rows": {split: len(value) for split, value in rows.items()},
        "nonempty_tool_result_rows": isolation["nonempty_tool_result_rows"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", default=RELEASE_ID)
    parser.add_argument("--parent-id", default=PARENT_ID)
    parser.add_argument("--release-root", type=Path, default=RELEASE_ROOT)
    parser.add_argument("--universe", type=Path, default=UNIVERSE_PATH)
    args = parser.parse_args()
    print(json.dumps(freeze(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
