#!/usr/bin/env python3
"""Freeze the bounded Chinese narration v2 corpus.

The v1 release proved that a verified ToolResult can be rendered without an
executor.  V2 adds the semantic coverage needed by an actual small-device
runtime: scalar queries, set/adjust actions, power state, success/error/
cancelled outcomes, no-op/partial results, and multi-step summaries.  It is a
new immutable release; the existing v1 artifact is never rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from common._repo import ROOT

import sys

sys.path.insert(0, str(ROOT / "platform/_shared/runtime"))
from narration import NarrationProvider, narration_prompt_v2, verified_result_view  # noqa: E402


PARENT_AGENT_ID = "mei-1.0-51m-tool-sft-v2-agent300m-v1"
PARENT_NARRATION_ID = "mei-1.0-51m-narration-sft-agent300m-v1"
# The first v2-format artifact was frozen before the deterministic semantic
# renderer parity check existed.  Preserve it and emit the corrected revision
# under a new immutable release id.
RELEASE_ID = "mei-1.0-51m-narration-sft-agent300m-v3"
RELEASE_ROOT = ROOT / "artifacts/mei-1.2-51m/legacy/mei-1.0-51m/exp-00300m/corpus/sft-suite/historical-notebook-releases/releases"
PROVIDER_ID = "mei-zh-narration-adapter-r16-v2"
PROMPT_ID = "mei-verified-result-narration-prompt-v2"
WIRE = "mei-runtime-wire-v2"
SYNTHETIC_PER_SPLIT = {"train": 200, "valid": 25, "eval": 25}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(canonical(row).decode("utf-8") + "\n" for row in rows),
        encoding="utf-8",
    )


def call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"call_id": call_id, "name": name, "arguments": arguments}


def result(
    call_id: str,
    *,
    status: str,
    payload: Any,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "wire_version": WIRE,
        "call_id": call_id,
        "status": status,
        "payload": payload,
        "provenance": {"source": "frozen-narration-v2-simulator", "verified": True},
    }
    if error is not None:
        row["error"] = error
    return row


def prompt(query: str, calls: list[dict[str, Any]], results: list[dict[str, Any]]) -> str:
    views = []
    by_id = {str(item["call_id"]): item for item in results}
    for item in calls:
        one_result = by_id[str(item["call_id"])]
        views.append(
            {
                "call_id": item["call_id"],
                "tool_name": item["name"],
                "arguments": item["arguments"],
                "status": one_result["status"],
                "payload": one_result["payload"],
                "error": one_result.get("error"),
            }
        )
    return narration_prompt_v2(query, views)


def case(
    family: str,
    index: int,
    call_specs: list[tuple[str, dict[str, Any], str, Any, dict[str, str] | None]],
    query: str,
    target: str,
    required_facts: list[str],
    *,
    polarity: str,
) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    views: list[dict[str, Any]] = []
    for step, (name, arguments, status, payload, error) in enumerate(call_specs, 1):
        seed = hashlib.sha256(f"{family}:{index}:{step}".encode()).hexdigest()
        call_id = f"call-s{seed[:8]}-{step}-{seed[8:20]}"
        one_call = call(call_id, name, arguments)
        one_result = result(call_id, status=status, payload=payload, error=error)
        calls.append(one_call)
        results.append(one_result)
        views.append(
            {
                "call_id": call_id,
                "tool_name": name,
                "arguments": arguments,
                "status": status,
                "payload": payload,
                "error": error,
                "provenance": dict(one_result["provenance"]),
            }
        )
    return {
        "query": query,
        "calls": calls,
        "results": results,
        "views": views,
        "target": target,
        "required_facts": required_facts,
        "polarity": polarity,
    }


def semantic_cases(index: int) -> list[tuple[str, dict[str, Any]]]:
    temperature = 18 + index % 16
    target_temperature = 20 + index % 11
    humidity = 35 + index % 51
    brightness = 10 + (index % 10) * 10
    minutes = 1 + index % 90
    device = ["客厅空调", "卧室空调", "空气净化器", "加湿器"][index % 4]
    scene = ["回家", "离家", "睡眠", "观影"][index % 4]
    timer_id = f"T-{index:05d}"
    cases: list[tuple[str, dict[str, Any]]] = []

    def add(family: str, value: dict[str, Any]) -> None:
        cases.append((family, value))

    add("temperature_query", case(
        "temperature_query", index,
        [("get_temperature", {"zone": "客厅"}, "ok", {"temperature_c": temperature}, None)],
        "客厅现在多少度？", f"客厅当前温度为{temperature}℃。", [str(temperature), "客厅", "温度"], polarity="query",
    ))
    add("temperature_set", case(
        "temperature_set", index,
        [("set_temperature", {"zone": "客厅", "target_c": target_temperature}, "ok", {"applied": True, "temperature_c": target_temperature}, None)],
        f"把客厅温度设为{target_temperature}度", f"已将客厅温度设为{target_temperature}℃。", [str(target_temperature), "客厅", "已"], polarity="success",
    ))
    add("temperature_raise", case(
        "temperature_raise", index,
        [("adjust_temperature", {"zone": "客厅", "delta_c": 1}, "ok", {"temperature_c": target_temperature}, None)],
        "把客厅温度调高一度", f"已将客厅温度调高到{target_temperature}℃。", [str(target_temperature), "调高", "已"], polarity="success",
    ))
    add("temperature_lower", case(
        "temperature_lower", index,
        [("adjust_temperature", {"zone": "卧室", "delta_c": -1}, "ok", {"temperature_c": target_temperature}, None)],
        "把卧室温度调低一度", f"已将卧室温度调低到{target_temperature}℃。", [str(target_temperature), "调低", "已"], polarity="success",
    ))
    add("humidity_query", case(
        "humidity_query", index,
        [("get_humidity", {"zone": "卧室"}, "ok", {"humidity_percent": humidity}, None)],
        "卧室湿度是多少？", f"卧室当前湿度为{humidity}%。", [str(humidity), "卧室", "湿度"], polarity="query",
    ))
    add("device_start", case(
        "device_start", index,
        [("start_device", {"device": device}, "ok", {"state": "on"}, None)],
        f"启动{device}", f"{device}已启动。", [device, "已启动"], polarity="success_on",
    ))
    add("device_stop", case(
        "device_stop", index,
        [("stop_device", {"device": device}, "ok", {"state": "off"}, None)],
        f"关闭{device}", f"{device}已关闭。", [device, "已关闭"], polarity="success_off",
    ))
    add("device_start_error", case(
        "device_start_error", index,
        [("start_device", {"device": device}, "error", None, {"code": "device_offline", "message": "设备离线"})],
        f"启动{device}", f"{device}启动失败：设备离线。", [device, "失败", "设备离线"], polarity="failure_on",
    ))
    add("device_cancelled", case(
        "device_cancelled", index,
        [("start_device", {"device": device}, "cancelled", None, None)],
        f"启动{device}", f"{device}的启动操作已取消。", [device, "已取消"], polarity="cancelled",
    ))
    add("brightness_set", case(
        "brightness_set", index,
        [("set_brightness", {"light": "客厅主灯", "percent": brightness}, "ok", {"brightness_percent": brightness}, None)],
        f"把客厅主灯亮度调到{brightness}%", f"已将客厅主灯亮度调到{brightness}%。", [str(brightness), "客厅主灯", "已"], polarity="success",
    ))
    add("door_lock", case(
        "door_lock", index,
        [("lock_door", {"door": "入户门"}, "ok", {"locked": True}, None)],
        "锁上入户门", "入户门已锁定。", ["入户门", "已锁定"], polarity="success_lock",
    ))
    add("door_unlock_error", case(
        "door_unlock_error", index,
        [("unlock_door", {"door": "入户门"}, "error", None, {"code": "permission_denied", "message": "权限不足"})],
        "打开入户门", "入户门解锁失败：权限不足。", ["入户门", "失败", "权限不足"], polarity="failure_unlock",
    ))
    add("fan_speed", case(
        "fan_speed", index,
        [("set_fan_speed", {"device": "循环扇", "level": 3}, "ok", {"level": 3}, None)],
        "把循环扇调到三档", "已将循环扇调到3档。", ["循环扇", "3", "已"], polarity="success",
    ))
    add("timer_create", case(
        "timer_create", index,
        [("create_timer", {"minutes": minutes}, "ok", {"timer_id": timer_id, "minutes": minutes}, None)],
        f"设置一个{minutes}分钟的计时器", f"已设置{minutes}分钟计时器，编号为{timer_id}。", [str(minutes), timer_id, "已设置"], polarity="success",
    ))
    add("timer_cancel", case(
        "timer_cancel", index,
        [("cancel_timer", {"timer_id": timer_id}, "ok", {"cancelled": True, "timer_id": timer_id}, None)],
        f"取消计时器{timer_id}", f"计时器{timer_id}已取消。", [timer_id, "已取消"], polarity="success",
    ))
    add("scene_activate", case(
        "scene_activate", index,
        [("activate_scene", {"scene": scene}, "ok", {"scene": scene, "active": True}, None)],
        f"开启{scene}模式", f"{scene}模式已启动。", [scene, "已启动"], polarity="success_on",
    ))
    add("music_start", case(
        "music_start", index,
        [("play_music", {"playlist": "每日推荐"}, "ok", {"playing": True, "playlist": "每日推荐"}, None)],
        "播放每日推荐", "已开始播放每日推荐。", ["每日推荐", "已开始"], polarity="success_on",
    ))
    add("no_change", case(
        "no_change", index,
        [("set_temperature", {"zone": "客厅", "target_c": target_temperature}, "ok", {"applied": False, "reason": "already_at_target", "temperature_c": target_temperature}, None)],
        f"把客厅温度设为{target_temperature}度", f"客厅已经是{target_temperature}℃，无需调整。", [str(target_temperature), "无需调整"], polarity="no_change",
    ))
    add("partial_action", case(
        "partial_action", index,
        [("activate_scene", {"scene": scene}, "ok", {"status": "partial", "completed": 2, "failed": 1}, None)],
        f"开启{scene}模式", f"{scene}模式已部分启动：2项完成，1项失败。", [scene, "2", "1", "部分", "失败"], polarity="partial",
    ))
    add("multi_step_climate", case(
        "multi_step_climate", index,
        [
            ("get_temperature", {"zone": "客厅"}, "ok", {"temperature_c": temperature}, None),
            ("set_temperature", {"zone": "客厅", "target_c": target_temperature}, "ok", {"applied": True, "temperature_c": target_temperature}, None),
        ],
        f"看看客厅温度并调到{target_temperature}度",
        f"客厅当前温度为{temperature}℃。\n已将客厅温度设为{target_temperature}℃。",
        [str(temperature), str(target_temperature), "当前温度", "已将"], polarity="success_multi",
    ))
    return cases


def old_rows(parent: Path, split: str) -> list[dict[str, Any]]:
    rows = load_jsonl(parent / f"narration.{split}.jsonl")
    upgraded = []
    for row in rows:
        item = dict(row)
        item["provider_id"] = PROVIDER_ID
        item["prompt_id"] = PROMPT_ID
        item["semantic_family"] = "legacy_agent_result"
        item["polarity"] = "success"
        item["required_facts"] = []
        item["source_release"] = PARENT_NARRATION_ID
        item["grounding_target"] = "bounded-grounded-chinese-v2"
        upgraded.append(item)
    return upgraded


def build_rows(parent: Path) -> dict[str, list[dict[str, Any]]]:
    provider = NarrationProvider()
    output: dict[str, list[dict[str, Any]]] = {}
    split_offset = {"train": 0, "valid": 100_000, "eval": 200_000}
    for split in ("train", "valid", "eval"):
        rows = old_rows(parent, split)
        for local in range(SYNTHETIC_PER_SPLIT[split]):
            index = split_offset[split] + local
            for family, value in semantic_cases(index):
                views = value["views"]
                deterministic_target = "\n".join(
                    provider.narrate(verified_result_view(view)) for view in views
                )
                if value["target"] != deterministic_target:
                    raise RuntimeError(
                        f"semantic target drift for {family}: {value['target']!r} != {deterministic_target!r}"
                    )
                sample = {
                    "sample_id": f"NAR2-{family}-{index:06d}",
                    "trajectory_id": f"NAR2-{family}-{index:06d}",
                    "cf_group": f"NAR2-polarity-{index:06d}",
                    "family": "narration_device_control",
                    "semantic_family": family,
                    "polarity": value["polarity"],
                    "split": split,
                    "prompt": prompt(value["query"], value["calls"], value["results"]),
                    "target": value["target"],
                    "required_facts": value["required_facts"],
                    "verified_result_views": views,
                    "call_ids": [view["call_id"] for view in views],
                    "prompt_id": PROMPT_ID,
                    "provider_id": PROVIDER_ID,
                    "grounding_target": "bounded-grounded-chinese-v2",
                    "can_execute_tools": False,
                    "source_release": "synthetic-semantic-v2",
                }
                rows.append(sample)
        rows.sort(key=lambda row: str(row["sample_id"]).encode("utf-8"))
        output[split] = rows
    return output


def validate(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    groups: dict[str, set[str]] = {}
    prompts: dict[str, set[str]] = {}
    statuses: set[str] = set()
    families: set[str] = set()
    polarities: set[str] = set()
    for split, values in rows.items():
        groups[split] = {str(row["cf_group"]) for row in values}
        prompts[split] = {str(row["prompt"]) for row in values}
        if len(prompts[split]) != len(values):
            raise RuntimeError(f"duplicate narration prompts in {split}")
        for row in values:
            if row.get("can_execute_tools") is not False or not row.get("verified_result_views"):
                raise RuntimeError("unsafe narration row")
            if len(str(row["target"])) > 160:
                raise RuntimeError("narration target exceeds bounded character budget")
            families.add(str(row.get("semantic_family") or ""))
            polarities.add(str(row.get("polarity") or ""))
            for view in row["verified_result_views"]:
                if (view.get("provenance") or {}).get("verified") is not True:
                    raise RuntimeError("unverified narration result")
                statuses.add(str(view.get("status") or ""))
    pairs = (("train", "valid"), ("train", "eval"), ("valid", "eval"))
    if any(groups[left] & groups[right] for left, right in pairs):
        raise RuntimeError("narration counterfactual group crosses splits")
    if any(prompts[left] & prompts[right] for left, right in pairs):
        raise RuntimeError("narration prompt leakage across splits")
    required_statuses = {"ok", "error", "cancelled"}
    required_polarities = {
        "query", "success_on", "success_off", "failure_on", "cancelled",
        "no_change", "partial", "success_multi",
    }
    if not required_statuses <= statuses or not required_polarities <= polarities:
        raise RuntimeError("narration v2 semantic coverage is incomplete")
    return {
        "status_values": sorted(statuses),
        "semantic_families": sorted(families),
        "polarities": sorted(polarities),
        "group_cross_split": 0,
        "prompt_overlap": 0,
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    output = args.release_root / args.release_id
    if output.exists():
        raise RuntimeError("refusing to overwrite frozen narration v2 release")
    parent = args.release_root / args.parent_narration_id
    parent_manifest = parent / "manifest.json"
    agent_manifest = args.release_root / args.parent_agent_id / "manifest.json"
    rows = build_rows(parent)
    coverage = validate(rows)
    with tempfile.TemporaryDirectory(prefix="mei-narration-v2-freeze-", dir=args.release_root.parent) as name:
        temp = Path(name) / "release"
        temp.mkdir()
        files: dict[str, Any] = {}
        for split in ("train", "valid", "eval"):
            filename = f"narration.{split}.jsonl"
            dump_jsonl(temp / filename, rows[split])
            files[filename] = {"rows": len(rows[split]), "sha256": sha_file(temp / filename)}
        receipt = {
            "schema": "mei-narration-data-isolation-receipt-v2",
            "status": "passed",
            **coverage,
            "verified_result_only": True,
            "executor_access": False,
            "terminal_only": True,
            "max_target_characters": 160,
            "adapter_acceptance": "semantic-facts-or-deterministic-fallback",
        }
        (temp / "isolation-receipt.json").write_bytes(canonical(receipt))
        manifest = {
            "schema": "mei-narration-sft-data-release-v2",
            "release_id": args.release_id,
            "product": "mei-1.0-51m",
            "parents": {
                "agent": {"release_id": args.parent_agent_id, "manifest_sha256": sha_file(agent_manifest)},
                "narration_v1": {"release_id": args.parent_narration_id, "manifest_sha256": sha_file(parent_manifest)},
            },
            "provider_id": PROVIDER_ID,
            "prompt_id": PROMPT_ID,
            "adapter": {
                "kind": "frozen-backbone-logit-residual",
                "rank": 16,
                "parameter_count": 392_192,
            },
            "coverage": coverage,
            "files": files,
            "isolation_receipt": {"file": "isolation-receipt.json", "sha256": sha_file(temp / "isolation-receipt.json")},
        }
        (temp / "manifest.json").write_bytes(canonical(manifest))
        temp.replace(output)
    return {
        "ok": True,
        "release_id": args.release_id,
        "release_dir": str(output),
        "rows": {split: len(value) for split, value in rows.items()},
        "coverage": coverage,
        "manifest_sha256": sha_file(output / "manifest.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", default=RELEASE_ID)
    parser.add_argument("--parent-agent-id", default=PARENT_AGENT_ID)
    parser.add_argument("--parent-narration-id", default=PARENT_NARRATION_ID)
    parser.add_argument("--release-root", type=Path, default=RELEASE_ROOT)
    args = parser.parse_args()
    print(json.dumps(freeze(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
