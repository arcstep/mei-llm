"""Deterministic Chinese narration over verified tool-result views only."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

NARRATION_ID = "mei-zh-deterministic-narration-v1"
NARRATION_ADAPTER_ID = "mei-zh-narration-adapter-r16-v2"
NARRATION_PROMPT_ID = "mei-verified-result-narration-prompt-v2"
_VIEW_SEAL = object()


def _compact_value(value: Any, *, limit: int = 240) -> str:
    if isinstance(value, str):
        text = value.strip()
    else:
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def narration_prompt_v2(user_request: str, views: list[dict[str, Any]]) -> str:
    """Render the exact executor-free prompt consumed by the rank-16 adapter."""

    calls = []
    results = []
    for view in views:
        calls.append(
            {
                "call_id": str(view.get("call_id") or ""),
                "name": str(view.get("tool_name") or "该工具"),
                "arguments": dict(view.get("arguments") or {}),
            }
        )
        result = {
            "call_id": str(view.get("call_id") or ""),
            "status": str(view.get("status") or ""),
            "payload": view.get("payload"),
        }
        if view.get("error") is not None:
            result["error"] = view.get("error")
        results.append(result)
    public = {
        "user_request": str(user_request or ""),
        "calls": calls,
        "verified_results": results,
    }
    return (
        "任务：只依据用户请求、已执行调用和 <verified_results> 中的已验证结果，"
        "生成不超过48个token的简洁中文结果说明。必须保留数值、单位、标识符和成功/失败极性；"
        "不得补充结果中不存在的事实，不得调用工具。\n"
        "<verified_results>"
        + json.dumps(
            public,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "</verified_results>\n解说："
    )


@dataclass(frozen=True)
class VerifiedResultView:
    value: dict[str, Any]
    _seal: object


def verified_result_view(value: dict[str, Any]) -> VerifiedResultView:
    """Internal Session boundary: seal one already accepted ToolResult view."""

    return VerifiedResultView(value=dict(value), _seal=_VIEW_SEAL)


@dataclass(frozen=True)
class NarrationProvider:
    """Pure formatter: deliberately has no engine, session or executor handle."""

    locale: str = "zh-CN"

    def narrate(self, result_view: VerifiedResultView) -> str:
        if not isinstance(result_view, VerifiedResultView) or result_view._seal is not _VIEW_SEAL:
            raise ValueError("narration_requires_session_result_view")
        value = result_view.value
        provenance = value.get("provenance")
        verified = bool(
            isinstance(provenance, dict)
            and provenance.get("verified") is True
            and isinstance(provenance.get("source"), str)
            and provenance.get("source")
        )
        if not verified:
            raise ValueError("narration_requires_verified_result")
        call_id = str(value.get("call_id") or "")
        if not call_id:
            raise ValueError("narration_requires_call_id")
        status = str(value.get("status") or "")
        payload = value.get("payload")
        tool_name = str(value.get("tool_name") or "该工具")
        arguments = value.get("arguments") if isinstance(value.get("arguments"), dict) else {}
        payload_object = payload if isinstance(payload, dict) else {}
        device = str(arguments.get("device") or "设备")
        zone = str(arguments.get("zone") or "")
        door = str(arguments.get("door") or "门")
        if status == "error":
            error = value.get("error") if isinstance(value.get("error"), dict) else {}
            detail = str(error.get("message") or _compact_value(error or payload))
            if tool_name == "start_device":
                return f"{device}启动失败：{detail}。"
            if tool_name == "stop_device":
                return f"{device}关闭失败：{detail}。"
            if tool_name == "unlock_door":
                return f"{door}解锁失败：{detail}。"
            return f"{tool_name}执行未成功：{detail}。"
        if status == "cancelled":
            if tool_name == "start_device":
                return f"{device}的启动操作已取消。"
            return f"{tool_name}已取消，未继续执行。"
        if status != "ok":
            raise ValueError("narration_requires_terminal_status")
        if tool_name == "get_temperature" and "temperature_c" in payload_object:
            prefix = f"{zone}当前" if zone else "当前"
            return f"{prefix}温度为{_compact_value(payload_object['temperature_c'])}℃。"
        if tool_name == "set_temperature" and "temperature_c" in payload_object:
            temperature = _compact_value(payload_object["temperature_c"])
            prefix = zone or "目标区域"
            if payload_object.get("applied") is False:
                return f"{prefix}已经是{temperature}℃，无需调整。"
            return f"已将{prefix}温度设为{temperature}℃。"
        if tool_name == "adjust_temperature" and "temperature_c" in payload_object:
            temperature = _compact_value(payload_object["temperature_c"])
            delta = arguments.get("delta_c")
            direction = "调低" if isinstance(delta, (int, float)) and delta < 0 else "调高"
            return f"已将{zone or '目标区域'}温度{direction}到{temperature}℃。"
        if tool_name == "get_humidity" and "humidity_percent" in payload_object:
            prefix = f"{zone}当前" if zone else "当前"
            return f"{prefix}湿度为{_compact_value(payload_object['humidity_percent'])}%。"
        if tool_name == "start_device" and payload_object.get("state") == "on":
            return f"{device}已启动。"
        if tool_name == "stop_device" and payload_object.get("state") == "off":
            return f"{device}已关闭。"
        if tool_name == "set_brightness" and "brightness_percent" in payload_object:
            light = str(arguments.get("light") or "灯光")
            return f"已将{light}亮度调到{_compact_value(payload_object['brightness_percent'])}%。"
        if tool_name == "lock_door" and payload_object.get("locked") is True:
            return f"{door}已锁定。"
        if tool_name == "set_fan_speed" and "level" in payload_object:
            return f"已将{device}调到{_compact_value(payload_object['level'])}档。"
        if tool_name == "create_timer" and "timer_id" in payload_object:
            minutes = _compact_value(payload_object.get("minutes", arguments.get("minutes")))
            return f"已设置{minutes}分钟计时器，编号为{_compact_value(payload_object['timer_id'])}。"
        if tool_name == "cancel_timer" and payload_object.get("cancelled") is True:
            return f"计时器{_compact_value(payload_object.get('timer_id', arguments.get('timer_id')))}已取消。"
        if tool_name == "activate_scene":
            scene = str(payload_object.get("scene") or arguments.get("scene") or "场景")
            if payload_object.get("status") == "partial":
                return (
                    f"{scene}模式已部分启动："
                    f"{_compact_value(payload_object.get('completed'))}项完成，"
                    f"{_compact_value(payload_object.get('failed'))}项失败。"
                )
            if payload_object.get("active") is True:
                return f"{scene}模式已启动。"
        if tool_name == "play_music" and payload_object.get("playing") is True:
            return f"已开始播放{_compact_value(payload_object.get('playlist', arguments.get('playlist')))}。"
        if payload is None or payload == {} or payload == []:
            return f"{tool_name}已执行完成。"
        if isinstance(payload, dict):
            pairs = []
            for key in sorted(payload):
                pairs.append(f"{key}为{_compact_value(payload[key], limit=96)}")
            return f"{tool_name}已执行完成，" + "；".join(pairs) + "。"
        return f"{tool_name}已执行完成，结果为{_compact_value(payload)}。"

    def capabilities(self) -> dict[str, Any]:
        return {
            "provider_id": NARRATION_ID,
            "locale": self.locale,
            "deterministic": True,
            "can_execute_tools": False,
        }
