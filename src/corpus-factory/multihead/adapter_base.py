#!/usr/bin/env python3
"""Adapter 接口（intake）与注册表。

每个数据集一个适配器，读原始字节 → 产出统一中间表示（TaskEvidence）。
适配器不做五头派生、不做语义审核，只做「忠实映射到契约」；映射里有依据的
归一化（枚举映射、单位换算、搜索改写）必须记录派生规则，无依据的猜值不允许。

交付物（0405 第 12 节）：源格式说明、目标字段映射、不能恢复的信息、例子与
反例、自动验证器版本、试批报告。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from evidence import TaskEvidence


class AdapterError(RuntimeError):
    pass


@dataclass
class AdapterLimits:
    """试批上限。新来源/适配器最多 200 条跑通映射（0405 第 9 节）。"""
    max_records: int = 200


@dataclass
class AdapterManifest:
    """适配器交付物：让新数据集接入可审计、可复核。"""
    source_id: str
    source_format: str = ""                       # 源格式说明
    field_mapping: dict[str, Any] = field(default_factory=dict)   # 目标字段映射
    unrecoverable: list[str] = field(default_factory=list)        # 不能恢复的信息
    examples: list[dict[str, Any]] = field(default_factory=list)  # 例子
    counterexamples: list[dict[str, Any]] = field(default_factory=list)  # 反例
    verifier_version: str = ""                    # 自动验证器版本
    trial_report: str = ""                        # 试批报告

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_format": self.source_format,
            "field_mapping": self.field_mapping,
            "unrecoverable": self.unrecoverable,
            "examples": self.examples,
            "counterexamples": self.counterexamples,
            "verifier_version": self.verifier_version,
            "trial_report": self.trial_report,
        }


@runtime_checkable
class Adapter(Protocol):
    """数据源适配器协议。每个公开工具调用数据集实现一个。"""
    source_id: str

    def load(self, raw_path: Path, *, limits: AdapterLimits) -> list[TaskEvidence]:
        """读原始字节 → 产出统一中间表示。不做五头派生，不做语义审核。"""
        ...

    def manifest(self) -> AdapterManifest:
        """源格式说明、字段映射、恢复不了的信息、例子与反例、验证器版本、试批报告。"""
        ...


# source_id -> Adapter 类。pipeline 按 source_id 查找适配器。
_ADAPTERS: dict[str, type[Adapter]] = {}


def register(source_id: str) -> Any:
    """类装饰器：把适配器按 source_id 登记进注册表。"""
    def decorate(cls: type[Adapter]) -> type[Adapter]:
        if source_id in _ADAPTERS:
            raise AdapterError(f"适配器 source_id 重复: {source_id}")
        cls.source_id = source_id
        _ADAPTERS[source_id] = cls
        return cls
    return decorate


def adapter_for(source_id: str) -> type[Adapter]:
    try:
        return _ADAPTERS[source_id]
    except KeyError:
        raise AdapterError(f"未注册的 source_id: {source_id}") from None


def registered_ids() -> list[str]:
    return sorted(_ADAPTERS)
