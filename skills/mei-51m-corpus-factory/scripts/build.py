#!/usr/bin/env python3
"""Stable skill entrypoint for the mei-1.0-51m SFT corpus builder."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType

from _shared import find_repo_root


TARGET_NAMES = (
    "RETRIEVAL_TARGETS",
    "FULLCALL_TARGETS",
    "AGENT_TARGETS",
    "MW_TARGETS",
    "NARRATION_TARGETS",
    "CONFIDENCE_TARGETS",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "运行 corpus-factory/rebuild_zh_v1，并以显式、不可覆盖的 release ID "
            "写入 .local/artifacts。"
        )
    )
    parser.add_argument(
        "--release-id",
        required=True,
        help="新的不可变 release ID；必须以 mei-1.0-51m- 开头",
    )
    parser.add_argument(
        "--cycle-id",
        default="exp-000600m",
        help="产物所属累计 exposure cycle（默认：exp-000600m）",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="把各目标计数封顶到 --pilot-limit，先做小样本验证",
    )
    parser.add_argument(
        "--pilot-limit",
        type=int,
        default=20,
        help="pilot 模式下每个目标的最大计数（默认：20）",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="只输出解析后的执行计划，不生成语料",
    )
    parser.add_argument(
        "--targets-json",
        type=Path,
        help="可选：由 register_targets.py 生成的 family/配额注册文件",
    )
    return parser.parse_args()


def validate_release_id(value: str) -> None:
    if not value.startswith("mei-1.0-51m-"):
        raise SystemExit("--release-id 必须以 mei-1.0-51m- 开头")
    if "/" in value or value in {".", ".."}:
        raise SystemExit("--release-id 不得包含路径分隔符")


def load_builder(root: Path) -> ModuleType:
    path = root / "corpus-factory/generators/rebuild_zh_v1/build.py"
    spec = importlib.util.spec_from_file_location("mei_51m_rebuild_zh", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"无法加载生成器：{path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cap_targets(module: ModuleType, limit: int) -> None:
    if limit < 1:
        raise SystemExit("--pilot-limit 必须大于 0")
    for name in TARGET_NAMES:
        targets = getattr(module, name)
        setattr(
            module,
            name,
            {
                key: min(value, limit) if isinstance(value, int) and value > 0 else value
                for key, value in targets.items()
            },
        )


def apply_registered_targets(module: ModuleType, path: Path) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "mei-51m-sft-target-registration-v1":
        raise SystemExit(f"invalid targets schema: {path}")
    targets = value.get("targets")
    if not isinstance(targets, dict) or set(targets) - set(TARGET_NAMES):
        raise SystemExit(f"unknown targets maps: {sorted(set(targets or {}) - set(TARGET_NAMES))}")
    for name, rows in targets.items():
        current = getattr(module, name)
        if not isinstance(rows, dict) or set(rows) - set(current):
            raise SystemExit(f"{name}: unknown target keys")
        merged = dict(current)
        for key, count in rows.items():
            if not isinstance(count, int) or count < 0:
                raise SystemExit(f"{name}.{key}: target must be a non-negative integer")
            merged[key] = count
        setattr(module, name, merged)


def main() -> int:
    args = parse_args()
    validate_release_id(args.release_id)
    if not re.fullmatch(r"exp-\d{6}m", args.cycle_id):
        raise SystemExit("--cycle-id 必须形如 exp-000900m")
    root = find_repo_root()
    builder = load_builder(root)
    cycle_root = root / ".local/artifacts/mei-1.0-51m" / args.cycle_id / "corpus"
    builder.C.CYCLE_ID = args.cycle_id
    builder.C.RELEASE_ROOT = cycle_root / "sft-suite"
    builder.C.EVAL_LOCK_ROOT = cycle_root / "eval-lock"
    builder.RELEASE_ID = args.release_id
    builder.RELEASE_DIR = builder.C.RELEASE_ROOT / args.release_id

    if builder.RELEASE_DIR.exists():
        raise SystemExit(
            f"拒绝覆盖已有不可变 release：{builder.RELEASE_DIR}\n"
            "请使用新的 --release-id。"
        )
    if args.targets_json:
        apply_registered_targets(builder, args.targets_json)
    if args.pilot:
        cap_targets(builder, args.pilot_limit)

    plan = {
        "repo_root": str(root),
        "cycle_id": args.cycle_id,
        "release_id": args.release_id,
        "release_dir": str(builder.RELEASE_DIR),
        "mode": "pilot" if args.pilot else "scale",
        "targets": {name: getattr(builder, name) for name in TARGET_NAMES},
    }
    if args.plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    print(json.dumps({"execution_plan": plan}, ensure_ascii=False, indent=2), flush=True)
    return int(builder.main())


if __name__ == "__main__":
    sys.exit(main())
