#!/usr/bin/env python3
"""Read-only environment checks for the mei-1.0-51m corpus factory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from _shared import find_repo_root, preferred_python


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读检查代码入口、Python 环境、工具注册表、tokenizer 与产物目录。"
    )
    parser.add_argument(
        "--strict-tokenizer",
        action="store_true",
        help="tokenizer 回退为字符计数时返回失败",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = find_repo_root()
    generators = root / "corpus-factory/generators"
    sys.path.insert(0, str(generators))

    checks: dict[str, dict[str, Any]] = {}

    def record(name: str, ok: bool, detail: Any) -> None:
        checks[name] = {"ok": ok, "detail": detail}

    required = (
        "corpus-factory/FACTORY.json",
        "corpus-factory/generators/rebuild_zh_v1/build.py",
        "corpus-factory/verifiers/verify_sft_zh_rebuild_v1.py",
        "CURRENT.json",
    )
    missing = [relative for relative in required if not (root / relative).is_file()]
    record("required_files", not missing, {"missing": missing})

    expected_python = preferred_python(root)
    record(
        "python",
        Path(sys.executable).resolve() == expected_python.resolve(),
        {
            "running": sys.executable,
            "preferred": str(expected_python),
            "hint": f"{expected_python} {Path(__file__).name}",
        },
    )

    try:
        from rebuild_zh_v1 import common as common

        deploy = common.load_deploy_tools()
        training = common.load_training_tools()
        fallback = bool(common.tokenizer_is_fallback())
        record(
            "tool_registries",
            deploy.n_tools > 0 and training.n_tools > 0,
            {
                "deploy_tools": deploy.n_tools,
                "deploy_families": len(deploy.families),
                "training_tools": training.n_tools,
                "training_families": len(training.families),
            },
        )
        record(
            "tokenizer",
            not fallback or not args.strict_tokenizer,
            {
                "fallback_used": fallback,
                "strict": args.strict_tokenizer,
            },
        )
        output_parent = common.RELEASE_ROOT.parent
        writable_parent = next(
            (path for path in (output_parent, *output_parent.parents) if path.exists()),
            root,
        )
        record(
            "artifact_output",
            os.access(writable_parent, os.W_OK),
            {
                "release_root": str(common.RELEASE_ROOT),
                "existing_parent": str(writable_parent),
            },
        )
    except Exception as error:  # doctor must report import/configuration failures
        record("factory_import", False, f"{type(error).__name__}: {error}")

    ok = all(item["ok"] for item in checks.values())
    print(
        json.dumps(
            {"ok": ok, "repo_root": str(root), "checks": checks},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
