#!/usr/bin/env python3
"""多head流水线编排：intake → derive → audit。

五段流水线（0405 第 8 节）在此只跑前三段（intake/derive/audit），
split/compile/freeze 由后续模块接续。标杆阶段验证「一个适配器 + 复用核心管线」
即可把新数据集跑通，其余四个数据集只需新增适配器。

用法（脚本目录自动进 sys.path，平铺 import）：
    python3 pipeline.py --source-id moss --raw <moss.zip> --out <out_dir>
    python3 pipeline.py --source-id toolace --raw <toolace.json> --out <out_dir>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import adapter_moss        # noqa: F401  触发注册
import adapter_toolace     # noqa: F401  触发注册
from adapter_base import adapter_for, AdapterLimits, registered_ids
from derive_tool_lm import ToolLmDeriver
from derive_retrieval import RetrievalDeriver
from derive_disposition import DispositionDeriver
from derive_narration import NarrationDeriver
from audit import structural_gate, timeline_structural_checks

DERIVERS = [ToolLmDeriver(), RetrievalDeriver(), DispositionDeriver(), NarrationDeriver()]


def run_pipeline(source_id: str, raw_path: Path, out_dir: Path, *, max_records: int = 200) -> dict[str, int]:
    """跑通 intake → derive → audit，返回各产出行数。"""
    adapter = adapter_for(source_id)()
    evidences = adapter.load(raw_path, limits=AdapterLimits(max_records=max_records))
    out_dir.mkdir(parents=True, exist_ok=True)

    views: list[dict] = []
    audit_rows: list[dict] = []
    for evidence in evidences:
        for issue in timeline_structural_checks(evidence):
            audit_rows.append({"case_id": evidence.identity.case_id, "view_id": "", "head": "timeline", "issues": [issue]})
        for deriver in DERIVERS:
            for view in deriver.derive(evidence):
                report = structural_gate(evidence, view)
                views.append(view.to_dict())
                audit_rows.append(report.to_dict())

    (out_dir / "evidences.jsonl").write_text(
        "\n".join(json.dumps(e.to_dict(), ensure_ascii=False) for e in evidences) + "\n",
        encoding="utf-8",
    )
    (out_dir / "views.jsonl").write_text(
        "\n".join(json.dumps(v, ensure_ascii=False) for v in views) + "\n",
        encoding="utf-8",
    )
    (out_dir / "audit.jsonl").write_text(
        "\n".join(json.dumps(a, ensure_ascii=False) for a in audit_rows) + "\n",
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(adapter.manifest().to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "evidences": len(evidences),
        "views": len(views),
        "audit_rows": len(audit_rows),
        "admitted_or_pending": sum(1 for a in audit_rows if "passed" in a and a["passed"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="多head流水线 intake→derive→audit")
    parser.add_argument("--source-id", required=True, choices=registered_ids(),
                        help="已注册的 source_id")
    parser.add_argument("--raw", required=True, help="原始数据文件路径（zip/jsonl/json）")
    parser.add_argument("--out", required=True, help="产出目录")
    parser.add_argument("--max-records", type=int, default=200, help="试批上限")
    args = parser.parse_args()

    counts = run_pipeline(args.source_id, Path(args.raw), Path(args.out), max_records=args.max_records)
    print(json.dumps(counts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
