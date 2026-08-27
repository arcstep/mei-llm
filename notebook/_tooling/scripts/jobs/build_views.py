#!/usr/bin/env python3
"""Build human-browsable Job topic views from machine registries.

This creates small artifact descriptors under jobs/<topic>/outbox. It never
moves or copies corpus shards, SFT packs, checkpoints, or eval banks.
"""

from __future__ import annotations

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))

from jobs.registry import load_corpora, load_jobs  # noqa: E402
from repo_paths import ROOT  # noqa: E402

MANAGED = ".mei-llm-view"

TOPICS = {
    "colloquial-cpt": {
        "title": "中文口语 CPT",
        "purpose": "生产、审计和收口 spoken-role CPT；正式合同与混合模型草案分账。",
        "entrypoints": {
            "produce": [
                "scripts/produce_colloquial_qwen.py",
                "scripts/run_colloquial_synth_sidecars.py",
                "scripts/run_colloquial_synth_staircase.py",
            ],
            "collect": [
                "scripts/pack_colloquial_pooled.py",
                "scripts/mix_colloquial_synth_releases.py",
            ],
            "gates": [
                "scripts/audit_zh_pretrain_colloquial.py",
                "scripts/record_colloquial_isolation.py",
                "scripts/review_colloquial_blind.py",
                "scripts/validate_approved_colloquial.py",
            ],
            "publish": ["scripts/admit_colloquial_qwen.py"],
            "diagnostics": [
                "scripts/report_colloquial_synth_fleet.py",
                "scripts/bakeoff_colloquial_ollama.py",
                "scripts/correct_colloquial_offline_accounting.py",
            ],
        },
    },
    "general-cpt": {
        "title": "通用中文与结构 CPT",
        "purpose": "组织 vocab、wiki、HQ、结构语料、v4 atlas 与 CPT 阶梯。",
        "entrypoints": {
            "build": [
                "scripts/build_zh_pretrain_v0.py",
                "scripts/build_zh_pretrain_v1.py",
                "scripts/build_zh_pretrain_v2.py",
                "scripts/build_zh_pretrain_v3_structure.py",
                "scripts/build_zh_pretrain_v4.py",
            ],
            "train": [
                "scripts/train_needle_zh_pretrain.py",
                "scripts/run_cpt_v2_staircase.py",
            ],
            "gates": [
                "scripts/check_train_eval_isolation.py",
                "scripts/eval_cpt_v2_5m_utility.py",
                "scripts/eval_needle_zh_pretrain_valid.py",
                "scripts/eval_needle_pretrain_probes.py",
            ],
        },
    },
    "toolcall-sft": {
        "title": "mei-1.0-58m 工具调用 SFT",
        "purpose": "组织 task-local SFT seed、candidate、review、accepted pack 与训练入口。",
        "entrypoints": {
            "build": [
                "scripts/build_needle_home_sft_packs.py",
                "scripts/build_needle_mw_sft_v0.py",
            ],
            "validate": [
                "scripts/validate_needle_home_sft_pack.py",
                "scripts/validate_needle_mw_sft_v0.py",
                "scripts/validate_mei_toolcall_v2_pack.py",
                "scripts/check_train_eval_isolation.py",
            ],
            "train": [
                "scripts/train_needle_zh_sft.py",
                "scripts/train_mei_58m_sft.py",
            ],
        },
    },
    "mei-expert-sft": {
        "title": "MEI 专家行为 SFT",
        "purpose": "组织 Qwen3.5 0.8B 专家行为 seed、MLX adapter 和评测工作。",
        "entrypoints": {
            "export": ["scripts/export_mlx_sft_seed_v0.py"],
            "evaluate": [
                "scripts/run_eval_mlx_v0.py",
                "scripts/compare_mlx_base_lora_v0.py",
            ],
            "gates": ["scripts/check_train_eval_isolation.py"],
        },
    },
}

GENERAL_IDS = {
    "_probe",
    "zh-vocab-v0",
    "zh-pretrain-v0",
    "zh-pretrain-v1",
    "zh-pretrain-v2",
    "zh-pretrain-v3",
    "zh-pretrain-v4",
}


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def topic_for_corpus(row: dict) -> str:
    corpus_id = str(row.get("id") or "")
    role = str(row.get("role") or "")
    if corpus_id in GENERAL_IDS or role in {"vocab", "cpt-wiki", "cpt-atlas", "cpt-structure", "probe"}:
        return "general-cpt"
    if role == "cpt-colloquial" or corpus_id.startswith("zh-pretrain-colloquial"):
        return "colloquial-cpt"
    if role.startswith("sft"):
        return "toolcall-sft"
    return "general-cpt"


def outbox_state(state: str) -> str:
    if state == "accepted":
        return "accepted"
    if state in {"draft", "fail_closed", "job_complete", "empty"}:
        return "draft"
    return "archive"


def clear_managed_views(topic_root: Path) -> None:
    outbox = topic_root / "outbox"
    for state in ("accepted", "draft", "archive"):
        state_dir = outbox / state
        state_dir.mkdir(parents=True, exist_ok=True)
        for child in state_dir.iterdir():
            if child.is_dir() and (child / MANAGED).is_file():
                shutil.rmtree(child)


def write_artifact(topic: str, state: str, artifact_id: str, payload: dict) -> None:
    dest = ROOT / "jobs" / topic / "outbox" / state / artifact_id
    dest.mkdir(parents=True, exist_ok=True)
    (dest / MANAGED).write_text("generated by scripts/jobs/build_views.py\n", encoding="utf-8")
    dump_json(dest / "artifact.json", payload)
    path = str(payload.get("path") or "")
    lines = [
        f"# {artifact_id}",
        "",
        f"- topic: `{topic}`",
        f"- outbox: `{state}`",
        f"- source state: `{payload.get('source_state')}`",
        f"- role: `{payload.get('role')}`",
        f"- artifact: `{path}`",
    ]
    if payload.get("unique_train_tokens") is not None:
        lines.append(f"- unique train tokens: `{payload['unique_train_tokens']}`")
    if payload.get("formal_eligible") is not None:
        lines.append(f"- formal eligible: `{str(bool(payload['formal_eligible'])).lower()}`")
    if payload.get("reason"):
        lines.append(f"- disposition: {payload['reason']}")
    lines.extend(
        [
            "",
            "This directory is a lifecycle view. The large artifact remains at the path above.",
            "",
        ]
    )
    (dest / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_topic_readme(topic: str, counts: Counter) -> None:
    meta = TOPICS[topic]
    root = ROOT / "jobs" / topic
    lines = [
        f"# {topic} · {meta['title']}",
        "",
        meta["purpose"],
        "",
        "## 从这里开始",
        "",
        "- `inbox/`：合同、输入引用、混合策略；不放评测 gold。",
        "- `jobs/`：可执行入口与每次 Job Card。",
        "- `outbox/accepted/`：已可供训练任务订阅的成果。",
        "- `outbox/draft/`：已生成但门禁或合同未闭合的成果。",
        "- `outbox/archive/`：对照、smoke、invalid、历史产物。",
        "",
        "## 当前成果",
        "",
        f"- accepted: {counts['accepted']}",
        f"- draft: {counts['draft']}",
        f"- archive: {counts['archive']}",
        "",
        "机器 registry 不是人工入口；人工从本目录和 `outbox/INDEX.md` 阅读。",
        "",
    ]
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_entrypoints(topic: str) -> None:
    root = ROOT / "jobs" / topic / "jobs"
    root.mkdir(parents=True, exist_ok=True)
    entrypoints = TOPICS[topic]["entrypoints"]
    dump_json(root / "entrypoints.json", entrypoints)
    lines = ["# 可执行入口", ""]
    for group, paths in entrypoints.items():
        lines.extend([f"## {group}", ""])
        lines.extend(f"- `{path}`" for path in paths)
        lines.append("")
    lines.append("脚本物理路径暂不移动；本文件是按任务归类后的作者入口。")
    lines.append("")
    (root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def build() -> dict:
    corpora = load_corpora()
    jobs = load_jobs()
    counts: dict[str, Counter] = {topic: Counter() for topic in TOPICS}
    artifacts: dict[str, list[dict]] = {topic: [] for topic in TOPICS}

    for topic in TOPICS:
        topic_root = ROOT / "jobs" / topic
        (topic_root / "inbox").mkdir(parents=True, exist_ok=True)
        clear_managed_views(topic_root)
        write_entrypoints(topic)

    for row in corpora.get("corpora") or []:
        topic = topic_for_corpus(row)
        state = outbox_state(str(row.get("state") or "draft"))
        payload = {
            "id": row["id"],
            "kind": "corpus",
            "topic": topic,
            "outbox_state": state,
            "source_state": row.get("state"),
            "role": row.get("role"),
            "path": row.get("path"),
            "release": row.get("release"),
            "job_id": row.get("job_id"),
            "generators": row.get("generators") or [],
            "unique_train_tokens": row.get("unique_train_tokens"),
            "formal_eligible": bool(row.get("formal_eligible")),
            "subscribers": row.get("subscribers") or [],
        }
        write_artifact(topic, state, str(row["id"]), payload)
        counts[topic][state] += 1
        artifacts[topic].append(payload)

    for row in jobs.get("sft_packs") or []:
        topic = "mei-expert-sft" if row.get("task_id") == "mei-expert-qwen35-0p8b" else "toolcall-sft"
        source_state = str(row.get("state") or "draft")
        state = "accepted" if source_state == "accepted" else ("draft" if source_state == "draft" else "archive")
        artifact_id = str(row["id"])
        payload = {
            "id": artifact_id,
            "kind": "sft-pack",
            "topic": topic,
            "outbox_state": state,
            "source_state": source_state,
            "role": "sft-pack",
            "path": row.get("path"),
            "task_id": row.get("task_id"),
            "formal_eligible": state == "accepted",
            "reason": "invalid or diagnostic only" if source_state == "invalid" else None,
        }
        write_artifact(topic, state, artifact_id, payload)
        counts[topic][state] += 1
        artifacts[topic].append(payload)

    for topic in TOPICS:
        write_topic_readme(topic, counts[topic])
        rows = sorted(artifacts[topic], key=lambda r: (str(r["outbox_state"]), str(r["id"])))
        lines = [
            f"# {TOPICS[topic]['title']}成果索引",
            "",
            "| 状态 | ID | 角色 | 实体路径 |",
            "|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['outbox_state']} | `{row['id']}` | `{row['role']}` | `{row['path']}` |"
            )
        lines.append("")
        (ROOT / "jobs" / topic / "outbox" / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "version": 1,
        "entry": "jobs/README.md",
        "topics": {
            topic: {
                "path": f"jobs/{topic}",
                "accepted": counts[topic]["accepted"],
                "draft": counts[topic]["draft"],
                "archive": counts[topic]["archive"],
            }
            for topic in TOPICS
        },
    }
    dump_json(ROOT / "jobs" / "catalog.json", summary)
    return summary


def main() -> int:
    print(json.dumps({"ok": True, "catalog": build()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
