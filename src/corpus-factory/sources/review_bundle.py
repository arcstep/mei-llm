"""Prepare a bounded, blinded human review entry point; never sign it off."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random


def build(surveys: list[Path], audit: dict, out: Path, *, seed=20260913, per_source=6):
    out.mkdir(parents=True, exist_ok=False)
    population = {}
    for survey in surveys:
        lock = json.loads((survey / "survey.json").read_text())
        for rel, expected in lock["artifacts"].items():
            if not Path(rel).name.startswith("sample-"): continue
            path = (survey / rel).resolve()
            if not path.is_relative_to(survey.resolve()): raise ValueError("unsafe sample path")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected: raise ValueError("sample hash changed")
            sample = json.loads(raw)
            source = Path(rel).parts[0]
            for row in sample["samples"]:
                if row.get("purpose") != "population": continue
                # This entry packet deduplicates exact text within each source.
                key = row["text_sha256"]
                population.setdefault(source, {}).setdefault(key, {
                    "source_id": source, "sample_path": str(path), "sample_sha256": expected,
                    "shard": sample["shard"], "record": row,
                    "human_review": {"reviewer": None, "reviewed_at": None, "decision": None,
                                     "naturalness": None, "meaning_preserved": None,
                                     "mei_use": None, "notes": None}})
    cases = []
    for source, records in sorted(population.items()):
        keys = sorted(records)
        rng = random.Random(f"{seed}:{source}")
        cases.extend(records[k] for k in rng.sample(keys, min(per_source, len(keys))))
    def write(name, value):
        with (out/name).open("x") as f: json.dump(value, f, ensure_ascii=False, indent=2)
    (out/"implementation.py.snapshot").write_bytes(Path(__file__).read_bytes())
    write("coverage-audit.json", audit)
    write("cases.json", cases)
    lines = ["# 来源人工审核入口", "",
             "这是一份导航包，每个有正文的来源最多6例，不能替代每来源/适配器的分层正式审核。",
             "原始样本在cases.json完整保留；机器标签暂不展示，减少先入为主。所有审核字段为空，不代表通过。", "",
             "依次判断：原件是否可追溯；语义/上下文是否保留；表达自然度；适合哪些Mei能力；应继续、限制、扩样或暂缓。",
             "权利与版本裁决按来源档案处理，不能从一条顺口的句子推断。", ""]
    for i, case in enumerate(cases, 1):
        row = case["record"]
        link = os.path.relpath(case["sample_path"], out.resolve())
        lines += [f"## {i}. {case['source_id']} · 原始位置 {row['row_index']}", "",
                  f"[完整原件与上下文]({link})；文本SHA256 `{row['text_sha256']}`。", "",
                  "以下仅展示前600字符，正式判断请看完整记录；此处不用于比例估计。", ""]
        lines += ["> " + line for line in row["text"][:600].splitlines()]
        lines += ["", "审核人/日期：待填；结论/依据：待填。", ""]
    (out/"REVIEW.md").write_text("\n".join(lines))
    artifacts = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    result = {"schema": "mei-source-human-review-entry-v1", "seed": seed, "case_count": len(cases),
              "per_source": per_source, "artifacts": artifacts, "human_review_passed": False,
              "m1_passed": False, "m2_passed": False,
              "scope": "blinded review navigation, not a qualification sample or training release"}
    write("review-bundle.json", result)
    return result
