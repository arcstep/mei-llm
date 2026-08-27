#!/usr/bin/env python3
"""Coverage + round-trip report for needle-zh tokenizer (v1 by default)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import CORPUS_ZH_VOCAB, ROOT

PROBES = [
    "北京现在天气怎么样",
    "把客厅灯调到30",
    "get_weather",
    "city",
    "set_lights",
    "nod",
    "order_food",
    "兰州拉面",
    "麦当劳",
    "牛肉面",
    "巨无霸",
    "kitchen_light",
    "<tools>",
    "<tool_call>",
    "<act_execute>",
    "<act_refuse>",
    "2026-08-24",
    '{"name":"nod","arguments":{}}',
    "请把厨房灯打开",
]


def load_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def core_pieces(spp, text: str) -> list[str]:
    raw = spp.encode(text, out_type=str)
    out: list[str] = []
    for p in raw:
        if p == "▁":
            continue
        out.append(p[1:] if p.startswith("▁") else p)
    return out


def main() -> int:
    try:
        import sentencepiece as spm
    except ImportError:
        print("install sentencepiece: pip install -r requirements-corpus.txt", file=sys.stderr)
        return 1

    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", action="store_true", help="Use frozen zh-24k-v1 (default)")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--seeds", type=Path, default=CORPUS_ZH_VOCAB / "seeds")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.model is None:
        args.model = CORPUS_ZH_VOCAB / "zh-24k-v1.model"
    if args.out is None:
        args.out = CORPUS_ZH_VOCAB / "coverage-v1.json"

    if not args.model.is_file():
        print(f"missing model {args.model}", file=sys.stderr)
        return 1

    spp = spm.SentencePieceProcessor(model_file=str(args.model))
    ids = {
        "pad": spp.pad_id(),
        "eos": spp.eos_id(),
        "bos": spp.bos_id(),
        "unk": spp.unk_id(),
    }
    hanzi = load_lines(args.seeds / "hanzi-level1.txt")

    missing = []
    for ch in hanzi:
        core = core_pieces(spp, ch)
        if core != [ch]:
            missing.append({"char": ch, "pieces": spp.encode(ch, out_type=str)})

    atomic_need = [
        "nod",
        "shake_head",
        "order_food",
        "kitchen_light",
        "兰州拉面",
        "麦当劳",
        "牛肉面",
        "巨无霸",
        "<tools>",
        "<tool_call>",
        "<act_execute>",
        "get_weather",
        "city",
    ]
    atomic_fail = []
    for t in atomic_need:
        core = core_pieces(spp, t)
        pieces = spp.encode(t, out_type=str)
        unk = spp.unk_id()
        ids_t = spp.encode(t, out_type=int)
        if unk in ids_t or core != [t]:
            atomic_fail.append({"text": t, "core": core, "pieces": pieces, "ids": ids_t})

    probe_rows = []
    roundtrip_fail = []
    for text in PROBES:
        pieces = spp.encode(text, out_type=str)
        ids_t = spp.encode(text, out_type=int)
        back = spp.decode(ids_t)
        row = {
            "text": text,
            "n_chars": len(text),
            "n_tokens": len(pieces),
            "pieces": pieces,
            "core": core_pieces(spp, text),
            "roundtrip": back,
            "roundtrip_ok": back.replace(" ", "") == text.replace(" ", "") or back == text,
        }
        probe_rows.append(row)
        if not row["roundtrip_ok"] and text not in ("{", "}", "[", "]"):
            # JSON punctuation may pick up SPM space marks; still record.
            if text[0] not in "{[":
                roundtrip_fail.append(text)

    try:
        model_rel = str(args.model.relative_to(ROOT))
    except ValueError:
        model_rel = str(args.model)
    report = {
        "model": model_rel,
        "vocab_size": spp.get_piece_size(),
        "control_ids": ids,
        "control_ids_ok": ids == {"pad": 0, "eos": 1, "bos": 2, "unk": 3},
        "hanzi_level1_n": len(hanzi),
        "hanzi_level1_covered": len(hanzi) - len(missing),
        "hanzi_level1_coverage": round((len(hanzi) - len(missing)) / max(len(hanzi), 1), 4),
        "hanzi_missing_head": missing[:50],
        "atomic_fail": atomic_fail,
        "tools_atomic": not atomic_fail,
        "probes": probe_rows,
        "roundtrip_fail": roundtrip_fail,
    }
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {k: report[k] for k in report if k not in {"probes", "hanzi_missing_head"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote {args.out}")
    ok = (
        report["hanzi_level1_coverage"] >= 0.99
        and report["tools_atomic"]
        and report["control_ids_ok"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
