#!/usr/bin/env python3
"""Correct offline colloquial-synth-v1 unique accounting. Engineering contrast only.

Does not delete bins. Does not mark the pack formal-CPT eligible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_V1, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from colloquial_synth_lib import unique_by_first_frame  # noqa: E402
from zh_pretrain_ingest import dump_json, file_sha256  # noqa: E402


def iter_accepted(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_V1)
    args = ap.parse_args()
    corpus = args.corpus_dir if args.corpus_dir.is_absolute() else ROOT / args.corpus_dir
    raw = corpus / "raw" / "accepted.jsonl"
    if not raw.is_file():
        print(f"missing {raw}", file=sys.stderr)
        return 2
    gens = {}
    snapshots = {}
    n_ok = 0
    rows_meta: list[dict] = []
    for row in iter_accepted(raw):
        if not (row.get("filter") or {}).get("ok"):
            continue
        n_ok += 1
        g = str(row.get("generator") or "")
        gens[g] = gens.get(g, 0) + 1
        snap = str(row.get("model_snapshot") or "")
        snapshots[snap] = snapshots.get(snap, 0) + 1
        rows_meta.append(
            {
                "doc_id": row.get("doc_id"),
                "frame": {"frame_id": (row.get("frame") or {}).get("frame_id")},
                "n_tokens": int(row.get("n_tokens") or 0),
                "split": row.get("split") or "train",
            }
        )
    counts = unique_by_first_frame(rows_meta)
    claimed = 0
    release_path = corpus / "RELEASE.json"
    release = json.loads(release_path.read_text(encoding="utf-8")) if release_path.is_file() else {}
    claimed = int(release.get("n_unique_train_tokens") or 0)
    accounting = {
        "corpus": str(corpus.relative_to(ROOT)) if corpus.is_relative_to(ROOT) else str(corpus),
        "engineering_contrast_only": True,
        "formal_cpt_eligible": False,
        "do_not_register_v4": True,
        "do_not_resume_qwen_into_this_dir": True,
        "claimed_n_unique_train_tokens": claimed,
        "claimed_is_exposure_including_duplicate_frame_id": True,
        "n_accepted_ok_rows": n_ok,
        "generators": gens,
        "model_snapshots": snapshots,
        **counts,
        "inflation_ratio": round(
            (claimed / counts["unique_train_tokens"]) if counts["unique_train_tokens"] else 0.0, 4
        ),
        "note": (
            "The previously published 38.1M figure is exposure (sum of train n_tokens) "
            "and includes duplicate frame_id. Unique-by-first-frame is the corrected unique."
        ),
    }
    dump_json(corpus / "ACCOUNTING.json", accounting)
    dump_json(
        corpus / "ENGINEERING_ONLY.json",
        {
            "engineering_contrast_only": True,
            "formal_cpt_eligible": False,
            "do_not_register_v4": True,
            "do_not_resume_qwen_into_this_dir": True,
            "corrected_unique_train_tokens_by_first_frame": counts["unique_train_tokens"],
            "claimed_exposure_train_tokens": claimed,
            "n_duplicate_frame_ids": counts["n_duplicate_frame_ids"],
        },
    )
    if release_path.is_file():
        release["engineering_contrast_only"] = True
        release["formal_cpt_eligible"] = False
        release["register_v4"] = False
        release["n_unique_train_tokens_claimed_exposure"] = claimed
        release["n_unique_train_tokens_by_first_frame_id"] = counts["unique_train_tokens"]
        release["n_duplicate_frame_ids"] = counts["n_duplicate_frame_ids"]
        release["accounting_corrected"] = True
        release["eligibility_reason"] = (
            "offline_engineering_contrast_only;duplicate_frame_id;"
            "claimed_unique_is_exposure_not_first_frame"
        )
        dump_json(release_path, release)
    ledger_path = corpus / "unique-ledger.json"
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["unique_by_first_frame_id"] = counts["unique_train_tokens"]
        ledger["n_duplicate_frame_ids"] = counts["n_duplicate_frame_ids"]
        ledger["accounting_corrected"] = True
        ledger["note"] = (
            "Ledger unique_train_tokens remains bin exposure. See ACCOUNTING.json for first-frame unique. "
            "This pack is engineering contrast only."
        )
        dump_json(ledger_path, ledger)
    hashes_path = corpus / "hashes.json"
    if hashes_path.is_file():
        stored = json.loads(hashes_path.read_text(encoding="utf-8"))
        if release_path.is_file():
            stored["RELEASE.json"] = file_sha256(release_path)
        if ledger_path.is_file():
            stored["unique-ledger.json"] = file_sha256(ledger_path)
        stored["ACCOUNTING.json"] = file_sha256(corpus / "ACCOUNTING.json")
        stored["ENGINEERING_ONLY.json"] = file_sha256(corpus / "ENGINEERING_ONLY.json")
        dump_json(hashes_path, stored)
    readme = corpus / "README.md"
    readme.write_text(
        """# zh-pretrain-colloquial-synth-v1 (engineering contrast only)

Offline `offline-frame-renderer` pack. **Not** a formal CPT spoken source.

- Claimed `n_unique_train_tokens` ≈ 38.1M is **exposure**, including duplicate `frame_id`.
- Corrected unique-by-first-frame is recorded in `ACCOUNTING.json` (≈23.2M) and is still below a quality bar.
- Do not mix or resume qwen production into `raw/accepted.jsonl`.
- Formal spoken role is `notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen-plus/` after quality/isolation/blind gates.
""",
        encoding="utf-8",
    )
    print(json.dumps(accounting, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
