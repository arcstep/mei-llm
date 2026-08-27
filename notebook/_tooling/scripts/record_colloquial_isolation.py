#!/usr/bin/env python3
"""Record cpt-v2 / sft-v2 isolation onto a colloquial release reviews/isolation.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from repo_paths import CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1, ROOT, SCRIPTS_ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))

from zh_pretrain_ingest import dump_json  # noqa: E402

ISOLATION = SCRIPTS_ROOT / "check_train_eval_isolation.py"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_ZH_PRETRAIN_COLLOQUIAL_SYNTH_QWEN_V1)
    args = ap.parse_args()
    corpus = args.corpus_dir if args.corpus_dir.is_absolute() else ROOT / args.corpus_dir
    cpt = subprocess.run([sys.executable, str(ISOLATION), "--scope", "cpt-v2"], cwd=str(ROOT), capture_output=True, text=True)
    sft = subprocess.run([sys.executable, str(ISOLATION), "--scope", "sft-v2"], cwd=str(ROOT), capture_output=True, text=True)
    ok = cpt.returncode == 0 and sft.returncode == 0
    report = {
        "ok": ok,
        "cpt_v2": {"returncode": cpt.returncode, "stdout_tail": (cpt.stdout or "")[-2000:]},
        "sft_v2": {"returncode": sft.returncode, "stdout_tail": (sft.stdout or "")[-2000:]},
    }
    dump_json(corpus / "reviews" / "isolation.json", report)
    print(json.dumps({"ok": ok, "cpt_v2": cpt.returncode, "sft_v2": sft.returncode}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
