#!/usr/bin/env python3
"""Download and checksum Needle-zh v2 CPT sources. Does not tokenize."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from repo_paths import CORPUS_ZH_PRETRAIN_V1, CORPUS_ZH_PRETRAIN_V2, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))

from zh_pretrain_ingest import dump_json, file_sha256, hardlink_or_copy, load_json, rel

CWT2_REPO = "CASIA-LM/ChineseWebText2.0"
CWT2_FILE = "ChineseWebText2.0/part-0001.jsonl.gz"
GITHUB_REPO = "codeparrot/github-code"
STACK_REPOS = (
    ("bigcode/the-stack", "data/json/train-00000-of-01329.parquet"),
    ("bigcode/starcoderdata", "json/train-00000-of-00006.parquet"),
)


def hf_get(repo: str, filename: str, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    local = dest_dir / Path(filename).name
    if local.is_file() and local.stat().st_size > 1_000_000:
        return local
    nested = dest_dir / filename
    if nested.is_file() and nested.stat().st_size > 1_000_000:
        return nested
    from huggingface_hub import hf_hub_download

    print(f"download {repo} {filename}", flush=True)
    path = Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=filename,
            local_dir=str(dest_dir),
        )
    )
    if path != local and path.is_file() and not local.exists():
        try:
            local.hardlink_to(path)
        except OSError:
            pass
    return path


def record(entry: dict, path: Path | None, *, ok: bool, error: str | None = None) -> dict:
    row = dict(entry)
    row["ok"] = ok
    row["error"] = error
    if path is not None and path.is_file():
        row["path"] = rel(path, ROOT)
        row["bytes"] = path.stat().st_size
        row["sha256"] = file_sha256(path)
    return row


def try_gated(repo: str, filename: str) -> dict:
    try:
        from huggingface_hub import HfApi

        HfApi().get_paths_info(repo, [filename], repo_type="dataset")
        return {"repo": repo, "filename": filename, "ok": True, "status": "available"}
    except Exception as exc:
        return {
            "repo": repo,
            "filename": filename,
            "ok": False,
            "status": "blocked",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CORPUS_ZH_PRETRAIN_V2)
    ap.add_argument("--github-shards", type=int, default=16)
    args = ap.parse_args()
    corpus = args.out if args.out.is_absolute() else ROOT / args.out
    t0 = time.time()
    rows: list[dict] = []

    cwt_dir = corpus / "raw" / "cwt2"
    try:
        cwt = hf_get(CWT2_REPO, CWT2_FILE, cwt_dir)
        rows.append(
            record(
                {
                    "source_id": "chinesewebtext2-part-0001",
                    "repo": CWT2_REPO,
                    "filename": CWT2_FILE,
                    "band": "C",
                    "license": "Apache-2.0 metadata tag; webpage copyright not cleared",
                    "role": "cpt_colloquial",
                },
                cwt,
                ok=True,
            )
        )
        print(json.dumps({"cwt2": rows[-1]["bytes"]}), flush=True)
    except Exception as exc:
        rows.append(
            record(
                {"source_id": "chinesewebtext2-part-0001", "repo": CWT2_REPO, "filename": CWT2_FILE},
                None,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        )
        print(f"CWT2 failed: {exc}", flush=True)

    gh_dir = corpus / "raw" / "github-code"
    for i in range(max(1, args.github_shards)):
        name = f"data/train-{i:05d}-of-01126.parquet"
        try:
            path = hf_get(GITHUB_REPO, name, gh_dir)
            rows.append(
                record(
                    {
                        "source_id": f"github-code-{i:05d}",
                        "repo": GITHUB_REPO,
                        "filename": name,
                        "band": "C",
                        "license": "per-file detected license; ingest permissive only",
                        "role": "cpt_structure",
                    },
                    path,
                    ok=True,
                )
            )
            print(json.dumps({"github": name, "bytes": rows[-1]["bytes"]}), flush=True)
        except Exception as exc:
            rows.append(
                record(
                    {"source_id": f"github-code-{i:05d}", "repo": GITHUB_REPO, "filename": name},
                    None,
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            print(f"github-code {name} failed: {exc}", flush=True)
            break

    blocked = []
    for repo, filename in STACK_REPOS:
        info = try_gated(repo, filename)
        blocked.append(info)
        print(json.dumps({"gated": info["repo"], "status": info.get("status")}), flush=True)

    schema_src = CORPUS_ZH_PRETRAIN_V1 / "raw" / "schemastore" / "src" / "schemas" / "json"
    oai_src = CORPUS_ZH_PRETRAIN_V1 / "raw" / "oai-examples"
    schema_dest = corpus / "raw" / "schemastore" / "src" / "schemas" / "json"
    oai_dest = corpus / "raw" / "oai-examples"
    if schema_src.is_dir():
        schema_dest.parent.mkdir(parents=True, exist_ok=True)
        if not schema_dest.is_dir():
            shutil_copy = __import__("shutil").copytree
            shutil_copy(schema_src, schema_dest, dirs_exist_ok=True)
        rows.append(
            {
                "source_id": "schemastore-json",
                "band": "B",
                "ok": True,
                "path": rel(schema_dest, ROOT),
                "role": "cpt_structure",
            }
        )
    if oai_src.is_dir():
        oai_dest.mkdir(parents=True, exist_ok=True)
        for path in oai_src.glob("*.yaml"):
            hardlink_or_copy(path, oai_dest / path.name)
        rows.append(
            {
                "source_id": "oai-openapi-examples",
                "band": "B",
                "ok": True,
                "path": rel(oai_dest, ROOT),
                "role": "cpt_structure",
            }
        )

    hq_src = CORPUS_ZH_PRETRAIN_V1 / "raw" / "fineweb2-hq" / "cmn_Hani"
    hq_dest = corpus / "raw" / "fineweb2-hq" / "cmn_Hani"
    if hq_src.is_dir():
        hq_dest.mkdir(parents=True, exist_ok=True)
        for path in sorted(hq_src.glob("*.parquet")):
            hardlink_or_copy(path, hq_dest / path.name)
        rows.append(
            {
                "source_id": "fineweb2-hq-cmn-hani-v1-snapshot",
                "band": "C",
                "ok": True,
                "path": rel(hq_dest, ROOT),
                "role": "cpt_hq",
                "note": "hardlink/copy of v1 frozen parquets; not re-downloaded",
            }
        )

    man = {
        "id": "zh-pretrain-v2-download",
        "elapsed_s": round(time.time() - t0, 1),
        "sources": rows,
        "blocked": blocked,
        "required_ok": all(
            r.get("ok") for r in rows if r.get("source_id") in {"chinesewebtext2-part-0001"}
        )
        and any(str(r.get("source_id") or "").startswith("github-code") and r.get("ok") for r in rows),
    }
    dump_json(corpus / "download-manifest.json", man)
    print(json.dumps({"required_ok": man["required_ok"], "n_sources": len(rows), "elapsed_s": man["elapsed_s"]}), flush=True)
    return 0 if man["required_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
