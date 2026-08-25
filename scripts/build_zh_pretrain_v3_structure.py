#!/usr/bin/env python3
"""Build zh-pretrain-v3 clean structure slot. Does not rewrite v2 RELEASE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_PRETRAIN_PROBES,
    CORPORA_ROOT,
    EVAL_SHARED_ROOT,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
    TOKENIZER_ZH_V1,
    all_eval_jsonl,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from zh_pretrain_ingest import (
    TOKENS_PER_SHARD,
    UNK_TOKEN_MAX,
    SplitWriters,
    dump_json,
    file_sha256,
    is_clean_structure_text,
    sha256_text,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
from data import leak_strings_from_rows, document_leaks_eval  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402

CORPUS = CORPORA_ROOT / "zh-pretrain-v3"


def iter_schema_docs(n: int) -> list[dict]:
    tool_dir = EVAL_SHARED_ROOT / "toolsets"
    templates = []
    for path in sorted(tool_dir.glob("*.json")):
        templates.append(json.loads(path.read_text(encoding="utf-8")))
    docs: list[dict] = []
    kinds = ("object", "string", "boolean", "integer", "number")
    i = 0
    while len(docs) < n:
        i += 1
        ts = templates[i % len(templates)]
        name = f"tool_{i % 97}"
        typ = kinds[i % len(kinds)]
        required = ["slot"] if i % 3 else []
        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"中文工具描述 {i}",
            "type": "object",
            "properties": {
                "slot": {"type": typ, "description": f"参数类型 {typ}"},
                "tag": {"type": "string", "enum": ["alpha", "beta", "gamma"]},
            },
            "required": required,
        }
        instance_ok = {"slot": {"string": "alpha", "boolean": True, "integer": 3, "number": 1.5, "object": {"k": 1}}[typ]}
        instance_bad = {"slot": ["array-not-allowed"]}
        openapi = {
            "openapi": "3.0.3",
            "info": {"title": f"工具面 {name}", "version": "1.0.0"},
            "paths": {
                f"/{name}": {
                    "post": {
                        "summary": "调用已登记工具",
                        "requestBody": {"content": {"application/json": {"schema": schema}}},
                    }
                }
            },
        }
        text = (
            json.dumps(schema, ensure_ascii=False)
            + "\n合法实例："
            + json.dumps(instance_ok, ensure_ascii=False)
            + "\n非法实例："
            + json.dumps(instance_bad, ensure_ascii=False)
            + "\n"
            + json.dumps(openapi, ensure_ascii=False)
        )
        if ts.get("tools"):
            text += "\n已登记工具：" + json.dumps(
                [{"name": t.get("name"), "description": t.get("description")} for t in ts["tools"][:4]],
                ensure_ascii=False,
            )
        docs.append({"page_id": f"v3-struct-{i:06d}", "title": name, "text": text, "sha256": sha256_text(text)})
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=8000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    n_docs = 80 if args.smoke else args.n_docs
    tok = ZhTokenizerV1()
    token_dir = CORPUS / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    (CORPUS / "raw").mkdir(parents=True, exist_ok=True)
    writers = SplitWriters(token_dir, "structure", TOKENS_PER_SHARD, ROOT)
    leak_rows = []
    for path in all_eval_jsonl():
        if path.is_file():
            leak_rows.extend([json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()][:8])
    if BANK_NEEDLE_PRETRAIN_PROBES.is_file():
        leak_rows.extend([json.loads(l) for l in BANK_NEEDLE_PRETRAIN_PROBES.read_text(encoding="utf-8").splitlines() if l.strip()])
    leaks = leak_strings_from_rows(leak_rows)
    docs = iter_schema_docs(n_docs)
    raw_path = CORPUS / "raw" / "structure.jsonl"
    n_train = n_valid = n_unk = 0
    kept = []
    for doc in docs:
        text = doc["text"]
        if not is_clean_structure_text(text):
            continue
        if document_leaks_eval(text, leaks) is not None:
            continue
        ids = tok.encode_document(text)
        n_unk += sum(1 for t in ids if t == tok.unk_id)
        split = "valid" if int(doc["sha256"][:8], 16) % 100 < 2 else "train"
        row = {**doc, "split": split}
        kept.append(row)
        if split == "valid":
            writers.valid.write(ids, row, tok.unk_id)
            n_valid += len(ids)
        else:
            writers.train.write(ids, row, tok.unk_id)
            n_train += len(ids)
    writers.close()
    raw_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    n_tok = n_train + n_valid
    unk_rate = (n_unk / n_tok) if n_tok else 1.0
    ledger = {
        "unique_train_tokens": n_train,
        "exposure_train_tokens": n_train,
        "by_source": {"structure_clean": n_train},
        "valid": {"structure_clean": n_valid},
        "parent_checkpoint": None,
        "tokenizer": str(TOKENIZER_ZH_V1.relative_to(ROOT)),
        "seq_len": 256,
        "note": "Do not multiply unique by epochs to name 2B/10B. 2B/10B need new immutable batches.",
    }
    dump_json(CORPUS / "unique-ledger.json", ledger)
    release = {
        "id": "zh-pretrain-v3-structure",
        "promote_structure": unk_rate <= UNK_TOKEN_MAX and n_train > 0,
        "n_unique_train_tokens": n_train,
        "n_valid_tokens": n_valid,
        "unk_rate": unk_rate,
        "source": "synthetic JSON Schema / OpenAPI / schema-instance pairs + registered tool descriptions",
        "rewrites_v2": False,
        "unique_ledger": "corpora/zh-pretrain-v3/unique-ledger.json",
    }
    dump_json(CORPUS / "RELEASE.json", release)
    dump_json(
        CORPUS / "manifest.json",
        {
            "id": "zh-pretrain-v3",
            "n_train_tokens": n_train,
            "n_unique_train_tokens": n_train,
            "n_valid_tokens": n_valid,
            "unk_rate": unk_rate,
            "tokenizer_sha256": file_sha256(TOKENIZER_ZH_V1) if TOKENIZER_ZH_V1.is_file() else "",
        },
    )
    print(json.dumps(release, indent=2))
    return 0 if release["promote_structure"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
