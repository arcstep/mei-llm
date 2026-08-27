#!/usr/bin/env python3
"""Validate grounded route SFT: provenance round-trip, isolation, no meta stems, no overflow."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from mei_tool_grounded_lib import FORBIDDEN, normalized_body
from repo_paths import EVAL_BANKS_ROOT, ROOT, TASKS_ROOT, TASK_NEEDLE_ZH

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
from candidates import ToolContext, entities_from_mode, load_entity_catalog, load_lexicon  # noqa: E402
from data import encode_sft_row  # noqa: E402
from route_compiler import compile_routes, gold_route_id  # noqa: E402
from schema_render import PRODUCT_SFT_SEQ_LEN, load_toolset_json  # noqa: E402
from tokenizer import ZhTokenizerV1  # noqa: E402

BANK = EVAL_BANKS_ROOT / "mei-tool-grounded-v1" / "eval-bank-v0.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--limit-encode", type=int, default=64)
    args = ap.parse_args()
    pack = args.pack if args.pack.is_absolute() else ROOT / args.pack
    rows = load_jsonl(pack)
    eval_rows = load_jsonl(BANK) if BANK.is_file() else []
    eval_q = {(r["query"], r["toolset_id"]) for r in eval_rows}
    eval_body = {normalized_body(r["query"]) for r in eval_rows}
    errors: list[str] = []
    kinds = Counter()
    for i, row in enumerate(rows):
        q = str(row.get("query") or "")
        if any(tok in q for tok in FORBIDDEN):
            errors.append(f"{row.get('sample_id')}:forbidden_meta")
        key = (q, row.get("toolset_id"))
        if key in eval_q:
            errors.append(f"{row.get('sample_id')}:eval_query_overlap")
        if normalized_body(q) in eval_body and key not in eval_q:
            # body match with different toolset is a counterfactual; allow if toolset differs
            bodies = [er for er in eval_rows if normalized_body(er["query"]) == normalized_body(q)]
            if any(er.get("toolset_id") == row.get("toolset_id") for er in bodies):
                errors.append(f"{row.get('sample_id')}:normalized_body_overlap")
        ts = load_toolset_json(str(row["toolset_id"]))
        ctx = ToolContext(
            query=q,
            scene=row.get("scene") if isinstance(row.get("scene"), str) else None,
            toolset=ts,
            entities=list(row.get("entities") or entities_from_mode(str(row.get("entity_mode") or "train"))),
            lexicon=dict(row.get("lexicon") or load_lexicon()),
            param_types=dict(row.get("param_types") or load_entity_catalog().get("param_types") or {}),
        )
        manifest = compile_routes(ctx)
        answers = row.get("answers") or []
        kinds[str(row.get("kind") or "?")] += 1
        if answers:
            rid = gold_route_id(manifest, answers[0])
            if rid is None:
                errors.append(f"{row.get('sample_id')}:gold_not_in_manifest")
            else:
                route = manifest.by_id(rid)
                for p, prov in (route.provenance or {}).items():
                    if prov.get("canonical_value") != route.arguments.get(p):
                        errors.append(f"{row.get('sample_id')}:prov_mismatch:{p}")
                    if not prov.get("evidence_source"):
                        errors.append(f"{row.get('sample_id')}:no_source:{p}")
        elif manifest.routes:
            errors.append(f"{row.get('sample_id')}:refuse_has_routes")
        if len(errors) > 40:
            break
    tok = ZhTokenizerV1()
    overflow = 0
    for row in rows[: args.limit_encode]:
        packed = encode_sft_row(tok, row, PRODUCT_SFT_SEQ_LEN)
        if packed.get("reject_reason"):
            overflow += 1
            errors.append(f"{row.get('sample_id')}:encode:{packed.get('reject_reason')}")
    out = {
        "ok": not errors,
        "n": len(rows),
        "kinds": dict(kinds),
        "overflow_or_reject_in_sample": overflow,
        "errors": errors[:40],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
