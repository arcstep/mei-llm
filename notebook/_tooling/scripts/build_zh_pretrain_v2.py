#!/usr/bin/env python3
"""Assemble zh-pretrain-v2: frozen wiki/v1 HQ + colloquial CWT2 + structure/tech.

Does not rewrite zh-pretrain-v0 or v1 shards. HQ/schema/oai are hash-split into v2 bins.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from repo_paths import (
    BANK_NEEDLE_PRETRAIN_PROBES,
    CORPUS_ZH_PRETRAIN,
    CORPUS_ZH_PRETRAIN_V1,
    CORPUS_ZH_PRETRAIN_V2,
    ROOT,
    TASKS_ROOT,
    TASK_NEEDLE_ZH,
    all_eval_jsonl,
)

sys.path.insert(0, str(TASKS_ROOT / TASK_NEEDLE_ZH / "model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import (  # noqa: E402
    file_sha256,
    iter_jsonl,
    leak_strings_from_rows,
)
from tokenizer import ZhTokenizerV1  # noqa: E402
from zh_pretrain_ingest import (  # noqa: E402
    CHUNK_CHARS,
    MAX_CHARS,
    MAX_STRUCT_CHARS,
    TOKENS_PER_SHARD,
    UNK_TOKEN_MAX,
    Ingestor,
    NearDupIndex,
    SplitWriters,
    dump_json,
    extract_signatures,
    is_struct_path,
    iter_jsonl_gz,
    iter_parquet_rows,
    license_ok,
    load_hash_file,
    load_json,
    domain_labels,
    rel,
    shard_token_count,
    split_existing_bins,
)

WIKI_TRAIN_TOKENS = 649_904_474
WIKI_VALID_TOKENS = 34_610_229
WIKI_UNK = 295_816
COLLOQUIAL_TARGET = 80_000_000
STRUCTURE_TARGET = 40_000_000
PROBE_WIKI_HASHES = ROOT / "notebook/archive/corpus/_probe/wiki-hashes/wiki-norm.sha256"


def collect_leak_strings() -> list[str]:
    rows: list[dict] = []
    for path in all_eval_jsonl():
        if "pretrain-probes" in str(path):
            continue
        rows.extend(iter_jsonl(path))
    if BANK_NEEDLE_PRETRAIN_PROBES.is_file():
        rows.extend(iter_jsonl(BANK_NEEDLE_PRETRAIN_PROBES))
    return leak_strings_from_rows(rows)


def load_wiki_hashes() -> set[str]:
    if PROBE_WIKI_HASHES.is_file():
        return load_hash_file(PROBE_WIKI_HASHES)
    hashes: set[str] = set()
    for idx in (CORPUS_ZH_PRETRAIN / "tokens").glob("*.idx.jsonl"):
        for row in iter_jsonl(idx):
            h = str(row.get("sha256") or "")
            if h:
                hashes.add(h)
    if not hashes:
        raise FileNotFoundError("missing wiki hashes")
    return hashes


def list_wiki_bins(split: str) -> list[Path]:
    return sorted((CORPUS_ZH_PRETRAIN / "tokens").glob(f"{split}-*.bin"))


def list_v1_bins(prefix: str) -> list[Path]:
    return sorted((CORPUS_ZH_PRETRAIN_V1 / "tokens").glob(f"{prefix}-*.bin"))


def bins(token_dir: Path, prefix: str) -> list[Path]:
    return sorted(token_dir.glob(f"{prefix}-*.bin"))


def recount(token_dir: Path, prefix: str) -> int:
    return sum(shard_token_count(p) for p in bins(token_dir, prefix))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=CORPUS_ZH_PRETRAIN_V2)
    ap.add_argument("--colloquial-target", type=int, default=COLLOQUIAL_TARGET)
    ap.add_argument("--structure-target", type=int, default=STRUCTURE_TARGET)
    args = ap.parse_args()
    corpus = args.out if args.out.is_absolute() else ROOT / args.out
    token_dir = corpus / "tokens"
    token_dir.mkdir(parents=True, exist_ok=True)
    (corpus / "hashes").mkdir(parents=True, exist_ok=True)
    (corpus / "reviews").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    tok = ZhTokenizerV1()
    leaks = collect_leak_strings()
    wiki = load_wiki_hashes()
    print(f"wiki hashes {len(wiki)} leak_strings {len(leaks)}", flush=True)
    hash_path = corpus / "hashes" / "supplement.sha256"
    seen = set(wiki) | load_hash_file(CORPUS_ZH_PRETRAIN_V1 / "hashes" / "supplement.sha256")
    seen |= load_hash_file(hash_path)
    near = NearDupIndex()
    stats_all: dict = load_json(corpus / "ingest-stats.json")
    cursor = load_json(corpus / "ingest-cursor.json")

    wiki_train = list_wiki_bins("train")
    wiki_valid = list_wiki_bins("valid")
    if not wiki_train or not wiki_valid:
        print("missing frozen v0 shards", file=sys.stderr)
        return 2

    if not cursor.get("hq_split_done"):
        hq_bins = list_v1_bins("hq-train")
        if not hq_bins:
            print("missing v1 HQ bins", file=sys.stderr)
            return 2
        print(f"split HQ bins {len(hq_bins)}", flush=True)
        stats_all["hq_split"] = split_existing_bins(hq_bins, token_dir, "hq", root=ROOT)
        cursor["hq_split_done"] = True
        cursor["n_hq_train_tokens"] = recount(token_dir, "hq-train")
        cursor["n_hq_valid_tokens"] = recount(token_dir, "hq-valid")
        dump_json(corpus / "ingest-cursor.json", cursor)
        dump_json(corpus / "ingest-stats.json", stats_all)
        print(json.dumps({"hq_split": stats_all["hq_split"]}), flush=True)

    if not cursor.get("schema_split_done"):
        schema_bins = list_v1_bins("schema-train") + list_v1_bins("oai-train")
        print(f"split schema/oai bins {len(schema_bins)}", flush=True)
        stats_all["schema_split"] = split_existing_bins(schema_bins, token_dir, "structure", root=ROOT)
        cursor["schema_split_done"] = True
        cursor["n_structure_seed_train_tokens"] = recount(token_dir, "structure-train")
        cursor["n_structure_seed_valid_tokens"] = recount(token_dir, "structure-valid")
        dump_json(corpus / "ingest-cursor.json", cursor)
        dump_json(corpus / "ingest-stats.json", stats_all)

    writers_struct = SplitWriters(token_dir, "structure", TOKENS_PER_SHARD, ROOT)
    writers_struct.train.idx = max((int(p.stem.split("-")[-1]) + 1 for p in bins(token_dir, "structure-train")), default=0)
    writers_struct.valid.idx = max((int(p.stem.split("-")[-1]) + 1 for p in bins(token_dir, "structure-valid")), default=0)

    n_struct = recount(token_dir, "structure-train")
    if not cursor.get("schema_chunk_done"):
        schema_dir = corpus / "raw" / "schemastore" / "src" / "schemas" / "json"
        if not schema_dir.is_dir():
            schema_dir = CORPUS_ZH_PRETRAIN_V1 / "raw" / "schemastore" / "src" / "schemas" / "json"
        ing = Ingestor(
            tok,
            writers_struct,
            seen,
            wiki,
            leaks,
            hash_path,
            near,
            require_cjk=False,
            max_chars=MAX_STRUCT_CHARS,
            style="structure",
        )
        files = sorted(p for p in schema_dir.glob("*.json") if p.is_file()) if schema_dir.is_dir() else []
        print(f"chunk large schemas {len(files)}", flush=True)
        for path in files:
            text = path.read_text(encoding="utf-8", errors="replace")
            if len(text) <= MAX_STRUCT_CHARS:
                continue
            for i, chunk in enumerate(text[j : j + CHUNK_CHARS] for j in range(0, min(len(text), 200_000), CHUNK_CHARS)):
                ing.accept_text(chunk, page_id=f"{path.stem}:big:{i}", title=path.name)
        stats_all["schema_chunk"] = ing.stats
        cursor["schema_chunk_done"] = True
        writers_struct.train.flush()
        writers_struct.valid.flush()
        dump_json(corpus / "ingest-cursor.json", cursor)
        dump_json(corpus / "ingest-stats.json", stats_all)
        dump_json(corpus / "reviews" / "schema-chunk.json", {"samples": ing.reviews})

    if not cursor.get("github_done"):
        ing = Ingestor(
            tok,
            writers_struct,
            seen,
            wiki,
            leaks,
            hash_path,
            near,
            require_cjk=False,
            max_chars=MAX_STRUCT_CHARS,
            style="structure",
        )
        gh_dir = corpus / "raw" / "github-code"
        parquets = sorted(gh_dir.rglob("*.parquet")) if gh_dir.is_dir() else []
        print(f"github-code parquets {len(parquets)} target={args.structure_target}", flush=True)
        n_struct = recount(token_dir, "structure-train") + ing.stats["n_train_tokens"]
        consumed = list(cursor.get("consumed_github") or [])
        for path in parquets:
            if n_struct >= args.structure_target:
                break
            if path.name in consumed:
                continue
            print(f"ingest {path.name} struct_train={n_struct}", flush=True)
            n_rows = 0
            for row in iter_parquet_rows(path, columns=["code", "content", "text", "license", "language", "path", "repo_name"]):
                n_rows += 1
                code = row.get("code") or row.get("content") or row.get("text") or ""
                if not isinstance(code, str) or not code.strip():
                    continue
                if not license_ok(row.get("license")):
                    continue
                path_name = str(row.get("path") or row.get("repo_name") or path.name)
                lang = row.get("language")
                if is_struct_path(path_name, lang):
                    payload = code
                else:
                    payload = extract_signatures(code)
                    if len(payload) < 80:
                        continue
                added = ing.accept_text(
                    payload,
                    page_id=f"{path.name}:{n_rows}",
                    title=path_name,
                )
                n_struct += added
                if n_struct >= args.structure_target:
                    break
                if n_rows % 5000 == 0:
                    print(f"  {path.name} rows={n_rows} keep={ing.stats['n_keep']} tok={ing.stats['n_tokens']}", flush=True)
            consumed.append(path.name)
            cursor["consumed_github"] = consumed
            stats_all[path.name] = dict(ing.stats)
            writers_struct.train.flush()
            writers_struct.valid.flush()
            dump_json(corpus / "ingest-cursor.json", cursor)
            dump_json(corpus / "ingest-stats.json", stats_all)
            print(json.dumps({"parquet": path.name, "n_structure_train": n_struct, **ing.stats}, ensure_ascii=False), flush=True)
        cursor["github_done"] = n_struct >= args.structure_target
        dump_json(corpus / "reviews" / "structure.json", {"samples": ing.reviews})
    writers_struct.close()
    cursor["n_structure_train_tokens"] = recount(token_dir, "structure-train")
    cursor["n_structure_valid_tokens"] = recount(token_dir, "structure-valid")

    if not cursor.get("cwt_done"):
        writers = SplitWriters(token_dir, "colloquial", TOKENS_PER_SHARD, ROOT)
        ing = Ingestor(
            tok,
            writers,
            seen,
            wiki,
            leaks,
            hash_path,
            near,
            require_cjk=True,
            max_chars=MAX_CHARS,
            style="colloquial",
        )
        cwt_files = sorted((corpus / "raw" / "cwt2").rglob("part-0001.jsonl.gz"))
        if not cwt_files:
            print("missing CWT2 part-0001.jsonl.gz", file=sys.stderr)
            return 2
        path = cwt_files[0]
        print(f"ingest CWT2 {path} target={args.colloquial_target}", flush=True)
        n_col = 0
        n_rows = 0
        for row in iter_jsonl_gz(path):
            n_rows += 1
            text = row.get("text") or row.get("content") or ""
            domain = row.get("domain") or row.get("source_domain")
            quality = row.get("quality_score") or row.get("score")
            try:
                quality_f = float(quality) if quality is not None else None
            except (TypeError, ValueError):
                quality_f = None
            tox = row.get("toxicity")
            tox_f = None
            if isinstance(tox, dict):
                try:
                    tox_f = float(tox.get("score")) if tox.get("score") is not None else None
                except (TypeError, ValueError):
                    tox_f = None
                if tox.get("label") not in (None, 0, "0"):
                    tox_f = max(tox_f or 0.0, 1.0)
            elif tox is not None:
                try:
                    tox_f = float(tox)
                except (TypeError, ValueError):
                    tox_f = None
            added = ing.accept_text(
                str(text),
                page_id=f"cwt2:{n_rows}",
                title=str((domain_labels(domain) or ["cwt2"])[0]),
                domain=domain,
                quality=quality_f,
                toxicity=tox_f,
            )
            n_col += added
            if n_col >= args.colloquial_target:
                break
            if n_rows % 2000 == 0:
                print(
                    f"  cwt2 rows={n_rows} keep={ing.stats['n_keep']} tok={ing.stats['n_tokens']} skip_style={ing.stats['n_skip_style']}",
                    flush=True,
                )
        writers.close()
        cursor["cwt_done"] = True
        cursor["cwt_rows_seen"] = n_rows
        cursor["cwt_file"] = rel(path, ROOT)
        stats_all["cwt2"] = ing.stats
        dump_json(corpus / "reviews" / "colloquial.json", {"samples": ing.reviews})
        dump_json(corpus / "ingest-stats.json", stats_all)
        print(json.dumps({"cwt2": ing.stats}, ensure_ascii=False), flush=True)

    n_hq_train = recount(token_dir, "hq-train")
    n_hq_valid = recount(token_dir, "hq-valid")
    n_col_train = recount(token_dir, "colloquial-train")
    n_col_valid = recount(token_dir, "colloquial-valid")
    n_st_train = recount(token_dir, "structure-train")
    n_st_valid = recount(token_dir, "structure-valid")
    n_unique = WIKI_TRAIN_TOKENS + n_hq_train + n_col_train + n_st_train
    n_unk = 0
    for prefix in ("hq-train", "hq-valid", "colloquial-train", "colloquial-valid", "structure-train", "structure-valid"):
        for path in bins(token_dir, prefix):
            idx = path.with_name(path.stem + ".idx.jsonl")
            if not idx.is_file():
                continue
            for row in iter_jsonl(idx):
                n_unk += int(row.get("n_unk") or 0)
    n_tokens_all = n_unique + WIKI_VALID_TOKENS + n_hq_valid + n_col_valid + n_st_valid
    unk_rate = (WIKI_UNK + n_unk) / n_tokens_all if n_tokens_all else 0.0

    remaining_wiki = max(0, WIKI_TRAIN_TOKENS - 300_000_000)
    weight_sum = remaining_wiki + n_hq_train + n_col_train + n_st_train
    def w(n: int) -> float:
        return round(n / weight_sum, 6) if weight_sum else 0.0

    sources = {
        "wiki": {
            "band": "A",
            "n_train_tokens": WIKI_TRAIN_TOKENS,
            "skip_tokens_after_300m": 300_000_000,
            "train_shards": [rel(p, ROOT) for p in wiki_train],
            "valid_shards": [rel(p, ROOT) for p in wiki_valid],
        },
        "hq": {
            "band": "C",
            "license": "ODC-By-1.0 + Common-Crawl-ToU",
            "n_train_tokens": n_hq_train,
            "n_valid_tokens": n_hq_valid,
            "train_shards": [rel(p, ROOT) for p in bins(token_dir, "hq-train")],
            "valid_shards": [rel(p, ROOT) for p in bins(token_dir, "hq-valid")],
        },
        "colloquial": {
            "band": "C",
            "license": "ChineseWebText2.0 Apache-2.0 tag; webpage copyright not cleared",
            "file": cursor.get("cwt_file"),
            "n_train_tokens": n_col_train,
            "n_valid_tokens": n_col_valid,
            "train_shards": [rel(p, ROOT) for p in bins(token_dir, "colloquial-train")],
            "valid_shards": [rel(p, ROOT) for p in bins(token_dir, "colloquial-valid")],
        },
        "structure": {
            "band": "B/C",
            "license": "SchemaStore/OAI Apache-2.0; github-code per-file permissive",
            "n_train_tokens": n_st_train,
            "n_valid_tokens": n_st_valid,
            "train_shards": [rel(p, ROOT) for p in bins(token_dir, "structure-train")],
            "valid_shards": [rel(p, ROOT) for p in bins(token_dir, "structure-valid")],
        },
    }
    schedule = {
        "stage_id": "cpt-v2-after-300m",
        "parent_rung": "300m",
        "parent_tokens_seen": 300_000_000,
        "sampler_seed": 0,
        "skip_seen_wiki": True,
        "sources": {
            "wiki": {"weight": w(remaining_wiki), "skip_tokens": 300_000_000, "max_epochs": 1.0},
            "hq": {"weight": w(n_hq_train), "skip_tokens": 0, "max_epochs": 1.0},
            "colloquial": {"weight": w(n_col_train), "skip_tokens": 0, "max_epochs": 1.0},
            "structure": {"weight": w(n_st_train), "skip_tokens": 0, "max_epochs": 1.2},
        },
    }
    mix = {
        "id": "zh-pretrain-v2",
        "tokenizer": "zh-24k-v1",
        "parent_release": "zh-pretrain-v1",
        "valid_source": "zh-pretrain-v0 wiki valid (frozen, longitudinal)",
        "n_train_tokens": n_unique,
        "n_unique_train_tokens": n_unique,
        "n_valid_tokens": WIKI_VALID_TOKENS,
        "n_wiki_train_tokens": WIKI_TRAIN_TOKENS,
        "n_hq_train_tokens": n_hq_train,
        "n_colloquial_train_tokens": n_col_train,
        "n_structure_train_tokens": n_st_train,
        "target_train_tokens": [1_000_000_000, 1_500_000_000],
        "reached_target": 1_000_000_000 <= n_unique <= 1_500_000_000,
        "train_shards": (
            [rel(p, ROOT) for p in wiki_train]
            + [rel(p, ROOT) for p in bins(token_dir, "hq-train")]
            + [rel(p, ROOT) for p in bins(token_dir, "colloquial-train")]
            + [rel(p, ROOT) for p in bins(token_dir, "structure-train")]
        ),
        "valid_shards": [rel(p, ROOT) for p in wiki_valid],
        "valid_sets": {
            "wiki": [rel(p, ROOT) for p in wiki_valid],
            "hq": [rel(p, ROOT) for p in bins(token_dir, "hq-valid")],
            "colloquial": [rel(p, ROOT) for p in bins(token_dir, "colloquial-valid")],
            "structure": [rel(p, ROOT) for p in bins(token_dir, "structure-valid")],
        },
        "sources": sources,
        "schedule": rel(corpus / "schedule.json", ROOT),
    }
    cursor.update(
        {
            "n_hq_train_tokens": n_hq_train,
            "n_colloquial_train_tokens": n_col_train,
            "n_structure_train_tokens": n_st_train,
            "n_train_tokens": n_unique,
            "elapsed_s": round(time.time() - t0, 1),
        }
    )
    man = {
        "id": "zh-pretrain-v2",
        "source": "wiki-v0 + fineweb2-hq-v1 + cwt2-part-0001 + schemastore/oai + github-code",
        "wiki_manifest": rel(CORPUS_ZH_PRETRAIN / "manifest.json", ROOT),
        "parent_release": "zh-pretrain-v1",
        "n_wiki_train_tokens": WIKI_TRAIN_TOKENS,
        "n_valid_tokens": WIKI_VALID_TOKENS,
        "n_hq_train_tokens": n_hq_train,
        "n_colloquial_train_tokens": n_col_train,
        "n_structure_train_tokens": n_st_train,
        "n_train_tokens": n_unique,
        "n_unique_train_tokens": n_unique,
        "n_tokens": n_tokens_all,
        "n_unk_tokens_supplement": n_unk,
        "unk_token_rate_approx": round(unk_rate, 8),
        "unk_gate_ok": unk_rate <= UNK_TOKEN_MAX,
        "unk_token_max": UNK_TOKEN_MAX,
        "tokenizer_sha256": tok.model_sha256,
        "binary": True,
        "mix": rel(corpus / "mix.json", ROOT),
        "reached_target": mix["reached_target"],
        "in_band_1b_1p5b": mix["reached_target"],
        "gap_to_1b": max(0, 1_000_000_000 - n_unique),
        "gap_to_1p2b": max(0, 1_200_000_000 - n_unique),
    }
    release = {
        "release_id": "zh-pretrain-v2",
        "parent_release": "zh-pretrain-v1",
        "tokenizer": "zh-24k-v1",
        "tokenizer_sha256": tok.model_sha256,
        "unique_train_tokens": n_unique,
        "ledger": "unique counts first-seen SHA256 chunks; exposure is trainer-side",
        "blocked": ["bigcode/the-stack", "bigcode/starcoderdata"],
        "created_unix": int(time.time()),
    }
    dump_json(corpus / "mix.json", mix)
    dump_json(corpus / "schedule.json", schedule)
    dump_json(corpus / "manifest.json", man)
    dump_json(corpus / "RELEASE.json", release)
    dump_json(
        corpus / "source-role.json",
        {
            "wiki": {"role": "cpt_encyclopedia", "band": "A"},
            "hq": {"role": "cpt_hq_web", "band": "C"},
            "colloquial": {"role": "cpt_colloquial_web", "band": "C"},
            "structure": {"role": "cpt_structure_tech", "band": "B/C"},
        },
    )
    dump_json(
        corpus / "unique-ledger.json",
        {
            "unique_train_tokens": n_unique,
            "by_source": {
                "wiki": WIKI_TRAIN_TOKENS,
                "hq": n_hq_train,
                "colloquial": n_col_train,
                "structure": n_st_train,
            },
            "valid": {
                "wiki": WIKI_VALID_TOKENS,
                "hq": n_hq_valid,
                "colloquial": n_col_valid,
                "structure": n_st_valid,
            },
            "exposure_note": "trainer records per-source window counts; structure max_epochs=1.2 counts as exposure only",
        },
    )
    dump_json(corpus / "ingest-cursor.json", cursor)
    dump_json(corpus / "ingest-stats.json", stats_all)
    print(
        json.dumps(
            {
                "n_train_tokens": n_unique,
                "n_hq_train_tokens": n_hq_train,
                "n_colloquial_train_tokens": n_col_train,
                "n_structure_train_tokens": n_st_train,
                "unk_token_rate_approx": man["unk_token_rate_approx"],
                "reached_target": mix["reached_target"],
                "elapsed_s": cursor["elapsed_s"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if n_col_train < 30_000_000:
        print(f"colloquial train {n_col_train} < 30M", file=sys.stderr)
        return 2
    if n_st_train < 20_000_000:
        print(f"structure train {n_st_train} < 20M", file=sys.stderr)
        return 2
    if not (1_000_000_000 <= n_unique <= 1_500_000_000):
        print(f"unique {n_unique} outside 1.0–1.5B", file=sys.stderr)
        return 2
    if unk_rate > UNK_TOKEN_MAX:
        print(f"UNK {unk_rate} exceeds {UNK_TOKEN_MAX}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
