"""Paired Float Base comparison on identical text across different tokenizers."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np

from common.paths import ensure_formal_on_path
from common.evidence_stage import sha256, write_json, stage


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(item: dict) -> Path:
    path = Path(item["path"])
    if sha256(path) != item["sha256"]:
        raise ValueError(f"hash mismatch: {path}")
    return path


def _indices(n: int, limit: int) -> list[int]:
    count = min(int(limit), int(n))
    return [i * n // count for i in range(count)] if count else []


def _decode_record(sp, ids: list[int]) -> str:
    body = list(ids)
    if body and body[0] == 2:
        body = body[1:]
    if body and body[-1] == 1:
        body = body[:-1]
    text = sp.decode(body)
    rebuilt = [2] + list(sp.encode(text, out_type=int)) + [1]
    if rebuilt != ids:
        raise ValueError("lossless source record failed token roundtrip")
    return text


def _bounded_text(text: str, key: str, max_chars: int) -> tuple[str, int]:
    if len(text) <= max_chars:
        return text, 0
    span = len(text) - max_chars + 1
    start = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16) % span
    return text[start : start + max_chars], start


def _indexed_panel(spec: dict) -> tuple[list[dict], dict]:
    import sentencepiece as sentencepiece

    index = _check(spec["index"])
    tokenizer = _check(spec["source_tokenizer"])
    sp = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
    con = sqlite3.connect(index)
    con.row_factory = sqlite3.Row
    limit = int(spec.get("records_per_source", 32))
    max_chars = int(spec.get("max_chars_per_record", 8192))
    sources = [r[0] for r in con.execute(
        "select distinct source from records where split=? order by source", (spec.get("split", "dev"),)
    )]
    maps: dict[str, np.memmap] = {}
    snippets: list[dict] = []
    selection: list[dict] = []
    for source in sources:
        rows = list(con.execute(
            """select r.id,r.file_index,r.row_number,r.source,r.domain,r.language,
                      r.token_offset,r.token_length,i.bin_path
               from records r join input_files i using(file_index)
               where r.split=? and r.source=? order by r.id""",
            (spec.get("split", "dev"), source),
        ))
        for pos in _indices(len(rows), limit):
            row = rows[pos]
            bin_path = str(row["bin_path"])
            if bin_path not in maps:
                maps[bin_path] = np.memmap(bin_path, dtype="<u2", mode="r")
            ids = maps[bin_path][
                int(row["token_offset"]) : int(row["token_offset"]) + int(row["token_length"])
            ].astype(np.int32).tolist()
            text = _decode_record(sp, ids)
            text, char_start = _bounded_text(text, f"indexed:{row['id']}", max_chars)
            sid = f"v13-dev:{source}:{row['id']}"
            meta = {
                "snippet_id": sid,
                "panel": spec["panel_id"],
                "source": source,
                "domain": row["domain"],
                "language": row["language"],
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "utf8_bytes": len(text.encode("utf-8")),
                "characters": len(text),
            }
            snippets.append({**meta, "text": text})
            selection.append({
                **meta,
                "record_id": row["id"],
                "file_index": row["file_index"],
                "row_number": row["row_number"],
                "bin_path": bin_path,
                "token_offset": row["token_offset"],
                "token_length": row["token_length"],
                "char_start": char_start,
            })
    con.close()
    return snippets, {"kind": "indexed_records", "items": selection}


def _decode_window(sp, ids: list[int]) -> str:
    pieces: list[str] = []
    current: list[int] = []
    for token in ids:
        if token == 0:
            continue
        if token in {1, 2}:
            if current:
                pieces.append(sp.decode(current))
                current = []
            continue
        current.append(int(token))
    if current:
        pieces.append(sp.decode(current))
    return "\n".join(piece for piece in pieces if piece)


def _legacy_panel(spec: dict) -> tuple[list[dict], dict]:
    import sentencepiece as sentencepiece
    from common.data import PackedTokenSource

    mix_path = _check(spec["mix"])
    tokenizer = _check(spec["source_tokenizer"])
    mix = _load_json(mix_path)
    sp = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
    limit = int(spec.get("windows_per_source", 32))
    snippets: list[dict] = []
    selection: list[dict] = []
    for source, item in sorted((mix.get("sources") or {}).items()):
        paths = [Path(p) for p in item.get("valid_shards") or []]
        packed = PackedTokenSource(paths, int(spec.get("seq_len", 2048)))
        for index in _indices(len(packed), limit):
            window = packed[index]
            ids = list(window["x"]) + [window["y"][-1]]
            text = _decode_window(sp, ids)
            if not text:
                continue
            sid = f"v12-valid:{source}:{index}"
            meta = {
                "snippet_id": sid,
                "panel": spec["panel_id"],
                "source": source,
                "domain": source,
                "language": "en" if source in {"code", "wiki_en"} else "zh_or_mixed",
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "utf8_bytes": len(text.encode("utf-8")),
                "characters": len(text),
            }
            snippets.append({**meta, "text": text})
            selection.append({**meta, "window_index": index, "valid_shards": [str(p) for p in paths]})
    return snippets, {"kind": "legacy_token_windows_decoded_to_text", "items": selection}


def _score_model(
    model_spec: dict,
    snippets: list[dict],
    device: str,
    partial_path: Path,
) -> tuple[list[dict], dict]:
    import mlx.core as mx
    import sentencepiece as sentencepiece
    from architecture import NeedleZh
    from config import NeedleZhConfig
    from common.checkpoint import load_params
    from common.train_common import masked_lm_loss

    mx.set_default_device(mx.cpu if device == "cpu" else mx.gpu)
    weights = _check(model_spec["weights"])
    tokenizer = _check(model_spec["tokenizer"])
    sp = sentencepiece.SentencePieceProcessor(model_file=str(tokenizer))
    config = NeedleZhConfig(**model_spec["model_config"])
    model = NeedleZh(config)
    model.eval()
    loaded = load_params(model, weights, strict=True, return_report=True)
    if loaded["missing"] or loaded["unexpected"]:
        raise ValueError(f"tensor contract mismatch for {model_spec['model_id']}")
    rows: list[dict] = []
    for n, snippet in enumerate(snippets, 1):
        text = snippet["text"]
        content = list(sp.encode(text, out_type=int))
        ids = [int(model_spec.get("bos_id", 2))] + content + [int(model_spec.get("eos_id", 1))]
        rebuilt = sp.decode(content)
        nll = 0.0
        predicted = 0
        seq_len = int(config.max_seq_len)
        for start in range(0, len(ids) - 1, seq_len):
            length = min(seq_len, len(ids) - start - 1)
            if length <= 0:
                continue
            x_ids = ids[start : start + length]
            y_ids = ids[start + 1 : start + 1 + length]
            pad = seq_len - length
            x = mx.array([x_ids + [0] * pad], dtype=mx.int32)
            y = mx.array([y_ids + [0] * pad], dtype=mx.int32)
            mask = mx.array([[1.0] * length + [0.0] * pad], dtype=mx.float32)
            loss = float(masked_lm_loss(model(x)["logits"], y, mask))
            nll += loss * len(y_ids)
            predicted += len(y_ids)
        rows.append({
            "model_id": model_spec["model_id"],
            "snippet_id": snippet["snippet_id"],
            "panel": snippet["panel"],
            "source": snippet["source"],
            "domain": snippet["domain"],
            "language": snippet["language"],
            "text_sha256": snippet["text_sha256"],
            "utf8_bytes": snippet["utf8_bytes"],
            "characters": snippet["characters"],
            "content_tokens": len(content),
            "predicted_tokens": predicted,
            "nll_sum": nll,
            "roundtrip_exact": rebuilt == text,
        })
        if n % 16 == 0 or n == len(snippets):
            write_json(partial_path, {"complete": n == len(snippets), "rows": rows})
            mx.clear_cache()
            print(f"{model_spec['model_id']}: {n}/{len(snippets)} snippets", flush=True)
    del model
    mx.clear_cache()
    gc.collect()
    return rows, loaded


def _aggregate(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, values in sorted(groups.items()):
        nll = sum(float(v["nll_sum"]) for v in values)
        raw_bytes = sum(int(v["utf8_bytes"]) for v in values)
        predicted = sum(int(v["predicted_tokens"]) for v in values)
        content = sum(int(v["content_tokens"]) for v in values)
        row = {k: v for k, v in zip(keys, key)}
        row.update({
            "snippets": len(values),
            "utf8_bytes": raw_bytes,
            "predicted_tokens": predicted,
            "content_tokens": content,
            "nll_sum": nll,
            "token_loss": nll / max(predicted, 1),
            "nats_per_byte": nll / max(raw_bytes, 1),
            "bits_per_byte": nll / max(raw_bytes * math.log(2), 1e-12),
            "content_tokens_per_1000_bytes": content * 1000.0 / max(raw_bytes, 1),
            "roundtrip_exact": sum(bool(v["roundtrip_exact"]) for v in values),
        })
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = _load_json(args.config)
    if cfg.get("action") != "cross_tokenizer_base_compare":
        raise ValueError("wrong comparison action")
    if len(cfg.get("models") or []) != 2:
        raise ValueError("comparison requires exactly two models")
    ensure_formal_on_path()
    with stage(cfg, track="diagnostic", pipeline_id="mei-51m-cross-tokenizer-base-compare-v1") as out:
        started = time.monotonic()
        snippets: list[dict] = []
        selections = []
        for panel in cfg["panels"]:
            if panel["kind"] == "indexed_records":
                rows, selection = _indexed_panel(panel)
            elif panel["kind"] == "legacy_token_windows":
                rows, selection = _legacy_panel(panel)
            else:
                raise ValueError(f"unsupported panel kind {panel['kind']}")
            snippets.extend(rows)
            selections.append({"panel_id": panel["panel_id"], **selection})
        write_json(out / "selection.json", {"panels": selections})
        model_rows: dict[str, list[dict]] = {}
        loads = {}
        for model_spec in cfg["models"]:
            partial = out / f"partial-{model_spec['model_id']}.json"
            rows, loaded = _score_model(
                model_spec,
                snippets,
                str(cfg.get("device") or "gpu"),
                partial,
            )
            model_rows[model_spec["model_id"]] = rows
            loads[model_spec["model_id"]] = loaded
            write_json(out / f"details-{model_spec['model_id']}.json", {"rows": rows})
            write_json(out / "progress.json", {
                "completed_models": list(model_rows),
                "elapsed_seconds": time.monotonic() - started,
            })
        exact_ids = set.intersection(*[
            {r["snippet_id"] for r in rows if r["roundtrip_exact"]}
            for rows in model_rows.values()
        ])
        all_rows = [row for rows in model_rows.values() for row in rows]
        exact_rows = [row for row in all_rows if row["snippet_id"] in exact_ids]
        summary = {
            "schema": "mei-51m-cross-tokenizer-base-compare-v1",
            "ok": True,
            "scope": "paired Float Base LM comparison on identical decoded text; no SFT, QAT or tool-use claim",
            "models": [{k: v for k, v in item.items() if k not in {"model_config"}} for item in cfg["models"]],
            "panels": [{"panel_id": p["panel_id"], "kind": p["kind"]} for p in cfg["panels"]],
            "snippets": len(snippets),
            "common_roundtrip_exact_snippets": len(exact_ids),
            "all_text": {
                "overall": _aggregate(all_rows, ("model_id", "panel")),
                "by_source": _aggregate(all_rows, ("model_id", "panel", "source")),
                "by_domain": _aggregate(all_rows, ("model_id", "panel", "domain")),
            },
            "common_roundtrip_exact_text": {
                "overall": _aggregate(exact_rows, ("model_id", "panel")),
                "by_source": _aggregate(exact_rows, ("model_id", "panel", "source")),
                "by_domain": _aggregate(exact_rows, ("model_id", "panel", "domain")),
            },
            "load_reports": loads,
            "elapsed_seconds": time.monotonic() - started,
            "limitations": [
                "Different deterministic tokenizers define different token sequences; bits-per-byte is the primary normalized diagnostic.",
                "The v1.3 panel is held-out from v1.3 training but source-aligned to its recipe; the legacy panel is source-aligned to v1.2.",
                "Legacy valid windows are decoded to common text before re-encoding, so their original packed-token loss is not reused.",
                "Base LM likelihood is not evidence of tool-call exactness, disposition, confidence, narration, QAT quality or release eligibility.",
            ],
        }
        write_json(out / "receipt.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
