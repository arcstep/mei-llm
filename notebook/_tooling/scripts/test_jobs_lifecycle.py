#!/usr/bin/env python3
"""Tests for jobs/ path resolution, registry, promote rewrite, and corpus serving layout."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from repo_paths import ROOT, SCRIPTS_ROOT, resolve_rel
sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT / "jobs"))

from jobs.paths import job_work, receipt_path, rel_to_root  # noqa: E402
from jobs.registry import load_jobs  # noqa: E402
from jobs.schema import validate_corpus_entry, validate_job  # noqa: E402
from jobs.promote_corpus import rewrite_rel  # noqa: E402

FORBIDDEN_CORPUS_NAMES = {
    "raw",
    "reviews",
    "shards",
    "ingest-cursor.json",
    "ingest-stats.json",
    "extract-manifest.json",
    "recipe-lock.json",
    "parents.json",
    "unique-ledger.json",
    "source-role.json",
    "approved-colloquial.json",
}


def test_resolve_rel_round_trip() -> None:
    published = (ROOT / "corpus/lm-v1/RELEASE.json").resolve()
    got = resolve_rel("corpus/lm-v1/RELEASE.json")
    assert got == published
    receipt = ROOT / "notebook/corpus/lm-v1/assemble/work/zh-pretrain-v4/receipt.json"
    assert receipt.is_file() and not receipt.is_symlink()
    pointer = json.loads(receipt.read_text(encoding="utf-8"))
    assert pointer["published_path"] == "corpus/lm-v1"
    assert pointer.get("writable") is False
    assemble_release = ROOT / "notebook/corpus/lm-v1/assemble/work/zh-pretrain-v4/RELEASE.json"
    assert not assemble_release.exists()
    try:
        resolve_rel("../secret")
    except ValueError:
        pass
    else:
        raise AssertionError("expected escape to fail")


def test_job_paths() -> None:
    work = job_work("colloquial-cpt", "260826-01-synthesize")
    assert work.as_posix().endswith(
        "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work"
    )
    rec = receipt_path("colloquial-cpt", "260826-01-synthesize", "draft")
    assert rec.name == "receipt.json"
    from jobs.paths import topic_dir

    toolcall = topic_dir("toolcall-sft", root=ROOT)
    assert toolcall.as_posix().endswith("notebook/jobs/toolcall-sft")
    assert "/mei-llm/jobs/toolcall-sft" not in toolcall.as_posix()


def test_promote_rewrite() -> None:
    text = '{"train_shards": ["jobs/t/jobs/j/work/tokens/a.bin"]}'
    out = rewrite_rel(text, "jobs/t/jobs/j/work", "corpus/new-id")
    assert "corpus/new-id/tokens/a.bin" in out
    assert "jobs/t/jobs/j/work" not in out


def test_create_close_temp() -> None:
    from jobs.create_job import ensure_topic
    from jobs.registry import upsert_job, upsert_receipt_pointer
    from jobs.schema import validate_receipt

    tmp = Path(tempfile.mkdtemp(prefix="mei-jobs-"))
    try:
        ensure_topic("demo", root=tmp)
        work = job_work("demo", "260101-01-x", root=tmp)
        work.mkdir(parents=True)
        card = {
            "job_id": "260101-01-x",
            "topic": "demo",
            "kind": "produce",
            "status": "running",
            "work_dir": rel_to_root(work, root=tmp),
        }
        validate_job(card)
        upsert_job(card, root=tmp)
        receipt = {
            "job_id": "260101-01-x",
            "topic": "demo",
            "status": "draft",
            "publish_policy": "in_place_register",
        }
        validate_receipt(receipt)
        upsert_receipt_pointer(receipt, root=tmp)
        jobs = load_jobs(tmp)
        assert jobs["jobs"][0]["receipt"].endswith("outbox/draft/260101-01-x/receipt.json")
        entry = {
            "id": "demo-corpus",
            "role": "cpt-colloquial",
            "state": "draft",
            "path": "corpus/demo-corpus",
        }
        validate_corpus_entry(entry)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_repo_indexes_exist() -> None:
    current = json.loads((ROOT / "CURRENT.json").read_text(encoding="utf-8"))
    assert current["tokenizer"] == "tokenizer/zh-24k-v1"
    assert current["corpus"] == "corpus/lm-v1"
    assert current["architecture"] == "architecture/mei-1.0-58m-arch-v1"
    assert current["training"] == "training/mei-1.0-58m-train-v1"
    assert current["base"] is None
    assert current["sft"] is None
    assert current["runtime"] is None
    assert current["blocked"] == []
    assert current["plan"]["mode"] == "scratch"
    assert current["plan"]["parent_checkpoint"] is None
    card = json.loads(
        (
            ROOT
            / "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/job.json"
        ).read_text(encoding="utf-8")
    )
    assert card["status"] == "draft"
    assert card["formal_cpt_eligible"] is False
    assert card["publish_path"] == (
        "corpus/lm-v1/colloquial/zh-pretrain-colloquial-synth-pooled-v1"
    )
    draft = ROOT / "notebook/corpus/lm-v1/colloquial/outbox/draft/colloquial-v1"
    stored = ROOT / "corpus/lm-v1/colloquial/zh-pretrain-colloquial-synth-pooled-v1"
    assert draft.is_dir() and not draft.is_symlink()
    assert stored.is_dir() and not stored.is_symlink()
    assert draft.resolve() != stored.resolve()
    published = json.loads((draft / "published.json").read_text(encoding="utf-8"))
    assert published["published_path"] == (
        "corpus/lm-v1/colloquial/zh-pretrain-colloquial-synth-pooled-v1"
    )
    assert published["status"] == "active"


def test_serving_layout() -> None:
    sys.path.insert(0, str(ROOT / "training" / "mei-1.0-58m-train-v1"))
    from _repo import ensure_formal_on_path  # noqa: E402

    ensure_formal_on_path()
    from data import file_sha256, list_token_shards  # noqa: E402

    root = ROOT / "corpus" / "lm-v1"
    mix = json.loads((root / "mix.json").read_text(encoding="utf-8"))
    release = json.loads((root / "RELEASE.json").read_text(encoding="utf-8"))
    assert not (root / "schedule.json").exists()
    assert mix["id"] == "lm-v1"
    assert mix["roles_complete"] is True
    assert mix["colloquial_promoted"] is True
    assert mix["n_colloquial_train_tokens"] == 30108616
    assert mix["colloquial_stored"]["status"] == "active"
    assert mix["colloquial_stored"]["admitted"] is True
    assert release["roles_complete"] is True
    assert release["training_mode"] == "scratch"
    scratch = json.loads((root / "schedule-scratch.json").read_text(encoding="utf-8"))
    assert scratch["kind"] == "scratch"
    assert scratch["sampler"] == "quota_plan"
    assert int(scratch["parent_tokens_seen"] or 0) == 0
    assert set(scratch["sources"]) == {"wiki", "hq", "structure", "colloquial"}
    assert mix["schedule_scratch"] == "corpus/lm-v1/schedule-scratch.json"
    structure_rel = json.loads(
        (root / "structure/zh-pretrain-v3/RELEASE.json").read_text(encoding="utf-8")
    )
    assert (ROOT / structure_rel["unique_ledger"]).is_file()
    leftovers: list[str] = []
    for path in root.rglob("*"):
        if path.name in FORBIDDEN_CORPUS_NAMES or path.name.endswith(".idx.jsonl"):
            leftovers.append(str(path.relative_to(ROOT)))
        if path.is_dir() and path.name in {"raw", "reviews", "shards"}:
            leftovers.append(str(path.relative_to(ROOT)))
    assert leftovers == []
    accepted = [
        ROOT / "notebook/corpus/lm-v1/language/outbox/accepted/zh-pretrain-v0",
        ROOT / "notebook/corpus/lm-v1/language/outbox/accepted/zh-pretrain-v1",
        ROOT / "notebook/corpus/lm-v1/structure/outbox/accepted/zh-pretrain-v3",
    ]
    for link in accepted:
        assert link.is_symlink()
        assert str(link.resolve()).startswith(str((ROOT / "corpus/lm-v1").resolve()))
    hashes = json.loads((root / "hashes.json").read_text(encoding="utf-8"))
    for name, expected in hashes.items():
        got = file_sha256(root / name)
        assert got == expected, f"{name} hash mismatch"

    trains = list_token_shards(root, "train")
    valids = list_token_shards(root, "valid")
    assert trains and valids
    assert all(p.is_file() for p in trains + valids)
    assert any("colloquial" in p.as_posix() for p in trains)
    sources = set()
    for p in trains:
        rel = p.relative_to(ROOT).as_posix()
        if "/zh-pretrain-v0/" in rel:
            sources.add("wiki")
        elif "/language/hq/" in rel:
            sources.add("hq")
        elif "/zh-pretrain-v3/" in rel:
            sources.add("structure")
        elif "/colloquial/" in rel:
            sources.add("colloquial")
        else:
            raise AssertionError(f"unexpected active shard {rel}")
    assert sources == {"wiki", "hq", "structure", "colloquial"}


def main() -> int:
    test_resolve_rel_round_trip()
    test_job_paths()
    test_promote_rewrite()
    test_create_close_temp()
    test_repo_indexes_exist()
    test_serving_layout()
    print(json.dumps({"ok": True, "tests": 6}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
