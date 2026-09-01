#!/usr/bin/env python3
"""Scan existing corpora and SFT packs into registries. Does not move files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jobs.registry import empty_jobs_index, save_corpora, save_jobs  # noqa: E402
from jobs.schema import validate_corpus_entry  # noqa: E402
from repo_paths import ROOT  # noqa: E402

CORPUS_OVERRIDES = {
    "zh-vocab-v0": {"role": "vocab", "state": "accepted", "subscribers": ["needle-zh"]},
    "zh-pretrain-v0": {"role": "cpt-wiki", "state": "accepted", "subscribers": ["needle-zh"]},
    "zh-pretrain-v1": {"role": "cpt-atlas", "state": "accepted", "subscribers": ["needle-zh"]},
    "zh-pretrain-v2": {"role": "cpt-atlas", "state": "accepted", "subscribers": ["needle-zh"]},
    "zh-pretrain-v3": {"role": "cpt-structure", "state": "accepted", "subscribers": ["needle-zh"]},
    "zh-pretrain-v4": {"role": "cpt-atlas", "state": "fail_closed", "subscribers": ["needle-zh"]},
    "zh-pretrain-colloquial-synth-pooled-v1": {
        "role": "cpt-colloquial",
        "state": "draft",
        "job_id": "260826-01-colloquial-30m",
        "subscribers": [],
    },
    "zh-pretrain-colloquial-synth-qwen-v1": {"role": "cpt-colloquial", "state": "job_complete", "subscribers": []},
    "zh-pretrain-colloquial-synth-dsflash-v1": {"role": "cpt-colloquial", "state": "job_complete", "subscribers": []},
    "zh-pretrain-colloquial-synth-qwen36plus-v1": {"role": "cpt-colloquial", "state": "job_complete", "subscribers": []},
    "zh-pretrain-colloquial-synth-qwen37plus-v1": {"role": "cpt-colloquial", "state": "job_complete", "subscribers": []},
    "zh-pretrain-colloquial-synth-glm52-v1": {"role": "cpt-colloquial", "state": "job_complete", "subscribers": []},
    "zh-pretrain-colloquial-synth-kimi-k3-v1": {"role": "cpt-colloquial", "state": "job_complete", "subscribers": []},
    "zh-pretrain-colloquial-synth-v1": {"role": "cpt-colloquial", "state": "engineering_contrast", "subscribers": []},
    "zh-pretrain-colloquial-synth-v2": {"role": "cpt-colloquial", "state": "engineering_contrast", "subscribers": []},
    "zh-pretrain-colloquial-synth-ollama-bakeoff": {
        "role": "cpt-colloquial",
        "state": "engineering_contrast",
        "subscribers": [],
    },
    "zh-pretrain-colloquial-synth-smoke": {"role": "cpt-colloquial", "state": "archive", "subscribers": []},
    "zh-pretrain-colloquial-synth-qwen-smoke": {"role": "cpt-colloquial", "state": "archive", "subscribers": []},
    "sft-style-v0": {"role": "sft-style", "state": "accepted", "subscribers": []},
    "shared-tool-traces-v0": {"role": "sft-traces", "state": "empty", "subscribers": []},
    "_probe": {"role": "probe", "state": "probe", "subscribers": []},
}

TOPICS = {
    "colloquial-cpt": "Spoken-role CPT synthesis",
    "general-cpt": "Wiki / HQ / structure CPT atlas",
    "toolcall-sft": "needle-zh / mei-1.0-51m SFT packs",
    "mei-expert-sft": "Qwen3.5 0.8B expert SFT",
}


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def live_unique(corpus: Path) -> tuple[int | None, str | None]:
    cursor = read_json(corpus / "state" / "cursor.json")
    release = read_json(corpus / "RELEASE.json")
    manifest = read_json(corpus / "manifest.json")
    gen = cursor.get("generator") or release.get("generator")
    for blob in (cursor, release, manifest):
        for key in ("n_unique_train_tokens", "unique_train_tokens", "n_train_tokens"):
            if blob.get(key) is not None:
                return int(blob[key]), str(gen) if gen else None
    return None, str(gen) if gen else None


def scan_corpora(root: Path) -> list[dict]:
    rows = []
    corpora = root / "corpora"
    if not corpora.is_dir():
        return rows
    for path in sorted(p for p in corpora.iterdir() if p.is_dir()):
        cid = path.name
        override = CORPUS_OVERRIDES.get(cid, {"role": "unknown", "state": "draft", "subscribers": []})
        unique, gen = live_unique(path)
        release = path / "RELEASE.json"
        formal = bool(read_json(release).get("formal_cpt_eligible"))
        generators = [gen] if gen else []
        entry = {
            "id": cid,
            "role": override["role"],
            "state": override["state"],
            "path": f"corpora/{cid}",
            "job_id": override.get("job_id"),
            "release": f"corpora/{cid}/RELEASE.json" if release.is_file() else None,
            "generators": generators,
            "unique_train_tokens": unique,
            "formal_eligible": formal,
            "subscribers": list(override.get("subscribers") or []),
        }
        rows.append(validate_corpus_entry(entry))
    return rows


def scan_sft_packs(root: Path) -> list[dict]:
    packs_dir = root / "tasks" / "needle-zh" / "train" / "packs"
    expert_seed = root / "tasks" / "mei-expert-qwen35-0p8b" / "train" / "seed" / "sft-smoke-v0.jsonl"
    rows = []
    if packs_dir.is_dir():
        invalid_v1 = (packs_dir / "mei-tool-sft-v1.INVALID_FOR_PUBLISH.json").is_file()
        invalid_route = (packs_dir / "mei-tool-route-sft-v1.INVALID_FOR_V2_PUBLISH.json").is_file()
        for path in sorted(packs_dir.glob("*.jsonl")):
            rel = path.relative_to(root).as_posix()
            name = path.name
            state = "draft"
            if name.endswith(".candidates.jsonl") or ".review-sample." in name:
                state = "draft"
            elif name.startswith("mei-tool-sft-v1") and invalid_v1:
                state = "invalid"
            elif name.startswith("mei-tool-route-sft-v1") and invalid_route:
                state = "invalid"
            elif name == "mei-toolcall-v2-smoke.jsonl":
                state = "smoke"
            elif name in {"home-sft-2k.jsonl", "home-sft-10k.jsonl"}:
                state = "accepted"
            elif name == "mw-sft-v0-2k.jsonl":
                state = "draft"
            rows.append({"id": path.stem, "path": rel, "task_id": "needle-zh", "state": state})
    seed = root / "tasks" / "needle-zh" / "train" / "seed" / "sft-phase1-v0.jsonl"
    if seed.is_file():
        legacy = (seed.parent / "sft-phase1-v0.LEGACY_DIAGNOSTIC_ONLY.json").is_file()
        rows.append(
            {
                "id": "sft-phase1-v0",
                "path": seed.relative_to(root).as_posix(),
                "task_id": "needle-zh",
                "state": "invalid" if legacy else "draft",
            }
        )
    if expert_seed.is_file():
        rows.append(
            {
                "id": "sft-smoke-v0",
                "path": expert_seed.relative_to(root).as_posix(),
                "task_id": "mei-expert-qwen35-0p8b",
                "state": "accepted",
            }
        )
    return rows


def seed_topics(root: Path) -> None:
    from jobs.create_job import ensure_topic

    for topic, title in TOPICS.items():
        ensure_topic(topic, root=root)
        from jobs.paths import topic_dir

        readme = topic_dir(topic, root=root) / "README.md"
        if readme.is_file() and title not in readme.read_text(encoding="utf-8"):
            continue
        if not readme.is_file():
            readme.write_text(f"# {topic}\n\n{title}\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()
    root = args.root.resolve()
    seed_topics(root)
    corpora = {
        "version": 1,
        "note": "Process state lives in jobs/. This file is the publish catalog. Do not treat raw/accepted.jsonl as outbox/accepted.",
        "corpora": scan_corpora(root),
    }
    save_corpora(corpora, root=root)
    jobs = empty_jobs_index()
    jobs["topics"] = {
        topic: {
            "inbox": f"jobs/{topic}/inbox",
            "jobs_glob": f"jobs/{topic}/jobs/*",
            "outbox": f"jobs/{topic}/outbox",
        }
        for topic in TOPICS
    }
    jobs["sft_packs"] = scan_sft_packs(root)
    save_jobs(jobs, root=root)
    print(json.dumps({"ok": True, "n_corpora": len(corpora["corpora"]), "n_packs": len(jobs["sft_packs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
