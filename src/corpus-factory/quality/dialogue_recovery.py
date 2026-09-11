from __future__ import annotations

import hashlib
import json
import re
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            value.update(block)
    return value.hexdigest()


def normalized(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"<sup>.*?</sup>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|[^>]+\|>|<eo[hmtrc]>", "", text)
    return re.sub(r"\s+", " ", text).strip().lstrip(":").strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(normalized(text).encode()).hexdigest()


def role_turns(row: dict):
    for turn in row.get("chat", {}).values():
        for role in ("Human", "MOSS"):
            text = str(turn.get(role) or "")
            if re.search(r"[\u4e00-\u9fff]", text):
                yield role, text


def audit(manifest_path: Path, config_path: Path) -> dict:
    config = json.loads(config_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    output = Path(config["output_directory"])
    output.mkdir(parents=True, exist_ok=False)
    baseline = Path(config["baseline_raw"])
    archive_path = Path(config["archive"])
    filtered = Path(config["previous_filtered"])
    inputs = [manifest_path, config_path, baseline, archive_path, filtered,
              Path(config["terminal_summary"]), Path(config["layout_mix"]),
              Path(manifest["clearance_receipt"]["path"])]
    forbidden = {}
    for name in config["quarantined_artifacts"]:
        path = Path(name)
        forbidden[name] = digest(path)
    binding = {str(path): digest(path) for path in inputs}
    if binding[str(archive_path)] != config["archive_sha256"]:
        raise ValueError("archive hash mismatch")
    old_hashes = set()
    with baseline.open(encoding="utf-8") as stream:
        for line in stream:
            old_hashes.add(text_hash(json.loads(line)["text"]))
    counts = Counter()
    for line in filtered.open(encoding="utf-8"):
        text = json.loads(line)["text"]
        counts["previous_filtered_documents"] += 1
        counts["documents_with_dangling_citations"] += bool(re.search(r"<sup>|<\|\d+\|>", text))
        counts["documents_with_any_marker"] += bool(re.search(r"<\|[^>]+\|>|<eo[hmtrc]>", text))
    clean_hashes = set()
    examples = []
    retained_path = output / "non_tool_unseen_candidates.jsonl"
    with zipfile.ZipFile(archive_path) as archive, retained_path.open("x", encoding="utf-8") as retained:
        members = [name for name in archive.namelist() if name.endswith(".jsonl")]
        for member in members:
            with archive.open(member) as stream:
                for line in stream:
                    row = json.loads(line)
                    counts["archive_conversations"] += 1
                    turns = list(role_turns(row))
                    if not turns:
                        continue
                    counts["chinese_conversations"] += 1
                    shared = []
                    for role, text in turns:
                        counts[f"{role}_turns"] += 1
                        overlap = text_hash(text) in old_hashes
                        counts[f"{role}_turns_shared_with_old"] += overlap
                        shared.append(overlap)
                    counts["conversations_with_any_shared_turn"] += any(shared)
                    counts["conversations_all_retained_turns_shared"] += all(shared)
                    uses_tools = any(turn.get("Commands") or turn.get("Tool Responses")
                                     for turn in row.get("chat", {}).values())
                    counts["chinese_conversations_with_tool_dependency"] += uses_tools
                    if uses_tools or any(shared):
                        if len(examples) < 12:
                            examples.append({"conversation_id": row.get("conversation_id"),
                                             "tool_dependency": uses_tools, "shared_turns": sum(shared),
                                             "retained_turns": len(turns)})
                        continue
                    text = "\n".join(normalized(text) for _, text in turns)
                    key = text_hash(text)
                    if key in clean_hashes:
                        counts["duplicate_non_tool_conversations"] += 1
                        continue
                    clean_hashes.add(key)
                    counts["non_tool_unseen_candidate_documents"] += 1
                    retained.write(json.dumps({"text": text, "conversation_id": row.get("conversation_id"),
                                               "source_archive_sha256": config["archive_sha256"],
                                               "member": member, "status": "unreviewed_candidate"},
                                              ensure_ascii=False) + "\n")
    terminal = json.loads(Path(config["terminal_summary"]).read_text())
    mix = json.loads(Path(config["layout_mix"]).read_text())
    remaining = {role: source["n_train_tokens"] - terminal["source_token_cursors"][role]
                 for role, source in mix["sources"].items()}
    result = {
        "schema": "mei-dialogue-supplement-recovery-audit-v1", "status": "blocked",
        "corpus_reuse_eligible": False,
        "decision": "retire_previous_candidates_pending_replacement",
        "quarantined_artifacts": forbidden,
        "inputs": binding, "implementation_sha256": digest(Path(__file__)),
        "counts": dict(counts), "examples": examples,
        "terminal_exposure": terminal["tokens_seen"],
        "target_1800m_increment": 1800000000 - terminal["tokens_seen"],
        "physical_train_remaining": remaining,
        "dialogue_shortage_1800m": max(0, 40000000 - remaining["dialogue"]),
        "dialogue_shortage_through_2100m": max(0, 80000000 - remaining["dialogue"]),
        "new_admitted_usable_tokens": None,
        "candidate_file": str(retained_path), "candidate_sha256": digest(retained_path),
        "remaining_requirements": ["source-bound named clearance", "semantic quality review",
                                   "near-duplicate and evaluation leakage audit", "group-isolated split",
                                   "tokenize accepted split and recalculate capacity"],
        "limitations": ["normalized exact turn overlap is not a near-duplicate audit",
                        "MOSS is assistant-generated dialogue, not a verified natural spoken corpus",
                        "existing canonical and shared seen-ledger bytes are preserved"],
    }
    (output / "config.json").write_bytes(config_path.read_bytes())
    (output / "implementation.py.snapshot").write_bytes(Path(__file__).read_bytes())
    return result
