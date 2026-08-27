#!/usr/bin/env python3
"""Hard cutover: corpora/tasks/scripts/eval → tokenizer/corpus/base/sft/notebook.

Same-disk rename only. Do not import repo_paths (old layout).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MIG = Path(__file__).resolve().parent
NEW_REPO_PATHS = MIG / "repo_paths.py"

TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".sh", ".txt", ".gitignore", ".yml", ".yaml"}
SKIP_REWRITE_PARTS = (
    "/.venv/",
    "/__pycache__/",
    "/notebook/_migration/260826-root-cutover/",
    "/.git/",
)

DIR_MOVES: list[tuple[str, str]] = [
    ("scripts", "notebook/_tooling/scripts"),
    ("skills", "notebook/_tooling/skills"),
    ("eval", "notebook/evaluation"),
    ("tasks/needle-zh/model", "notebook/_tooling/model/mei-1.0-58m"),
    ("tasks/needle-zh/spec", "notebook/base/pretrain-v1/spec"),
    ("tasks/needle-zh/recipes", "notebook/sft/mei-1.0-58m/recipes"),
    ("tasks/needle-zh/train", "notebook/sft/mei-1.0-58m/train"),
    ("tasks/needle-zh/eval", "notebook/evaluation/jobs/mei-1.0-58m"),
    ("tasks/needle-zh/checkpoints", "notebook/archive/base/mei-1.0-58m-checkpoints"),
    ("tasks/mei-expert-qwen35-0p8b", "notebook/archive/legacy-products/mei-expert-qwen35-0p8b"),
    ("experiments/runs", "notebook/archive/runs"),
    ("corpora/zh-pretrain-v0", "notebook/corpus/lm-v1/language/outbox/accepted/zh-pretrain-v0"),
    ("corpora/zh-pretrain-v1", "notebook/corpus/lm-v1/language/outbox/accepted/zh-pretrain-v1"),
    ("corpora/zh-pretrain-v3", "notebook/corpus/lm-v1/structure/outbox/accepted/zh-pretrain-v3"),
    ("corpora/zh-pretrain-v4", "notebook/corpus/lm-v1/assemble/work/zh-pretrain-v4"),
    ("corpora/zh-pretrain-colloquial-synth-pooled-v1", "notebook/corpus/lm-v1/colloquial/outbox/draft/colloquial-v1"),
    ("corpora/zh-pretrain-colloquial-synth-qwen-v1", "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen-plus"),
    ("corpora/zh-pretrain-colloquial-synth-dsflash-v1", "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/dsflash"),
    ("corpora/zh-pretrain-colloquial-synth-qwen36plus-v1", "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen36"),
    ("corpora/zh-pretrain-colloquial-synth-qwen37plus-v1", "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen37"),
    ("corpora/zh-pretrain-colloquial-synth-glm52-v1", "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/glm52"),
    ("corpora/zh-pretrain-colloquial-synth-kimi-k3-v1", "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/kimi"),
    ("corpora/zh-pretrain-v2", "notebook/archive/corpus/zh-pretrain-v2"),
    ("corpora/zh-pretrain-colloquial-synth-v1", "notebook/archive/corpus/zh-pretrain-colloquial-synth-v1"),
    ("corpora/zh-pretrain-colloquial-synth-v2", "notebook/archive/corpus/zh-pretrain-colloquial-synth-v2"),
    ("corpora/zh-pretrain-colloquial-synth-smoke", "notebook/archive/corpus/zh-pretrain-colloquial-synth-smoke"),
    ("corpora/zh-pretrain-colloquial-synth-qwen-smoke", "notebook/archive/corpus/zh-pretrain-colloquial-synth-qwen-smoke"),
    ("corpora/zh-pretrain-colloquial-synth-ollama-bakeoff", "notebook/archive/corpus/zh-pretrain-colloquial-synth-ollama-bakeoff"),
    ("corpora/_probe", "notebook/archive/corpus/_probe"),
    ("corpora/sft-style-v0", "notebook/sft/style-v0"),
    ("corpora/shared-tool-traces-v0", "notebook/sft/shared-tool-traces-v0"),
]

FILE_MOVES: list[tuple[str, str]] = [
    ("tasks/needle-zh/README.md", "notebook/base/pretrain-v1/README.md"),
    ("tasks/needle-zh/DESIGN.md", "notebook/base/pretrain-v1/DESIGN.md"),
    ("requirements.txt", "notebook/_tooling/requirements/requirements.txt"),
    ("requirements-corpus.txt", "notebook/_tooling/requirements/requirements-corpus.txt"),
]

TOKENIZER_FILES = [
    "zh-24k-v1.model",
    "zh-24k-v1.vocab",
    "tokenizer-v1-manifest.json",
    "coverage-v1.json",
    "vocab.txt",
]

ARCHIVE_PACK_NAMES = [
    "home-sft-10k.candidates.jsonl",
    "home-sft-10k.review-sample.jsonl",
    "home-sft-2k.candidates.jsonl",
    "home-sft-2k.review-sample.jsonl",
    "mei-tool-route-sft-v1-10k.jsonl",
    "mei-tool-route-sft-v1-2k.jsonl",
    "mei-tool-route-sft-v1-valid.jsonl",
    "mei-tool-route-sft-v1.INVALID_FOR_V2_PUBLISH.json",
    "mei-tool-sft-v1-10k.jsonl",
    "mei-tool-sft-v1-2k.jsonl",
    "mei-tool-sft-v1.INVALID_FOR_PUBLISH.json",
    "mw-sft-v0-2k.review-sample.jsonl",
]

# Longest-first string replacements for live path literals.
PATH_REPLACEMENTS: list[tuple[str, str]] = [
    ("corpora/zh-vocab-v0/zh-24k-v1.model", "tokenizer/zh-24k-v1/zh-24k-v1.model"),
    ("corpora/zh-vocab-v0/zh-24k-v1.vocab", "tokenizer/zh-24k-v1/zh-24k-v1.vocab"),
    ("corpora/zh-vocab-v0/tokenizer-v1-manifest.json", "tokenizer/zh-24k-v1/tokenizer-v1-manifest.json"),
    ("corpora/zh-vocab-v0/coverage-v1.json", "tokenizer/zh-24k-v1/coverage-v1.json"),
    ("corpora/zh-vocab-v0/vocab.txt", "tokenizer/zh-24k-v1/vocab.txt"),
    ("corpora/zh-vocab-v0", "notebook/corpus/tokenizer-v1/work/zh-vocab-v0"),
    (
        "corpora/zh-pretrain-colloquial-synth-pooled-v1",
        "notebook/corpus/lm-v1/colloquial/outbox/draft/colloquial-v1",
    ),
    (
        "corpora/zh-pretrain-colloquial-synth-qwen36plus-v1",
        "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen36",
    ),
    (
        "corpora/zh-pretrain-colloquial-synth-qwen37plus-v1",
        "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen37",
    ),
    (
        "corpora/zh-pretrain-colloquial-synth-qwen-v1",
        "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen-plus",
    ),
    (
        "corpora/zh-pretrain-colloquial-synth-dsflash-v1",
        "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/dsflash",
    ),
    (
        "corpora/zh-pretrain-colloquial-synth-glm52-v1",
        "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/glm52",
    ),
    (
        "corpora/zh-pretrain-colloquial-synth-kimi-k3-v1",
        "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/kimi",
    ),
    (
        "corpora/zh-pretrain-colloquial-synth-qwen-smoke",
        "notebook/archive/corpus/zh-pretrain-colloquial-synth-qwen-smoke",
    ),
    (
        "corpora/zh-pretrain-colloquial-synth-ollama-bakeoff",
        "notebook/archive/corpus/zh-pretrain-colloquial-synth-ollama-bakeoff",
    ),
    (
        "corpora/zh-pretrain-colloquial-synth-smoke",
        "notebook/archive/corpus/zh-pretrain-colloquial-synth-smoke",
    ),
    (
        "corpora/zh-pretrain-colloquial-synth-v1",
        "notebook/archive/corpus/zh-pretrain-colloquial-synth-v1",
    ),
    (
        "corpora/zh-pretrain-colloquial-synth-v2",
        "notebook/archive/corpus/zh-pretrain-colloquial-synth-v2",
    ),
    ("corpora/zh-pretrain-v0", "notebook/corpus/lm-v1/language/outbox/accepted/zh-pretrain-v0"),
    ("corpora/zh-pretrain-v1", "notebook/corpus/lm-v1/language/outbox/accepted/zh-pretrain-v1"),
    ("corpora/zh-pretrain-v2", "notebook/archive/corpus/zh-pretrain-v2"),
    ("corpora/zh-pretrain-v3", "notebook/corpus/lm-v1/structure/outbox/accepted/zh-pretrain-v3"),
    ("corpora/zh-pretrain-v4", "notebook/corpus/lm-v1/assemble/work/zh-pretrain-v4"),
    ("corpora/sft-style-v0", "notebook/sft/style-v0"),
    ("corpora/shared-tool-traces-v0", "notebook/sft/shared-tool-traces-v0"),
    ("corpora/_probe", "notebook/archive/corpus/_probe"),
    ("tasks/needle-zh/checkpoints", "notebook/archive/base/mei-1.0-58m-checkpoints"),
    ("tasks/needle-zh/model", "notebook/_tooling/model/mei-1.0-58m"),
    ("tasks/needle-zh/spec", "notebook/base/pretrain-v1/spec"),
    ("tasks/needle-zh/recipes", "notebook/sft/mei-1.0-58m/recipes"),
    ("tasks/needle-zh/train", "notebook/sft/mei-1.0-58m/train"),
    ("tasks/needle-zh/eval", "notebook/evaluation/jobs/mei-1.0-58m"),
    ("tasks/mei-expert-qwen35-0p8b", "notebook/archive/legacy-products/mei-expert-qwen35-0p8b"),
    ("tasks/needle-zh", "notebook/base/pretrain-v1"),
    ("eval/banks", "notebook/evaluation/banks"),
    ("eval/shared", "notebook/evaluation/shared"),
    ("eval/playground", "notebook/evaluation/playground"),
    ("experiments/runs", "notebook/archive/runs"),
    ("python3 scripts/", "python3 notebook/_tooling/scripts/"),
    ("python3 tasks/needle-zh/model/", "python3 notebook/_tooling/model/mei-1.0-58m/"),
    ('ROOT / "scripts"', "SCRIPTS_ROOT"),
    ('ROOT / "eval"', "EVAL_ROOT"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def inventory_entry(src: Path) -> dict:
    if not src.exists():
        return {"path": rel(src) if src.exists() else src.as_posix(), "missing": True}
    if src.is_file():
        return {
            "path": rel(src),
            "kind": "file",
            "bytes": src.stat().st_size,
            "sha256": sha256_file(src) if src.stat().st_size <= 8_000_000 else None,
        }
    n_files = 0
    n_bytes = 0
    for p in src.rglob("*"):
        if p.is_file():
            n_files += 1
            try:
                n_bytes += p.stat().st_size
            except OSError:
                pass
    return {"path": rel(src), "kind": "dir", "n_files": n_files, "bytes": n_bytes}


def rename(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        raise FileExistsError(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.rename(src, dst)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rewrite_text(text: str) -> str:
    for old, new in PATH_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def should_rewrite(path: Path) -> bool:
    posix = path.as_posix()
    if any(part in posix for part in SKIP_REWRITE_PARTS):
        return False
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
        return False
    return True


def patch_python_imports(text: str) -> str:
    """Ensure MODEL_MEI_58M / SCRIPTS_ROOT are imported when used after rewrite."""
    if "from repo_paths import" in text:
        needed = []
        if "MODEL_MEI_58M" in text and "MODEL_MEI_58M" not in text.split("from repo_paths import", 1)[1].split("\n", 1)[0]:
            needed.append("MODEL_MEI_58M")
        if "SCRIPTS_ROOT" in text and "SCRIPTS_ROOT" not in text.split("from repo_paths import", 1)[1].split("\n", 1)[0]:
            needed.append("SCRIPTS_ROOT")
        if "EVAL_ROOT" in text and "EVAL_ROOT" not in text.split("from repo_paths import", 1)[1].split("\n", 1)[0]:
            needed.append("EVAL_ROOT")
        if needed:

            def add_imports(m: re.Match[str]) -> str:
                body = m.group(1)
                extras = [n for n in needed if n not in body]
                if not extras:
                    return m.group(0)
                if body.rstrip().endswith("(") or "\n" in body:
                    return m.group(0)
                return f"from repo_paths import {body}, {', '.join(extras)}"

            text = re.sub(r"from repo_paths import ([^\n]+)", add_imports, text, count=1)
    text = text.replace(
        'sys.path.insert(0, str(ROOT / "notebook/_tooling/scripts"))',
        "sys.path.insert(0, str(SCRIPTS_ROOT))",
    )
    text = text.replace(
        'sys.path.insert(0, str(ROOT / "notebook/_tooling/model/mei-1.0-58m"))',
        "sys.path.insert(0, str(MODEL_MEI_58M))",
    )
    text = text.replace('ROOT / "notebook/_tooling/scripts"', "SCRIPTS_ROOT")
    return text


def patch_test_roots(text: str, path: Path) -> str:
    if path.name.startswith("test_") and 'Path(__file__).resolve().parents[1]' in text:
        text = text.replace(
            "ROOT = Path(__file__).resolve().parents[1]\n"
            'sys.path.insert(0, str(ROOT / "scripts"))\n',
            "from repo_paths import MODEL_MEI_58M, ROOT, SCRIPTS_ROOT\n"
            "sys.path.insert(0, str(SCRIPTS_ROOT))\n",
        )
        # After path rewrite, scripts path may already be new:
        text = text.replace(
            "ROOT = Path(__file__).resolve().parents[1]\n",
            "from repo_paths import MODEL_MEI_58M, ROOT, SCRIPTS_ROOT\n",
        )
        text = text.replace('sys.path.insert(0, str(ROOT / "scripts"))\n', "sys.path.insert(0, str(SCRIPTS_ROOT))\n")
        text = text.replace(
            'sys.path.insert(0, str(ROOT / "notebook/_tooling/scripts"))\n',
            "sys.path.insert(0, str(SCRIPTS_ROOT))\n",
        )
        text = text.replace(
            'sys.path.insert(0, str(ROOT / "notebook/_tooling/model/mei-1.0-58m"))\n',
            "sys.path.insert(0, str(MODEL_MEI_58M))\n",
        )
        text = text.replace(
            'sys.path.insert(0, str(ROOT / "scripts" / "jobs"))\n',
            "sys.path.insert(0, str(SCRIPTS_ROOT / \"jobs\"))\n",
        )
        # Avoid duplicate imports
        lines = text.splitlines(keepends=True)
        seen_import = False
        out = []
        for line in lines:
            if line.startswith("from repo_paths import"):
                if seen_import:
                    continue
                seen_import = True
            out.append(line)
        text = "".join(out)
    return text


def patch_model_file(path: Path, text: str) -> str:
    old_scripts = 'Path(__file__).resolve().parents[3] / "scripts"'
    new_scripts = 'Path(__file__).resolve().parents[2] / "scripts"'
    if old_scripts in text:
        text = text.replace(old_scripts, new_scripts)
    if path.name == "config.py" and 'parents[1] / "spec"' in text:
        bootstrap = '''
def _spec_dir() -> Path:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from repo_paths import SPEC_NEEDLE_ZH
    return SPEC_NEEDLE_ZH
'''
        if "import sys" not in text:
            text = text.replace("from pathlib import Path\n", "import sys\nfrom pathlib import Path\n")
        if "_spec_dir" not in text:
            text = text.replace(
                "@classmethod\n    def from_spec(cls, path: Path | None = None) -> NeedleZhConfig:\n        if path is None:\n            path = Path(__file__).resolve().parents[1] / \"spec\" / \"model.json\"\n",
                bootstrap + "\n    @classmethod\n    def from_spec(cls, path: Path | None = None) -> NeedleZhConfig:\n        if path is None:\n            path = _spec_dir() / \"model.json\"\n",
            )
            text = text.replace(
                'path = Path(__file__).resolve().parents[1] / "spec" / "model-target-v2.json"',
                'path = _spec_dir() / "model-target-v2.json"',
            )
            text = text.replace(
                'cfg = cls.from_spec(Path(__file__).resolve().parents[1] / "spec" / "model.json")',
                'cfg = cls.from_spec(_spec_dir() / "model.json")',
            )
    if path.name == "candidates.py":
        text = text.replace(
            'path = Path(__file__).resolve().parents[3] / "eval" / "shared" / "entities" / "mei-grounded-v1.json"',
            'from repo_paths import EVAL_SHARED_ROOT\n        path = EVAL_SHARED_ROOT / "entities" / "mei-grounded-v1.json"',
        )
        text = text.replace(
            'path = Path(__file__).resolve().parents[3] / "eval" / "shared" / "lexicon" / "mei-tool-intents-v1.json"',
            'from repo_paths import EVAL_SHARED_ROOT\n        path = EVAL_SHARED_ROOT / "lexicon" / "mei-tool-intents-v1.json"',
        )
        if "parents[2] / \"scripts\"" not in text and "from repo_paths import EVAL_SHARED_ROOT" in text:
            insert = (
                "import sys\n"
                "_SCRIPTS = Path(__file__).resolve().parents[2] / \"scripts\"\n"
                "if str(_SCRIPTS) not in sys.path:\n"
                "    sys.path.insert(0, str(_SCRIPTS))\n"
            )
            if "import sys" not in text:
                text = text.replace("from pathlib import Path\n", insert + "from pathlib import Path\n")
            else:
                text = insert + text
    if path.name == "check_student.py":
        text = text.replace("ROOT = HERE.parents[2]\n", "ROOT = HERE.parents[3]\n")
        text = text.replace(
            'sys.path.insert(0, str(ROOT / "scripts"))\n',
            'sys.path.insert(0, str(HERE.parents[2] / "scripts"))\n',
        )
    if path.name == "tokenizer.py":
        text = text.replace(
            "from repo_paths import CORPUS_ZH_VOCAB, TOKENIZER_ZH_V1  # noqa: E402\n\n"
            "MANIFEST_PATH = CORPUS_ZH_VOCAB / \"tokenizer-v1-manifest.json\"\n",
            "from repo_paths import TOKENIZER_DIR, TOKENIZER_ZH_V1  # noqa: E402\n\n"
            "MANIFEST_PATH = TOKENIZER_DIR / \"tokenizer-v1-manifest.json\"\n",
        )
    return text


def rehash_sidecar(dir_path: Path) -> None:
    hashes_file = dir_path / "hashes.json"
    if not hashes_file.is_file():
        return
    data = json.loads(hashes_file.read_text(encoding="utf-8"))
    changed = False
    if not isinstance(data, dict):
        return
    for key, old in list(data.items()):
        target = dir_path / key
        if target.is_file():
            digest = sha256_file(target)
            if digest != old:
                data[key] = digest
                changed = True
    if changed:
        hashes_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_current_and_readmes() -> None:
    write_json(
        ROOT / "CURRENT.json",
        {
            "tokenizer": "tokenizer/zh-24k-v1",
            "corpus": None,
            "base": None,
            "sft": None,
            "stage": "corpus-preparation",
            "blocked": ["colloquial-contract-not-promoted"],
            "product": "mei-1.0-58m",
        },
    )
    (ROOT / "corpus").mkdir(parents=True, exist_ok=True)
    (ROOT / "base").mkdir(parents=True, exist_ok=True)
    (ROOT / "sft").mkdir(parents=True, exist_ok=True)
    (ROOT / "tokenizer" / "zh-24k-v1").mkdir(parents=True, exist_ok=True)
    (ROOT / "corpus" / "README.md").write_text(
        "# corpus/\n\nPublished train mix only. Empty until a corpus is promoted into CURRENT.json.\n"
        "Process shards live under `notebook/corpus/`.\n",
        encoding="utf-8",
    )
    (ROOT / "base" / "README.md").write_text(
        "# base/\n\nPublished base checkpoints (random-init pretrain and later CPT). Empty until promote.\n"
        "Training process lives under `notebook/base/`.\n",
        encoding="utf-8",
    )
    (ROOT / "sft" / "README.md").write_text(
        "# sft/\n\nPublished SFT models only. Empty until promote.\n"
        "Packs and recipes live under `notebook/sft/`.\n",
        encoding="utf-8",
    )
    tok_readme = ROOT / "tokenizer" / "zh-24k-v1" / "README.md"
    tok_readme.write_text(
        "# zh-24k-v1\n\nFrozen SentencePiece tokenizer. PAD/EOS/BOS/UNK = 0/1/2/3.\n"
        "Process dumps and draft vocabs: `notebook/corpus/tokenizer-v1/work/zh-vocab-v0/`.\n",
        encoding="utf-8",
    )
    (ROOT / "requirements.txt").write_text(
        "-r notebook/_tooling/requirements/requirements.txt\n",
        encoding="utf-8",
    )
    (ROOT / "requirements-corpus.txt").write_text(
        "-r notebook/_tooling/requirements/requirements-corpus.txt\n",
        encoding="utf-8",
    )


def write_product_json() -> None:
    payload = {
        "version": 1,
        "product": "mei-1.0-58m",
        "historical_alias": "needle-zh",
        "shared": {
            "tokenizer": "tokenizer/zh-24k-v1",
            "eval_banks_root": "notebook/evaluation/banks",
            "eval_shared_root": "notebook/evaluation/shared",
            "runs_root": "notebook/archive/runs",
        },
        "tasks": [
            {
                "id": "mei-1.0-58m",
                "canonical_id": "mei-1.0-58m",
                "aliases": ["needle-zh", "mei-1.0-58m"],
                "title": "MEI 1.0 58M",
                "kind": "from-scratch-router",
                "spec": "notebook/base/pretrain-v1/spec",
                "model": "notebook/_tooling/model/mei-1.0-58m",
                "sft_mixture": "notebook/sft/mei-1.0-58m/recipes/sft-mixture-v1.json",
                "train_seed": "notebook/sft/mei-1.0-58m/train/seed/sft-smoke-v0.jsonl",
                "train_seeds": [
                    "notebook/sft/mei-1.0-58m/train/packs/home-sft-2k.jsonl",
                    "notebook/sft/mei-1.0-58m/train/packs/home-sft-10k.jsonl",
                    "notebook/sft/mei-1.0-58m/train/packs/mw-sft-v0-2k.jsonl",
                    "notebook/sft/mei-1.0-58m/train/packs/mei-toolcall-v2-smoke.jsonl",
                ],
                "eval_banks": [
                    "notebook/evaluation/banks/needle-toolcall-v0/eval-bank-v0.jsonl",
                    "notebook/evaluation/banks/needle-vrm-agent-v0/eval-bank-v0.jsonl",
                    "notebook/evaluation/banks/needle-vrm-agent-v0/eval-bank-v2.jsonl",
                    "notebook/evaluation/banks/mei-tool-schema-v1/eval-bank-v0.jsonl",
                ],
            }
        ],
    }
    write_json(ROOT / "notebook/base/pretrain-v1/product.json", payload)


def write_job_card() -> None:
    card = {
        "job_id": "260826-01-synthesize",
        "topic": "colloquial",
        "kind": "pack",
        "status": "draft",
        "product": "mei-1.0-58m",
        "work_dir": "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work",
        "publish_path": "notebook/corpus/lm-v1/colloquial/outbox/draft/colloquial-v1",
        "formal_cpt_eligible": False,
        "parents": [
            "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen-plus",
            "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/dsflash",
            "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen36",
            "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/qwen37",
            "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/glm52",
            "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/work/lanes/kimi",
        ],
        "note": "Pooled unique 30M mixed-generator pack. Not admitted to formal CPT.",
    }
    write_json(ROOT / "notebook/corpus/lm-v1/colloquial/jobs/260826-01-synthesize/job.json", card)


GITIGNORE = """__pycache__/
*.py[cod]
.DS_Store
.venv/
.env
*.egg-info/

# Published tokenizer: keep v1 model + sidecar
!tokenizer/zh-24k-v1/zh-24k-v1.model
!tokenizer/zh-24k-v1/zh-24k-v1.vocab
!tokenizer/zh-24k-v1/*.json
!tokenizer/zh-24k-v1/*.md
!tokenizer/zh-24k-v1/vocab.txt

# Published corpus/base/sft stay empty until promote
corpus/**
!corpus/README.md
base/**
!base/README.md
sft/**
!sft/README.md

# Notebook process artifacts
notebook/**/*.parquet
notebook/**/*.bin
notebook/**/*.model
notebook/**/*.vocab
notebook/**/*.safetensors
notebook/**/*.npz
notebook/**/*.sqlite
notebook/**/*.xml*
notebook/**/*.bz2
notebook/**/*.gz
notebook/**/*.zst
notebook/**/*.7z
notebook/**/state/
notebook/**/shards/
notebook/**/raw/
notebook/**/dumps/
notebook/**/tokens/
notebook/**/hashes/
notebook/**/work/
notebook/archive/runs/**
notebook/archive/base/**
!notebook/archive/base/**/registry/
!notebook/archive/base/**/registry/**
notebook/**/*.jsonl
!notebook/evaluation/banks/**/*.jsonl
!notebook/evaluation/banks/**/*.json
!notebook/evaluation/banks/**/README.md
!notebook/evaluation/shared/**
!notebook/_tooling/scripts/**
!notebook/_tooling/model/**
!notebook/_tooling/skills/**
!notebook/_tooling/requirements/**
!notebook/_tooling/model/**/*.py
!notebook/base/pretrain-v1/spec/**
!notebook/base/pretrain-v1/*.md
!notebook/base/pretrain-v1/*.json
!notebook/sft/**/*.md
!notebook/sft/**/*.json
!notebook/corpus/**/*.md
!notebook/corpus/**/*.json
!notebook/_migration/**
!notebook/**/README.md
!notebook/**/job.json
!notebook/**/receipt.json
!notebook/**/seeds/**

# Job scratch
notebook/**/_/
"""


DESIGN = """# mei-llm 布局

仓根只放**已发布资产**。过程、评测、脚本、归档一律进 `notebook/`。

```text
CURRENT.json            # 现行指针；未升格则为 null
tokenizer/zh-24k-v1/    # 冻结词表
corpus/                 # 正式训练 mix（未升格则空）
base/                   # 正式 base（pretrain 与 CPT 都发布到这里）
sft/                    # 正式 SFT 模型
notebook/
  corpus/               # 语料过程：accepted / draft / jobs / assemble
  base/pretrain-v1/     # 随机初始化训练规格
  sft/                  # SFT pack / recipe 过程面
  evaluation/           # banks / shared / jobs
  _tooling/             # scripts / model / skills / requirements
  archive/              # 脏语料、旧 checkpoint、legacy 产品、runs
```

约定：

1. **Pretrain** = 随机初始化；**CPT** = 从已有 base 续训。二者发布目标都是 `base/`，不是 `cpt/`。
2. `CURRENT.json` 是人读入口。禁止再维护根目录 `corpora/index.json` 或 `tasks/` 双入口。
3. 口语合成六车道与 pooled 包在 `notebook/corpus/lm-v1/colloquial/`；未满足合同不得写入 `corpus/` 或 `CURRENT.corpus`。
4. 隔离门禁：`notebook/_tooling/scripts/check_train_eval_isolation.py --scope cpt-v2|sft-v2`。
5. 现行产品只登记 `mei-1.0-58m`。0.8B 专家线在 `notebook/archive/legacy-products/`。
"""


README = """# mei-llm

MEI 1.0 58M 及其训练、评测工作线的代码根（独立 Git 仓）。本仓不是在线模型 provider SDK。

现行指针见 `CURRENT.json`。布局合同见 `DESIGN.md`。

## 目录

```text
CURRENT.json
tokenizer/zh-24k-v1/     # 冻结词表
corpus/                  # 正式训练 mix（当前为空）
base/                    # 正式 base（当前为空）
sft/                     # 正式 SFT 模型（当前为空）
notebook/_tooling/       # scripts / model / skills
notebook/corpus/         # 语料过程与 draft
notebook/evaluation/     # 评测题库
notebook/archive/        # 不进 CURRENT 的历史资产
```

当前阶段：`corpus-preparation`。口语合同未升格，`CURRENT.corpus` 为 null。

## 常用命令

```bash
cd mei-llm

python3 notebook/_tooling/scripts/check_train_eval_isolation.py --all
python3 notebook/_tooling/scripts/check_train_eval_isolation.py --scope cpt-v2
python3 notebook/_tooling/scripts/check_train_eval_isolation.py --scope sft-v2

python3 notebook/_tooling/model/mei-1.0-58m/check_student.py
python3 notebook/_tooling/scripts/test_mei_route_runtime.py
python3 notebook/_tooling/scripts/test_mei_v2_runtime.py
python3 notebook/_tooling/scripts/test_colloquial_synth_pipeline.py
```

依赖见 `requirements.txt`（转发到 `notebook/_tooling/requirements/`）。
"""


def write_gitignore_docs() -> None:
    (ROOT / ".gitignore").write_text(GITIGNORE, encoding="utf-8")
    (ROOT / "DESIGN.md").write_text(DESIGN, encoding="utf-8")
    (ROOT / "README.md").write_text(README, encoding="utf-8")


def empty_or_remove(path: Path, leftover_ok: tuple[str, ...] = (".DS_Store",)) -> list[str]:
    if not path.exists():
        return []
    leftover = []
    for p in path.rglob("*"):
        if p.is_file() and p.name not in leftover_ok:
            leftover.append(rel(p))
    if leftover:
        return leftover
    shutil.rmtree(path)
    return []


def main() -> int:
    if not (ROOT / "DESIGN.md").is_file():
        print("not mei-llm root", ROOT, file=sys.stderr)
        return 2

    inv = {
        "root": str(ROOT),
        "tokenizer_expected_sha256": "fcd07b3d49f5174bb60e81996f4d3f2d55f458f5b8420a271aea59ac5dc58629",
        "moves": [],
    }

    # --- tokenizer split ---
    vocab_src = ROOT / "corpora" / "zh-vocab-v0"
    tok_dst = ROOT / "tokenizer" / "zh-24k-v1"
    tok_dst.mkdir(parents=True, exist_ok=True)
    for name in TOKENIZER_FILES:
        src = vocab_src / name
        dst = tok_dst / name
        inv["moves"].append({"src": inventory_entry(src), "dst": rel(dst)})
        if src.exists() and not dst.exists():
            rename(src, dst)

    vocab_work = ROOT / "notebook/corpus/tokenizer-v1/work/zh-vocab-v0"
    if vocab_src.exists() and not vocab_work.exists():
        inv["moves"].append({"src": inventory_entry(vocab_src), "dst": rel(vocab_work)})
        rename(vocab_src, vocab_work)

    # --- file moves ---
    for src_rel, dst_rel in FILE_MOVES:
        src, dst = ROOT / src_rel, ROOT / dst_rel
        if src.exists() and not dst.exists():
            inv["moves"].append({"src": inventory_entry(src), "dst": dst_rel})
            rename(src, dst)

    # --- dir moves ---
    for src_rel, dst_rel in DIR_MOVES:
        src, dst = ROOT / src_rel, ROOT / dst_rel
        if not src.exists():
            inv["moves"].append({"src": src_rel, "missing": True, "dst": dst_rel})
            continue
        if dst.exists():
            raise FileExistsError(dst)
        inv["moves"].append({"src": inventory_entry(src), "dst": dst_rel})
        print(f"mv {src_rel} -> {dst_rel}", flush=True)
        rename(src, dst)

    # --- SFT invalid/candidates to archive ---
    packs = ROOT / "notebook/sft/mei-1.0-58m/train/packs"
    arch_packs = ROOT / "notebook/archive/sft/mei-1.0-58m/packs"
    arch_packs.mkdir(parents=True, exist_ok=True)
    for name in ARCHIVE_PACK_NAMES:
        src = packs / name
        if src.exists():
            rename(src, arch_packs / name)
    seed = ROOT / "notebook/sft/mei-1.0-58m/train/seed"
    arch_seed = ROOT / "notebook/archive/sft/mei-1.0-58m/seed"
    arch_seed.mkdir(parents=True, exist_ok=True)
    for name in ("sft-phase1-v0.jsonl", "sft-phase1-v0.LEGACY_DIAGNOSTIC_ONLY.json"):
        src = seed / name
        if src.exists():
            rename(src, arch_seed / name)

    # --- install new repo_paths ---
    dest_repo_paths = ROOT / "notebook/_tooling/scripts/repo_paths.py"
    shutil.copy2(NEW_REPO_PATHS, dest_repo_paths)

    # --- copy colloquial inbox then archive old jobs overlay ---
    old_jobs = ROOT / "jobs"
    new_inbox = ROOT / "notebook/corpus/lm-v1/colloquial/inbox"
    if (old_jobs / "colloquial-cpt/inbox").is_dir():
        if new_inbox.exists():
            shutil.rmtree(new_inbox)
        shutil.copytree(old_jobs / "colloquial-cpt/inbox", new_inbox)
    overlay = ROOT / "notebook/_migration/260826-jobs-overlay"
    if old_jobs.exists() and not overlay.exists():
        rename(old_jobs, overlay)

    write_job_card()
    write_product_json()
    write_current_and_readmes()
    write_gitignore_docs()

    # --- rewrite live text files ---
    n_rewrite = 0
    for path in ROOT.rglob("*"):
        if not path.is_file() or not should_rewrite(path):
            continue
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        new = rewrite_text(text)
        new = patch_python_imports(new)
        new = patch_test_roots(new, path)
        if "notebook/_tooling/model/mei-1.0-58m" in path.as_posix():
            new = patch_model_file(path, new)
        if new != text:
            path.write_text(new, encoding="utf-8")
            n_rewrite += 1
    print(f"rewrote {n_rewrite} text files", flush=True)

    # --- rehash JSON sidecars whose contents include paths ---
    for hashes in ROOT.rglob("hashes.json"):
        if hashes.is_file():
            rehash_sidecar(hashes.parent)

    # --- delete stubs / empty old roots ---
    leftovers = {}
    for name in ("train", "mlx", "data", "corpora", "tasks", "eval", "scripts", "skills", "experiments", "jobs"):
        leftovers[name] = empty_or_remove(ROOT / name)

    # --- verify tokenizer ---
    tok = ROOT / "tokenizer/zh-24k-v1/zh-24k-v1.model"
    got = sha256_file(tok)
    expected = inv["tokenizer_expected_sha256"]
    if got != expected:
        raise SystemExit(f"tokenizer hash mismatch: {got} != {expected}")

    inv["rewritten_files"] = n_rewrite
    inv["tokenizer_sha256"] = got
    inv["leftovers"] = leftovers
    write_json(MIG / "inventory.json", inv)
    write_json(MIG / "moves.json", {"dir_moves": DIR_MOVES, "file_moves": FILE_MOVES})
    (MIG / "README.md").write_text(
        "# 260826 root cutover\n\nHard cutover of mei-llm to tokenizer/corpus/base/sft/notebook.\n"
        "See inventory.json. CURRENT.json must not list draft or archive.\n",
        encoding="utf-8",
    )
    print("cutover physical moves complete", flush=True)
    print("leftovers", json.dumps(leftovers, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
