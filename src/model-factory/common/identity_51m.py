"""Immutable 51M identity helpers. Fail closed on identity drift or base overwrites."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ["MEI_ARCHITECTURE_ID"] = "mei-1.0-51m-arch-v1"

_HERE = Path(__file__).resolve().parent

from common._repo import ROOT, TRAIN_RUNS, ensure_formal_on_path

_ARCH = ROOT / "src/architecture/mei-1.2-51m"
for path in (_ARCH, ROOT / "src/model-factory"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

ensure_formal_on_path()

ARCHITECTURE_ID = "mei-1.0-51m-arch-v1"
MODEL_ID = "mei-1.0-51m-base-scratch300m-v1"
RUN_NAME = "pretrain-mei-1.0-51m-base-scratch300m-v1"
EXPECTED_PARAMS = 51_463_797
ARCHITECTURE_DIR = ROOT / "src/architecture/mei-1.2-51m"
ARCHITECTURE_SPEC = ARCHITECTURE_DIR / "spec" / "model.json"
CYCLES_ROOT = ROOT / "cycles/mei-1.1-51m/exp-00300m"
BASE_DIR = CYCLES_ROOT / "models/base" / MODEL_ID
WEIGHTS_PATH = BASE_DIR / f"{MODEL_ID}.npz"
RELEASE_PATH = BASE_DIR / "RELEASE.json"
RUN_DIR = TRAIN_RUNS / RUN_NAME
SUMMARY_PATH = RUN_DIR / "summary.json"
RUN_PROBES_PATH = RUN_DIR / "probes.json"
JOBS_DIR = ROOT / "cycles/mei-1.1-51m/_legacy/notebook/evaluation/jobs/mei-1.0-51m"
PROBE_BANK = ROOT / "cycles/mei-1.1-51m/_legacy/notebook/evaluation/banks/needle-pretrain-probes-v0/probes-v0.jsonl"
FLOAT_ANCHOR_NAME = "float-base-lm-anchor.json"
PTQ_SCAN_NAME = "ptq-scan.json"
CANDIDATE_MAP_NAME = "q2q4-candidate-bit-map.json"
PACK_BUDGET_NAME = "pack-budget-scan.json"
Q4_BASELINE_MAP_NAME = "q4-baseline-bit-map.json"
PRODUCT_MIXED_MAP_NAME = "q2q4-product-candidate-bit-map.json"
Q4_PACKAGE_ID = "mei-1.0-51m-base-scratch300m-q4-v1"
Q4_PACKAGE_DIR = CYCLES_ROOT / "packages" / Q4_PACKAGE_ID
QAT_Q4_MODEL_ID = "mei-1.0-51m-base-scratch300m-qat-q4-v1"
QAT_Q4_BASE_DIR = CYCLES_ROOT / "models/qat" / QAT_Q4_MODEL_ID
QAT_Q4_WEIGHTS_PATH = QAT_Q4_BASE_DIR / f"{QAT_Q4_MODEL_ID}.npz"
QAT_Q4_RELEASE_PATH = QAT_Q4_BASE_DIR / "RELEASE.json"
QAT_Q4_PACKAGE_ID = "mei-1.0-51m-base-scratch300m-qat-q4-v1"
QAT_Q4_PACKAGE_DIR = CYCLES_ROOT / "packages" / QAT_Q4_PACKAGE_ID
QAT_CQ2_MODEL_ID = "mei-1.0-51m-base-scratch300m-qat-cq2-v1"
QAT_CQ2_BASE_DIR = CYCLES_ROOT / "models/qat" / QAT_CQ2_MODEL_ID
QAT_CQ2_WEIGHTS_PATH = QAT_CQ2_BASE_DIR / f"{QAT_CQ2_MODEL_ID}.npz"
QAT_CQ2_PACKAGE_ID = "mei-1.0-51m-base-scratch300m-qat-cq2-v1"
QAT_CQ2_PACKAGE_DIR = CYCLES_ROOT / "packages" / QAT_CQ2_PACKAGE_ID
SFT_QAT_MODEL_ID = "mei-1.0-51m-sft-qat-q4-v1"
SFT_QAT_BASE_DIR = CYCLES_ROOT / "models/product" / SFT_QAT_MODEL_ID
SFT_QAT_WEIGHTS_PATH = SFT_QAT_BASE_DIR / f"{SFT_QAT_MODEL_ID}.npz"
SFT_QAT_PACKAGE_DIR = CYCLES_ROOT / "packages" / SFT_QAT_MODEL_ID
PRODUCT_MIXED_FINAL_NAME = "q2q4-product-bit-map.json"
NOT_A_CLAIM = "LM loss/probes are not tool-calling ability."
QAT_MANDATORY_NOTE = "CQ2-first mixed Q2/Q4 QAT is mandatory for the on-device 51M core. PTQ is diagnostic only."


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def fail(message: str) -> int:
    print(message, file=sys.stderr)
    return 2


def assert_51m_architecture_id(architecture_id: str) -> str | None:
    if architecture_id != ARCHITECTURE_ID:
        return f"architecture_id mismatch: got={architecture_id!r} expected={ARCHITECTURE_ID!r}"
    return None


def qat_checkpoint_improved(previous: dict, score: float, *, product_ok: bool) -> bool:
    """Keep the product-passing checkpoint with the lowest pre-registered score."""
    if not product_ok:
        return False
    previous_score = previous.get("selection_score")
    return previous_score is None or score < float(previous_score)


def refuse_base_write(path: Path) -> str | None:
    """Block writes only into the frozen scratch300m-v1 directory.

    New QAT/SFT model_ids may live under base/ as siblings. Never overwrite MODEL_ID.
    """
    resolved = path.resolve()
    frozen = BASE_DIR.resolve()
    if resolved == frozen or resolved.is_relative_to(frozen):
        return f"refusing to write under immutable {MODEL_ID}: {resolved}"
    return None


def write_json(path: Path, payload: dict) -> str | None:
    blocked = refuse_base_write(path)
    if blocked:
        return blocked
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return None


def validate_release(release: dict, weights: Path) -> str | None:
    identity = assert_51m_architecture_id(str(release.get("architecture_id") or ""))
    if identity:
        return identity
    if release.get("model_id") != MODEL_ID:
        return f"model_id mismatch: got={release.get('model_id')!r} expected={MODEL_ID!r}"
    params = int(release.get("params") or 0)
    if params != EXPECTED_PARAMS:
        return f"params mismatch: got={params} expected={EXPECTED_PARAMS}"
    if not weights.is_file():
        return f"missing 51M weights: {weights}"
    got_hash = sha256_file(weights)
    expected = str(release.get("weights_sha256") or "")
    if not expected or got_hash != expected:
        return f"weights_sha256 mismatch: got={got_hash} expected={expected}"
    return None


def propose_candidate_bit_map(components: dict) -> dict:
    ranked = sorted(
        (
            (name, row)
            for name, row in components.items()
            if int((row or {}).get("n_tensors") or 0) > 0
        ),
        key=lambda item: float((item[1] or {}).get("max_abs_logit_delta") or 0.0),
        reverse=True,
    )
    keep_q4 = max(1, (len(ranked) + 1) // 2)
    bits: dict[str, int] = {}
    for index, (name, _row) in enumerate(ranked):
        bits[name] = 4 if index < keep_q4 else 2
    if "embedding" in bits:
        bits["embedding"] = 4
    return {
        "architecture_id": ARCHITECTURE_ID,
        "model_id": MODEL_ID,
        "kind": "q2q4-candidate-bit-map",
        "candidate": True,
        "product_final": False,
        "qat_mandatory": True,
        "first_method": "4bit_ptq_diagnostic",
        "bits": bits,
        "ranking": [name for name, _row in ranked],
        "rule": (
            "Top-half logit-delta components stay 4-bit; the rest are proposed 2-bit. "
            "Tied embedding always stays 4-bit. This is not a product mixed-bit map."
        ),
        "note": QAT_MANDATORY_NOTE,
        "not_a_claim": "PTQ sensitivity is not QAT, packed-kernel, or tool-calling ability.",
    }
