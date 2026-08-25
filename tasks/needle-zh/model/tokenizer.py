"""Frozen zh-24k-v1 SentencePiece wrapper (PAD=0 EOS=1 BOS=2 UNK=3)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from repo_paths import CORPUS_ZH_VOCAB, TOKENIZER_ZH_V1  # noqa: E402

MANIFEST_PATH = CORPUS_ZH_VOCAB / "tokenizer-v1-manifest.json"
USER_PREFIX = "<|im_start|>user\n"
ASSISTANT_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"
TURN_END = "<|im_end|>"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ZhTokenizerV1:
    def __init__(self, model_path: Path | None = None, *, require_manifest: bool = True):
        import sentencepiece as spm

        self.model_path = Path(model_path or TOKENIZER_ZH_V1)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"missing tokenizer {self.model_path}; train with train_zh_vocab_spm.py --freeze-v1"
            )
        self.model_sha256 = sha256_file(self.model_path)
        self.manifest: dict = {}
        if require_manifest:
            if not MANIFEST_PATH.is_file():
                raise FileNotFoundError(f"missing tokenizer manifest {MANIFEST_PATH}")
            self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            expected = str(self.manifest.get("model_sha256") or "")
            if expected and expected != self.model_sha256:
                raise ValueError(
                    f"tokenizer hash mismatch: file={self.model_sha256} manifest={expected}"
                )
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        self.pad_id = 0
        self.eos_id = 1
        self.bos_id = 2
        self.unk_id = 3
        n = int(self.sp.get_piece_size())
        if n != 24000:
            raise ValueError(f"expected 24000 pieces, got {n}")
        if (
            int(self.sp.pad_id()) != 0
            or int(self.sp.eos_id()) != 1
            or int(self.sp.bos_id()) != 2
            or int(self.sp.unk_id()) != 3
        ):
            raise ValueError("special ids must be PAD=0 EOS=1 BOS=2 UNK=3")

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.sp.encode(text, out_type=int))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i not in {self.pad_id, self.bos_id}]
        if ids and ids[-1] == self.eos_id:
            ids = ids[:-1]
        return self.sp.decode(ids)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def encode_document(self, text: str) -> list[int]:
        return self.encode(text, add_bos=True, add_eos=True)

    def encode_chat(self, user: str, assistant: str | None = None) -> dict[str, list[int]]:
        prompt = USER_PREFIX + user + ASSISTANT_PREFIX
        prompt_ids = self.encode(prompt, add_bos=True, add_eos=False)
        if assistant is None:
            return {"prompt_ids": prompt_ids, "ids": prompt_ids, "n_prompt": len(prompt_ids)}
        ans_ids = (
            self.encode(assistant, add_bos=False, add_eos=False)
            + self.encode(TURN_END, add_bos=False, add_eos=False)
            + [self.eos_id]
        )
        ids = prompt_ids + ans_ids
        return {
            "prompt_ids": prompt_ids,
            "answer_ids": ans_ids,
            "ids": ids,
            "n_prompt": len(prompt_ids),
        }
