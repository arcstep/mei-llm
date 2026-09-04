"""Frozen zh-24k-v1 SentencePiece wrapper shared by 51m training.

V1 stays untouched; ZhTokenizerV2 is the parameterized successor that the
frozen-tokenizer pointer (TOKENIZER.json) selects when the v2 vocabulary is
trained and frozen.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

MODEL_ROOT = Path(__file__).resolve().parent.parent
TOKENIZER_DIR = MODEL_ROOT / "tokenizer"
TOKENIZER_ZH_V1 = TOKENIZER_DIR / "zh-24k-v1.model"

MANIFEST_PATH = TOKENIZER_DIR / "tokenizer-v1-manifest.json"
USER_PREFIX = "<|im_start|>user\n"
ASSISTANT_PREFIX = "<|im_end|>\n<|im_start|>assistant\n"
TURN_END = "<|im_end|>"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ZhTokenizerV1:
    def __init__(self, model_path: Path | None = None, *, require_manifest: bool = True):
        import sentencepiece as spm

        self.model_path = Path(model_path or TOKENIZER_ZH_V1)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"missing tokenizer {self.model_path}")
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
        self.pad_id, self.eos_id, self.bos_id, self.unk_id = 0, 1, 2, 3
        if int(self.sp.get_piece_size()) != 24_000:
            raise ValueError(f"expected 24000 pieces, got {self.sp.get_piece_size()}")
        actual = (
            int(self.sp.pad_id()),
            int(self.sp.eos_id()),
            int(self.sp.bos_id()),
            int(self.sp.unk_id()),
        )
        if actual != (0, 1, 2, 3):
            raise ValueError("special ids must be PAD=0 EOS=1 BOS=2 UNK=3")

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.sp.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        clean = [token for token in ids if token not in {self.pad_id, self.bos_id}]
        if clean and clean[-1] == self.eos_id:
            clean.pop()
        return self.sp.decode(clean)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def encode_document(self, text: str) -> list[int]:
        return self.encode(text, add_bos=True, add_eos=True)

    def encode_chat(self, user: str, assistant: str | None = None) -> dict[str, list[int]]:
        prompt_ids = self.encode(USER_PREFIX + user + ASSISTANT_PREFIX, add_bos=True)
        if assistant is None:
            return {"prompt_ids": prompt_ids, "ids": prompt_ids, "n_prompt": len(prompt_ids)}
        answer_ids = self.encode(assistant) + self.encode(TURN_END) + [self.eos_id]
        return {
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "ids": prompt_ids + answer_ids,
            "n_prompt": len(prompt_ids),
        }


class ZhTokenizerV2:
    """Parameterized successor selected by the frozen-tokenizer pointer.

    Same self-validation contract as V1 (file present, hash matches manifest,
    piece size and special ids as declared); tokenizer_id must match the
    pointer that selected it, which the caller enforces.
    """

    def __init__(
        self,
        *,
        tokenizer_id: str,
        vocab_size: int,
        manifest_path: Path | None = None,
    ):
        import sentencepiece as spm

        self.tokenizer_id = tokenizer_id
        self.model_path = TOKENIZER_DIR / f"{tokenizer_id}.model"
        if not self.model_path.is_file():
            raise FileNotFoundError(f"missing tokenizer {self.model_path}")
        self.model_sha256 = sha256_file(self.model_path)
        self.manifest: dict = {}
        manifest_file = manifest_path or (TOKENIZER_DIR / "tokenizer-v2-manifest.json")
        if manifest_file.is_file():
            self.manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            expected = str(self.manifest.get("model_sha256") or "")
            if expected and expected != self.model_sha256:
                raise ValueError(
                    f"tokenizer hash mismatch: file={self.model_sha256} manifest={expected}"
                )
            if str(self.manifest.get("tokenizer_id") or "") not in ("", tokenizer_id):
                raise ValueError(
                    f"manifest tokenizer_id {self.manifest.get('tokenizer_id')!r} "
                    f"!= {tokenizer_id!r}"
                )
        self.sp = spm.SentencePieceProcessor(model_file=str(self.model_path))
        if vocab_size and int(self.sp.get_piece_size()) != vocab_size:
            raise ValueError(
                f"expected {vocab_size} pieces, got {self.sp.get_piece_size()}"
            )
        self.pad_id, self.eos_id, self.bos_id, self.unk_id = 0, 1, 2, 3
        actual = (
            int(self.sp.pad_id()),
            int(self.sp.eos_id()),
            int(self.sp.bos_id()),
            int(self.sp.unk_id()),
        )
        if actual != (0, 1, 2, 3):
            raise ValueError("special ids must be PAD=0 EOS=1 BOS=2 UNK=3")

    @property
    def vocab_size(self) -> int:
        return int(self.sp.get_piece_size())

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self.sp.encode(text, out_type=int))
        if add_bos:
            ids.insert(0, self.bos_id)
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        clean = [token for token in ids if token not in {self.pad_id, self.bos_id}]
        if clean and clean[-1] == self.eos_id:
            clean.pop()
        return self.sp.decode(clean)

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def encode_document(self, text: str) -> list[int]:
        return self.encode(text, add_bos=True, add_eos=True)

    def encode_chat(self, user: str, assistant: str | None = None) -> dict[str, list[int]]:
        prompt_ids = self.encode(USER_PREFIX + user + ASSISTANT_PREFIX, add_bos=True)
        if assistant is None:
            return {"prompt_ids": prompt_ids, "ids": prompt_ids, "n_prompt": len(prompt_ids)}
        answer_ids = self.encode(assistant) + self.encode(TURN_END) + [self.eos_id]
        return {
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids,
            "ids": prompt_ids + answer_ids,
            "n_prompt": len(prompt_ids),
        }
