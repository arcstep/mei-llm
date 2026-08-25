from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class NeedleZhConfig:
    vocab_size: int = 24000
    d_model: int = 512
    n_layers: int = 27
    n_heads: int = 8
    n_kv_heads: int = 4
    rope_theta: float = 100000.0
    mlp: str = "HadamardMLP"
    engram_layers: tuple[int, ...] = (2, 15)
    engram_orders: tuple[int, ...] = (2, 3)
    engram_slots: int = 8192
    mhc_lanes: int = 4
    max_seq_len: int = 2048
    kv_window: int = 512
    tie_embeddings: bool = True
    mtp_enabled: bool = False
    confidence_head: bool = True
    pad_id: int = 0
    eos_id: int = 1
    bos_id: int = 2
    unk_id: int = 3
    rms_eps: float = 1e-6
    conf_probes: int = 8

    def __post_init__(self) -> None:
        self.engram_layers = tuple(self.engram_layers)
        self.engram_orders = tuple(self.engram_orders)
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must divide n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must divide n_kv_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_spec(cls, path: Path | None = None) -> NeedleZhConfig:
        if path is None:
            path = Path(__file__).resolve().parents[1] / "spec" / "model.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        arch = data["architecture"]
        tok = data.get("tokenizer") or {}
        return cls(
            vocab_size=int(arch["vocab_size"]),
            d_model=int(arch["d_model"]),
            n_layers=int(arch["n_layers"]),
            n_heads=int(arch["n_heads"]),
            n_kv_heads=int(arch["n_kv_heads"]),
            rope_theta=float(arch["rope_theta"]),
            mlp=str(arch.get("mlp") or "HadamardMLP"),
            engram_layers=tuple(arch["engram_layers"]),
            engram_orders=tuple(arch.get("engram_orders") or (2, 3)),
            engram_slots=int(arch.get("engram_slots") or 8192),
            mhc_lanes=int(arch["mhc_lanes"]),
            max_seq_len=int(arch["max_seq_len"]),
            kv_window=int(arch.get("kv_window") or 512),
            tie_embeddings=bool(arch.get("tie_embeddings", True)),
            mtp_enabled=bool(arch.get("mtp_enabled", False)),
            confidence_head=bool(arch.get("confidence_head", True)),
            pad_id=int(tok.get("pad_id", 0)),
            eos_id=int(tok.get("eos_id", 1)),
            bos_id=int(tok.get("bos_id", 2)),
            unk_id=int(tok.get("unk_id", 3)),
        )

    def tiny(self, **overrides) -> NeedleZhConfig:
        base = asdict(self)
        base.update(
            {
                "d_model": 64,
                "n_layers": 2,
                "n_heads": 4,
                "n_kv_heads": 2,
                "engram_layers": (0, 1),
                "engram_slots": 128,
                "mhc_lanes": 2,
                "max_seq_len": 64,
                "kv_window": 32,
            }
        )
        base.update(overrides)
        return NeedleZhConfig(**base)
