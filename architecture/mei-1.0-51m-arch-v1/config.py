from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent


@dataclass
class NeedleZhConfig:
    architecture_id: str = "mei-1.0-51m-arch-v1"
    vocab_size: int = 24000
    d_model: int = 512
    n_layers: int = 27
    n_heads: int = 8
    n_kv_heads: int = 4
    rope_theta: float = 100000.0
    mlp: str = "FixedWalshHadamardMLP"
    engram_layers: tuple[int, ...] = (2, 15)
    engram_orders: tuple[int, ...] = (2, 3)
    engram_slots: int = 8192
    engram_conv_taps: int = 4
    mhc_lanes: int = 4
    sinkhorn_iters: int = 20
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
        self.engram_layers = tuple(int(x) for x in self.engram_layers)
        self.engram_orders = tuple(int(x) for x in self.engram_orders)
        if self.d_model % self.n_heads != 0:
            raise ValueError("n_heads must divide d_model")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_kv_heads must divide n_heads")
        if not self.engram_orders or min(self.engram_orders) <= 0:
            raise ValueError("engram_orders must be positive")
        if any(layer < 0 or layer >= self.n_layers for layer in self.engram_layers):
            raise ValueError("engram layer index out of range")
        if self.mhc_lanes <= 0:
            raise ValueError("mhc_lanes must be positive")
        if self.sinkhorn_iters <= 0:
            raise ValueError("sinkhorn_iters must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def engram_history(self) -> int:
        return self.engram_conv_taps * max(self.engram_orders)

    @classmethod
    def from_spec(cls, path: Path | None = None) -> NeedleZhConfig:
        source = path or (_HERE / "spec/model.json")
        data = json.loads(source.read_text(encoding="utf-8"))
        arch = data["architecture"]
        tok = data.get("tokenizer") or {}
        return cls(
            architecture_id=str(data.get("architecture_id") or data.get("spec_id") or cls.architecture_id),
            vocab_size=int(arch["vocab_size"]),
            d_model=int(arch["d_model"]),
            n_layers=int(arch["n_layers"]),
            n_heads=int(arch["n_heads"]),
            n_kv_heads=int(arch["n_kv_heads"]),
            rope_theta=float(arch["rope_theta"]),
            mlp=str(arch.get("mlp") or "FixedWalshHadamardMLP"),
            engram_layers=tuple(arch["engram_layers"]),
            engram_orders=tuple(arch.get("engram_orders") or (2, 3)),
            engram_slots=int(arch.get("engram_slots") or 8192),
            engram_conv_taps=int(arch.get("engram_conv_taps") or 4),
            mhc_lanes=int(arch["mhc_lanes"]),
            sinkhorn_iters=int(arch.get("sinkhorn_iters") or 20),
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
