"""Bounded KV: fixed system/tool sinks + ordinary ring of 256."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx


@dataclass
class KVManager:
    ordinary_cap: int = 256
    sink_ids: list[int] = field(default_factory=list)
    ordinary_ids: list[int] = field(default_factory=list)
    sink_pos: list[int] = field(default_factory=list)
    ordinary_pos: list[int] = field(default_factory=list)
    next_pos: int = 0
    sink_cache: list[tuple[mx.array, mx.array]] | None = None
    ordinary_cache: list[tuple[mx.array, mx.array]] | None = None
    engram_tail: list[int] = field(default_factory=list)

    @property
    def visible_ids(self) -> list[int]:
        return list(self.sink_ids) + list(self.ordinary_ids)

    @property
    def visible_positions(self) -> list[int]:
        return list(self.sink_pos) + list(self.ordinary_pos)

    def remember_engram(self, token_id: int) -> None:
        self.engram_tail = (self.engram_tail + [int(token_id)])[-2:]

    def prefill_sinks(self, ids: list[int]) -> None:
        self.sink_ids = list(ids)
        self.sink_pos = list(range(len(ids)))
        self.next_pos = len(ids)
        for tok in ids:
            self.remember_engram(tok)

    def append_ordinary(self, token_id: int) -> None:
        pos = self.next_pos
        self.next_pos += 1
        self.ordinary_ids.append(int(token_id))
        self.ordinary_pos.append(pos)
        self.remember_engram(token_id)
        overflow = len(self.ordinary_ids) - self.ordinary_cap
        if overflow > 0:
            self.ordinary_ids = self.ordinary_ids[overflow:]
            self.ordinary_pos = self.ordinary_pos[overflow:]
            if self.ordinary_cache is not None:
                trimmed = []
                for k, v in self.ordinary_cache:
                    trimmed.append((k[:, :, overflow:, :], v[:, :, overflow:, :]))
                self.ordinary_cache = trimmed

    def packed_cache(self) -> list[tuple[mx.array, mx.array]] | None:
        if self.sink_cache is None and self.ordinary_cache is None:
            return None
        n = len(self.sink_cache or self.ordinary_cache or [])
        out = []
        for i in range(n):
            parts_k = []
            parts_v = []
            if self.sink_cache is not None:
                parts_k.append(self.sink_cache[i][0])
                parts_v.append(self.sink_cache[i][1])
            if self.ordinary_cache is not None and self.ordinary_cache[i][0].shape[2] > 0:
                parts_k.append(self.ordinary_cache[i][0])
                parts_v.append(self.ordinary_cache[i][1])
            out.append((mx.concatenate(parts_k, axis=2), mx.concatenate(parts_v, axis=2)))
        return out

    def split_layer_cache(self, new_cache: list[tuple[mx.array, mx.array]], *, n_new_sink: int, n_new_ord: int) -> None:
        sink_part = []
        ord_part = []
        for k, v in new_cache:
            if n_new_sink:
                sink_part.append((k[:, :, :n_new_sink, :], v[:, :, :n_new_sink, :]))
            if n_new_ord:
                start = n_new_sink
                ord_part.append((k[:, :, start : start + n_new_ord, :], v[:, :, start : start + n_new_ord, :]))
        if n_new_sink:
            self.sink_cache = sink_part
        if n_new_ord:
            if self.ordinary_cache is None:
                self.ordinary_cache = ord_part
            else:
                merged = []
                for (ok, ov), (nk, nv) in zip(self.ordinary_cache, ord_part):
                    cat_k = mx.concatenate([ok, nk], axis=2)
                    cat_v = mx.concatenate([ov, nv], axis=2)
                    cap = self.ordinary_cap
                    if cat_k.shape[2] > cap:
                        cat_k = cat_k[:, :, -cap:, :]
                        cat_v = cat_v[:, :, -cap:, :]
                    merged.append((cat_k, cat_v))
                self.ordinary_cache = merged

    def absorb_packed_cache(self, new_cache: list[tuple[mx.array, mx.array]]) -> None:
        n_sink = len(self.sink_ids)
        n_ord = len(self.ordinary_ids)
        sink_part = []
        ord_part = []
        for k, v in new_cache:
            sink_part.append((k[:, :, :n_sink, :], v[:, :, :n_sink, :]))
            rest_k = k[:, :, n_sink:, :]
            rest_v = v[:, :, n_sink:, :]
            if int(rest_k.shape[2]) > n_ord:
                rest_k = rest_k[:, :, -n_ord:, :]
                rest_v = rest_v[:, :, -n_ord:, :]
            ord_part.append((rest_k, rest_v))
        self.sink_cache = sink_part if n_sink else None
        self.ordinary_cache = ord_part if n_ord else None

    def prefill_forward(self, model, sink_ids: list[int], ordinary_ids: list[int]) -> dict[str, Any]:
        self.sink_cache = None
        self.ordinary_cache = None
        self.engram_tail = []
        self.prefill_sinks(sink_ids)
        for tok in ordinary_ids:
            self.append_ordinary(tok)
        ids = self.visible_ids
        pos = self.visible_positions
        if not ids:
            raise ValueError("empty prefill")
        out = model(
            mx.array([ids], dtype=mx.int32),
            position_ids=mx.array(pos, dtype=mx.int32),
        )
        mx.eval(out["logits"])
        self.absorb_packed_cache(out["cache"])
        return out

    def decode_step(self, model, token_id: int) -> dict[str, Any]:
        cache_pos = self.visible_positions
        packed = self.packed_cache()
        prefix = list(self.engram_tail)
        pos = self.next_pos
        kwargs = {
            "cache": packed,
            "position_ids": mx.array([pos], dtype=mx.int32),
        }
        if cache_pos:
            kwargs["cache_position_ids"] = mx.array(cache_pos, dtype=mx.int32)
        if prefix:
            kwargs["engram_prefix_ids"] = prefix
        out = model(mx.array([[int(token_id)]], dtype=mx.int32), **kwargs)
        mx.eval(out["logits"])
        self.append_ordinary(int(token_id))
        self.absorb_packed_cache(out["cache"])
        return out

    def ram_bound_ok(self, n_layers: int, n_kv_heads: int, head_dim: int) -> bool:
        cap = len(self.sink_ids) + min(len(self.ordinary_ids), self.ordinary_cap)
        packed = self.packed_cache()
        if packed is None:
            return True
        for k, v in packed:
            if int(k.shape[2]) > cap + 1:
                return False
            if int(k.shape[1]) != n_kv_heads or int(k.shape[-1]) != head_dim:
                return False
        return len(packed) == n_layers


def causal_mask(q_pos: list[int], k_pos: list[int]) -> Any:
    q = mx.array(q_pos, dtype=mx.int32)
    k = mx.array(k_pos, dtype=mx.int32)
    return (q[:, None] >= k[None, :])[None, None, :, :]
