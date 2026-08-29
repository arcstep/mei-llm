"""Bounded KV: fixed system/tool sinks + ordinary ring of 256."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

PAD_POS = 1_048_576


def _decode_forward(model):
    existing = getattr(model, "_mei_compiled_decode", None)
    if existing is not None:
        return existing

    def fwd(tokens, cache, pos, cpos, widx, prefix):
        out = model(
            tokens,
            cache=cache,
            position_ids=pos,
            cache_position_ids=cpos,
            cache_write_index=widx,
            engram_prefix_ids=prefix,
        )
        return out["logits"], out["cache"]

    if os.environ.get("MEI_SDK_NO_COMPILE") == "1":
        model._mei_compiled_decode = fwd
        return fwd
    try:
        model._mei_compiled_decode = mx.compile(fwd)
    except Exception:
        model._mei_compiled_decode = fwd
    return model._mei_compiled_decode


def _decode_chunk_forward(model, chunk_size: int):
    cache = getattr(model, "_mei_compiled_decode_chunks", None)
    if cache is None:
        cache = {}
        object.__setattr__(model, "_mei_compiled_decode_chunks", cache)
    if chunk_size in cache:
        return cache[chunk_size]

    def fwd(start_logits, kv_cache, pos_buf, start_pos, write_index, prefix):
        logits = start_logits
        tokens = []
        logprobs = []
        for offset in range(chunk_size):
            row = logits[0]
            chosen = mx.argmax(row).astype(mx.int32)
            tokens.append(chosen)
            logprobs.append(row[chosen] - mx.logsumexp(row))
            pos = start_pos + mx.array([offset], dtype=mx.int32)
            index = write_index + mx.array([offset], dtype=mx.int32)
            pos_buf = mx.slice_update(pos_buf, pos, index, [0])
            out = model(
                chosen.reshape(1, 1),
                cache=kv_cache,
                position_ids=pos,
                cache_position_ids=pos_buf,
                cache_write_index=index,
                engram_prefix_ids=prefix,
            )
            logits = out["logits"][:, -1, :]
            kv_cache = out["cache"]
            prefix = mx.concatenate([prefix[-1:], chosen.reshape(1)])
        return (
            logits,
            kv_cache,
            pos_buf,
            mx.stack(tokens),
            mx.stack(logprobs),
            prefix,
        )

    compiled = fwd if os.environ.get("MEI_SDK_NO_COMPILE") == "1" else mx.compile(fwd)
    cache[chunk_size] = compiled
    return compiled


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
    last_logits: Any = field(default=None, repr=False)
    decode_slots: int = 0
    _buffer: Any = field(default=None, repr=False)
    _pos_buf: Any = field(default=None, repr=False)
    _valid_len: int = 0
    _reserve_tokens: int | None = field(default=None, repr=False)

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
        if self._buffer is not None:
            n_valid = len(self.sink_ids) + len(self.ordinary_ids)
            if n_valid <= 0:
                return self._buffer
            return [(k[:, :, :n_valid, :], v[:, :, :n_valid, :]) for k, v in self._buffer]
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

    def _init_decode_buffer(self) -> None:
        packed = None
        if self.sink_cache is not None or self.ordinary_cache is not None:
            packed = []
            n = len(self.sink_cache or self.ordinary_cache or [])
            for i in range(n):
                parts_k = []
                parts_v = []
                if self.sink_cache is not None:
                    parts_k.append(self.sink_cache[i][0])
                    parts_v.append(self.sink_cache[i][1])
                if self.ordinary_cache is not None and self.ordinary_cache[i][0].shape[2] > 0:
                    parts_k.append(self.ordinary_cache[i][0])
                    parts_v.append(self.ordinary_cache[i][1])
                packed.append((mx.concatenate(parts_k, axis=2), mx.concatenate(parts_v, axis=2)))
        if not packed:
            raise ValueError("decode buffer requires prefill cache")
        n_sink = len(self.sink_ids)
        n_valid = n_sink + len(self.ordinary_ids)
        ordinary_slots = self.ordinary_cap + 1
        if self._reserve_tokens is not None:
            ordinary_slots = min(
                ordinary_slots,
                len(self.ordinary_ids) + max(1, self._reserve_tokens) + 1,
            )
        slots = n_sink + ordinary_slots
        buf = []
        for k, v in packed:
            have = int(k.shape[2])
            pad = slots - have
            if pad > 0:
                zk = mx.zeros((k.shape[0], k.shape[1], pad, k.shape[3]), dtype=k.dtype)
                zv = mx.zeros((v.shape[0], v.shape[1], pad, v.shape[3]), dtype=v.dtype)
                k = mx.concatenate([k, zk], axis=2)
                v = mx.concatenate([v, zv], axis=2)
            elif pad < 0:
                k = k[:, :, :slots, :]
                v = v[:, :, :slots, :]
            buf.append((k, v))
        pos = list(self.visible_positions) + [PAD_POS] * (slots - n_valid)
        self._buffer = buf
        self._pos_buf = mx.array(pos, dtype=mx.int32)
        self.decode_slots = slots
        self._valid_len = n_valid
        self.ordinary_cache = None
        mx.eval(self._buffer, self._pos_buf)

    def _compact_overflow(self) -> None:
        ns = len(self.sink_ids)
        slots = self.decode_slots
        new_buf = []
        for k, v in self._buffer:
            zk = mx.zeros((k.shape[0], k.shape[1], 1, k.shape[3]), dtype=k.dtype)
            zv = mx.zeros((v.shape[0], v.shape[1], 1, v.shape[3]), dtype=v.dtype)
            k2 = mx.concatenate([k[:, :, :ns, :], k[:, :, ns + 1 :, :], zk], axis=2)
            v2 = mx.concatenate([v[:, :, :ns, :], v[:, :, ns + 1 :, :], zv], axis=2)
            new_buf.append((k2, v2))
        self._buffer = new_buf
        vis = self.visible_positions
        pad = [PAD_POS] * (slots - len(vis))
        self._pos_buf = mx.array(vis + pad, dtype=mx.int32)
        self._valid_len = len(vis)
        mx.eval(self._buffer, self._pos_buf)

    def prefill_forward(
        self,
        model,
        sink_ids: list[int],
        ordinary_ids: list[int],
        *,
        reserve_tokens: int | None = None,
    ) -> dict[str, Any]:
        self.sink_cache = None
        self.ordinary_cache = None
        self._buffer = None
        self._pos_buf = None
        self._reserve_tokens = reserve_tokens
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
        mx.eval(out["logits"], out["cache"])
        self.last_logits = out["logits"][:, -1, :]
        self.absorb_packed_cache(out["cache"])
        self._init_decode_buffer()
        return out

    def decode_step(self, model, token_id: int) -> dict[str, Any]:
        if self._buffer is None:
            self._init_decode_buffer()
        write_idx = len(self.sink_ids) + len(self.ordinary_ids)
        pos = self.next_pos
        widx = mx.array([write_idx], dtype=mx.int32)
        self._pos_buf = mx.slice_update(self._pos_buf, mx.array([pos], dtype=mx.int32), widx, [0])
        tokens = mx.array([[int(token_id)]], dtype=mx.int32)
        pos_arr = mx.array([pos], dtype=mx.int32)
        prefix = list(self.engram_tail)
        prefix_arr = mx.array(prefix, dtype=mx.int32) if prefix else mx.zeros((0,), dtype=mx.int32)
        at_cap = len(self.ordinary_ids) >= self.ordinary_cap
        fwd = _decode_forward(model)
        logits, cache = fwd(tokens, self._buffer, pos_arr, self._pos_buf, widx, prefix_arr)
        mx.eval(logits, cache)
        self._buffer = cache
        self.last_logits = logits[:, -1, :]
        self.append_ordinary(int(token_id))
        if at_cap:
            self._compact_overflow()
        else:
            self._valid_len = len(self.sink_ids) + len(self.ordinary_ids)
        return {"logits": logits, "cache": cache}

    def decode_chunk(self, model, start_logits, *, chunk_size: int = 8) -> dict[str, Any]:
        """Generate a fixed raw-greedy chunk with one CPU/Metal synchronization."""
        if self._buffer is None:
            self._init_decode_buffer()
        capacity = self.ordinary_cap - len(self.ordinary_ids)
        n = min(max(1, int(chunk_size)), max(0, capacity))
        if n <= 0 or len(self.engram_tail) < 2:
            return {"available": False}
        write_idx = len(self.sink_ids) + len(self.ordinary_ids)
        fwd = _decode_chunk_forward(model, n)
        result = fwd(
            start_logits,
            self._buffer,
            self._pos_buf,
            mx.array([self.next_pos], dtype=mx.int32),
            mx.array([write_idx], dtype=mx.int32),
            mx.array(self.engram_tail[-2:], dtype=mx.int32),
        )
        logits, cache, pos_buf, tokens, logprobs, prefix = result
        mx.eval(logits, cache, pos_buf, tokens, logprobs, prefix)
        token_ids = [int(x) for x in tokens.tolist()]
        self._buffer = cache
        self._pos_buf = pos_buf
        self.last_logits = logits
        for token_id in token_ids:
            self.append_ordinary(token_id)
        self.engram_tail = [int(x) for x in prefix.tolist()]
        self._valid_len = len(self.sink_ids) + len(self.ordinary_ids)
        return {
            "available": True,
            "logits": logits,
            "cache": cache,
            "ids": token_ids,
            "logprobs": [float(x) for x in logprobs.tolist()],
        }

    def ram_bound_ok(self, n_layers: int, n_kv_heads: int, head_dim: int) -> bool:
        max_len = len(self.sink_ids) + self.ordinary_cap + 1
        packed = self._buffer if self._buffer is not None else self.packed_cache()
        if packed is None:
            return True
        if len(packed) != n_layers:
            return False
        for k, v in packed:
            if int(k.shape[2]) > max_len:
                return False
            if int(k.shape[1]) != n_kv_heads or int(k.shape[-1]) != head_dim:
                return False
        return len(self.ordinary_ids) <= self.ordinary_cap


def causal_mask(q_pos: list[int], k_pos: list[int]) -> Any:
    q = mx.array(q_pos, dtype=mx.int32)
    k = mx.array(k_pos, dtype=mx.int32)
    return (q[:, None] >= k[None, :])[None, None, :, :]
