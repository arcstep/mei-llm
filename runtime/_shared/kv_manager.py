"""Bounded stable-sink plus rolling-token context manager.

The portable contract is a maximum 1024-token stable prefix, a 256-token
ordinary ring and at most 2048 total context tokens.  The Python MLX adapter
uses the same fixed physical ring during incremental decode; reference rebuild
is reserved for a position-domain rollover or an unavailable cache.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable

SINK_CAP = 1024
ORDINARY_CAP = 256
MAX_CONTEXT = 2048
DEFAULT_OUTPUT_RESERVE = 128
TARGET_KV_DTYPE = "int8"
TARGET_ACTIVATION_DTYPE = "int8"


def _decode_forward(model):
    existing = getattr(model, "_mei_compiled_decode", None)
    if existing is not None:
        return existing

    def fwd(tokens, cache, pos, cpos, write_index, prefix):
        out = model(
            tokens,
            cache=cache,
            position_ids=pos,
            cache_position_ids=cpos,
            cache_write_index=write_index,
            engram_prefix_ids=prefix,
        )
        return out["logits"], out["cache"]

    compiled = fwd
    if os.environ.get("MEI_SDK_NO_COMPILE") != "1":
        try:
            import mlx.core as mx

            compiled = mx.compile(fwd)
        except Exception:
            compiled = fwd
    object.__setattr__(model, "_mei_compiled_decode", compiled)
    return compiled


def _decode_chunk_forward(model, chunk_size: int, history_tokens: int):
    compiled_chunks = getattr(model, "_mei_compiled_decode_chunks", None)
    if compiled_chunks is None:
        compiled_chunks = {}
        object.__setattr__(model, "_mei_compiled_decode_chunks", compiled_chunks)
    key = (int(chunk_size), int(history_tokens))
    if key in compiled_chunks:
        return compiled_chunks[key]

    import mlx.core as mx

    def fwd(start_logits, cache, pos_buf, start_pos, start_write, prefix):
        logits = start_logits
        tokens = []
        logprobs = []
        for offset in range(int(chunk_size)):
            row = logits[0]
            chosen = mx.argmax(row).astype(mx.int32)
            tokens.append(chosen)
            logprobs.append(row[chosen] - mx.logsumexp(row))
            pos = start_pos + mx.array([offset], dtype=mx.int32)
            write_index = start_write + mx.array([offset], dtype=mx.int32)
            pos_buf = mx.slice_update(pos_buf, pos, write_index, [0])
            out = model(
                chosen.reshape(1, 1),
                cache=cache,
                position_ids=pos,
                cache_position_ids=pos_buf,
                cache_write_index=write_index,
                engram_prefix_ids=prefix,
            )
            logits = out["logits"][:, -1, :]
            cache = out["cache"]
            prefix = mx.concatenate([prefix, chosen.reshape(1)])[-history_tokens:]
        return logits, cache, pos_buf, mx.stack(tokens), mx.stack(logprobs)

    compiled = fwd if os.environ.get("MEI_SDK_NO_COMPILE") == "1" else mx.compile(fwd)
    compiled_chunks[key] = compiled
    return compiled


@dataclass(frozen=True)
class QuantizedInt8:
    values: tuple[int, ...]
    scale: float

    def dequantize(self) -> list[float]:
        return [value * self.scale for value in self.values]


def quantize_int8(values: Iterable[float]) -> QuantizedInt8:
    row = [float(value) for value in values]
    if any(not math.isfinite(value) for value in row):
        raise ValueError("int8 cache input must be finite")
    maximum = max((abs(value) for value in row), default=0.0)
    scale = maximum / 127.0 if maximum else 1.0
    quantized = tuple(max(-127, min(127, int(round(value / scale)))) for value in row)
    return QuantizedInt8(quantized, scale)


@dataclass
class KVManager:
    ordinary_cap: int = ORDINARY_CAP
    sink_cap: int = SINK_CAP
    max_context: int = MAX_CONTEXT
    output_reserve: int = DEFAULT_OUTPUT_RESERVE
    sink_ids: list[int] = field(default_factory=list)
    ordinary_ids: list[int] = field(default_factory=list)
    sink_pos: list[int] = field(default_factory=list)
    ordinary_pos: list[int] = field(default_factory=list)
    next_pos: int = 0
    last_logits: Any = field(default=None, repr=False)
    _cache: Any = field(default=None, repr=False)
    _cache_positions: list[int] = field(default_factory=list, repr=False)
    _cache_write_cursor: int = field(default=0, repr=False)
    _incremental_decode_steps: int = field(default=0, repr=False)
    _recomputed_decode_steps: int = field(default=0, repr=False)
    _activation_dtype: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        for name in ("ordinary_cap", "sink_cap", "max_context", "output_reserve"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.sink_cap + self.ordinary_cap + self.output_reserve > self.max_context:
            raise ValueError("sink + ordinary + output reserve exceeds max context")

    @property
    def visible_ids(self) -> list[int]:
        return list(self.sink_ids) + list(self.ordinary_ids)

    @property
    def visible_positions(self) -> list[int]:
        return list(self.sink_pos) + list(self.ordinary_pos)

    @property
    def bounded(self) -> bool:
        return (
            len(self.sink_ids) <= self.sink_cap
            and len(self.ordinary_ids) <= self.ordinary_cap
            and len(self.visible_ids) + self.output_reserve <= self.max_context
        )

    def prefill_sinks(self, token_ids: Iterable[int]) -> None:
        values = [int(value) for value in token_ids]
        if len(values) > self.sink_cap:
            raise ValueError("tool_schema_budget_exceeded")
        self.sink_ids = values
        self.sink_pos = list(range(len(values)))
        self.next_pos = len(values)

    def replace_ordinary(self, token_ids: Iterable[int]) -> None:
        values = [int(value) for value in token_ids]
        position_start = self.next_pos
        self.next_pos += len(values)
        kept = values[-self.ordinary_cap :] if self.ordinary_cap else []
        self.ordinary_ids = kept
        kept_start = position_start + len(values) - len(kept)
        if self.next_pos <= self.max_context:
            self.ordinary_pos = list(range(kept_start, self.next_pos))
        else:
            # Inputs beyond the declared 2048 position domain are represented
            # by a freshly re-based bounded window, never by unbounded RoPE.
            self.ordinary_pos = list(
                range(len(self.sink_ids), len(self.sink_ids) + len(kept))
            )

    def append_ordinary(self, token_id: int) -> None:
        next_position = self.next_pos
        self.ordinary_ids.append(int(token_id))
        self.ordinary_pos.append(next_position)
        self.next_pos += 1
        overflow = len(self.ordinary_ids) - self.ordinary_cap
        if overflow > 0:
            del self.ordinary_ids[:overflow]
            del self.ordinary_pos[:overflow]
        if self.next_pos > self.max_context:
            self.ordinary_pos = list(
                range(len(self.sink_ids), len(self.sink_ids) + len(self.ordinary_ids))
            )

    def prepare(self, sink_ids: Iterable[int], ordinary_ids: Iterable[int]) -> dict[str, Any]:
        self.prefill_sinks(sink_ids)
        self.replace_ordinary(ordinary_ids)
        if not self.bounded:
            raise ValueError("context_budget_exceeded")
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        measured = self.measured_cache_dtype()
        physical_bounded = self._cache is not None
        if physical_bounded:
            try:
                physical_cap = len(self.sink_ids) + self.ordinary_cap
                physical_bounded = all(
                    len(layer) in {2, 4}
                    and int(layer[0].shape[2]) <= physical_cap
                    and int(layer[2 if len(layer) == 4 else 1].shape[2]) <= physical_cap
                    for layer in self._cache
                )
            except (AttributeError, TypeError, ValueError):
                physical_bounded = False
        return {
            "sink_ids": list(self.sink_ids),
            "ordinary_ids": list(self.ordinary_ids),
            "visible_ids": self.visible_ids,
            "visible_positions": self.visible_positions,
            "sink_tokens": len(self.sink_ids),
            "ordinary_tokens": len(self.ordinary_ids),
            "stable_prefix_tokens": len(self.sink_ids),
            "rolling_tokens": len(self.ordinary_ids),
            "visible_tokens": len(self.visible_ids),
            "output_reserve": self.output_reserve,
            "total_tokens_seen": self.next_pos,
            "max_context": self.max_context,
            "target_kv_dtype": TARGET_KV_DTYPE,
            "target_activation_dtype": TARGET_ACTIVATION_DTYPE,
            "measured_cache_dtype": measured,
            "kv_dtype": measured,
            "kv_storage_dtype": measured,
            "activation_dtype": self._activation_dtype,
            "activation_quantization": self._activation_dtype,
            "incremental_decode_steps": self._incremental_decode_steps,
            "recomputed_decode_steps": self._recomputed_decode_steps,
            "bounded": self.bounded,
            "cache_growth_bounded": bool(self.bounded and physical_bounded),
        }

    def measured_cache_dtype(self) -> str | None:
        if self._cache is None:
            return None
        dtypes: set[str] = set()
        try:
            int8_cache = True
            for layer in self._cache:
                if len(layer) == 4:
                    key_codes, key_scales, value_codes, value_scales = layer
                    code_dtypes = {
                        str(key_codes.dtype).lower(),
                        str(value_codes.dtype).lower(),
                    }
                    scale_dtypes = {
                        str(key_scales.dtype).lower(),
                        str(value_scales.dtype).lower(),
                    }
                    int8_cache = int8_cache and all("int8" in dtype for dtype in code_dtypes)
                    int8_cache = int8_cache and all("float" in dtype for dtype in scale_dtypes)
                    dtypes.update(code_dtypes)
                elif len(layer) == 2:
                    key, value = layer
                    int8_cache = False
                    dtypes.add(str(key.dtype).lower())
                    dtypes.add(str(value.dtype).lower())
                else:
                    return "unknown"
        except (AttributeError, TypeError, ValueError):
            return "unknown"
        if int8_cache and dtypes:
            return "mlx.core.int8"
        if not dtypes:
            return "unknown"
        return next(iter(dtypes)) if len(dtypes) == 1 else "mixed"

    def prefill_forward(
        self,
        model,
        sink_ids: list[int],
        ordinary_ids: list[int],
        *,
        reserve_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Correctness-first MLX prefill over the bounded visible window."""
        import mlx.core as mx

        if reserve_tokens is not None:
            reserve = int(reserve_tokens)
            if reserve < 0 or self.sink_cap + self.ordinary_cap + reserve > self.max_context:
                raise ValueError("context_budget_exceeded")
            self.output_reserve = reserve
        self.prepare(sink_ids, ordinary_ids)
        if not self.visible_ids:
            raise ValueError("empty prefill")
        out = model(
            mx.array([self.visible_ids], dtype=mx.int32),
            position_ids=mx.array(self.visible_positions, dtype=mx.int32),
        )
        mx.eval(out["logits"], out.get("cache"))
        self._activation_dtype = (
            "int8-qdq"
            if bool(getattr(model, "_mei_activation_int8", False))
            else str(out["hidden"].dtype).lower()
        )
        self.last_logits = out["logits"][:, -1, :]
        raw_cache = out.get("cache")
        capacity = len(self.sink_ids) + self.ordinary_cap
        visible = len(self.visible_ids)
        if raw_cache is not None and capacity >= visible:
            pad = capacity - visible
            if pad:
                raw_cache = [
                    tuple(
                        mx.pad(value, [(0, 0), (0, 0), (0, pad), (0, 0)])
                        for value in layer
                    )
                    for layer in raw_cache
                ]
                mx.eval(raw_cache)
            self._cache_positions = self.visible_positions + [self.max_context + 1] * pad
            self._cache_write_cursor = len(self.ordinary_ids) % max(1, self.ordinary_cap)
        else:
            self._cache_positions = self.visible_positions
            self._cache_write_cursor = 0
        self._cache = raw_cache
        out["cache"] = raw_cache
        self._incremental_decode_steps = 0
        self._recomputed_decode_steps = 0
        return out

    def absorb_cache(self, cache: Any, *, last_logits: Any = None) -> None:
        """Attach a freshly computed cache without changing window identity."""

        self._cache = cache
        if last_logits is not None:
            self.last_logits = last_logits

    def decode_step(self, model, token_id: int) -> dict[str, Any]:
        """Append one token using the existing KV cache whenever possible.

        The normal product path evaluates a genuine one-token decode against a
        fixed physical KV ring, including after ordinary-token eviction.  A
        bounded rebuild is only used when no cache is available or the declared
        2048-position domain must be re-based.
        """
        import mlx.core as mx

        new_position = self.next_pos
        can_append_cache = (
            self._cache is not None
            and self.ordinary_cap > 0
            and len(self._cache_positions) == len(self.sink_ids) + self.ordinary_cap
            and new_position < self.max_context
        )
        if len(self.ordinary_ids) < self.ordinary_cap:
            write_index = len(self.sink_ids) + len(self.ordinary_ids)
        else:
            write_index = len(self.sink_ids) + self._cache_write_cursor
        self.append_ordinary(int(token_id))
        if can_append_cache:
            # Build Engram history from the post-eviction logical window, but
            # exclude the token currently being decoded.  Using the old window
            # would leak the just-evicted oldest token into n-gram/tap state.
            prefix_ids = self.visible_ids[:-1]
            self._cache_positions[write_index] = new_position
            cfg = getattr(model, "cfg", None)
            orders = tuple(getattr(cfg, "engram_orders", (2, 3))) or (1,)
            history = max(
                1,
                int(getattr(cfg, "engram_history", 0))
                or int(getattr(cfg, "engram_conv_taps", 4)) * max(orders),
            )
            call = _decode_forward(model)
            logits, cache = call(
                mx.array([[int(token_id)]], dtype=mx.int32),
                self._cache,
                mx.array([self.visible_positions[-1]], dtype=mx.int32),
                mx.array(self._cache_positions, dtype=mx.int32),
                mx.array([write_index], dtype=mx.int32),
                mx.array([prefix_ids[-history:]], dtype=mx.int32),
            )
            out = {"logits": logits, "cache": cache}
            self._cache_write_cursor = (self._cache_write_cursor + 1) % self.ordinary_cap
            self._incremental_decode_steps += 1
        else:
            out = model(
                mx.array([self.visible_ids], dtype=mx.int32),
                position_ids=mx.array(self.visible_positions, dtype=mx.int32),
            )
            self._recomputed_decode_steps += 1
        mx.eval(out["logits"], out.get("cache"))
        self.last_logits = out["logits"][:, -1, :]
        self._cache = out.get("cache")
        return out

    def decode_chunk(self, model, start_logits, *, chunk_size: int = 8) -> dict[str, Any]:
        """Generate a raw greedy chunk with one CPU/Metal synchronization.

        A chunk never crosses a physical ring wrap or the 2048-position
        rollover, keeping tensor shapes and write indices static for MLX.
        """
        import mlx.core as mx

        if self._cache is None or self.ordinary_cap <= 0 or self.next_pos >= self.max_context:
            return {"available": False}
        cfg = getattr(model, "cfg", None)
        orders = tuple(getattr(cfg, "engram_orders", (2, 3))) or (1,)
        history = max(
            1,
            int(getattr(cfg, "engram_history", 0))
            or int(getattr(cfg, "engram_conv_taps", 4)) * max(orders),
        )
        prefix_ids = self.visible_ids
        if len(prefix_ids) < history:
            return {"available": False}
        if len(self.ordinary_ids) < self.ordinary_cap:
            contiguous = self.ordinary_cap - len(self.ordinary_ids)
            write_index = len(self.sink_ids) + len(self.ordinary_ids)
        else:
            contiguous = self.ordinary_cap - self._cache_write_cursor
            write_index = len(self.sink_ids) + self._cache_write_cursor
        n = min(
            max(1, int(chunk_size)),
            contiguous,
            self.max_context - self.next_pos,
        )
        if n <= 0:
            return {"available": False}
        fwd = _decode_chunk_forward(model, n, history)
        logits, cache, pos_buf, tokens, logprobs = fwd(
            start_logits,
            self._cache,
            mx.array(self._cache_positions, dtype=mx.int32),
            mx.array([self.next_pos], dtype=mx.int32),
            mx.array([write_index], dtype=mx.int32),
            mx.array(prefix_ids[-history:], dtype=mx.int32),
        )
        mx.eval(logits, cache, pos_buf, tokens, logprobs)
        token_ids = [int(value) for value in tokens.tolist()]
        for token_id in token_ids:
            self.append_ordinary(token_id)
        self._cache = cache
        self._cache_positions = [int(value) for value in pos_buf.tolist()]
        self._cache_write_cursor = (self._cache_write_cursor + n) % self.ordinary_cap
        self._incremental_decode_steps += n
        self.last_logits = logits
        return {
            "available": True,
            "logits": logits,
            "cache": cache,
            "ids": token_ids,
            "logprobs": [float(value) for value in logprobs.tolist()],
        }

    def ram_bound_ok(self, n_layers: int, n_kv_heads: int, head_dim: int) -> bool:
        if not self.bounded:
            return False
        if self._cache is None:
            return False
        if self.measured_cache_dtype() not in {"int8", "mlx.core.int8"}:
            return False
        if len(self._cache) != int(n_layers):
            return False
        maximum = len(self.sink_ids) + self.ordinary_cap
        for layer in self._cache:
            if len(layer) == 4:
                key, _, value, _ = layer
            elif len(layer) == 2:
                key, value = layer
            else:
                return False
            if int(key.shape[2]) > maximum or int(value.shape[2]) > maximum:
                return False
            if int(key.shape[1]) != int(n_kv_heads) or int(key.shape[-1]) != int(head_dim):
                return False
        return True


BoundedKVManager = KVManager
