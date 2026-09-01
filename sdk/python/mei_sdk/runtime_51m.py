"""Python+MLX numerical oracle for the 51M portable CQ2 runtime."""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

from .errors import SdkError
from .package import ModelPackage
from .protocol import MAX_SELECTED_TOOLS, adapt_v1_request, render_request
from .shared import (
    BoundedKVManager,
    ToolIndex,
    catalog_fingerprint,
    compile_byte_grammar_cached,
    token_to_bytes,
    validate_generated_call,
)
from .version import SDK_ROOT

_MEI_LLM = SDK_ROOT.parent
_ARCH_51 = _MEI_LLM / "architecture" / "mei-1.0-51m-arch-v1"
_TRAIN = _MEI_LLM / "training" / "mei-1.0-51m-train-v1"
_RUNTIME_SHARED = _MEI_LLM / "runtime" / "_shared"

EXECUTE_HIGH = 0.70
ESCALATE_LOW = 0.35
MW_N_CLASSES = 20
MW_CONTINUE_CLASS = 0
MW_CONTINUE_MIN_PROBABILITY = 0.70
MW_CONTINUE_MIN_MARGIN = 0.15


def _ensure(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


def _token_bytes(tokenizer, tok_id: int, *, at_start: bool = False) -> bytes:
    return token_to_bytes(tokenizer, tok_id, at_start=at_start)


def validate_call(text: str, *, tools: list[dict[str, Any]], query: str, system_facts: str = "") -> dict[str, Any]:
    """Read-only v1 training adapter over the canonical v2 validator.

    Historical training/evaluation scripts still pass text facts instead of
    structured evidence.  Marking the request as v1 makes that degradation
    explicit while keeping grammar, schema and safety decisions in one place.
    """

    request = adapt_v1_request(
        {
            "wire_version": "mei-runtime-wire-v1",
            "query": query,
            "system_facts": system_facts,
        }
    )
    return validate_generated_call(
        text,
        tools=tools,
        request=request,
        confidence=None,
        enforce_confidence=False,
    )


def apply_confidence_gate(validated: dict[str, Any], confidence: float | None) -> dict[str, Any]:
    """confidence actually changes execute / escalate / refuse."""
    out = dict(validated)
    if confidence is not None and (
        isinstance(confidence, bool) or not isinstance(confidence, (int, float))
    ):
        value = math.nan
    else:
        value = None if confidence is None else float(confidence)
    out["confidence_value"] = value
    if not validated.get("ok") or validated.get("refuse"):
        out["execution"] = "refuse"
        return out
    if value is None:
        out["ok"] = False
        out["refuse"] = True
        out["function_calls"] = []
        out["error"] = "confidence_unavailable"
        out["execution"] = "refuse"
        return out
    if value is None or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        out["ok"] = False
        out["refuse"] = True
        out["function_calls"] = []
        out["error"] = "confidence_invalid" if value is not None else "confidence_unavailable"
        out["execution"] = "refuse"
        return out
    if value >= EXECUTE_HIGH:
        out["execution"] = "execute"
        return out
    if value >= ESCALATE_LOW:
        out["execution"] = "escalate"
        out["ok"] = True
        out["refuse"] = True
        out["function_calls"] = []
        out["error"] = None
        out["escalate"] = True
        return out
    out["execution"] = "refuse"
    out["ok"] = True
    out["refuse"] = True
    out["function_calls"] = []
    return out


def mw_disposition_from_probabilities(
    probabilities: list[float], *, receipt_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project the canonical 20-class MW reason head onto the safe wire gate.

    The reason codebook is intentionally richer than the three wire
    dispositions.  Only calibrated class 0 (``ready_to_execute``) may
    continue.  A reason head can never synthesize ``constrain`` because it has
    no authority to invent the required ``allowed_tools`` set.
    """

    if (
        len(probabilities) != MW_N_CLASSES
        or any(not math.isfinite(float(value)) or float(value) < 0.0 for value in probabilities)
        or not math.isclose(sum(float(value) for value in probabilities), 1.0, abs_tol=1e-4)
    ):
        raise SdkError("package_invalid", "MW disposition head returned invalid probabilities")
    ranked = sorted(
        ((float(value), index) for index, value in enumerate(probabilities)),
        key=lambda row: (-row[0], row[1]),
    )
    top_probability, reason_code = ranked[0]
    second_probability = ranked[1][0]
    margin = top_probability - second_probability
    decision = (
        "continue"
        if reason_code == MW_CONTINUE_CLASS
        and top_probability >= MW_CONTINUE_MIN_PROBABILITY
        and margin >= MW_CONTINUE_MIN_MARGIN
        else "stop"
    )
    wire = {
        "decision": decision,
        "source": "mw-head",
        "receipt_sha256": receipt_sha256,
    }
    audit = {
        **wire,
        "reason_code": reason_code,
        "top_probability": top_probability,
        "margin": margin,
        "continue_min_probability": MW_CONTINUE_MIN_PROBABILITY,
        "continue_min_margin": MW_CONTINUE_MIN_MARGIN,
    }
    return wire, audit


class PortableMWDispositionHead:
    """Numerical oracle for the portable 512→20 reason-code projection."""

    def __init__(self, weight, bias, *, receipt_sha256: str):
        if tuple(weight.shape) != (MW_N_CLASSES, 512) or tuple(bias.shape) != (MW_N_CLASSES,):
            raise SdkError("package_invalid", "portable MW disposition tensor shape mismatch")
        if (
            not isinstance(receipt_sha256, str)
            or len(receipt_sha256) != 64
            or any(char not in "0123456789abcdef" for char in receipt_sha256)
        ):
            raise SdkError("package_invalid", "portable MW disposition receipt is missing")
        self.weight = weight
        self.bias = bias
        self.receipt_sha256 = receipt_sha256

    def __call__(self, cells) -> tuple[dict[str, Any], dict[str, Any]]:
        import mlx.core as mx

        stacked = mx.stack(list(cells), axis=2) if isinstance(cells, (list, tuple)) else cells
        if int(stacked.ndim) != 4 or int(stacked.shape[-1]) != 512:
            raise SdkError("package_invalid", "MW disposition cells must be [B,T,L,512]")
        pooled = mx.mean(mx.stop_gradient(stacked).astype(mx.float32)[:, -1, :, :], axis=1)
        logits = pooled @ self.weight.astype(mx.float32).T + self.bias.astype(mx.float32)
        probabilities = mx.softmax(logits, axis=-1)
        mx.eval(probabilities)
        rows = probabilities.tolist()
        if len(rows) != 1:
            raise SdkError("package_invalid", "MW disposition runtime requires batch size 1")
        return mw_disposition_from_probabilities(
            [float(value) for value in rows[0]], receipt_sha256=self.receipt_sha256
        )


class Runtime51M:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        contrastive=None,
        mw_disposition=None,
        conf_v2=None,
        narration_adapter=None,
        index=None,
        model_hash: str = "",
        head_hash: str = "",
        tokenizer_hash: str = "",
        release_class: str = "experimental",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.contrastive = contrastive
        self.mw_disposition = mw_disposition
        self.conf_v2 = conf_v2
        self.narration_adapter = narration_adapter
        self.release_class = release_class
        self.retrieval_source = "contrastive" if contrastive is not None else "backbone_fallback"
        self.index = index or ToolIndex(
            model_hash=model_hash,
            head_hash=head_hash,
            tokenizer_hash=tokenizer_hash,
        )
        self._catalog_fp = self.index.catalog_hash

    def embed_text(self, text: str):
        import mlx.core as mx

        ids = self.tokenizer.encode(text, add_bos=True, add_eos=False)[:256]
        arr = mx.array([ids], dtype=mx.int32)
        out = self.model(arr, return_cells=True)
        cells = out.get("cells")
        if self.contrastive is None or cells is None:
            hidden = out["hidden"][:, -1, :].astype(mx.float32)
            norm = mx.sqrt(mx.sum(hidden * hidden, axis=-1, keepdims=True) + 1e-8)
            return hidden / norm
        vec = self.contrastive(cells)
        mx.eval(vec)
        return vec[0]

    def build_index(self, catalog: list[dict[str, Any]]) -> None:
        fp = catalog_fingerprint(catalog)
        if fp == self._catalog_fp and len(self.index.records) == len(catalog):
            return
        self.index.build(catalog, self.embed_text)
        self._catalog_fp = fp

    def search_top_k(self, query: str, catalog: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
        self.build_index(catalog)
        q = self.embed_text(query)
        return self.index.select_tools(catalog, q, k=k)

    def _forward(
        self,
        ids: list[int],
        *,
        cache=None,
        position_ids: list[int] | None = None,
        return_confidence: bool = False,
        return_cells: bool = False,
    ):
        import mlx.core as mx

        arr = mx.array([ids], dtype=mx.int32)
        return self.model(
            arr,
            cache=cache,
            position_ids=position_ids,
            return_confidence=return_confidence,
            return_cells=return_cells,
        )

    def generate_narration(self, prompt: str, *, max_new: int = 48) -> dict[str, Any]:
        """Greedy short-text generation with the frozen-backbone rank-16 residual."""

        import mlx.core as mx

        if self.narration_adapter is None:
            raise SdkError("capability_missing", "narration adapter is unavailable")
        max_new = min(48, max(1, int(max_new)))
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        if not prompt_ids:
            raise SdkError("invalid_argument", "narration prompt produced no tokens")
        sink_ids = prompt_ids[: min(1024, len(prompt_ids))]
        ordinary_ids = prompt_ids[len(sink_ids) :]
        kv = BoundedKVManager(output_reserve=max_new)
        out = kv.prefill_forward(
            self.model, list(sink_ids), list(ordinary_ids), reserve_tokens=max_new
        )
        pieces: list[int] = []
        prefix = b""
        for _ in range(max_new):
            hidden = out["hidden"][:, -1, :]
            residual = self.narration_adapter(hidden).astype(mx.float32)
            logits = out["logits"][:, -1, :].astype(mx.float32) + residual
            mx.eval(logits)
            token = int(mx.argmax(logits[0]).item())
            if token == self.tokenizer.eos_id:
                break
            prefix += _token_bytes(self.tokenizer, token, at_start=not pieces)
            pieces.append(token)
            out = kv.decode_step(self.model, token)
        return {
            "text": prefix.decode("utf-8", errors="ignore").strip(),
            "ids": pieces,
            "n_out": len(pieces),
            "bounded": kv.bounded,
        }

    def greedy(
        self,
        prompt_ids: list[int],
        *,
        tools: list[dict[str, Any]],
        max_new: int = 128,
        decode_mode: str = "constrained",
        sink_ids: list[int] | None = None,
        ordinary_ids: list[int] | None = None,
        benchmark_ignore_eos: bool = False,
    ) -> dict[str, Any]:
        import mlx.core as mx
        from byte_grammar import select_legal_token

        max_new = min(128, max(1, int(max_new)))
        grammar = compile_byte_grammar_cached(tools, self.tokenizer)
        pieces: list[int] = []
        prefix = b""
        logprob_sum = 0.0
        n_out = 0
        kv = BoundedKVManager(output_reserve=max_new)
        if sink_ids is None:
            sink_ids = list(prompt_ids[: min(1024, len(prompt_ids))])
        if ordinary_ids is None:
            ordinary_ids = list(prompt_ids[len(sink_ids) :])
        out = kv.prefill_forward(self.model, list(sink_ids), list(ordinary_ids), reserve_tokens=max_new)
        # Heads are read from the bounded visible prompt, separate from the cache prefill.
        head_out = self._forward(
            kv.visible_ids,
            position_ids=kv.visible_positions,
            return_confidence=True,
            return_cells=True,
        )
        logits = out["logits"][:, -1, :]
        cells = head_out.get("cells")
        conf_logit = head_out.get("confidence_logit")
        if self.mw_disposition is not None and cells is not None:
            mw_disposition, mw_audit = self.mw_disposition(cells)
        else:
            mw_disposition = {"decision": "stop", "source": "deterministic-policy"}
            mw_audit = {
                **mw_disposition,
                "reason_code": None,
                "detail": "mw_head_unavailable",
            }
        t0 = time.perf_counter()
        remaining = max_new
        use_chunk = (
            decode_mode == "raw"
            and getattr(self.model, "_inference_backend", "mlx-reference")
            in {"mlx-fused", "mlx-cq2"}
            and int(os.environ.get("MEI_SDK_DECODE_CHUNK", "24")) > 1
        )
        stopped = False
        while remaining > 0:
            if use_chunk:
                chunk = kv.decode_chunk(
                    self.model,
                    logits,
                    chunk_size=min(
                        int(os.environ.get("MEI_SDK_DECODE_CHUNK", "24")),
                        remaining,
                    ),
                )
                if chunk.get("available"):
                    logits = chunk["logits"]
                    remaining -= len(chunk["ids"])
                    for tok, selected_logp in zip(chunk["ids"], chunk["logprobs"]):
                        if tok == self.tokenizer.eos_id and not benchmark_ignore_eos:
                            stopped = True
                            break
                        extra = _token_bytes(self.tokenizer, tok, at_start=not pieces)
                        prefix = prefix + extra
                        pieces.append(tok)
                        logprob_sum += float(selected_logp)
                        n_out += 1
                    if stopped:
                        break
                    continue
                use_chunk = False
            row = logits[0]
            if decode_mode == "raw":
                tok = int(mx.argmax(row).item())
            else:
                selected = select_legal_token(
                    logits,
                    self.tokenizer,
                    grammar,
                    prefix,
                    generated_token_count=len(pieces),
                )
                if selected is None:
                    break
                tok = int(selected)
            logp = float((row[tok] - mx.logsumexp(row)).item())
            if tok == self.tokenizer.eos_id and not benchmark_ignore_eos:
                break
            extra = _token_bytes(self.tokenizer, tok, at_start=not pieces)
            prefix = prefix + extra
            pieces.append(tok)
            logprob_sum += logp
            n_out += 1
            step = kv.decode_step(self.model, tok)
            logits = step["logits"][:, -1, :]
            remaining -= 1
        decode_ms = (time.perf_counter() - t0) * 1000
        text = prefix.decode("utf-8", errors="ignore")
        conf = None
        if self.conf_v2 is not None and cells is not None:
            logit = self.conf_v2(cells)
            mx.eval(logit)
            conf = combine_conf(float(logit.item()), logprob_sum)
        elif conf_logit is not None:
            mx.eval(conf_logit)
            conf = combine_conf(float(conf_logit[0].item()), logprob_sum)
        return {
            "text": text,
            "ids": pieces,
            "n_out": n_out,
            "decode_ms": decode_ms,
            "confidence": conf,
            "mw_disposition": mw_disposition,
            "mw_audit": mw_audit,
            "logprob_sum": logprob_sum,
            "kv": kv.snapshot(),
        }


def combine_conf(head_logit: float, decode_logprob: float) -> float:
    value = float(head_logit)
    p_head = (
        1.0 / (1.0 + math.exp(-value))
        if value >= 0.0
        else math.exp(value) / (1.0 + math.exp(value))
    )
    p_dec = float(math.exp(min(0.0, decode_logprob)))
    return min(p_head, p_dec)


def load_51m_runtime(package: ModelPackage, *, backend: str = "mlx-reference"):
    import mlx.core as mx
    import mlx.utils as xu
    import numpy as np

    _ensure(_ARCH_51)
    _ensure(_TRAIN)
    from architecture import NeedleZh, count_params
    from config import NeedleZhConfig
    from tokenizer import ZhTokenizerV1

    from quant_pack_51m import load_pack_file

    is_v2 = package.manifest.get("package_format") == "mei-model-package-v2"
    weight_spec = (
        package.manifest.get("tensor_container") or {}
        if is_v2
        else package.manifest.get("weights") or {}
    )
    weights = (package.path / str(weight_spec["file"])).resolve()
    tokenizer_path = (package.path / str(package.manifest["tokenizer"]["file"])).resolve()
    if not weights.is_file():
        raise SdkError("file_not_found", f"missing weights: {weights}")
    tok = ZhTokenizerV1(tokenizer_path)
    arch = package.manifest.get("architecture") or {}
    cfg = NeedleZhConfig.from_spec(_ARCH_51 / "spec" / "model.json")
    if str(arch.get("id") or cfg.architecture_id) != cfg.architecture_id:
        raise SdkError("package_invalid", f"architecture id mismatch: {arch.get('id')}")
    model = NeedleZh(cfg)
    mx.eval(model.parameters())
    fmt = str(weight_spec.get("format") or "")
    portable_head_blobs: dict[str, dict[str, Any]] = {
        "contrastive": {},
        "mw_disposition": {},
        "confidence": {},
        "narration_adapter": {},
    }
    portable_head_hashes: dict[str, str] = {}
    if fmt == "mei-cq-tensor-v2":
        from .cq2 import TensorContainer
        from checkpoint import flatten_params
        from cq2_metal import PackedCqMatrix

        container = TensorContainer.load(weights)
        current = flatten_params(model)
        loaded_lm: set[str] = set()
        packed_lm: dict[str, Any] = {}
        roles = {
            str(entry.get("name")): str(entry.get("role") or "lm")
            for entry in ((package.manifest.get("tensor_container") or {}).get("directory") or [])
        }
        for packed_name, entry in container.entries.items():
            role = roles.get(packed_name, entry.role or "lm")
            if role != "lm":
                values = np.asarray(
                    container.dequantize(packed_name), dtype=np.float32
                ).reshape(entry.shape)
                portable_head_blobs.setdefault(role, {})[packed_name] = mx.array(values)
                continue
            candidates = [packed_name]
            for prefix in ("lm.", "model."):
                if packed_name.startswith(prefix):
                    candidates.append(packed_name[len(prefix) :])
            target_name = next((name for name in candidates if name in current), None)
            if target_name is None:
                raise SdkError("package_invalid", f"unknown LM tensor {packed_name}")
            if tuple(entry.shape) != tuple(current[target_name].shape):
                raise SdkError("package_invalid", f"tensor shape mismatch {packed_name}")
            eligible_packed = (
                target_name == "embed.weight"
                or (
                    target_name.startswith("blocks.")
                    and ".attn." in target_name
                    and target_name.endswith(".weight")
                )
                or (
                    target_name.startswith("engrams.")
                    and (
                        target_name.endswith(".tables")
                        or target_name.endswith(".key_proj.weight")
                        or target_name.endswith(".value_proj.weight")
                    )
                )
            )
            if (
                backend == "mlx-cq2"
                and eligible_packed
                and entry.dtype in {"cq2", "cq4"}
            ):
                try:
                    packed_lm[target_name] = PackedCqMatrix.from_container(container, entry)
                except ValueError as exc:
                    raise SdkError("package_invalid", str(exc)) from exc
                loaded_lm.add(target_name)
                continue
            values = np.asarray(
                container.dequantize(packed_name), dtype=np.float32
            ).reshape(entry.shape)
            current[target_name] = mx.array(values)
            loaded_lm.add(target_name)
        missing = sorted(set(current) - loaded_lm)
        if missing:
            raise SdkError("package_invalid", f"portable container missing LM tensors: {missing[:4]}")
        model.update(xu.tree_unflatten(list(current.items())))
        mx.eval(model.parameters())
        if packed_lm:
            try:
                model.install_packed_cq2(packed_lm)
            except ValueError as exc:
                raise SdkError("package_invalid", str(exc)) from exc
        n_loaded = len(loaded_lm)
        for role, names in {
            role: sorted(name for name, declared_role in roles.items() if declared_role == role)
            for role in portable_head_blobs
        }.items():
            digest = hashlib.sha256()
            for name in names:
                entry = container.entries[name]
                digest.update(name.encode("utf-8"))
                for offset, nbytes in (
                    (entry.data_offset, entry.data_nbytes),
                    (entry.scales_offset, entry.scales_nbytes),
                    (entry.bit_map_offset, entry.bit_map_nbytes),
                ):
                    if nbytes:
                        digest.update(container.blob[offset : offset + nbytes])
            portable_head_hashes[role] = digest.hexdigest()
    elif fmt == "mei-q4-packed-v1":
        _header, arrays = load_pack_file(weights)
        from checkpoint import flatten_params

        current = flatten_params(model)
        n = 0
        for name, val in current.items():
            if name not in arrays:
                continue
            arr = arrays[name]
            if tuple(arr.shape) != tuple(val.shape):
                raise SdkError("package_invalid", f"tensor shape mismatch {name}")
            current[name] = mx.array(np.asarray(arr, dtype=np.float32))
            n += 1
        model.update(xu.tree_unflatten(list(current.items())))
        mx.eval(model.parameters())
        n_loaded = n
    else:
        from checkpoint import load_params

        report = load_params(model, weights, strict=False, return_report=True)
        n_loaded = (report or {}).get("n_loaded") if isinstance(report, dict) else report
    if count_params(model) != 51_463_797:
        raise SdkError("package_invalid", "51M param count mismatch after load")
    model.set_inference_backend(backend)
    contrastive = None
    mw_disposition = None
    conf_v2 = None
    narration_adapter = None
    head_path = package.path / "heads.npz"
    if any(portable_head_blobs.values()) or head_path.is_file():
        from heads import ConfidenceV2Head, ContrastiveHead, NarrationAdapterHead

        if any(portable_head_blobs.values()):
            contrastive_blob = portable_head_blobs.get("contrastive") or {}
            confidence_blob = portable_head_blobs.get("confidence") or {}
            mw_blob = portable_head_blobs.get("mw_disposition") or {}
            narration_blob = portable_head_blobs.get("narration_adapter") or {}
            contrastive_params = _extract_portable_head_params(
                contrastive_blob,
                ("tok_probes", "lay_probes", "proj.weight"),
                role="contrastive",
            ) if contrastive_blob else {}
            confidence_params = _extract_portable_head_params(
                confidence_blob,
                ("cell_probes", "proj.weight", "proj.bias"),
                role="confidence",
            ) if confidence_blob else {}
            mw_params = _extract_portable_head_params(
                mw_blob,
                ("proj.weight", "proj.bias"),
                role="mw_disposition",
            ) if mw_blob else {}
            narration_params = _extract_portable_head_params(
                narration_blob,
                ("down.weight", "up.weight"),
                role="narration_adapter",
            ) if narration_blob else {}
        else:
            try:
                blob = mx.load(str(head_path))
            except Exception:
                blob = {}
            contrastive_params = {}
            confidence_params = {}
            mw_params = {}
            narration_params = {}
        if contrastive_params:
            probes = int(contrastive_params["tok_probes"].shape[0])
            dim = int(contrastive_params["proj.weight"].shape[0])
            contrastive = ContrastiveHead(cfg.d_model, cfg.n_layers, dim=dim, probes=probes)
            mx.eval(contrastive.parameters())
            contrastive.update(xu.tree_unflatten(list(contrastive_params.items())))
            mx.eval(contrastive.parameters())
        elif not any(portable_head_blobs.values()) and any(k.startswith("contrastive") for k in blob):
            contrastive_meta = (package.manifest.get("heads") or {}).get("contrastive") or {}
            contrastive = ContrastiveHead(
                cfg.d_model,
                cfg.n_layers,
                dim=int(contrastive_meta.get("dim") or 128),
                probes=int(contrastive_meta.get("probes") or 4),
            )
            mx.eval(contrastive.parameters())
            contrastive.update(_unflatten_prefix(blob, "contrastive"))
            mx.eval(contrastive.parameters())
        if confidence_params:
            probes = int(confidence_params["cell_probes"].shape[0])
            conf_v2 = ConfidenceV2Head(cfg.d_model, probes=probes)
            mx.eval(conf_v2.parameters())
            conf_v2.update(xu.tree_unflatten(list(confidence_params.items())))
            mx.eval(conf_v2.parameters())
        elif not any(portable_head_blobs.values()):
            confidence_prefix = next(
                (prefix for prefix in ("confidence", "conf_v2", "conf") if any(k == prefix or k.startswith(prefix + ".") for k in blob)),
                None,
            )
            if confidence_prefix is not None:
                conf_v2 = ConfidenceV2Head(cfg.d_model)
                mx.eval(conf_v2.parameters())
                conf_v2.update(_unflatten_prefix(blob, confidence_prefix))
                mx.eval(conf_v2.parameters())
        if mw_params:
            mw_meta = (package.manifest.get("heads") or {}).get("mw_disposition") or {}
            mw_disposition = PortableMWDispositionHead(
                mw_params["proj.weight"],
                mw_params["proj.bias"],
                receipt_sha256=str(mw_meta.get("training_receipt_sha256") or ""),
            )
        if narration_params:
            narration_adapter = NarrationAdapterHead(cfg.d_model, cfg.vocab_size, rank=16)
            mx.eval(narration_adapter.parameters())
            narration_adapter.update(xu.tree_unflatten(list(narration_params.items())))
            mx.eval(narration_adapter.parameters())
    tokenizer_spec = package.manifest.get("tokenizer") or {}
    weights_spec = weight_spec
    heads_spec = package.manifest.get("heads") or {}
    head_hash = str(
        portable_head_hashes.get("contrastive")
        or heads_spec.get("sha256")
        or (heads_spec.get("artifact") or {}).get("sha256")
        or "missing"
    )
    model_hash = str(weights_spec.get("sha256") or "")
    tokenizer_hash = str(
        tokenizer_spec.get("sha256") or tokenizer_spec.get("vocab_sha256") or ""
    )
    index = None
    index_files = [
        str(row.get("path") or "")
        for row in (package.manifest.get("files") or [])
        if isinstance(row, dict) and row.get("role") == "tool_index"
    ]
    if len(index_files) > 1:
        raise SdkError("package_invalid", "portable runtime supports one canonical tool index")
    if index_files:
        try:
            index = ToolIndex.load(
                package.path / index_files[0],
                expected_serializer="mei-tool-call-serializer-v2",
            )
        except (OSError, TypeError, ValueError) as exc:
            raise SdkError("package_invalid", f"invalid portable tool index: {exc}") from exc
        if (
            index.model_hash != model_hash
            or index.head_hash != head_hash
            or index.tokenizer_hash != tokenizer_hash
        ):
            raise SdkError("package_hash_mismatch", "tool index model/head/tokenizer fingerprint mismatch")
    runtime = Runtime51M(
        model,
        tok,
        contrastive=contrastive,
        mw_disposition=mw_disposition,
        conf_v2=conf_v2,
        narration_adapter=narration_adapter,
        index=index,
        model_hash=model_hash,
        head_hash=head_hash,
        tokenizer_hash=tokenizer_hash,
        release_class=str(package.manifest.get("release_class") or "experimental"),
    )
    return runtime, {
        "n_loaded": n_loaded,
        "weights": str(weights),
        "tokenizer": str(tokenizer_path),
        "package_id": package.package_id,
        "backend": backend,
        "product": "mei-1.0-51m",
        "inference": True,
        "mw_disposition_head": mw_disposition is not None,
        "narration_adapter": narration_adapter is not None,
    }


def _unflatten_prefix(blob: dict, prefix: str):
    import mlx.utils as xu

    items = []
    for key, val in blob.items():
        if key == prefix or key.startswith(prefix + "."):
            rel = key[len(prefix) :].lstrip(".")
            items.append((rel or key, val))
    return xu.tree_unflatten(items) if items else {}


def _extract_portable_head_params(
    blob: dict[str, Any], expected: tuple[str, ...], *, role: str
) -> dict[str, Any]:
    """Map role-tagged portable tensor names onto the local head parameter tree."""

    out: dict[str, Any] = {}
    used: set[str] = set()
    for parameter in expected:
        matches = [
            name
            for name in blob
            if name == parameter or name.endswith("." + parameter)
        ]
        if len(matches) != 1:
            raise SdkError(
                "package_invalid",
                f"portable {role} head requires exactly one {parameter} tensor",
            )
        name = matches[0]
        out[parameter] = blob[name]
        used.add(name)
    unexpected = sorted(set(blob) - used)
    if unexpected:
        raise SdkError(
            "package_invalid", f"unknown portable {role} tensor: {unexpected[0]}"
        )
    return out


def complete_51m(runtime: Runtime51M, request: dict[str, Any]) -> dict[str, Any]:
    query = str(request.get("query") or "")
    oracle = request.get("oracle_tools")
    catalog = list(request.get("catalog") or [])
    if oracle is not None:
        selected = list(oracle)
        if len(selected) > MAX_SELECTED_TOOLS:
            raise SdkError("too_many_tools")
    else:
        selected = runtime.search_top_k(query, catalog, k=MAX_SELECTED_TOOLS) if catalog else []
        if not selected:
            selected = catalog[:MAX_SELECTED_TOOLS]
    rendered = render_request(request, selected, already_normalized=True)
    sink_ids = runtime.tokenizer.encode(rendered["sink"], add_bos=True, add_eos=False)
    if len(sink_ids) > 1024:
        return {
            "selected_tools": [str(t.get("name") or "") for t in selected],
            "text": None,
            "validated": {
                "ok": False,
                "refuse": True,
                "function_calls": [],
                "error": "tool_schema_budget_exceeded",
                "gates": [{"gate": "retrieval", "ok": True}, {"gate": "sink_budget", "ok": False}],
            },
            "confidence": None,
            "execution": "refuse",
            "prompt_tokens": len(sink_ids),
            "output_tokens": 0,
            "timings": {},
            "decode": {"mode": str(request.get("decode_mode") or "constrained")},
            "error": "tool_schema_budget_exceeded",
        }
    ordinary_ids = runtime.tokenizer.encode(rendered["ordinary"], add_bos=False, add_eos=False)
    ids = sink_ids + ordinary_ids[-256:]
    t_prefill = time.perf_counter()
    decode = runtime.greedy(
        ids,
        tools=selected,
        max_new=min(128, int(request.get("max_new") or 128)),
        decode_mode=str(request.get("decode_mode") or "constrained"),
        sink_ids=sink_ids,
        ordinary_ids=ordinary_ids,
        benchmark_ignore_eos=bool(request.get("_kernel_benchmark_ignore_eos", False)),
    )
    prefill_ms = (time.perf_counter() - t_prefill) * 1000 - float(decode.get("decode_ms") or 0)
    validated = validate_generated_call(
        decode.get("text") or "",
        tools=selected,
        request=_runtime_gate_request(
            rendered["request"],
            decode.get("mw_disposition") or {
                "decision": "stop",
                "source": "deterministic-policy",
            },
            release_class=runtime.release_class,
        ),
        confidence=decode.get("confidence"),
        enforce_confidence=True,
    )
    validated["mw_audit"] = decode.get("mw_audit")
    return {
        "selected_tools": [str(t.get("name") or "") for t in selected],
        "text": decode.get("text"),
        "validated": validated,
        "confidence": validated.get("confidence_value"),
        "execution": validated.get("execution"),
        "prompt_tokens": len(ids),
        "output_tokens": decode.get("n_out"),
        "timings": {"prefill_ms": prefill_ms, "decode_ms": decode.get("decode_ms")},
        "decode": {"mode": str(request.get("decode_mode") or "constrained")},
        "kv": decode.get("kv"),
        "mw_disposition": decode.get("mw_disposition"),
        "error": validated.get("error"),
    }


def _runtime_gate_request(
    request: dict[str, Any],
    disposition: dict[str, Any],
    *,
    release_class: str,
) -> dict[str, Any]:
    """Bind a trusted MW head result without accepting caller spoofing.

    ``mw`` remains available only as an explicit protocol-test override for an
    experimental package.  Candidate/release packages retain both values so
    the canonical validator deterministically returns ``mw_invalid`` rather
    than silently ignoring the attempted override.
    """

    out = dict(request)
    out.pop("mw_disposition", None)
    if out.get("mw") is not None and release_class == "experimental":
        return out
    out["mw_disposition"] = dict(disposition)
    return out
