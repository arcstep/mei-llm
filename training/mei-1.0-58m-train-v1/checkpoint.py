"""Named npz save/load for Needle-zh parameters and full train state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.utils as xu


def flatten_tree(tree) -> dict[str, mx.array]:
    leaves = xu.tree_flatten(tree)
    if isinstance(leaves, tuple):
        leaves = leaves[0]
    out: dict[str, mx.array] = {}
    for i, item in enumerate(leaves):
        if isinstance(item, tuple) and len(item) == 2:
            key, val = item
            out[str(key)] = val
        else:
            out[f"p{i}"] = item
    return out


def flatten_params(model) -> dict[str, mx.array]:
    return flatten_tree(model.parameters())


def _atomic_savez(dest: Path, arrays: dict) -> None:
    """Write a zip of .npy arrays. Avoid mx.savez(**kwargs) which caps at 1024 keys."""
    import io
    import zipfile

    import numpy as np

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.stem + ".saving.npz")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for key, val in arrays.items():
            arr = np.asarray(val)
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            zf.writestr(f"{key}.npy", buf.getvalue())
    tmp.replace(dest)


def save_params(model, path: Path) -> None:
    path = Path(path)
    _atomic_savez(path, flatten_params(model))


def load_params(
    model,
    path: Path,
    *,
    strict: bool = True,
    allow_missing_prefixes: tuple[str, ...] = (),
    return_report: bool = False,
):
    path = Path(path)
    blob = mx.load(str(path))
    current = flatten_params(model)
    n = 0
    missing = []
    unexpected = [k for k in blob if k not in current]
    skipped = []
    for key, val in current.items():
        if key in blob and tuple(blob[key].shape) == tuple(val.shape):
            current[key] = blob[key]
            n += 1
        else:
            if allow_missing_prefixes and any(key.startswith(p) for p in allow_missing_prefixes):
                skipped.append(key)
                continue
            missing.append(key)
    if strict and missing:
        raise KeyError(f"checkpoint missing {len(missing)} tensors, e.g. {missing[:5]}")
    tree = xu.tree_unflatten(list(current.items()))
    model.update(tree)
    mx.eval(model.parameters())
    if return_report:
        return {
            "n_loaded": n,
            "missing": missing,
            "skipped_new": skipped,
            "unexpected": unexpected,
        }
    return n


def save_train_state(path: Path, model, optimizer, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {f"p.{k}": v for k, v in flatten_params(model).items()}
    payload.update({f"o.{k}": v for k, v in flatten_tree(optimizer.state).items()})
    _atomic_savez(path, payload)
    meta_path = path.with_suffix(".meta.json")
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
    meta_tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    meta_tmp.replace(meta_path)


CURRICULUM_FLEX_META = ("seq_len", "batch_size", "grad_accum")
CONTINUATION_FLEX_META = (
    "corpus_sha256",
    "schedule_sha256",
    "manifest_sha256",
    "lr_horizon_tokens",
    "seq_len",
    "batch_size",
    "grad_accum",
)
STRICT_META_KEYS = (
    "tokenizer_sha256",
    "corpus_sha256",
    "seq_len",
    "manifest_sha256",
    "lr_horizon_tokens",
    "batch_size",
    "grad_accum",
    "precision",
    "shuffle_seed",
    "architecture_id",
    "architecture_sha256",
    "params",
)
LOAD_MODES = {"strict", "weights_only", "curriculum", "continuation"}


def validate_expected_meta(
    meta: dict[str, Any],
    expected_meta: dict[str, Any] | None,
    mode: str,
) -> None:
    if not expected_meta:
        return
    if mode == "weights_only":
        return
    keys = list(expected_meta.keys())
    for key in STRICT_META_KEYS:
        if key not in keys:
            keys.append(key)
    flex = set()
    if mode == "curriculum":
        flex = set(CURRICULUM_FLEX_META)
    elif mode == "continuation":
        flex = set(CONTINUATION_FLEX_META)
    for key in keys:
        if key not in expected_meta:
            continue
        if key in flex:
            continue
        if key not in meta:
            raise ValueError(f"train state missing {key}")
        if str(meta[key]) != str(expected_meta[key]):
            raise ValueError(f"train state {key} mismatch: ckpt={meta[key]} expected={expected_meta[key]}")


def load_train_state(
    path: Path,
    model,
    optimizer,
    *,
    strict: bool = True,
    expected_meta: dict[str, Any] | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    path = Path(path)
    resolved = mode or ("strict" if strict else "weights_only")
    if resolved not in LOAD_MODES:
        raise ValueError(f"unsupported load mode {resolved}")
    load_opt = resolved in {"strict", "curriculum", "continuation"}
    check_meta = resolved in {"strict", "curriculum", "continuation"}
    blob = mx.load(str(path))
    p_cur = flatten_params(model)
    p_next = {}
    missing_p = []
    for key, val in p_cur.items():
        bkey = f"p.{key}"
        if bkey in blob and tuple(blob[bkey].shape) == tuple(val.shape):
            p_next[key] = blob[bkey]
        else:
            missing_p.append(key)
    if missing_p:
        raise KeyError(f"train state missing params {missing_p[:5]}")
    extra_p = [k[2:] for k in blob if k.startswith("p.") and k[2:] not in p_cur]
    if load_opt and extra_p:
        raise KeyError(f"train state extra params {extra_p[:5]}")
    model.update(xu.tree_unflatten(list((p_next or p_cur).items())))
    o_blob = {k[2:]: blob[k] for k in blob if k.startswith("o.")}
    if load_opt and not o_blob:
        raise KeyError("train state missing optimizer tensors")
    if load_opt and o_blob:
        optimizer.state = xu.tree_unflatten(list(o_blob.items()))
        mx.eval(model.parameters(), optimizer.state)
    else:
        mx.eval(model.parameters())
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if check_meta:
        validate_expected_meta(meta, expected_meta, resolved)
    return meta
