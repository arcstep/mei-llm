from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from .registry import Registry


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _cycle_list(registry: Registry) -> int:
    print("cycle\ttarget_tokens\tactual_tokens\tstatus")
    for item in registry.cycles()["cycles"]:
        print(
            f"{item['cycle_id']}\t{item['target_exposure_tokens']}\t"
            f"{item.get('actual_exposure_tokens', '-')}\t{item['status']}"
        )
    return 0


def _model_status(registry: Registry) -> int:
    current = json.loads((registry.root / "CURRENT.json").read_text(encoding="utf-8"))
    model = registry.models()["models"][0]
    model_factory = json.loads(
        (registry.root / "model-factory" / "FACTORY.json").read_text(encoding="utf-8")
    )
    resolved_current = {
        key: str(registry.resolve(value)) if isinstance(value, str) else None
        for key, value in current.items()
        if key in {"architecture", "base", "corpus", "runtime", "sft", "tokenizer", "training"}
    }
    _print_json(
        {
            "model": model,
            "model_factory": model_factory,
            "current": current,
            "current_sha256": registry.current_sha256(),
            "resolved_current_base": str(registry.resolve(current["base"])),
            "resolved_current_paths": resolved_current,
            "cycles": registry.cycles()["cycles"],
        }
    )
    return 0


def _cycle_show(registry: Registry, cycle_id: str) -> int:
    _print_json(registry.cycle(cycle_id))
    return 0


def _cycle_compare(registry: Registry, left: str, right: str) -> int:
    a, b = registry.cycle(left), registry.cycle(right)
    metrics_a, metrics_b = a.get("metrics", {}), b.get("metrics", {})
    rows = []
    for key in sorted(set(metrics_a) | set(metrics_b)):
        av, bv = metrics_a.get(key), metrics_b.get(key)
        delta = bv - av if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else None
        rows.append({"metric": key, left: av, right: bv, "delta": delta})
    _print_json({"left": left, "right": right, "metrics": rows})
    return 0


def _artifact_resolve(registry: Registry, value: str, verify: bool) -> int:
    path = registry.resolve(value)
    result = {"input": value, "resolved": str(path), "exists": path.exists()}
    if verify and path.exists():
        manifest_path = next(
            (
                parent / "ASSETS.json"
                for parent in (path if path.is_dir() else path.parent, *path.parents)
                if (parent / "ASSETS.json").is_file()
            ),
            None,
        )
        if manifest_path is None:
            result["integrity"] = "not_manifested"
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            cycle_root = manifest_path.parent
            selected = []
            for item in manifest.get("assets", []):
                asset_path = cycle_root / item["path"]
                if path == asset_path or (path.is_dir() and asset_path.is_relative_to(path)):
                    selected.append((item, asset_path))
            errors = []
            for item, asset_path in selected:
                if not asset_path.is_file():
                    errors.append(f"missing:{item['path']}")
                    continue
                if asset_path.stat().st_size != item["bytes"]:
                    errors.append(f"size:{item['path']}")
                    continue
                digest = hashlib.sha256()
                with asset_path.open("rb") as handle:
                    for block in iter(lambda: handle.read(8 << 20), b""):
                        digest.update(block)
                if digest.hexdigest() != item["sha256"]:
                    errors.append(f"sha256:{item['path']}")
            result.update(
                {
                    "asset_manifest": str(manifest_path),
                    "verified_assets": len(selected),
                    "integrity": "verified" if selected and not errors else "failed",
                    "errors": errors or ([] if selected else ["no_manifest_entry_for_path"]),
                }
            )
    _print_json(result)
    if not verify:
        return 0
    return 0 if result["exists"] and result.get("integrity") != "failed" else 2


def _corpus_show(registry: Registry, cycle_id: str) -> int:
    cycle = registry.cycle(cycle_id)
    path = registry.root / cycle["corpus_contract"]
    _print_json(json.loads(path.read_text(encoding="utf-8")))
    return 0


def _forward(registry: Registry, script: str, arguments: list[str]) -> int:
    path = registry.root / script
    if not path.is_file():
        raise FileNotFoundError(path)
    return subprocess.run([sys.executable, str(path), *arguments], cwd=registry.root).returncode


def _forward_module(registry: Registry, module: str, arguments: list[str]) -> int:
    env = os.environ.copy()
    roots = [
        str(registry.root / "src"),
        str(registry.root / "model-factory"),
        str(registry.root / "platform/python-sdk"),
    ]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=registry.root,
        env=env,
    ).returncode


def _cycle_action(registry: Registry, action: str, run_id: str, rest: list[str]) -> int:
    return _forward_module(
        registry,
        "orchestration.lifecycle_51m",
        [action, "--run-id", run_id, *rest],
    )


def _corpus_action(registry: Registry, action: str, rest: list[str]) -> int:
    command = {
        "plan": "create-worklist",
        "build": "compile-shard",
        "audit": "audit-shard",
        "release": "freeze-release",
    }[action]
    return _forward(
        registry,
        "corpus-factory/generators/factory_51m.py",
        [command, *rest],
    )


def _corpus_diff(registry: Registry, left: Path, right: Path) -> int:
    a = json.loads(left.read_text(encoding="utf-8"))
    b = json.loads(right.read_text(encoding="utf-8"))
    keys = sorted(set(a) | set(b))
    _print_json(
        {
            "left": str(left),
            "right": str(right),
            "changed": [key for key in keys if a.get(key) != b.get(key)],
            "left_only": [key for key in keys if key in a and key not in b],
            "right_only": [key for key in keys if key in b and key not in a],
        }
    )
    return 0


def _release_prepare(args: argparse.Namespace) -> int:
    _print_json(
        {
            "schema": "mei-llm-release-v1",
            "release_id": args.release_id,
            "model_id": "mei-1.0-51m",
            "cycle_id": args.cycle_id,
            "package_uri": args.package_uri,
            "package_sha256": args.package_sha256,
            "status": "candidate",
            "clearance": False,
        }
    )
    return 0


def _release_verify(path: Path) -> int:
    value = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "release_id",
        "model_id",
        "cycle_id",
        "package_uri",
        "package_sha256",
        "status",
        "clearance",
    }
    errors = [f"missing:{key}" for key in sorted(required - set(value))]
    if value.get("schema") != "mei-llm-release-v1":
        errors.append("schema")
    if value.get("status") in {"publishable", "published"} and value.get("clearance") is not True:
        errors.append("publishable_without_clearance")
    _print_json({"manifest": str(path), "ok": not errors, "errors": errors})
    return 0 if not errors else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mei", description="MEI LLM five-domain control plane")
    top = parser.add_subparsers(dest="domain", required=True)

    model = top.add_parser("model")
    model_sub = model.add_subparsers(dest="action", required=True)
    model_sub.add_parser("status")

    cycle = top.add_parser("cycle")
    cycle_sub = cycle.add_subparsers(dest="action", required=True)
    cycle_sub.add_parser("list")
    show = cycle_sub.add_parser("show")
    show.add_argument("cycle_id")
    compare = cycle_sub.add_parser("compare")
    compare.add_argument("left")
    compare.add_argument("right")
    for action in ("plan", "run"):
        command = cycle_sub.add_parser(action)
        command.add_argument("--run-id", required=True)
        command.add_argument("arguments", nargs=argparse.REMAINDER)

    corpus = top.add_parser("corpus")
    corpus_sub = corpus.add_subparsers(dest="action", required=True)
    corpus_show = corpus_sub.add_parser("show")
    corpus_show.add_argument("cycle_id")
    for action in ("plan", "build", "audit", "release"):
        command = corpus_sub.add_parser(action)
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    corpus_diff = corpus_sub.add_parser("diff")
    corpus_diff.add_argument("left", type=Path)
    corpus_diff.add_argument("right", type=Path)

    artifact = top.add_parser("artifact")
    artifact_sub = artifact.add_subparsers(dest="action", required=True)
    for action in ("resolve", "verify"):
        command = artifact_sub.add_parser(action)
        command.add_argument("value")

    platform = top.add_parser("platform")
    platform_sub = platform.add_subparsers(dest="action", required=True)
    platform_test = platform_sub.add_parser("test")
    platform_test.add_argument("--scope", choices=("python-browser-wasm", "extended"), default="python-browser-wasm")
    platform_bench = platform_sub.add_parser("benchmark")
    platform_bench.add_argument("arguments", nargs=argparse.REMAINDER)

    release = top.add_parser("release")
    release_sub = release.add_subparsers(dest="action", required=True)
    release_prepare = release_sub.add_parser("prepare")
    release_prepare.add_argument("--cycle-id", required=True)
    release_prepare.add_argument("--release-id", required=True)
    release_prepare.add_argument("--package-uri", required=True)
    release_prepare.add_argument("--package-sha256", required=True)
    release_verify = release_sub.add_parser("verify")
    release_verify.add_argument("manifest", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry = Registry.open()
    if (args.domain, args.action) == ("model", "status"):
        return _model_status(registry)
    if (args.domain, args.action) == ("cycle", "list"):
        return _cycle_list(registry)
    if (args.domain, args.action) == ("cycle", "show"):
        return _cycle_show(registry, args.cycle_id)
    if (args.domain, args.action) == ("cycle", "compare"):
        return _cycle_compare(registry, args.left, args.right)
    if args.domain == "cycle" and args.action in {"plan", "run"}:
        return _cycle_action(registry, args.action, args.run_id, args.arguments)
    if (args.domain, args.action) == ("corpus", "show"):
        return _corpus_show(registry, args.cycle_id)
    if args.domain == "corpus" and args.action in {"plan", "build", "audit", "release"}:
        return _corpus_action(registry, args.action, args.arguments)
    if (args.domain, args.action) == ("corpus", "diff"):
        return _corpus_diff(registry, args.left, args.right)
    if args.domain == "artifact":
        return _artifact_resolve(registry, args.value, args.action == "verify")
    if (args.domain, args.action) == ("platform", "test"):
        return _forward(registry, "platform/_shared/tools/run_gates.py", ["--scope", args.scope])
    if (args.domain, args.action) == ("platform", "benchmark"):
        return _forward(registry, "platform/_shared/tools/bench_mlx_complete.py", args.arguments)
    if (args.domain, args.action) == ("release", "prepare"):
        return _release_prepare(args)
    if (args.domain, args.action) == ("release", "verify"):
        return _release_verify(args.manifest)
    raise AssertionError("unreachable")
