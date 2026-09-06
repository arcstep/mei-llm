from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# 控制面调用工厂：先把 model-factory 挂上 sys.path，再 import 编排模块
_FACTORY = Path(__file__).resolve().parents[2] / "src/model-factory"
if str(_FACTORY) not in sys.path:
    sys.path.insert(0, str(_FACTORY))

import orchestration.cycle_control as cycle_control
import orchestration.phase_binding as phase_binding
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
        (registry.root / "src/model-factory" / "FACTORY.json").read_text(encoding="utf-8")
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
        str(registry.root / "src/model-factory"),
        str(registry.root / "src/platform/python-sdk"),
    ]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    return subprocess.run(
        [sys.executable, "-m", module, *arguments],
        cwd=registry.root,
        env=env,
    ).returncode


def _training_arguments(arguments: list[str], *, action: str) -> list[str]:
    flag = "--confirm-training"
    confirmed = os.environ.get("MEI_TRAINING_CONFIRMED") == "1" or flag in arguments
    if not confirmed:
        raise SystemExit(f"{action} refused: pass {flag} explicitly")
    return [value for value in arguments if value != flag]


def _corpus_action(registry: Registry, action: str, rest: list[str]) -> int:
    command = {
        "plan": "create-worklist",
        "build": "compile-shard",
        "audit": "audit-shard",
        "release": "freeze-release",
    }[action]
    return _forward(
        registry,
        "src/corpus-factory/generators/factory_51m.py",
        [command, *rest],
    )


def _corpus_control(
    registry: Registry, family: str, action: str, rest: list[str]
) -> int:
    if family == "source":
        script = "src/corpus-factory/sources/source_manager.py"
    else:
        script = "src/corpus-factory/quality/audit.py"
    return _forward(registry, script, [action, *rest])


def _pipeline(registry: Registry, pipeline_id: str) -> dict:
    value = json.loads(
        (registry.root / "src/model-factory/contracts/PIPELINES.json").read_text(
            encoding="utf-8"
        )
    )
    matches = [
        row for row in value["pipelines"] if row.get("pipeline_id") == pipeline_id
    ]
    if len(matches) != 1 or matches[0].get("status") != "current":
        raise RuntimeError(f"no unique current pipeline: {pipeline_id}")
    return matches[0]


def _cpt_action(
    registry: Registry, action: str, run_id: str | None, rest: list[str]
) -> int:
    pipeline = _pipeline(registry, "mei-51m-cpt-lifecycle-v1")
    arguments = [action]
    if run_id is not None:
        arguments.extend(["--run-id", run_id])
    if action in {"plan", "run", "resume"}:
        arguments.extend(["--track", pipeline["track"]])
    arguments.extend(rest)
    return _forward_module(registry, pipeline["entrypoint"], arguments)


def _product_doctor(registry: Registry) -> int:
    pipeline = _pipeline(registry, "mei-51m-adaptive-productization-v5")
    recipe_path = registry.root / pipeline["recipe"]
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    entrypoint = (
        registry.root
        / "src/model-factory"
        / f"{pipeline['entrypoint'].replace('.', '/')}.py"
    )
    errors = []
    if not entrypoint.is_file():
        errors.append(f"missing entrypoint: {entrypoint}")
    if recipe.get("id") != pipeline["pipeline_id"]:
        errors.append("pipeline/recipe id mismatch")
    if recipe.get("stages") != pipeline.get("stages"):
        errors.append("pipeline/recipe stage mismatch")
    if recipe.get("changes_weights") != pipeline.get("changes_weights"):
        errors.append("pipeline/recipe weight-change scope mismatch")
    if recipe.get("child_pipelines") != pipeline.get("children"):
        errors.append("pipeline/recipe child pipeline mismatch")
    for child_id in pipeline.get("children", []):
        try:
            child = _pipeline(registry, child_id)
            child_recipe_path = registry.root / child["recipe"]
            child_recipe = json.loads(
                child_recipe_path.read_text(encoding="utf-8")
            )
            if child_recipe.get("phase") != child.get("phase"):
                errors.append(f"child phase mismatch: {child_id}")
            if child_recipe.get("entrypoint") != child.get("entrypoint"):
                errors.append(f"child entrypoint mismatch: {child_id}")
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            errors.append(f"child pipeline invalid {child_id}: {error}")
    for key, expected in (
        ("output_token_reserve", 128),
        ("package_bytes_max", 18 * 1024 * 1024),
        ("wasm_heap_bytes_max", 96 * 1024 * 1024),
    ):
        if recipe.get(key) != expected:
            errors.append(f"{key} mismatch")
    _print_json(
        {
            "ok": not errors,
            "pipeline": pipeline,
            "recipe": str(recipe_path),
            "entrypoint": str(entrypoint),
            "errors": errors,
            "current_sha256": registry.current_sha256(),
        }
    )
    return 0 if not errors else 2


def _product_status(run_dir: Path) -> int:
    run_dir = run_dir.resolve()
    files = {}
    for name in ("plan.json", "progress.json", "STATUS.json"):
        path = run_dir / name
        if path.is_file():
            files[name] = {
                "path": str(path),
                "value": json.loads(path.read_text(encoding="utf-8")),
            }
    receipts = sorted(run_dir.glob("stages/*/receipt.json"))
    _print_json(
        {
            "run_dir": str(run_dir),
            "exists": run_dir.is_dir(),
            "files": files,
            "stage_receipts": [
                {
                    "stage": path.parent.name,
                    "terminal_status": json.loads(
                        path.read_text(encoding="utf-8")
                    ).get("terminal_status"),
                }
                for path in receipts
            ],
        }
    )
    return 0 if run_dir.is_dir() else 2


def _product_action(registry: Registry, action: str, rest: list[str]) -> int:
    pipeline = _pipeline(registry, "mei-51m-adaptive-productization-v5")
    module = pipeline["entrypoint"]
    if action in {"run", "resume", "adopt"}:
        if any(value in {"--help", "-h"} for value in rest):
            return _forward_module(registry, module, ["--help"])
        compatibility_flag = "--legacy-compatibility-run"
        if compatibility_flag not in rest:
            raise SystemExit(
                "direct productization execution is compatibility-only; "
                "use qat/sft-alignment/model-evaluation/runtime-release bindings, "
                "or pass --legacy-compatibility-run for a historical run"
            )
        rest = [value for value in rest if value != compatibility_flag]
    if action == "plan":
        if "--help" in rest or "-h" in rest:
            return _forward_module(registry, module, ["--help"])
        if "--validate-inputs" in rest:
            validated_rest = [value for value in rest if value != "--validate-inputs"]
            return _forward_module(registry, module, [*validated_rest, "--dry-run"])
        recipe_path = registry.root / pipeline["recipe"]
        _print_json(
            {
                "schema": "mei-51m-productization-control-plan-v1",
                "pipeline": pipeline,
                "recipe": json.loads(recipe_path.read_text(encoding="utf-8")),
                "forwarded_arguments": rest,
                "input_validation": "deferred; pass --validate-inputs to run full preflight",
                "writes_artifacts": False,
                "changes_weights": False,
                "current_sha256": registry.current_sha256(),
            }
        )
        return 0
    if action == "resume":
        return _forward_module(registry, module, [*rest, "--resume"])
    if action in {"run", "adopt"}:
        return _forward_module(registry, module, rest)
    if action == "compare":
        return _forward_module(
            registry,
            "evaluation.alignment.compare_longitudinal_products_51m",
            rest,
        )
    if action == "final-audit":
        return _forward_module(
            registry, "release.final_audit_51m", rest
        )
    raise AssertionError(action)


def _base_action(registry: Registry, action: str, rest: list[str]) -> int:
    command = {
        "register": "register-base-candidate",
        "propose-freeze": "propose-freeze",
    }[action]
    return _forward_module(
        registry,
        "release.base_candidate_51m",
        [command, *rest],
    )


def _phase_status(registry: Registry, binding_path: Path) -> int:
    report = phase_binding.verify(
        registry, binding_path, require_inputs=False
    )
    binding = phase_binding.load_json(binding_path)
    run_value = (binding.get("outputs") or {}).get("run_dir")
    run_dir = (
        phase_binding.output_path(registry, binding["cycle_id"], str(run_value))
        if run_value
        else None
    )
    progress = None
    if run_dir is not None and (run_dir / "progress.json").is_file():
        progress = phase_binding.load_json(run_dir / "progress.json")
    _print_json(
        {
            "binding": report,
            "run_dir": str(run_dir) if run_dir else None,
            "progress": progress,
        }
    )
    return 0 if report["ok"] else 2


def _phase_action(
    registry: Registry,
    domain: str,
    action: str,
    binding_path: Path,
    *,
    confirmed: bool,
) -> int:
    expected_phase = {
        "qat": "qat",
        "sft-alignment": "sft_alignment",
        "model-evaluation": "model_evaluation",
        "runtime-release": "runtime_release",
    }[domain]
    if action == "status":
        return _phase_status(registry, binding_path)
    require_inputs = action != "doctor"
    report = phase_binding.verify(
        registry, binding_path, require_inputs=require_inputs
    )
    if report["phase"] != expected_phase:
        report["ok"] = False
        report["errors"].append(
            f"binding phase {report['phase']} cannot run through {domain}"
        )
    if action == "doctor":
        _print_json(report)
        return 0 if report["ok"] else 2
    if not report["ok"]:
        _print_json(report)
        return 2
    invocation = phase_binding.invocation(
        registry,
        binding_path,
        resume=action == "resume",
        dry_run=action == "plan",
    )
    if action == "plan":
        invocation = {**invocation, "execution_started": False}
        _print_json(invocation)
        return 0
    if report["pipeline"].get("changes_weights") is True and not (
        confirmed or os.environ.get("MEI_TRAINING_CONFIRMED") == "1"
    ):
        raise SystemExit(f"{domain} {action} refused: pass --confirm-training")
    phase_binding.record_for_run(registry, binding_path)
    env = os.environ.copy()
    env.update(invocation["environment"])
    return subprocess.run(
        invocation["command"],
        cwd=registry.root,
        env=env,
    ).returncode


def _phase_template(
    registry: Registry,
    domain: str,
    *,
    cycle_id: str,
    pipeline_id: str,
) -> int:
    value = phase_binding.template(
        registry, cycle_id=cycle_id, pipeline_id=pipeline_id
    )
    expected_phase = {
        "qat": "qat",
        "sft-alignment": "sft_alignment",
        "model-evaluation": "model_evaluation",
        "runtime-release": "runtime_release",
    }[domain]
    if value["phase"] != expected_phase:
        raise SystemExit(
            f"{pipeline_id} belongs to {value['phase']}, not {expected_phase}"
        )
    _print_json(value)
    return 0


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
    parser = argparse.ArgumentParser(
        prog="mei", description="MEI LLM lifecycle control plane"
    )
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
    init = cycle_sub.add_parser("init")
    init.add_argument("--cycle-id", required=True)
    init.add_argument("--out", type=Path)
    for action in ("plan", "resume", "status"):
        command = cycle_sub.add_parser(action)
        command.add_argument("--cycle-id", required=True)
    cycle_sub.add_parser("verify")

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
    for family, actions in (
        (
            "source",
            ("inventory", "plan-mix", "download-hq", "download", "admit", "freeze-pool"),
        ),
        (
            "evaluate",
            (
                "audit-source",
                "audit-structured",
                "audit-synthetic",
                "audit-sft",
                "decide-reuse",
                "compare",
            ),
        ),
    ):
        command = corpus_sub.add_parser(family)
        child = command.add_subparsers(dest="corpus_control_action", required=True)
        for action in actions:
            leaf = child.add_parser(action, add_help=False)
            leaf.add_argument("arguments", nargs=argparse.REMAINDER)

    cpt = top.add_parser("cpt")
    cpt_sub = cpt.add_subparsers(dest="action", required=True)
    cpt_sub.add_parser("doctor")
    init = cpt_sub.add_parser("init")
    init.add_argument("--run-id", required=True)
    init.add_argument("--cycle-id")
    init.add_argument("--corpus-dir", type=Path, required=True)
    init.add_argument("--target-exposure", type=int, required=True)
    init.add_argument("--resume-checkpoint", type=Path)
    for action in ("plan", "run", "resume", "status"):
        command = cpt_sub.add_parser(action)
        command.add_argument("--run-id", required=True)
        if action != "status":
            command.add_argument(
                "--until",
                choices=("corpus_freeze", "cpt_readiness", "cpt", "cpt_gate"),
            )
        if action in {"run", "resume"}:
            command.add_argument("--confirm-training", action="store_true")

    base = top.add_parser("base")
    base_sub = base.add_subparsers(dest="action", required=True)
    for action in ("register", "propose-freeze"):
        command = base_sub.add_parser(action, add_help=False)
        command.add_argument("arguments", nargs=argparse.REMAINDER)

    product = top.add_parser("productization")
    product_sub = product.add_subparsers(dest="action", required=True)
    product_sub.add_parser("doctor")
    status = product_sub.add_parser("status")
    status.add_argument("--run-dir", type=Path, required=True)
    for action in ("plan", "run", "resume", "adopt", "compare", "final-audit"):
        command = product_sub.add_parser(action, add_help=False)
        command.add_argument("arguments", nargs=argparse.REMAINDER)

    for domain in ("qat", "sft-alignment", "model-evaluation", "runtime-release"):
        phase = top.add_parser(domain)
        phase_sub = phase.add_subparsers(dest="action", required=True)
        template = phase_sub.add_parser("template")
        template.add_argument("--cycle-id", required=True)
        template.add_argument("--pipeline-id", required=True)
        for action in ("doctor", "plan", "run", "resume", "status"):
            command = phase_sub.add_parser(action)
            command.add_argument("--binding", type=Path, required=True)
            if action in {"run", "resume"}:
                command.add_argument("--confirm-training", action="store_true")

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


def parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    values = list(sys.argv[1:] if argv is None else argv)
    forward_at: int | None = None
    if values[:2] and values[0] == "corpus":
        if len(values) >= 3 and values[1] in {"source", "evaluate"}:
            forward_at = 3
        elif len(values) >= 2 and values[1] in {"plan", "build", "audit", "release"}:
            forward_at = 2
    elif len(values) >= 2 and values[0] == "base":
        forward_at = 2
    elif (
        len(values) >= 2
        and values[0] == "productization"
        and values[1] not in {"doctor", "status"}
    ):
        forward_at = 2
    if forward_at is not None and len(values) > forward_at:
        values.insert(forward_at, "--")
    args = build_parser().parse_args(values)
    if hasattr(args, "arguments") and args.arguments[:1] == ["--"]:
        args.arguments = args.arguments[1:]
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_cli_args(argv)
    registry = Registry.open()
    if (args.domain, args.action) == ("model", "status"):
        return _model_status(registry)
    if (args.domain, args.action) == ("cycle", "list"):
        return _cycle_list(registry)
    if (args.domain, args.action) == ("cycle", "show"):
        return _cycle_show(registry, args.cycle_id)
    if (args.domain, args.action) == ("cycle", "compare"):
        return _cycle_compare(registry, args.left, args.right)
    if (args.domain, args.action) == ("cycle", "init"):
        _print_json(cycle_control.init_proposal(registry, args.cycle_id, out=args.out))
        return 0
    if (args.domain, args.action) == ("cycle", "plan"):
        _print_json(cycle_control.plan(registry, args.cycle_id))
        return 0
    if args.domain == "cycle" and args.action in {"resume", "status"}:
        _print_json(cycle_control.status(registry, args.cycle_id))
        return 0
    if (args.domain, args.action) == ("cycle", "verify"):
        report = cycle_control.verify(registry)
        _print_json(report)
        return 0 if report["ok"] else 2
    if (args.domain, args.action) == ("corpus", "show"):
        return _corpus_show(registry, args.cycle_id)
    if args.domain == "corpus" and args.action in {"plan", "build", "audit", "release"}:
        return _corpus_action(registry, args.action, args.arguments)
    if (args.domain, args.action) == ("corpus", "diff"):
        return _corpus_diff(registry, args.left, args.right)
    if args.domain == "corpus" and args.action in {"source", "evaluate"}:
        return _corpus_control(
            registry,
            args.action,
            args.corpus_control_action,
            args.arguments,
        )
    if args.domain == "cpt":
        if args.action == "doctor":
            return _cpt_action(registry, "verify", None, [])
        if args.action == "init":
            init_args = [
                "init",
                "--run-id", args.run_id,
                "--corpus-dir", str(args.corpus_dir),
                "--target-exposure", str(args.target_exposure),
            ]
            if getattr(args, "cycle_id", None):
                init_args.extend(["--cycle-id", args.cycle_id])
            if getattr(args, "resume_checkpoint", None):
                init_args.extend(["--resume-checkpoint", str(args.resume_checkpoint)])
            return _forward_module(registry, "orchestration.lifecycle_51m", init_args)
        rest = []
        if getattr(args, "until", None):
            rest.extend(["--until", args.until])
        if args.action in {"run", "resume"}:
            raw = ["--confirm-training"] if args.confirm_training else []
            _training_arguments(raw, action=f"cpt {args.action}")
        return _cpt_action(
            registry, args.action, args.run_id, rest
        )
    if args.domain == "base":
        return _base_action(registry, args.action, args.arguments)
    if args.domain == "productization":
        if args.action == "doctor":
            return _product_doctor(registry)
        if args.action == "status":
            return _product_status(args.run_dir)
        if args.action in {"run", "resume"} and not any(
            value in {"--help", "-h"} for value in args.arguments
        ):
            args.arguments = _training_arguments(
                args.arguments, action=f"productization {args.action}"
            )
        return _product_action(registry, args.action, args.arguments)
    if args.domain in {
        "qat",
        "sft-alignment",
        "model-evaluation",
        "runtime-release",
    }:
        if args.action == "template":
            return _phase_template(
                registry,
                args.domain,
                cycle_id=args.cycle_id,
                pipeline_id=args.pipeline_id,
            )
        return _phase_action(
            registry,
            args.domain,
            args.action,
            args.binding,
            confirmed=getattr(args, "confirm_training", False),
        )
    if args.domain == "artifact":
        return _artifact_resolve(registry, args.value, args.action == "verify")
    if (args.domain, args.action) == ("platform", "test"):
        return _forward(registry, "src/platform/_shared/tools/run_gates.py", ["--scope", args.scope])
    if (args.domain, args.action) == ("platform", "benchmark"):
        return _forward(registry, "src/platform/_shared/tools/bench_mlx_complete.py", args.arguments)
    if (args.domain, args.action) == ("release", "prepare"):
        return _release_prepare(args)
    if (args.domain, args.action) == ("release", "verify"):
        return _release_verify(args.manifest)
    raise AssertionError("unreachable")
