from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from mei_llm.registry import Registry

from common.paths import cycle_artifacts


PHASES = {"qat", "sft_alignment", "model_evaluation", "runtime_release"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def pipeline(registry: Registry, pipeline_id: str) -> dict[str, Any]:
    value = load_json(registry.root / "src/model-factory/contracts/PIPELINES.json")
    matches = [
        row for row in value["pipelines"] if row.get("pipeline_id") == pipeline_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"pipeline must resolve uniquely: {pipeline_id}")
    selected = matches[0]
    if selected.get("status") != "current":
        raise RuntimeError(f"formal phase requires current pipeline: {pipeline_id}")
    return selected


def recipe(registry: Registry, selected: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = registry.root / str(selected.get("recipe") or "")
    if not path.is_file():
        raise RuntimeError(f"pipeline recipe missing: {path}")
    return path, load_json(path)


def template(
    registry: Registry, *, cycle_id: str, pipeline_id: str
) -> dict[str, Any]:
    registry.cycle(cycle_id)
    selected = pipeline(registry, pipeline_id)
    recipe_path, phase_recipe = recipe(registry, selected)
    selectors = [
        str(value)
        for value in (phase_recipe.get("argument_bindings") or {}).values()
    ]
    inputs = sorted(
        selector.partition(":")[2]
        for selector in selectors
        if selector.startswith("input:")
    )
    inputs = sorted(set(inputs) | set(phase_recipe.get("required_inputs", [])))
    parameters = sorted(
        selector.partition(":")[2]
        for selector in selectors
        if selector.startswith("parameter:")
    )
    outputs = sorted(
        selector.partition(":")[2]
        for selector in selectors
        if selector.startswith("output:")
    )
    resume_input = phase_recipe.get("resume_run_input")
    if resume_input and resume_input not in inputs:
        inputs.append(str(resume_input))
    return {
        "schema": "mei-51m-phase-binding-template-v1",
        "replace_schema_with": "mei-51m-phase-binding-v1",
        "binding_id": f"{cycle_id}-{phase_recipe['phase']}-REPLACE_ME",
        "model_id": "mei-1.0-51m",
        "cycle_id": cycle_id,
        "phase": phase_recipe["phase"],
        "pipeline_id": pipeline_id,
        "pipeline_recipe_sha256": sha256_file(recipe_path),
        "source": {
            "git_revision": "REPLACE_ME",
            "dirty_patch_sha256": None,
            "source_manifest": {
                "ref": "REPLACE_ME",
                "sha256": "REPLACE_ME",
            },
            "stage_source_closure": {
                "ref": "REPLACE_ME",
                "sha256": "REPLACE_ME",
            },
        },
        "inputs": {
            name: (
                {
                    "ref": "REPLACE_ME",
                    "target": "REPLACE_WITH_MANIFEST_OR_RECEIPT",
                    "pass_ref": True,
                    "sha256": "REPLACE_ME",
                }
                if name in set(phase_recipe.get("pass_ref_inputs", []))
                else {"ref": "REPLACE_ME", "sha256": "REPLACE_ME"}
            )
            for name in inputs
        },
        "outputs": {
            name: (
                f"{cycle_artifacts(cycle_id).relative_to(registry.root)}/runs/REPLACE_ME"
                if name == "run_dir"
                else "REPLACE_ME"
            )
            for name in outputs
        },
        "parameters": {name: "REPLACE_ME" for name in parameters},
        "current_sha256": registry.current_sha256(),
    }


def resolve_artifact(
    registry: Registry, descriptor: dict[str, Any], *, require_exists: bool
) -> Path:
    reference = str(descriptor.get("ref") or "")
    expected = str(descriptor.get("sha256") or "")
    if not reference or len(expected) != 64:
        raise RuntimeError("artifact descriptor requires ref and sha256")
    base = registry.resolve(reference)
    target = descriptor.get("target")
    if target:
        candidate = (base / str(target)).resolve()
        try:
            candidate.relative_to(base.resolve())
        except ValueError as error:
            raise RuntimeError(f"artifact target escapes ref: {target}") from error
    else:
        candidate = base.resolve()
    if not candidate.exists():
        if require_exists:
            raise RuntimeError(f"bound artifact missing: {candidate}")
        return base.resolve() if descriptor.get("pass_ref") else candidate
    if not candidate.is_file():
        raise RuntimeError(
            f"bound artifact must name a file or set target to its manifest: {candidate}"
        )
    actual = sha256_file(candidate)
    if actual != expected:
        raise RuntimeError(
            f"bound artifact hash drifted: {candidate}; expected={expected} actual={actual}"
        )
    return base.resolve() if descriptor.get("pass_ref") else candidate


def verify_source_closure(registry: Registry, path: Path) -> None:
    value = load_json(path)
    if value.get("schema") != "mei-51m-stage-source-closure-v1":
        raise RuntimeError(f"stage source closure schema mismatch: {path}")
    entries = value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(f"stage source closure has no entries: {path}")
    for entry in entries:
        relative = Path(str(entry.get("path") or ""))
        expected = str(entry.get("sha256") or "")
        if relative.is_absolute() or ".." in relative.parts or len(expected) != 64:
            raise RuntimeError(f"invalid source closure entry: {entry}")
        source = (registry.root / relative).resolve()
        try:
            source.relative_to(registry.root.resolve())
        except ValueError as error:
            raise RuntimeError(f"source closure path escapes repository: {source}") from error
        if not source.is_file():
            raise RuntimeError(f"source closure file missing: {source}")
        actual = sha256_file(source)
        if actual != expected:
            raise RuntimeError(
                f"source closure drifted: {relative}; expected={expected} actual={actual}"
            )


def output_path(registry: Registry, cycle_id: str, value: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = registry.root / candidate
    resolved = candidate.resolve()
    allowed = cycle_artifacts(cycle_id).resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as error:
        raise RuntimeError(
            f"phase output must stay inside cycle artifact root {allowed}: {resolved}"
        ) from error
    return resolved


def verify(
    registry: Registry,
    binding_path: Path,
    *,
    require_inputs: bool,
) -> dict[str, Any]:
    binding = load_json(binding_path)
    errors: list[str] = []
    if binding.get("schema") != "mei-51m-phase-binding-v1":
        errors.append("binding schema mismatch")
    if not str(binding.get("binding_id") or ""):
        errors.append("binding_id is required")
    if binding.get("model_id") != "mei-1.0-51m":
        errors.append("model identity mismatch")
    phase = str(binding.get("phase") or "")
    if phase not in PHASES:
        errors.append(f"unknown phase: {phase}")
    cycle_id = str(binding.get("cycle_id") or "")
    inputs_value = binding.get("inputs")
    outputs_value = binding.get("outputs")
    if not isinstance(inputs_value, dict):
        errors.append("inputs must be an object")
        inputs_value = {}
    if not isinstance(outputs_value, dict):
        errors.append("outputs must be an object")
        outputs_value = {}
    if not isinstance(binding.get("parameters"), dict):
        errors.append("parameters must be an object")
    try:
        registry.cycle(cycle_id)
    except (KeyError, OSError, ValueError) as error:
        errors.append(f"cycle binding invalid: {error}")
    selected: dict[str, Any] = {}
    recipe_path: Path | None = None
    phase_recipe: dict[str, Any] = {}
    try:
        selected = pipeline(registry, str(binding.get("pipeline_id") or ""))
        recipe_path, phase_recipe = recipe(registry, selected)
        actual_recipe_sha = sha256_file(recipe_path)
        if actual_recipe_sha != binding.get("pipeline_recipe_sha256"):
            errors.append("pipeline recipe hash drifted")
        if phase_recipe.get("phase") != phase:
            errors.append("pipeline recipe phase mismatch")
        if phase_recipe.get("entrypoint") != selected.get("entrypoint"):
            errors.append("pipeline recipe entrypoint mismatch")
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        errors.append(str(error))
    if binding.get("current_sha256") != registry.current_sha256():
        errors.append("CURRENT.json drifted from binding")
    resolved_inputs: dict[str, str] = {}
    missing_required = sorted(
        set(phase_recipe.get("required_inputs", [])) - set(inputs_value)
    )
    if missing_required:
        errors.append(f"required phase inputs missing: {missing_required}")
    for name in phase_recipe.get("pass_ref_inputs", []):
        descriptor = inputs_value.get(name)
        if not isinstance(descriptor, dict):
            continue
        if descriptor.get("pass_ref") is not True or not descriptor.get("target"):
            errors.append(
                f"input {name}: directory binding requires target and pass_ref=true"
            )
    for name, descriptor in inputs_value.items():
        if not isinstance(descriptor, dict):
            errors.append(f"input {name}: artifact descriptor must be an object")
            continue
        try:
            resolved_inputs[name] = str(
                resolve_artifact(
                    registry, descriptor, require_exists=require_inputs
                )
            )
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"input {name}: {error}")
    resolved_outputs: dict[str, str] = {}
    for name, value in outputs_value.items():
        try:
            resolved_outputs[name] = str(output_path(registry, cycle_id, str(value)))
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"output {name}: {error}")
    source = binding.get("source") or {}
    if not isinstance(source, dict):
        errors.append("source must be an object")
        source = {}
    if not source.get("git_revision"):
        errors.append("source binding missing: git_revision")
    dirty_patch_sha = source.get("dirty_patch_sha256")
    if dirty_patch_sha is not None and len(str(dirty_patch_sha)) != 64:
        errors.append("dirty_patch_sha256 must be null or a SHA256")
    for key in ("source_manifest", "stage_source_closure"):
        if not source.get(key):
            errors.append(f"source binding missing: {key}")
            continue
        if not isinstance(source[key], dict):
            errors.append(f"{key}: artifact descriptor must be an object")
            continue
        try:
            resolved_source = resolve_artifact(
                registry,
                source[key],
                require_exists=require_inputs,
            )
            if key == "stage_source_closure" and require_inputs:
                verify_source_closure(registry, resolved_source)
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"{key}: {error}")
    if dirty_patch_sha and not source.get("source_bundle"):
        errors.append("dirty source binding requires a recoverable source_bundle")
    if source.get("source_bundle"):
        if not isinstance(source["source_bundle"], dict):
            errors.append("source bundle: artifact descriptor must be an object")
            source["source_bundle"] = {}
        try:
            resolve_artifact(
                registry,
                source["source_bundle"],
                require_exists=require_inputs,
            )
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"source bundle: {error}")
    return {
        "schema": "mei-51m-phase-binding-verification-v1",
        "ok": not errors,
        "binding": str(binding_path.resolve()),
        "binding_sha256": sha256_file(binding_path),
        "phase": phase,
        "cycle_id": cycle_id,
        "pipeline": selected,
        "recipe": str(recipe_path) if recipe_path else None,
        "resolved_inputs": resolved_inputs,
        "resolved_outputs": resolved_outputs,
        "errors": errors,
    }


def invocation(
    registry: Registry,
    binding_path: Path,
    *,
    resume: bool,
    dry_run: bool,
) -> dict[str, Any]:
    report = verify(registry, binding_path, require_inputs=True)
    if not report["ok"]:
        raise RuntimeError("; ".join(report["errors"]))
    binding = load_json(binding_path)
    selected = report["pipeline"]
    recipe_path, phase_recipe = recipe(registry, selected)
    arguments: list[str] = []
    inputs = report["resolved_inputs"]
    outputs = report["resolved_outputs"]
    parameters = binding.get("parameters") or {}
    missing_inputs = sorted(
        set(phase_recipe.get("required_inputs", [])) - set(inputs)
    )
    if missing_inputs:
        raise RuntimeError(f"required phase inputs missing: {missing_inputs}")
    resume_run_input = phase_recipe.get("resume_run_input")
    if resume_run_input:
        if resume_run_input not in inputs or "run_dir" not in outputs:
            raise RuntimeError("resume phase must bind its existing run and run_dir")
        existing_run = Path(inputs[str(resume_run_input)]).resolve()
        if existing_run != Path(outputs["run_dir"]).resolve():
            raise RuntimeError("resume phase input run must equal output run_dir")
        for stage in phase_recipe.get("requires_existing_stages", []):
            receipt = existing_run / "stages" / str(stage) / "receipt.json"
            if not receipt.is_file():
                raise RuntimeError(f"required upstream stage receipt missing: {receipt}")
            status = load_json(receipt).get("terminal_status")
            if status not in {"passed", "degraded"}:
                raise RuntimeError(
                    f"required upstream stage is not reusable: {stage}={status}"
                )
    for flag, selector in (phase_recipe.get("argument_bindings") or {}).items():
        namespace, separator, name = str(selector).partition(":")
        if not separator:
            raise RuntimeError(f"invalid argument selector: {selector}")
        values = {
            "input": inputs,
            "output": outputs,
            "parameter": parameters,
        }
        if namespace not in values or name not in values[namespace]:
            raise RuntimeError(f"binding value missing for {flag}: {selector}")
        value = values[namespace][name]
        if isinstance(value, bool):
            if value:
                arguments.append(flag)
        else:
            arguments.extend([flag, str(value)])
    arguments.extend(str(value) for value in phase_recipe.get("fixed_arguments", []))
    if resume:
        resume_flag = str(phase_recipe.get("resume_flag") or "")
        if not resume_flag:
            raise RuntimeError(f"pipeline does not declare resume support: {selected['pipeline_id']}")
        arguments.append(resume_flag)
    if dry_run:
        dry_run_flag = str(phase_recipe.get("dry_run_flag") or "")
        if not dry_run_flag:
            raise RuntimeError(f"pipeline does not declare dry-run support: {selected['pipeline_id']}")
        arguments.append(dry_run_flag)
    module = str(selected["entrypoint"])
    env = os.environ.copy()
    roots = [
        str(registry.root / "src"),
        str(registry.root / "src/model-factory"),
        str(registry.root / "src/platform/python-sdk"),
    ]
    if env.get("PYTHONPATH"):
        roots.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(roots)
    return {
        "schema": "mei-51m-bound-phase-invocation-v1",
        "binding_verification": report,
        "recipe_sha256": sha256_file(recipe_path),
        "module": module,
        "arguments": arguments,
        "command": [sys.executable, "-m", module, *arguments],
        "environment": {
            "PYTHONPATH": env["PYTHONPATH"],
            "MEI_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
            "MEI_PHASE_BINDING": str(binding_path.resolve()),
            "MEI_PHASE_BINDING_SHA256": report["binding_sha256"],
            "MEI_PHASE_BINDING_ID": str(binding["binding_id"]),
            "MEI_PHASE_CYCLE_ID": str(binding["cycle_id"]),
            "MEI_PHASE": str(binding["phase"]),
            "MEI_PHASE_PIPELINE_ID": str(binding["pipeline_id"]),
        },
    }


def record_for_run(registry: Registry, binding_path: Path) -> Path:
    binding = load_json(binding_path)
    run_value = (binding.get("outputs") or {}).get("run_dir")
    if not run_value:
        raise RuntimeError("phase binding has no run_dir output")
    run_dir = output_path(registry, str(binding["cycle_id"]), str(run_value))
    digest = sha256_file(binding_path)
    target = run_dir / "bindings" / f"{binding['phase']}-{digest}.json"
    payload = binding_path.read_bytes()
    if target.is_file():
        if target.read_bytes() != payload:
            raise RuntimeError(f"recorded phase binding changed: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)
    return target
