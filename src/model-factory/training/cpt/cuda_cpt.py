"""CUDA scratch CPT adapter over the shared quota curriculum and evidence stage.

Research candidates are never registered as canonical bases by this worker.
An attempt may stop at an explicit update boundary; every resume is a new
attempt and requires the complete model/Adam/RNG/sampler/source fingerprint.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common.data import PackedTokenSource, QuotaPackedSources
from common.evidence_stage import freeze, sha256, write_json
from common.source_capture import json_digest
from training.torch_backend import checkpoint
from training.torch_backend.optim import configure, make_model, update
from training.torch_backend.performance import install_kernels
from training.torch_backend.precision import DtypeAudit, autocast, policy
from training.cpt.rolling_store import RollingStore, enough_space


def bound_file(binding):
    path = Path(binding["path"])
    if not path.is_absolute() or sha256(path) != binding["sha256"]:
        raise ValueError("input hash/path mismatch: " + str(path))
    return path


def logical_fingerprint(config, source_manifest_sha256):
    operational = {"out", "resume", "stop_after_steps", "compare_to"}
    return json_digest({"training": {k: v for k, v in config.items() if k not in operational},
                        "source_manifest_sha256": source_manifest_sha256,
                        "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu_capability": list(torch.cuda.get_device_capability())})


def validate_schedule(schedule, *, purpose):
    if schedule["kind"] != "scratch" or schedule["allow_repeat"] or schedule["parent_tokens_seen"] != 0:
        raise ValueError("CUDA candidate currently supports scratch, unique-token curriculum only")
    if schedule["sampler"] != "quota_plan" or schedule["lr"]["kind"] != "cosine_tokens":
        raise ValueError("unsupported sampler or LR")
    stages = schedule["curriculum"]
    names = set(schedule["sources"])
    total = 0
    for stage in stages:
        if set(stage["sources"]) != names or stage["grad_accum"] != 1:
            raise ValueError("source or accumulation contract differs")
        if not 0 < stage["seq_len"] <= 2048 or not 0 < stage["batch_size"] <= 8:
            raise ValueError("invalid sequence/batch")
        quotas = [int(v["token_quota"]) for v in stage["sources"].values()]
        if any(q <= 0 for q in quotas) or sum(quotas) != stage["stage_tokens"]:
            raise ValueError("stage quotas do not add up")
        total += stage["stage_tokens"]
        if total != stage["stop_at_tokens"]:
            raise ValueError("curriculum cumulative exposure mismatch")
    if total != schedule["exposure_tokens"] or schedule["lr"]["horizon_tokens"] != total:
        raise ValueError("total or LR horizon mismatch")
    for name in names:
        if sum(s["sources"][name]["token_quota"] for s in stages) != schedule["sources"][name]["token_quota"]:
            raise ValueError("lifetime role quotas mismatch")
    if purpose == "scratch_candidate":
        if total != 300000000 or [(s["seq_len"], s["batch_size"], s["stage_tokens"]) for s in stages] != [(512,8,150000000),(1024,2,100000000),(2048,1,50000000)]:
            raise ValueError("candidate must preserve the frozen 300M baseline curriculum")
    elif purpose == "scratch_campaign":
        if any(s["seq_len"] != 2048 for s in stages):
            raise ValueError("campaign input freeze requires 2048 windows")
        if any(q["token_quota"] % 2048 for s in stages for q in s["sources"].values()):
            raise ValueError("campaign quotas must use whole windows")
    elif purpose != "backend_probe" or total > 1000000:
        raise ValueError("invalid candidate/probe purpose")


def sampler_for(schedule, corpus, stage_index, prior=None, *, restore=False):
    stage = schedule["curriculum"][stage_index]
    sources = {name: PackedTokenSource([Path(p["path"]) for p in corpus["sources"][name]["train"]],
                  stage["seq_len"], 0) for name in schedule["sources"]}
    sampler = QuotaPackedSources(sources, {name: v["token_quota"] for name, v in stage["sources"].items()},
                 seed=schedule["sampler_seed"] + stage_index, seq_len=stage["seq_len"],
                 stage_id=stage["id"], stage_index=stage_index)
    sampler.schedule = schedule
    if prior is not None:
        (sampler.load_state_dict if restore else sampler.continue_from)(prior)
    return sampler


def tensors(windows):
    return tuple(torch.tensor([w[key] for w in windows], device="cuda",
                 dtype=torch.float32 if key == "mask" else torch.long) for key in ("x", "y", "mask"))


@torch.no_grad()
def monitor_loss(model, corpus, precision):
    result = {}
    model.eval()
    for name, source in corpus["sources"].items():
        if not source.get("dev"):
            continue
        data = PackedTokenSource([Path(p["path"]) for p in source["dev"]], 512)
        nll = tokens = 0.0
        for i in range(min(16, len(data))):
            x, y, mask = tensors([data[i]])
            with autocast(precision):
                logits = model(x)["logits"]
            terms = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
            nll += float((terms * mask.flatten()).sum())
            tokens += float(mask.sum())
        result[name] = {"nll_sum": nll, "predicted_tokens": int(tokens), "token_ce": nll / max(tokens, 1)}
    model.train()
    return {"scope": "fixed first16 x512 dev windows per role; health monitor, not cross-tokenizer score", "roles": result}


def train(config):
    out = freeze(config, config["out"])
    configure(config["seed"], deterministic=True)
    if config["precision"] != "bf16_amp" or config["vocab_size"] != 24000:
        raise ValueError("this candidate freezes 24K and BF16 AMP with FP32 master state")
    schedule = json.loads(bound_file(config["schedule"]).read_text())
    validate_schedule(schedule, purpose=config["purpose"])
    corpus = json.loads(bound_file(config["corpus"]).read_text())
    if corpus["schema"] != "mei-cuda-encoded-corpus-v1" or not corpus["ok"] or corpus["vocab_size"] != 24000:
        raise ValueError("invalid encoded corpus")
    if corpus['token_dtype'] != 'uint16':
        raise ValueError('shared sampler requires uint16 encoded inputs')
    if set(corpus["sources"]) != set(schedule["sources"]):
        raise ValueError("missing or additional corpus roles")
    if corpus["tokenizer"] != config["tokenizer"]:
        raise ValueError("encoded tokenizer differs")
    bound_file(config["tokenizer"])
    for source in corpus["sources"].values():
        for b in source["train"] + source.get("dev", []):
            bound_file(b)
    for name, source in corpus["sources"].items():
        arrays = [np.memmap(b["path"], dtype=corpus["token_dtype"], mode="r") for b in source["train"]]
        if any(a.size == 0 or int(a.min()) < 1 or int(a.max()) >= config["vocab_size"] for a in arrays):
            raise ValueError("invalid training IDs or unmasked PAD in source: " + name)
        required = schedule["sources"][name]["token_quota"]
        if config["purpose"] != "scratch_campaign":
            required += sum(s["seq_len"]-1 for s in schedule["curriculum"])
        if sum(a.size for a in arrays)-1 < required:
            raise ValueError("insufficient unique source capacity including alignment slack: " + name)
    if config["purpose"] == "scratch_candidate":
        readiness = json.loads(bound_file(config["readiness"]).read_text())
        if readiness.get("schema") != "mei-cuda-cpt-readiness-v1" or not readiness.get("ok"):
            raise ValueError("candidate needs a complete readiness receipt")
        if readiness.get("corpus") != config["corpus"] or readiness.get("tokenizer") != config["tokenizer"]:
            raise ValueError("readiness input bindings differ")
        for key in ("tokenizer_review", "source_review", "backend_restart", "curriculum_probe"):
            receipt = json.loads(bound_file(readiness[key]).read_text())
            if not receipt.get("ok"):
                raise ValueError("failed readiness prerequisite: " + key)
            if key == "tokenizer_review" and (receipt.get("schema") != "mei-tokenizer-training-review-v1" or
                    not receipt.get("training_candidate_accepted") or receipt.get("tokenizer") != config["tokenizer"]):
                raise ValueError("a completed review is not necessarily an accepted tokenizer")
            if key == "source_review" and (not receipt.get("research_training_accepted") or
                    receipt.get("sources") != {k:v["selection_receipt"] for k,v in corpus["sources"].items()} or
                    receipt.get("training_text_filter") != corpus.get("training_text_filter")):
                raise ValueError("source review does not accept these exact original-text sources")
    install_kernels(config["kernel_mode"])
    model, forward, opt = make_model(config)
    if sum(p.numel() for p in model.parameters()) != 51463797:
        raise ValueError("deployed LM parameter geometry changed")
    manifest = json.loads((out / "source-manifest.json").read_text())
    fingerprint = logical_fingerprint(config, manifest["manifest_sha256"])
    write_json(out / "training-binding.json", {"fingerprint": fingerprint, "precision": policy(config["precision"]),
               "model_config": asdict(model.cfg), "initial_parameters_sha256": model.initial_parameters_sha256,
               "initialization": "shared per-parameter SHA seed, differs from historic MLX random streams",
               "track": "research_candidate", "canonical_identity_adoption": False})
    step = tokens_seen = stage_index = 0
    sampler = sampler_for(schedule, corpus, 0)
    if config.get("resume"):
        state = checkpoint.load_state(bound_file(config["resume"]), model, opt, fingerprint=fingerprint)
        step, tokens_seen, stage_index = state["step"], state["tokens_seen"], state["stage_index"]
        sampler = sampler_for(schedule, corpus, stage_index, state["sampler"], restore=True)
    starting_step, starting_tokens = step, tokens_seen
    checkpoint_tokens = tokens_seen
    checkpoint_time = time.monotonic()
    started = time.perf_counter()
    audit = DtypeAudit(model)
    dtype_result = None
    observed_curriculum = []
    checkpoint_path = Path(config['resume']['path']) if config.get('resume') else None
    store = RollingStore(config["checkpoint_store"],fingerprint) if config.get("checkpoint_store") else None
    if config["purpose"] == "scratch_campaign" and store is None:
        raise ValueError("campaign needs bounded checkpoint retention")
    stop_reason = None
    last_loss = None
    trace = (out / "updates.jsonl").open("x")
    def save(milestone=False):
        nonlocal checkpoint_path, checkpoint_tokens, checkpoint_time
        candidate_path = (store.root if store else out) / f"state-{step:07d}.pt"
        if candidate_path != checkpoint_path:
            checkpoint.save_state(candidate_path, model, opt, fingerprint=fingerprint, step=step,
                        tokens_seen=tokens_seen, stage_index=stage_index, sampler=sampler.state_dict())
            checkpoint_path, checkpoint_tokens = candidate_path, tokens_seen
            checkpoint_time = time.monotonic()
        if store:
            store.committed(checkpoint_path,step=step,tokens=tokens_seen,milestone=milestone)
            if milestone:
                export_dir = store.root.parent / 'milestones' / f'tokens-{tokens_seen:010d}'
                export_dir.mkdir(parents=True,exist_ok=True)
                receipt_path = export_dir / 'SNAPSHOT.json'
                if receipt_path.exists():
                    receipt = json.loads(receipt_path.read_text())
                    if receipt['checkpoint']['sha256'] != sha256(checkpoint_path) or sha256(receipt['weights']['path']) != receipt['weights']['sha256']:
                        raise ValueError('milestone snapshot changed')
                else:
                    weights = export_dir / 'base-weights.npz'
                    partial = export_dir / 'base-weights.partial'
                    if weights.exists() or partial.exists():
                        raise ValueError('incomplete milestone export requires inspection')
                    with partial.open('xb') as handle:
                        np.savez(handle, **{k:p.detach().cpu().numpy() for k,p in model.state_dict().items()})
                    partial.rename(weights)
                    write_json(receipt_path,{'tokens_seen':tokens_seen,'stage_index':stage_index,
                        'model_config':asdict(model.cfg),'fingerprint':fingerprint,'tokenizer':config['tokenizer'],
                        'checkpoint':{'path':str(checkpoint_path),'sha256':sha256(checkpoint_path)},
                        'weights':{'path':str(weights),'sha256':sha256(weights)},
                        'format':'numpy named tensors; MLX adoption/parity pending',
                        'evaluation_blocks_training':False})
        return checkpoint_path
    try:
        while stage_index < len(schedule["curriculum"]):
            if store and not enough_space(store.root,reserve_bytes=int(config.get("disk_reserve_bytes",15*1024**3))):
                stop_reason = "disk_reserve_reached"
                break
            if (out/"STOP_REQUESTED").exists():
                stop_reason = "operator_stop_requested"
                break
            if config.get("stop_after_steps") is not None and step >= config["stop_after_steps"]:
                break
            if config["purpose"] == "backend_probe" and step >= 64:
                raise ValueError("backend probe exceeded64 optimizer updates")
            stage_cfg = schedule["curriculum"][stage_index]
            windows = sampler.take_windows(stage_cfg["batch_size"])
            if not windows:
                if any(q > 0 for q in sampler.quota_remaining.values()):
                    raise ValueError("quota plan ended before satisfying roles")
                if store:
                    save(milestone=True)
                if stage_index + 1 == len(schedule["curriculum"]):
                    break
                prior = sampler.state_dict()
                stage_index += 1
                sampler = sampler_for(schedule, corpus, stage_index, prior)
                continue
            progress = min(tokens_seen / schedule["lr"]["horizon_tokens"], 1.0)
            lr = schedule["lr"]["final"] + (schedule["lr"]["base"] - schedule["lr"]["final"]) * (1 + math.cos(math.pi * progress)) / 2
            for group in opt.param_groups:
                group["lr"] = lr
            torch.cuda.synchronize()
            before = time.perf_counter()
            x, y, mask = tensors(windows)
            last_loss, norm = update(forward, model, opt, x, y, mask, precision=config["precision"])
            torch.cuda.synchronize()
            if dtype_result is None:
                audit.close()
                dtype_result = audit.verify(config["precision"], model, opt)
                write_json(out / "dtype-audit.json", dtype_result)
                if not dtype_result["ok"]:
                    raise ValueError("observed precision policy failed")
            step += 1
            geometry = {"seq_len": stage_cfg["seq_len"], "batch_size": stage_cfg["batch_size"]}
            if geometry not in observed_curriculum:
                observed_curriculum.append(geometry)
            n = sum(int(w["n_predictable"]) for w in windows)
            tokens_seen += n
            row = {"step": step, "tokens_seen": tokens_seen, "stage": stage_cfg["id"], "loss": last_loss,
                   "lr": lr, "grad_norm": norm, "predicted_tokens": n, "update_seconds": time.perf_counter()-before,
                   "seconds": time.perf_counter()-started,
                   "process_tokens_per_second": (tokens_seen-starting_tokens)/(time.perf_counter()-started),
                   "sources": [w["source_id"] for w in windows], "source_cursors": [w["source_token_cursor"] for w in windows]}
            if step == starting_step+1 or step % int(config.get("trace_every_steps",1)) == 0:
                trace.write(json.dumps(row) + "\n")
                trace.flush()
            if step == starting_step + 1 or step % 100 == 0:
                write_json(out / "heartbeat.json", row)
            if tokens_seen-checkpoint_tokens >= config["checkpoint_tokens"] or time.monotonic()-checkpoint_time >= config.get("checkpoint_seconds",float('inf')):
                save()
                write_json(out / f"monitor-{step:07d}.json", monitor_loss(model, corpus, config["precision"]))
        if step == starting_step:
            raise ValueError("attempt performed no optimizer update")
        save(milestone=stage_index==len(schedule["curriculum"])-1 and all(q<=0 for q in sampler.quota_remaining.values()))
    except (InterruptedError, KeyboardInterrupt):
        # Unknown interruption point: the last committed state remains authoritative.
        raise
    finally:
        trace.close()
        audit.close()
    complete = stage_index == len(schedule["curriculum"])-1 and all(q <= 0 for q in sampler.quota_remaining.values())
    result = {"schema": "mei-cuda-cpt-attempt-v1", "ok": True, "cpt_complete": complete,
              "purpose": config["purpose"], "observed_curriculum": observed_curriculum,
              "track": "research_candidate", "release_eligible": False,
              "step": step, "tokens_seen": tokens_seen, "scheduled_tokens": schedule["exposure_tokens"],
              "last_loss": last_loss, "fingerprint": fingerprint, "sampler": sampler.state_dict(),
              "stop_reason":stop_reason,
              "elapsed_seconds": time.perf_counter()-started,
              "process_tokens_per_second": (tokens_seen-starting_tokens)/(time.perf_counter()-started),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "checkpoint": {"path": str(checkpoint_path), "sha256": sha256(checkpoint_path)}}
    if config.get("compare_to"):
        target = bound_file(config["compare_to"])
        comparison = checkpoint.compare_state(target, model, opt)
        meta = json.loads(target.with_suffix(".json").read_text())
        comparison["sampler_and_exposure_equal"] = meta["sampler"] == sampler.state_dict() and meta["tokens_seen"] == tokens_seen and meta["step"] == step
        result["restart_comparison"] = comparison
        result["ok"] = comparison["ok"] and comparison["sampler_and_exposure_equal"]
    if complete and config["purpose"] in ("scratch_candidate","scratch_campaign"):
        weights = out / "base-weights.npz"
        with weights.open("xb") as handle:
            np.savez(handle, **{k: p.detach().cpu().numpy() for k, p in model.state_dict().items()})
        result["base_weights"] = {"path": str(weights), "sha256": sha256(weights)}
    write_json(out / "receipt.json", result)
    return result
