"""Freeze Python golden cases and run Rust plus real-browser tokenizer parity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from profiling import ROOT, digest
from tokenizer_candidate import LosslessProcessor, rows, write_new


def run(config: dict, out: Path) -> dict:
    model = ROOT / config["model"]
    candidate = ROOT / config["candidate_manifest"]
    dev = ROOT / config["dev_jsonl"]
    if digest(model) != config["model_sha256"] or digest(candidate) != config["manifest_sha256"]:
        raise ValueError("candidate binding changed")
    out.mkdir(parents=True, exist_ok=False)
    write_new(out / "config.json", config)
    shutil.copyfile(__file__, out / "implementation.py.snapshot")
    processor = LosslessProcessor(model)
    cases = []
    for text in config["boundary_probes"]:
        cases.append({"text": text, "ids": processor.encode(text)})
    ranked = sorted(
        rows(dev), key=lambda row: hashlib.sha256(f"{config['seed']}:{row['text_sha256']}".encode()).hexdigest()
    )
    for row in ranked[: config["dev_cases"]]:
        cases.append({"text": row["text"], "ids": processor.encode(row["text"])})
    golden = {"schema": "mei-lossless-tokenizer-golden-v1", "model_sha256": digest(model), "cases": cases}
    write_new(out / "golden.json", golden)
    environment = {
        **__import__("os").environ,
        "MEI_LOSSLESS_TOKENIZER_MODEL": str(model.resolve()),
        "MEI_LOSSLESS_TOKENIZER_GOLDEN": str((out / "golden.json").resolve()),
    }
    commands = []
    for name, command in [
        ("rust", ["cargo", "test", "--manifest-path", "src/platform/_shared/rust/mei-sdk-core/Cargo.toml",
                  "--test", "lossless_tokenizer", "--", "--nocapture"]),
        ("wasm-build", ["cargo", "build", "--manifest-path", "src/platform/_shared/rust/mei-sdk-wasm/Cargo.toml",
                        "--target", "wasm32-unknown-unknown", "--release"]),
    ]:
        started = time.time()
        proc = subprocess.run(command, cwd=ROOT, env={**environment, "CARGO_TARGET_DIR": str((ROOT / ".local/cache/cargo-sdk-target").resolve())},
                              text=True, capture_output=True)
        log = out / f"{name}.log"
        log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
        commands.append({"name": name, "command": command, "exit_code": proc.returncode,
                         "seconds": time.time() - started, "log": str(log), "log_sha256": digest(log)})
        if proc.returncode:
            raise ValueError(f"{name} failed; see {log}")
    browser_script = ROOT / "src/platform/browser-sdk/run-tokenizer-audit.mjs"
    wasm = ROOT / ".local/cache/cargo-sdk-target/wasm32-unknown-unknown/release/mei_sdk_wasm.wasm"
    browser_json = out / "browser-result.json"
    command = ["node", str(browser_script), "--model", str(model), "--wasm", str(wasm),
               "--golden", str(out / "golden.json"), "--out", str(browser_json)]
    started = time.time()
    proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    log = out / "browser.log"
    log.write_text(proc.stdout + proc.stderr, encoding="utf-8")
    commands.append({"name": "real-browser", "command": command, "exit_code": proc.returncode,
                     "seconds": time.time() - started, "log": str(log), "log_sha256": digest(log)})
    if proc.returncode:
        raise ValueError(f"browser audit failed; see {log}")
    browser = json.loads(browser_json.read_text(encoding="utf-8"))
    report = {"schema": "mei-tokenizer-runtime-audit-v1", "status": "passed",
              "python_cases": len(cases), "rust_passed": True,
              "real_browser_passed": browser.get("passed") is True,
              "browser_user_agent": browser.get("user_agent"),
              "model_sha256": digest(model), "golden_sha256": digest(out / "golden.json"),
              "wasm_sha256": digest(wasm), "commands": commands}
    if not report["real_browser_passed"]:
        raise ValueError("browser result did not pass")
    write_new(out / "report.json", report)
    return report
