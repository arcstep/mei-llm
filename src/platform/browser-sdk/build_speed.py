"""Offline, immutable-ID builds of the experimental CPU-WASM speed profile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "src/platform/_shared"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--mode", choices=["packed", "approx"], default="approx")
    parser.add_argument("--numeric-bench", action="store_true", help="Include diagnostic raw forward ABI, not an Agent API")
    parser.add_argument("--parallel-kernels", action="store_true", help="Opt-in cooperative matrix workers")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,90}", args.id):
        parser.error("id must be an ASCII identifier")
    out = ROOT / ".local/cache/wasm-speed" / args.id
    out.mkdir(parents=True, exist_ok=False)
    files = sorted(p for base in [WORKSPACE / "rust", WORKSPACE / "spec", WORKSPACE / ".cargo"] for p in base.rglob("*") if p.is_file())
    files += [WORKSPACE / "Cargo.toml", WORKSPACE / "Cargo.lock", Path(__file__).resolve()]
    def manifest():
        return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    frozen = manifest()
    (out / "source-manifest.json").write_text(json.dumps(frozen, indent=2) + "\n")
    with tarfile.open(out / "source.tar.gz", "w:gz") as archive:
        for p in files:
            archive.add(p, arcname=p.relative_to(ROOT))
    features = "fast-kernels,prefix-cache,tiled-prefill"
    if args.mode == "approx":
        features += ",approx-kernels"
    if args.numeric_bench or args.parallel_kernels:
        features += ",numeric-bench"
    if args.parallel_kernels:
        features += ",parallel-kernels"
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(ROOT / ".local/cache/wasm-speed-20260914/target")
    # Explicit portable SIMD configuration; no inherited host CPU/relaxed flags.
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    env["RUSTFLAGS"] = "-C panic=abort -C target-feature=+simd128 -C link-arg=--initial-memory=33554432 -C link-arg=--max-memory=536870912"
    command = ["cargo", "build", "--offline", "--locked", "--release", "-p", "mei-sdk-wasm", "--target", "wasm32-unknown-unknown", "--features", features]
    with (out / "build.log").open("w") as log:
        subprocess.run(command, cwd=WORKSPACE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if manifest() != frozen:
        raise RuntimeError("source changed during build; this ID cannot be adopted")
    binary = Path(env["CARGO_TARGET_DIR"]) / "wasm32-unknown-unknown/release/mei_sdk_wasm.wasm"
    target = out / "runtime.wasm"
    shutil.copyfile(binary, target)
    receipt = {"schema": "mei-wasm-speed-experiment-v1", "id": args.id, "mode": args.mode,
               "features": features, "command": command, "rustflags": env["RUSTFLAGS"],
               "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
               "source_archive_sha256": hashlib.sha256((out / "source.tar.gz").read_bytes()).hexdigest(),
               "wasm_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "bytes": target.stat().st_size,
               "numeric_bench": args.numeric_bench or args.parallel_kernels, "parallel_kernels": args.parallel_kernels, "weights_changed": False, "experimental": True, "release_eligible": False,
               "canonical_numeric_parity_claimed": False, "requires_simd128": True,
               "approximate_integer_codebooks_and_exp": args.mode == "approx", "prefix_cache_max_tokens": 576,
               "retrieval_embedding_cache_entries": 8, "retrieval_embedding_cache_max_query_tokens": 128}
    (out / "runtime-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"directory": str(out), **receipt}, indent=2))


if __name__ == "__main__":
    main()
