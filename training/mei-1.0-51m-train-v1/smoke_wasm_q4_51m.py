#!/usr/bin/env python3
"""Build wasm32 mei-sdk-wasm and load the 51M Q4 package. Float npz is refused."""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from identity_51m import JOBS_DIR, QAT_Q4_PACKAGE_DIR, ROOT, write_json

SDK = Path(__file__).resolve().parents[2] / "sdk"
WASM_JS = SDK / "js" / "smoke_wasm_q4.mjs"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-dir", type=Path, default=QAT_Q4_PACKAGE_DIR)
    parser.add_argument("--jobs-dir", type=Path, default=JOBS_DIR)
    parser.add_argument("--skip-browser", action="store_true")
    args = parser.parse_args()
    if not args.package_dir.is_absolute():
        args.package_dir = (ROOT / args.package_dir).resolve()
    if not args.jobs_dir.is_absolute():
        args.jobs_dir = (ROOT / args.jobs_dir).resolve()
    cargo = shutil.which("cargo")
    node = shutil.which("node")
    if cargo is None or node is None:
        report = {
            "ok": False,
            "compiled": False,
            "quantized_only": True,
            "note": f"missing tools cargo={cargo} node={node}",
        }
        write_json(args.jobs_dir / "wasm-q4-smoke.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    env = dict(os.environ)
    env["CARGO_NET_OFFLINE"] = "true"
    env["MEI_51M_PACKAGE_DIR"] = str(args.package_dir)
    env["MEI_51M_JOBS_DIR"] = str(args.jobs_dir)
    build = subprocess.run(
        [
            cargo,
            "build",
            "-p",
            "mei-sdk-wasm",
            "--target",
            "wasm32-unknown-unknown",
            "--release",
        ],
        cwd=str(SDK),
        capture_output=True,
        text=True,
        env=env,
    )
    if build.returncode != 0:
        report = {
            "ok": False,
            "compiled": False,
            "quantized_only": True,
            "note": "cargo wasm32 build failed",
            "stderr_tail": (build.stderr or "")[-4000:],
        }
        write_json(args.jobs_dir / "wasm-q4-smoke.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 2
    smoke = subprocess.run([node, str(WASM_JS)], cwd=str(SDK / "js"), env=env)
    if smoke.returncode != 0:
        return 2
    if args.skip_browser:
        return 0
    chrome = next(
        (
            candidate
            for candidate in (
                shutil.which("google-chrome"),
                shutil.which("chromium"),
                shutil.which("chromium-browser"),
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            )
            if candidate and Path(candidate).is_file()
        ),
        None,
    )
    if not chrome:
        report = {"ok": False, "browser_worker": False, "error": "local Chromium executable missing"}
        write_json(args.jobs_dir / "wasm-q4-browser-smoke.json", report)
        return 2
    package_rel = args.package_dir.resolve().relative_to(ROOT)
    golden_rel = (args.jobs_dir / "mlx-qat-q4-golden.json").resolve().relative_to(ROOT)
    wasm_path = (
        Path(env.get("CARGO_TARGET_DIR") or SDK / "target")
        / "wasm32-unknown-unknown/release/mei_sdk_wasm.wasm"
    ).resolve()
    browser_wasm = None
    try:
        wasm_rel = wasm_path.relative_to(ROOT)
    except ValueError:
        browser_wasm = args.jobs_dir / "browser-assets/mei_sdk_wasm.wasm"
        browser_wasm.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(wasm_path, browser_wasm)
        wasm_rel = browser_wasm.resolve().relative_to(ROOT)
    from urllib.parse import urlencode

    query = urlencode({"package": str(package_rel), "golden": str(golden_rel), "wasm": str(wasm_rel)})
    report_ready = threading.Event()
    report_box: dict[str, object] = {}

    class ReportHandler(http.server.SimpleHTTPRequestHandler):
        def do_POST(self) -> None:
            if self.path != "/__mei_51m_report__":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("browser report must be an object")
                report_box.clear()
                report_box.update(payload)
                report_ready.set()
                self.send_response(204)
                self.end_headers()
            except (ValueError, json.JSONDecodeError) as exc:
                self.send_error(400, str(exc))

        def log_message(self, format: str, *args: object) -> None:
            return

    handler = functools.partial(ReportHandler, directory=str(ROOT))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="mei-51m-browser-smoke-http",
        daemon=True,
    )
    server_thread.start()
    port = int(server.server_address[1])
    try:
        with tempfile.TemporaryDirectory(prefix="mei-51m-chrome-") as profile:
            stderr_path = Path(profile) / "stderr.log"
            with stderr_path.open("w", encoding="utf-8") as stderr:
                page = subprocess.Popen(
                    [
                        chrome,
                        "--headless=new",
                        "--disable-gpu",
                        "--no-first-run",
                        f"--user-data-dir={profile}",
                        f"http://127.0.0.1:{port}/sdk/js/wasm-q4-smoke.html?{query}",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                )
                report_ready.wait(timeout=180)
                page.terminate()
                try:
                    page.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    page.kill()
                    page.wait(timeout=5)
            page_stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        report = dict(report_box) if report_ready.is_set() else {
            "ok": False,
            "browser_worker": False,
            "error": "browser worker report timed out",
            "stderr_tail": page_stderr[-2000:],
        }
        write_json(args.jobs_dir / "wasm-q4-browser-smoke.json", report)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report.get("ok") else 2
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        if browser_wasm is not None:
            browser_wasm.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
