# Mei browser data check experiment

Natural-language requirements → low-frequency planner → human confirmation → local Excel/CSV parsing → real 900M-SFT Mei-51m WASM tool selection → validated JS tools → result narration/report → local collector upload and receipt.

## Run

From the product repository root:

```sh
node src/demos/data-check/prepare.mjs
node src/demos/data-check/server.mjs
```

Open http://127.0.0.1:8765. Preparation is offline, CPU-only and writes a separate hash-bound tool-index adoption under `.local/cache/data-check/`. Canonical weights are verified and never changed. All independent heads and candidate execution gates are preserved.

The local server reads `../.env` on startup. Default: `QWEN_BASE_URL`, `QWEN_API_KEY`, `QWEN_COMPLETION_MODEL` (first comma-separated model). `MEI_DEMO_PROVIDER` selects another uppercase prefix. `PORT` defaults to 8765. `MEI_DEMO_PACKAGE` can explicitly select a prepared package directory. Secrets never reach the browser. Only the written requirement is sent to the planner; 10 requests/server lifetime maximum, 60s timeout, one concurrent request. Starting the server does not call the provider.

Fill the example requirement, generate and confirm the plan, download the problem sample, then select it. Each rule gets its own session; one WASM instance is shared serially. A refused/incorrect call is **incomplete**, not pass. The explicit assistance button executes the confirmed plan in JS and labels those checks `human-plan`, never `mei`. Download the corrected sample and repeat to exercise the local receipt.

The model is experimental: a real probe produced `[]` on an unseen required-column tool request. This demo deliberately exposes that gap; it does not silently mock model decisions or relax gates. Narration is attempted only after an actual validated call/result; SDK fallback and tool-generated summaries have distinct labels.

## Scope

Optional CPU-WASM speed experiment (2026-09-14):

```sh
.venv/bin/python src/platform/browser-sdk/build_speed.py --id my-speed-experiment --mode approx
MEI_DEMO_PACKAGE=.local/cache/data-check/data-check-d919985609f9 \
MEI_DEMO_WASM=.local/cache/wasm-speed/my-speed-experiment/runtime.wasm \
node src/demos/data-check/server.mjs
```

This preserves the canonical model and binary. The loader checks the experimental binary receipt; the page labels approximate execution and invalidates saved results after a runtime change. Without `MEI_DEMO_WASM`, the original runtime is used. Build IDs cannot be overwritten.

Alternating A/B on M4 Max: **78.35 → 144.72 raw decode tokens/s**, **3.60 → 2.03 s** for synthetic 512-token prefill plus first output, **3.89 → 2.26 s** for the first task request. Identical warm requests measured **13.65 ms** through KV and retrieval-vector reuse; changed context/tasks measured **0.38–0.45 s** in the cache checks. These are different workloads, not interchangeable throughput numbers. Measured heap peaked at **85.19 MiB**. First-request prefill still needs work; approximate arithmetic and full product quality are not certified. See [speed validation](validation/2026-09-14-wasm-speed.json).

No-provider validation entry points: `speed-checks.cjs`, `speed-paired.cjs`, and `profile.cjs`. Set `MEI_BENCH_WASM` to the experiment binary; the first two also accept `MEI_BENCH_OUTPUT` for a new JSON output. They use actual WASM. `speed-paired.cjs` compares against the original canonical binary, not a mocked result.

- XLSX/XLS/CSV, local SheetJS 0.20.3, browser parsing worker.
- Required, unique, numeric bounds, enum, quantity × price checks.
- Max 10 MiB, 10 sheets, 10,000 data rows and 100 columns/sheet. One selected sheet per report. No silent truncated pass.
- Formula values are cached values, not recalculated; formula workbooks require review and cannot receive pass receipt.
- IndexedDB stores task, parsed workbook and partial results locally. Service Worker caches app/model for offline use after initial loading. Closing a tab stops execution. No distributed fleet or background daemon claim.
- After an explicit upload click, the original file goes to the loopback collector. The server looks up the hash-bound planner contract, independently reruns checks for the selected sheet, and only then stores file bytes and a receipt. No file is sent to a cloud provider. This is not a multi-user production receiver.

## Tests

```sh
node --test src/demos/data-check/checks.test.mjs
node src/demos/data-check/e2e.cjs
```

E2E needs Playwright (`PLAYWRIGHT_PATH` can point to an installed package), a running server, and consumes **one authorized planner request**. It runs real browser WASM twice, tests problem/corrected files, explicit assistance, offline restore and receipt. Evidence is saved under `.local/cache/data-check/e2e-<timestamp>/`.

`probe.mjs <prepared-package>` is an offline single-request WASM diagnostic; not a product quality gate. `vendor/` contains unmodified SheetJS CE 0.20.3 and its Apache-2.0 license, obtained from the official SheetJS CDN. SHA256 of xlsx.full.min.js: `cc015130aa8521e7f088f88898eba949ccdcbfb38df0bd129b44b7273c3a6f41`.

Additional checks (no provider calls):

```sh
node src/demos/data-check/browser-checks.cjs
node src/demos/data-check/collector-checks.mjs
```

The collector checks require a previously generated sample plan. They verify that a forged client pass, mismatched file hash, and unknown plan cannot be accepted. Unit tests use an explicitly fake session only to test host sequencing; those results are never counted as real model quality.

## Measured outcome (2026-09-13)

See [validation/2026-09-13.json](validation/2026-09-13.json). Final real Qwen planning: 9.009 s / 958 total tokens. Both problem and corrected spreadsheets ran six real WASM session attempts: two empty refusals and four ABI `protocol_violation` errors per file. Autonomous completed checks: **0/6**. After explicit human-plan assistance, seven problem records were found; the corrected file passed collector revalidation and was stored with a receipt. Offline restore passed.

This validates the host workflow and exposes a model/runtime integration gap. It does **not** establish autonomous Mei tool competence or model narration quality. No release readiness claim follows from the demo. The existing SDK rejects/errs honestly; thresholds and frozen weights were not changed to make the demonstration pass.

Performance diagnostic: with the server running, `node src/demos/data-check/performance.cjs` measures the same bytes in Chromium (raw mode enabled only through an in-memory experimental manifest). On M4 Max with existing training still running, three warm short-context pairs measured **35.70–35.95 tokens/s**; synthetic 512-token prefill plus first output took **7.82 s**. The single long-context difference estimated 28.98 tokens/s and is sensitive to prefill timing noise. See [performance evidence](validation/2026-09-13-performance.json). The demo's 8.8–9.5 s per rule is whole-request latency. These measurements do not establish hundreds of tokens/s, constrained task throughput, or task quality.

2026-09-14 remeasurement, unchanged binary/weights: **77.77–78.09 raw tokens/s**, actual Worker request **3.85–3.91 s**. CPU sampling attributes about **3.82 s** of a 3.90 s one-output-token request to prefill and **57 ms** to retrieval. See [current throughput](validation/2026-09-14-performance.json) and [stage profile](validation/2026-09-14-profile.json). Run `node src/demos/data-check/profile.cjs` to reproduce profiles without a provider request. Sampling is approximate; no runtime optimization was made. Earlier training was concurrent, but the two dates were not a controlled load/power comparison. The benchmark now writes unique receipts and refuses to overwrite an existing output path.
