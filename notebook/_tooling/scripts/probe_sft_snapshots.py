#!/usr/bin/env python3
"""Probe named DashScope snapshots. Missing/denied/unknown price disables the lane. No silent replacement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from repo_paths import FLEET_SFT_SYNTH, ROOT

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sft_synth_lib import dump_json, lanes_of, load_fleet, require_spend_allowed  # noqa: E402


SNAPSHOTS = ("deepseek-v4-flash-0731", "qwen-plus-2025-12-01", "qwen3.7-plus")


def probe_lane(lane, *, allow_spend: bool) -> dict:
    require_spend_allowed(lane, allow_spend=allow_spend)
    sys.path.insert(0, str(ROOT.parent / "tools" / "mei-eval" / "python" / "src"))
    from mei_eval.chat import chat_complete, openai_client
    from mei_eval.endpoint import resolve_endpoint

    cfg = resolve_endpoint(provider=lane.provider, model=lane.model_snapshot)
    endpoint_name = cfg.base_url.split("//")[-1].split("/")[0]
    client = openai_client(cfg.base_url, cfg.api_key, timeout=30.0, max_retries=0)
    try:
        text, usage, latency_ms = chat_complete(
            client,
            model=lane.model_snapshot,
            messages=[{"role": "user", "content": '{"query":"ping"}'}],
            temperature=0.0,
            max_tokens=8,
            enable_thinking=False,
        )
        prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
        completion_tokens = int((usage or {}).get("completion_tokens") or 0)
        if prompt_tokens == 0 and completion_tokens == 0:
            return {
                "ok": False,
                "lane": lane.name,
                "endpoint": endpoint_name,
                "reason": "usage_missing",
                "latency_ms": latency_ms,
            }
        return {
            "ok": True,
            "lane": lane.name,
            "model_snapshot": lane.model_snapshot,
            "endpoint": endpoint_name,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "sample": (text or "")[:80],
        }
    except Exception as exc:  # noqa: BLE001
        err = str(exc)
        reason = "denied"
        if "404" in err or "not found" in err.lower():
            reason = "not_found"
        elif "403" in err or "401" in err:
            reason = "denied"
        return {"ok": False, "lane": lane.name, "endpoint": endpoint_name, "reason": reason, "error": err[:240]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", type=Path, default=FLEET_SFT_SYNTH)
    ap.add_argument("--allow-spend", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    fleet = load_fleet(args.fleet)
    lanes = lanes_of(fleet)
    reports = []
    enabled = []
    disabled = []
    for name in SNAPSHOTS:
        lane = lanes.get(name)
        if not lane:
            disabled.append({"lane": name, "reason": "missing_lane"})
            continue
        if lane.input_cny_per_million <= 0 and lane.output_cny_per_million <= 0:
            item = {"ok": False, "lane": name, "reason": "unknown_price"}
            reports.append(item)
            disabled.append(item)
            continue
        try:
            item = probe_lane(lane, allow_spend=args.allow_spend)
        except Exception as exc:  # noqa: BLE001
            item = {"ok": False, "lane": name, "reason": "probe_error", "error": str(exc)[:240]}
        reports.append(item)
        (enabled if item.get("ok") else disabled).append(item)
    out = {
        "ok": True,
        "probes": reports,
        "enabled_lanes": [r["lane"] for r in enabled],
        "disabled_lanes": disabled,
        "note": "Do not replace a disabled snapshot with Qwen3.6/GLM/Kimi.",
    }
    dest = args.out or (ROOT / "notebook/jobs/toolcall-sft/outbox/draft/snapshot-probe.json")
    dump_json(dest, out)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
