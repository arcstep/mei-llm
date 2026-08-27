#!/usr/bin/env python3
"""Unit tests for SFT synth fleet, compilers, isolation, and promotion refusals."""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from pathlib import Path

from repo_paths import (
    CODEBOOK_MW_DISPOSITION_V1,
    FLEET_SFT_SYNTH,
    ROOT,
    SCRIPTS_ROOT,
    SCHEMA_MW_GOVERNANCE,
)

sys.path.insert(0, str(SCRIPTS_ROOT))
sys.path.insert(0, str(SCRIPTS_ROOT / "jobs"))

from sft_canonical_lib import (  # noqa: E402
    compile_mw_row,
    compile_retrieval_row,
    freeze_mw_codebook,
    has_route_id_gold,
    leak_markers,
    load_jsonl,
    query_overlaps,
    retrieval_case_bank,
)
from sft_synth_lib import (  # noqa: E402
    ResumeMismatch,
    SpendNotAllowed,
    TokenBucket,
    bind_resume_contract,
    budget_status,
    claim_item,
    connect_queue,
    cost_per_1k_accepted,
    enqueue_cases,
    export_queue_rows,
    finish_item,
    fleet_contract,
    fleet_price_fingerprint,
    lanes_of,
    load_fleet,
    maybe_ramp,
    parse_teacher_query,
    require_spend_allowed,
    rewrite_query,
    spend_cny,
)


def test_budget_and_cost_unit() -> None:
    assert budget_status(0.0, 0.0) == "unlimited"
    assert budget_status(8.0, 10.0) == "warn"
    assert budget_status(10.0, 10.0) == "stop"
    assert budget_status(5.0, 10.0) == "ok"
    lane = lanes_of(load_fleet(FLEET_SFT_SYNTH))["deepseek-v4-flash-0731"]
    spend = spend_cny(prompt_tokens=1_000_000, completion_tokens=1_000_000, lane=lane)
    assert spend > 0
    assert cost_per_1k_accepted(2.0, 1000) == 2.0


def test_queue_resume_and_rate() -> None:
    TokenBucket(1000).take()
    tmp = Path(tempfile.mkdtemp(prefix="sft-queue-"))
    conn = connect_queue(tmp / "q.sqlite")
    lock = threading.Lock()
    fleet = load_fleet(FLEET_SFT_SYNTH)
    lane = lanes_of(fleet)["template"]
    bind_resume_contract(
        conn,
        fleet_id="sft-synth-fleet-v1",
        lane="template",
        model_snapshot="template",
        prompt_version="sft-teacher-v2-query-only",
        price_fingerprint=fleet_price_fingerprint(fleet),
    )
    try:
        bind_resume_contract(
            conn,
            fleet_id="sft-synth-fleet-v1",
            lane="template",
            model_snapshot="other-model",
            prompt_version="sft-teacher-v2-query-only",
            price_fingerprint=fleet_price_fingerprint(fleet),
        )
    except ResumeMismatch:
        pass
    else:
        raise AssertionError("expected resume mismatch")
    n = enqueue_cases(conn, lock, [{"case_id": "c1", "idx": 0, "query": "去厨房"}], lane="template")
    assert n == 1
    assert enqueue_cases(conn, lock, [{"case_id": "c1", "idx": 0, "query": "去厨房"}], lane="template") == 0
    item = claim_item(conn, lock)
    assert item and item["case_id"] == "c1"
    finish_item(
        conn,
        lock,
        "c1",
        status="accepted",
        query="去厨房",
        result_json=json.dumps({"sample_id": "r1", "query": "去厨房"}, ensure_ascii=False),
    )
    row = conn.execute("SELECT status FROM items WHERE case_id='c1'").fetchone()
    assert row["status"] == "accepted"
    accepted, rejected = export_queue_rows(conn)
    assert len(accepted) == 1 and accepted[0]["sample_id"] == "r1"
    assert not rejected


def test_parser_and_spend_gate() -> None:
    query, err = parse_teacher_query('{"query":"把前门打开"}')
    assert query == "把前门打开" and err is None
    _, err = parse_teacher_query('{"query":"x","reason_code":"offtopic"}')
    assert err == "gold_leak"
    _, err = parse_teacher_query('{"answers":[]}')
    assert err == "gold_leak"
    fleet = load_fleet(FLEET_SFT_SYNTH)
    paid = lanes_of(fleet)["deepseek-v4-flash-0731"]
    try:
        require_spend_allowed(paid, allow_spend=False)
    except SpendNotAllowed:
        pass
    else:
        raise AssertionError("paid lane must refuse without --allow-spend")
    template = lanes_of(fleet)["template"]
    require_spend_allowed(template, allow_spend=False)
    out = rewrite_query({"query": "去厨房", "stem": "去厨房"}, template, allow_spend=False)
    assert out["query"] == "去厨房"
    workers, qps = maybe_ramp(
        paid,
        {"n": 40, "workers": 6, "qps": 4, "rate_429": 0.0, "retry_error_rate": 0.0, "p95_stable": True},
        {"min_n": 20, "max_429_rate": 0.01, "max_retry_error_rate": 0.02},
    )
    assert workers == 12 and qps == 8.0


def test_compilers_protect_gold() -> None:
    cases = retrieval_case_bank()
    case = next(c for c in cases if c.get("zh_slots"))
    compile_retrieval_row(case, case["query"], teacher_model="template")
    try:
        compile_retrieval_row(case, "你好", teacher_model="template")
    except ValueError as exc:
        assert "protected_slot" in str(exc)
    else:
        raise AssertionError("expected protected_slot")
    codebook = json.loads(CODEBOOK_MW_DISPOSITION_V1.read_text(encoding="utf-8"))
    frozen = freeze_mw_codebook(json.loads(SCHEMA_MW_GOVERNANCE.read_text(encoding="utf-8")))
    assert frozen["n_classes"] == codebook["n_classes"] == 20
    assert [c["reason_code"] for c in frozen["classes"]] == [c["reason_code"] for c in codebook["classes"]]
    mw_case = {
        "case_id": "MWC-test",
        "reason_code": "ready_to_execute",
        "function_calls": [{"name": "nod", "arguments": {}}],
        "gaps": [],
        "zh_slots": {},
        "kind": "execute",
        "cf_group": "CFG-test",
        "split": "train",
    }
    row = compile_mw_row(mw_case, "点一下头", teacher_model="template")
    assert row["act"] == "execute" and row["cell"] == "MW.OK"
    assert row["head_target"] == "reason_code"
    leak_row = {"prompt_text": "任务 <routes>", "serializer": "mei-route-serializer-v1"}
    assert has_route_id_gold(leak_row)
    assert leak_markers("<routes> hello") == ["<routes>"]


def test_isolation_and_promote_refuse() -> None:
    train = [{"sample_id": "t1", "query": "把客厅那盏灯打开"}]
    eval_rows = load_jsonl(ROOT / "notebook/evaluation/banks/mei-toolcall-v2/eval-bank-smoke.jsonl")
    hits = query_overlaps(train, eval_rows)
    assert hits
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "jobs" / "promote_sft_pack.py"),
            "--pack",
            "notebook/evaluation/banks/mei-toolcall-v2/eval-bank-smoke.jsonl",
            "--task",
            "mei-1.0-58m",
            "--state",
            "draft",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 5
    proc2 = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "run_sft_synth_staircase.py"),
            "--tier",
            "2k",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 4


def test_fullcall_home_recompile() -> None:
    from build_mei_toolcall_v2 import build

    cases, rows = build(limit=4, source="home", seed=20260827)
    assert rows
    assert all(r.get("serializer") == "mei-tool-call-serializer-v2" for r in rows)
    assert all(not has_route_id_gold(r) for r in rows)
    assert all(len(r.get("retrieved_tools") or []) <= 5 for r in rows)


def test_ledger_atomic_phase_and_missing_usage() -> None:
    from sft_ledger import BudgetHardStop, ProviderDisabled, SharedLedger

    tmp = Path(tempfile.mkdtemp(prefix="sft-ledger-"))
    fleet = load_fleet(FLEET_SFT_SYNTH)
    paid = lanes_of(fleet)["deepseek-v4-flash-0731"]
    ledger = SharedLedger(
        tmp / "ledger.sqlite",
        global_limit=1.0,
        phase="canary",
        phase_limit=0.05,
        lane_caps={paid.name: 0.5},
    )
    reserved = ledger.reserve(
        task="retrieval",
        case_id="c1",
        lane=paid,
        prompt_version="sft-teacher-v2-query-only",
        reserved_cny=0.04,
    )
    assert reserved == 0.04
    billed = ledger.settle(
        task="retrieval",
        case_id="c1",
        lane=paid,
        prompt_version="sft-teacher-v2-query-only",
        actual_cny=0.0,
        reserved_cny=0.04,
        usage={},
    )
    assert billed == 0.04
    try:
        ledger.reserve(
            task="retrieval",
            case_id="c2",
            lane=paid,
            prompt_version="sft-teacher-v2-query-only",
            reserved_cny=0.01,
        )
    except ProviderDisabled:
        pass
    else:
        raise AssertionError("missing usage must disable provider")
    ledger.enable_provider(paid.provider)
    ledger.reserve(
        task="retrieval",
        case_id="c2-retry",
        lane=paid,
        prompt_version="sft-teacher-v2-query-only",
        reserved_cny=0.01,
    )
    ledger.void(
        task="retrieval",
        case_id="c2-retry",
        lane=paid,
        prompt_version="sft-teacher-v2-query-only",
    )
    ledger.reserve(
        task="retrieval",
        case_id="c2-after-void",
        lane=paid,
        prompt_version="sft-teacher-v2-query-only",
        reserved_cny=0.01,
    )
    ledger2 = SharedLedger(tmp / "ledger2.sqlite", global_limit=1.0, phase="canary", phase_limit=0.02)
    ledger2.reserve(task="fullcall", case_id="a", lane=paid, prompt_version="v", reserved_cny=0.02)
    ledger2.settle(
        task="fullcall",
        case_id="a",
        lane=paid,
        prompt_version="v",
        actual_cny=0.02,
        reserved_cny=0.02,
        usage={"prompt_tokens": 10, "completion_tokens": 10, "dry_run": True},
    )
    try:
        ledger2.reserve(task="fullcall", case_id="b", lane=paid, prompt_version="v", reserved_cny=0.01)
    except BudgetHardStop:
        pass
    else:
        raise AssertionError("phase cap must hard-stop")
    ledger3 = SharedLedger(tmp / "ledger2.sqlite", global_limit=1.0, phase="2k", phase_limit=0.3)
    ledger3.reserve(
        task="fullcall",
        case_id="c",
        lane=paid,
        prompt_version="v",
        reserved_cny=0.05,
    )


def test_park_fingerprint_prompt_and_split() -> None:
    from park_toolcall_lib import park_fingerprint, python_park_call_ok, rewrite_guard
    from sft_canonical_lib import ensure_train_valid_split, render_v2_prompt
    from build_mei_retrieval_v2 import build as build_ret

    fp = park_fingerprint()
    toolset = json.loads((ROOT / "notebook/evaluation/shared/toolsets/mei-park-room-v1.json").read_text(encoding="utf-8"))
    assert fp == toolset["fingerprint"]
    cases, rows = build_ret(limit=24, teacher_model="template")
    assert any(c.get("split") == "valid" for c in cases)
    assert all(r.get("engineering_smoke") for r in rows)
    rendered = render_v2_prompt(
        [{"name": "control_room_devices", "description": "d", "parameters": {"type": "object", "properties": {}}}],
        "请开灯",
        system_facts="室温 31℃",
        entities={"ac": "ac-1", "light": "light-1"},
        selected_entity=None,
        permissions=["control_room_devices"],
    )
    assert "系统事实：" in rendered["text"]
    assert "点选实体：无" in rendered["text"]
    assert "目录项不是点选证据" in rendered["text"]
    assert "scenario_id" not in rendered["text"]
    contract = fleet_contract(load_fleet(FLEET_SFT_SYNTH))
    assert contract["global_budget_cny"] == 1000.0
    assert rewrite_guard({"stem": "把灯关上", "task": "fullcall", "intent_id": "light_off"}, "把灯打开") == "negation_changed"
    assert rewrite_guard({"stem": "把灯关上", "task": "fullcall", "intent_id": "light_off"}, "把灯关掉") is None
    assert rewrite_guard({"stem": "麻烦看下南京现在热不热", "task": "retrieval"}, "南京现在热吗") is None
    env_q = "当前环境有变化：有人进入房间。请决定是否调整房间设备。"
    assert rewrite_guard({"stem": env_q, "task": "fullcall", "park_layer": "L2", "intent_id": "env-person_enter"}, env_q) is None
    precool_facts = {
        "ac": {"health": "ready", "power": False, "targetC": 26},
        "light": {"health": "ready", "on": False},
        "reservation": {"id": "rsv-train"},
    }
    assert python_park_call_ok(
        {"system_facts": precool_facts},
        {"name": "control_room_devices", "arguments": {"ac_power": True, "ac_target_c": 26}},
    )
    facts = {
        "ac": {"health": "fault", "power": False, "targetC": 26},
        "light": {"health": "ready", "on": False},
        "reservation": None,
    }
    assert python_park_call_ok({"system_facts": facts}, {"name": "control_room_devices", "arguments": {"ac_power": True}}) is False
    fixture = json.loads(
        (ROOT.parent / "tools/mei-park/tests/fixtures/cross-lang-validator-v1.json").read_text(encoding="utf-8")
    )
    assert fixture["cases"]
    from sft_synth_lib import assign_pareto_lanes

    buckets = assign_pareto_lanes(
        [{"kind": "execute", "prefer_template": True} for _ in range(160)]
        + [{"kind": "missing", "high_risk": True} for _ in range(40)]
    )
    assert set(buckets) == {"deepseek-v4-flash-0731", "qwen-plus-2025-12-01", "qwen3.7-plus"}
    assert sum(len(v) for v in buckets.values()) == 200
    assert buckets["qwen-plus-2025-12-01"]
    assert any(c.get("prefer_template") for c in buckets["deepseek-v4-flash-0731"])
    assert any(c.get("high_risk") for c in buckets["qwen-plus-2025-12-01"])
    assert abs(len(buckets["deepseek-v4-flash-0731"]) / 200 - 0.60) < 0.02
    assert abs(len(buckets["qwen-plus-2025-12-01"]) / 200 - 0.25) < 0.02
    assert abs(len(buckets["qwen3.7-plus"]) / 200 - 0.15) < 0.02


def test_promote_refuses_accepted_without_base() -> None:
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "promote_sft_v2_releases.py"),
            "--state",
            "accepted",
            "--allow-accepted",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    blob = proc.stdout + proc.stderr
    assert "0303" in blob or "missing" in blob or "sft_ready" in blob or "base" in blob
    proc10 = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_ROOT / "promote_sft_v2_releases.py"),
            "--state",
            "accepted",
            "--allow-accepted",
            "--paid",
            "--tier",
            "10k",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert proc10.returncode != 0
    blob10 = proc10.stdout + proc10.stderr
    assert "0303" in blob10 or "missing" in blob10 or "sft_ready" in blob10 or "base" in blob10


def test_10k_pack_contract_and_semantics() -> None:
    from repo_paths import (
        PACK_MEI_MW_DISPOSITION_V2_10K_PAID,
        PACK_MEI_RETRIEVAL_V2_10K_PAID,
        PACK_MEI_TOOLCALL_V2_ORACLE_10K_PAID,
        sft_v2_candidate_pack_name,
    )
    from park_toolcall_lib import park_fullcall_cases, park_retrieval_specs
    from build_mei_retrieval_v2 import build as build_ret
    from build_mei_mw_disposition_v2 import build as build_mw

    assert sft_v2_candidate_pack_name("mei-retrieval-v2", 24) == "mei-retrieval-v2-smoke.jsonl"
    assert sft_v2_candidate_pack_name("mei-retrieval-v2", 2000) == "mei-retrieval-v2-2k.candidates.jsonl"
    assert sft_v2_candidate_pack_name("mei-retrieval-v2", 10000) == "mei-retrieval-v2-10k.candidates.jsonl"
    assert PACK_MEI_RETRIEVAL_V2_10K_PAID.name.endswith("10k.paid.candidates.jsonl")
    assert PACK_MEI_TOOLCALL_V2_ORACLE_10K_PAID.name.endswith("10k.paid.candidates.jsonl")
    assert PACK_MEI_MW_DISPOSITION_V2_10K_PAID.name.endswith("10k.paid.candidates.jsonl")
    specs400 = park_retrieval_specs(400)
    specs2k = park_retrieval_specs(2000)
    assert len(specs400) == 400
    assert len(specs2k) == 2000
    assert [s["stem"] for s in specs400] == [s["stem"] for s in specs2k[:400]]
    assert len({s["stem"] for s in specs2k}) == 2000
    park = park_fullcall_cases(quota={"L0": 1280, "L1": 640, "L2": 640, "L3": 640})
    assert len(park) >= 3200
    from collections import Counter

    layers = Counter(c.get("park_layer") for c in park)
    assert layers["L0"] == 1280
    assert layers["L1"] == 640
    assert layers["L2"] == 640
    assert layers["L3"] == 640
    cases, rows = build_ret(limit=48, teacher_model="template")
    assert len(rows) == 48
    assert len({r["query"] for r in rows}) == 48
    mw_cases, mw_rows = build_mw(limit=48)
    assert len(mw_rows) == 48
    assert len({r["reason_code"] for r in mw_rows}) >= 8


def main() -> int:
    test_budget_and_cost_unit()
    test_queue_resume_and_rate()
    test_parser_and_spend_gate()
    test_compilers_protect_gold()
    test_isolation_and_promote_refuse()
    test_fullcall_home_recompile()
    test_ledger_atomic_phase_and_missing_usage()
    test_park_fingerprint_prompt_and_split()
    test_promote_refuses_accepted_without_base()
    test_10k_pack_contract_and_semantics()
    print(json.dumps({"ok": True, "tests": 10}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
