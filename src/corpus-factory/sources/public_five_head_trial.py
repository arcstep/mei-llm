"""Source-preserving five-head adaptation preview; never fabricate model outcomes."""
from collections import Counter
from pathlib import Path
import copy
import hashlib
import json
import re
import shutil
import sys

from profiling import resolve_path, ROOT, digest
from local_diagnostics import parse_declared_tool_calls
from tokenizer_candidate import write_new
from source_manager import load_tokenizer

sys.path.insert(0, str(ROOT / "src/platform/_shared/runtime"))
from canonical_json import dumps_canonical
from grammar import dump_calls
from provenance import validate_generated_call
from context_budget import validate_prompt_budget


def visible_spans(query):
    """Extract literal values without consulting a target, future result or model."""
    patterns = [r'"([^"\n]+)"', r"(?<!\w)'([^'\n]+)'(?!\w)",
                r'\b[A-Z]{1,2}\d[A-Z\d]?\s+\d[A-Z]{2}\b',
                r'\b[A-Z][A-Z0-9]{1,}\b', r'(?<!\w)\d+(?:\.\d+)?(?!\w)']
    found = {}
    for pattern in patterns:
        for m in re.finditer(pattern, query):
            a, b = m.span(1) if m.lastindex else m.span()
            found[(a,b)] = {"source": "user_query", "start": a, "end": b, "value": query[a:b]}
    return [dict(item, id=f"q{a}_{b}") for (a,b),item in sorted(found.items())]


def source_evidence(query, tools):
    spans = visible_spans(query)
    evidence = []
    for tool in tools:
        for argument, schema in tool["parameters"].get("properties", {}).items():
            for span in spans:
                value = span["value"]
                if schema.get("type") == "integer":
                    if not re.fullmatch(r"\d+", value): continue
                    value = int(value)
                elif schema.get("type") == "number":
                    if not re.fullmatch(r"\d+(?:\.\d+)?", value): continue
                    value = float(value)
                elif schema.get("type") != "string":
                    continue
                evidence.append({"tool": tool["name"], "argument": argument, "value": value,
                    "source": "user_query_literal", "locator": span["id"], "verified": True})
    return spans, evidence


def run(config, out):
    pilot_path = resolve_path(ROOT / config["pilot"])
    if digest(pilot_path) != config["pilot_sha256"]: raise ValueError("pilot changed")
    pilot = [json.loads(line) for line in pilot_path.open()]
    out = resolve_path(ROOT / out)
    if shutil.disk_usage(ROOT).free < 100*1024**3: raise ValueError("disk reserve")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "implementation.py.snapshot")
    write_new(out / "config.json", config)
    tok = load_tokenizer(resolve_path(ROOT / config["tokenizer_manifest"]))
    original_files = {}
    cases = []
    checks = Counter()
    review = ["# 公开工具语料五头转换试批", "", "12个源对话、4类各3个；保留英文原文，不新增问题或工具返回值。全部为可审核转换候选，不是正式SFT release或独立Eval。", "",
        "本次只选择单步、参数明确且原目录不超过5个工具的样本。选择具有目的性，不能推算整池通过率。", "",
        "检索仅给来源目标工具正例，其他工具未自动标负例。处置仅展示execute候选。confidence等待真实模型；英文原始解说不冒充48-token中文目标。", ""]
    for spec in config["cases"]:
        row = pilot[spec["pilot_row"]]
        if row["case_id"] != spec["case_id"]: raise ValueError("case changed")
        origin = row["origin"]
        if origin["path"] not in original_files:
            path = resolve_path(origin["path"])
            if digest(path) != origin["sha256"]: raise ValueError("original changed")
            original_files[origin["path"]] = json.loads(path.read_text())
        original = original_files[origin["path"]][origin["row_index"]]
        turns = original["conversations"]
        query = turns[0]["value"]
        if turns[0]["from"] != "user" or row["messages"] != [{"role":"user", "content":query}]:
            raise ValueError("not an exact first-user view")
        tools = copy.deepcopy(row["visible_tools"])
        if not 1 <= len(tools) <= 5: raise ValueError("catalog not a single batch")
        source_call = parse_declared_tool_calls(turns[1]["value"], [t["name"] for t in tools])
        if source_call != row["gold_lm_target"]: raise ValueError("source target mismatch")
        # An explicit alias registry changes only tool identifiers, never semantics or argument values.
        aliases = {t["name"]: "public_"+re.sub(r"[^a-z0-9]+", "_", t["name"].lower()).strip("_") for t in tools}
        if len(set(aliases.values())) != len(tools): raise ValueError("alias collision")
        for t in tools: t["name"] = aliases[t["name"]]
        target = [{"name": aliases[source_call[0]["name"]], "arguments":copy.deepcopy(source_call[0]["arguments"])}]
        spans, evidence = source_evidence(query, tools)
        request = {"query": query, "evidence": evidence}
        target_text = dump_calls(target)
        validation = validate_generated_call(target_text, tools=tools, request=request, enforce_confidence=False)
        if not validation["ok"] or validation.get("refuse"): raise ValueError(f"gold failed {spec}: {validation}")
        # Negatives are validator tests only, never additional source training examples.
        no_evidence = validate_generated_call(target_text, tools=tools, request={"query":query}, enforce_confidence=False)
        if no_evidence["ok"]: raise ValueError("missing-evidence control passed")
        checks.update(source_call_equal=1, grammar_schema_provenance_passed=1, missing_evidence_rejected=1)
        prompt = ('<|im_start|>system\nSelect one tool call or output []. Preserve parameter values.\n<tools>'
                  + dumps_canonical(tools) + '</tools>\n<literal_values>' + dumps_canonical(spans)
                  + '</literal_values><|im_end|>\n<|im_start|>user\n'+query+'<|im_end|>\n<|im_start|>assistant\n')
        input_tokens = validate_prompt_budget(tok, prompt, output_reserve=128)
        target_ids = tok.encode(target_text, add_eos=True)
        if len(target_ids)>128: raise ValueError("target reserve overflow")
        prompt_ids = tok.encode(prompt, add_bos=True)
        if tok.decode(tok.encode(prompt)) != prompt: raise ValueError("roundtrip failed")
        result_rows = json.loads(turns[2]["value"])
        if turns[2]["from"] != "tool" or len(result_rows)!=1 or result_rows[0]["name"] != source_call[0]["name"]:
            raise ValueError("result/call association mismatch")
        if turns[3]["from"] != "assistant": raise ValueError("narration turn missing")
        raw_result = result_rows[0]["results"]
        heads = {
            "lm": {"status":"preview_validated", "target":target, "target_text":target_text},
            "retrieval": {"status":"source_positive_only", "query":query,
                "positive_tool":target[0]["name"], "other_tools_status":"unlabelled_not_assumed_negative", "learned_batch":"pending_model"},
            "disposition": {"status":"mapping_candidate_not_final_codebook", "action":"execute",
                "reason":"explicit_request_arguments_supported", "legacy_reason":"ready_to_execute", "evidence_ids":[s["id"] for s in spans]},
            "confidence": {"status":"pending_model", "target":None, "reason":"No model-generated call or correctness outcome exists."},
            "narration": {"status":"public_result_and_text_reserved", "source_result":raw_result,
                "source_text":turns[3]["value"], "source_language":"en", "mei_chinese_target":None,
                "review":spec["narration_review"], "live_execution_verified":False,
                "reason":"Associated published synthetic result, not host-executed verified result; no Chinese rewriting performed."}}
        item = {"case_id":row["case_id"], "category":spec["category"], "label":spec["label"],
            "source":"ToolACE", "source_origin":origin, "source_generation":"publisher_synthetic", "language":"en",
            "source_association_group":row["association_group"], "split":"review_only_cpt_shared",
            "locked_eval_eligible":False, "formal_training_admitted":False,
            "query":query, "tools":tools, "tool_aliases":aliases, "visible_spans":spans,
            "evidence_rule":"Literal candidates from query and all catalog properties, independent of target; literal presence does not certify semantic slot assignment.",
            "request_for_validator":request, "heads":heads, "validation":validation,
            "validation_scope":"static grammar/schema/provenance; deterministic default MW and disabled confidence; NOT model E2E or real API execution",
            "preview_serializer":"mei-public-single-step-preview-v1-not-production",
            "prompt":prompt, "input_tokens":input_tokens, "output_reserve":128,
            "input_ids":prompt_ids+target_ids, "loss_mask":[0]*len(prompt_ids)+[1]*len(target_ids)}
        cases.append(item)
        review.extend([f'## {spec["category"]} · {spec["label"]}', "", "来源原文："+query, "",
            "原工具："+source_call[0]["name"], "", "Mei调用候选：", "```json", target_text, "```", "",
            f'输入 {input_tokens} token，输出预留128；语法/schema/参数字面来源验证通过。', "",
            "检索：原工具为正例，其余未标负例。处置：execute建议。Confidence：待模型。", "",
            "解说审阅："+spec["narration_review"], "", "原始解说："+turns[3]["value"], ""])
    for name, payload in [("cases.jsonl",cases), ("browser-audit-texts.jsonl",[
        {"text":c["prompt"]+c["heads"]["lm"]["target_text"],
         "text_sha256":hashlib.sha256((c["prompt"]+c["heads"]["lm"]["target_text"]).encode()).hexdigest()} for c in cases])]:
        with (out/name).open("x") as h:
            for item in payload: h.write(json.dumps(item,ensure_ascii=False)+"\n")
    with (out/"REVIEW.md").open("x") as h: h.write("\n".join(review))
    report = {"status":"reviewable_conversion_trial", "actual_cases":len(cases), "target_cases":12,
        "source_conversations":len({c["source_association_group"] for c in cases}),
        "semantic_independence":"not_certified", "categories":dict(Counter(c["category"] for c in cases)),
        "language":{"en":len(cases)}, "checks":dict(checks), "all_empty_baseline_exact":0,
        "head_counts":{"lm_preview":len(cases),"retrieval_positive":len(cases),"disposition_candidate":len(cases),
                       "narration_source_pairs":len(cases),"narration_mei_ready":0,"confidence_results":0},
        "formal_admitted_cases":0, "max_input_tokens":max(c["input_tokens"] for c in cases),
        "tokenizer_manifest":config["tokenizer_manifest"], "tokenizer_model_sha256":tok.model_sha256,
        "coverage_gaps":["Chinese requests/narration", "no-call branches", "multi-step", "learned retrieval", "model confidence", "production serializer integration"],
        "downloads":0,"teacher_calls":0,"external_tool_calls":0,"model_training":False,
        "evidence_files":{n:digest(out/n) for n in ("cases.jsonl","browser-audit-texts.jsonl","REVIEW.md","implementation.py.snapshot","config.json")}}
    write_new(out/"REPORT.json",report)
    return report
