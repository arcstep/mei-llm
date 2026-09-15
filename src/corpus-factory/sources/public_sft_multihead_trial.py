"""Adapt small public task-data batches without inventing missing head labels."""
from collections import Counter, defaultdict
from pathlib import Path
import ast
import copy
import hashlib
import heapq
import json
import re
import shutil
import sys
import warnings
import zipfile

from profiling import ROOT, digest, resolve_path
from source_manager import load_tokenizer
from tokenizer_candidate import write_new

sys.path.insert(0, str(ROOT / "src/platform/_shared/runtime"))
from canonical_json import dumps_canonical
from grammar import dump_calls


MOSS_POSITIONAL = {
    "Search": ["query"],
    "Calculate": ["expression"],
    "Solve": ["equation"],
    "Text2Image": ["description"],
}

DOMAIN_SLUGS = {
    "景点": "attraction", "旅游景点": "attraction", "餐馆": "restaurant", "餐厅": "restaurant",
    "酒店": "hotel", "地铁": "metro", "出租": "taxi", "医院": "hospital", "天气": "weather",
    "汽车": "car", "火车": "train", "电影": "movie", "电脑": "computer", "电视剧": "tv_series",
    "辅导班": "tutoring", "飞机": "flight",
}


def _sha_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean_marker(value, prefix, suffix):
    value = (value or "").strip()
    if value.startswith(prefix):
        value = value[len(prefix):]
    if value.endswith(suffix):
        value = value[:-len(suffix)]
    return value.strip()


def parse_python_calls(value, positional=None):
    """Parse only a literal list/call expression; never evaluate source code."""
    value = value.strip()
    if value.startswith("API-Request:"):
        value = value.split(":", 1)[1].strip()
    if value.startswith("<|Commands|>:"):
        value = value[len("<|Commands|>:"):]
    if value.endswith("<eoc>"):
        value = value[:-5]
    value = value.strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            tree = ast.parse(value, mode="eval").body
            nodes = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
        except SyntaxError:
            tree = ast.parse(value, mode="exec")
            nodes = [n.value for n in tree.body if isinstance(n, ast.Expr)]
    calls = []
    for node in nodes:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError("non-literal tool expression")
        name = node.func.id
        names = (positional or {}).get(name, [])
        if len(node.args) > len(names):
            raise ValueError("unknown positional arguments")
        arguments = {names[i]: ast.literal_eval(arg) for i, arg in enumerate(node.args)}
        for item in node.keywords:
            if item.arg is None:
                raise ValueError("expanded keyword arguments are not allowed")
            arguments[item.arg] = ast.literal_eval(item.value)
        calls.append({"name": name, "arguments": arguments})
    return calls


def stable_sample(items, count, key):
    """Choose the lowest fixed hashes without depending on source order."""
    heap = []
    for serial, item in enumerate(items):
        score = int(_sha_text(key(item)), 16)
        entry = (-score, serial, item)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    return [x[2] for x in sorted(heap, key=lambda x: (-x[0], x[1]))]


def _tool(name, description, properties, required=None):
    return {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties, "required": required or []}}


def moss_tools(meta):
    catalog = {
        "Search": _tool("public_search", "搜索与当前问题相关的资料。", {"query": {"type": "string"}}, ["query"]),
        "Calculate": _tool("public_calculate", "计算给定表达式。", {"expression": {"type": "string"}}, ["expression"]),
        "Solve": _tool("public_solve", "求解给定方程。", {"equation": {"type": "string"}}, ["equation"]),
        "Text2Image": _tool("public_text_to_image", "根据文字描述生成图片。", {"description": {"type": "string"}}, ["description"]),
    }
    enabled = []
    for line in meta.splitlines():
        if "enabled" not in line or "API:" not in line:
            continue
        for name in catalog:
            if name + "(" in line:
                enabled.append(catalog[name])
    return enabled


def alias_calls(calls, prefix="public_"):
    result = []
    for call in calls:
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", call["name"]).strip("_").lower()
        result.append({"name": prefix + name, "arguments": copy.deepcopy(call["arguments"])})
    return result


def argument_provenance(calls, visible_text, annotation_kind):
    literal = derived = 0
    for call in calls:
        for value in call["arguments"].values():
            values = value.values() if isinstance(value, dict) else value if isinstance(value, list) else [value]
            for item in values:
                if item in (None, "", []):
                    continue
                if str(item) in visible_text:
                    literal += 1
                else:
                    derived += 1
    return {"literal_values": literal, "source_annotation_values": derived,
            "annotation_kind": annotation_kind, "semantic_review_required": derived > 0}


def make_unit(*, unit_id, history, query, state, tools, calls, result, narration,
              disposition, provenance_kind, source_validation, tokenizer):
    if len(tools) > 5:
        raise ValueError("visible tool batch exceeds five")
    names = {tool["name"] for tool in tools}
    visible = all(call["name"] in names for call in calls)
    target_text = dump_calls(calls)
    prompt_object = {"history": history, "query": query, "state": state, "tools": tools}
    prompt = dumps_canonical(prompt_object)
    input_tokens = len(tokenizer.encode(prompt, add_bos=True))
    target_tokens = len(tokenizer.encode(target_text, add_eos=True))
    provenance = argument_provenance(calls, prompt, provenance_kind)
    result_present = result not in (None, "", [], {})
    narration_present = bool((narration or "").strip())
    return {
        "unit_id": unit_id,
        "history": history,
        "query": query,
        "visible_state": state,
        "visible_tools": tools,
        "source_result": result,
        "source_narration": narration,
        "source_validation": source_validation,
        "heads": {
            "lm": {"status": ("source_backed_needs_semantic_review" if provenance["semantic_review_required"]
                              else "source_backed_candidate") if visible else "rejected_tool_not_visible",
                   "target": calls, "target_text": target_text},
            "retrieval": {"status": "source_positive_candidate" if calls and visible else "not_applicable_no_call",
                          "positive_tools": [call["name"] for call in calls],
                          "other_visible_tools": "candidate_negatives_not_yet_certified"},
            "disposition": disposition,
            "confidence": {"status": "pending_model", "target": None},
            "narration": {"status": "source_result_pair_requires_grounding_review"
                                      if result_present and narration_present else "source_response_only_not_grounded",
                          "target": narration if narration_present else None},
        },
        "argument_provenance": provenance,
        "budget": {"input_tokens": input_tokens, "target_tokens": target_tokens,
                   "output_reserve": 128, "fits_2048": input_tokens + max(128, target_tokens) <= 2048},
    }


def _moss_groups(config, tokenizer):
    candidates = []
    for source in config["sources"]["moss"]["files"]:
        path = resolve_path(ROOT / source["path"])
        if digest(path) != source["sha256"]:
            raise ValueError("MOSS source changed")
        with zipfile.ZipFile(path) as archive:
            member = archive.namelist()[0]
            with archive.open(member) as handle:
                for line_number, raw in enumerate(handle):
                    row = json.loads(raw)
                    user_text = " ".join(turn.get("Human", "") for turn in row.get("chat", {}).values())
                    if not re.search(r"[\u3400-\u9fff]", user_text):
                        continue
                    parseable = 0
                    for turn in row.get("chat", {}).values():
                        try:
                            parseable += bool(parse_python_calls(turn.get("Commands", ""), MOSS_POSITIONAL))
                        except (SyntaxError, ValueError):
                            pass
                    if parseable:
                        candidates.append((source["path"], path.name, member, line_number, row))
    chosen = stable_sample(candidates, config["count_per_source"],
                           lambda x: "moss:" + x[1] + ":" + str(x[4].get("conversation_id")) + ":" + str(x[3]))
    groups = []
    for source_path, filename, member, line_number, row in chosen:
        history = []
        units = []
        tools = moss_tools(row.get("meta_instruction", ""))
        aliases = {"Search": "public_search", "Calculate": "public_calculate",
                   "Solve": "public_solve", "Text2Image": "public_text_to_image"}
        for turn_name, turn in row.get("chat", {}).items():
            query = _clean_marker(turn.get("Human"), "<|Human|>:", "<eoh>")
            try:
                raw_calls = parse_python_calls(turn.get("Commands", ""), MOSS_POSITIONAL)
            except (SyntaxError, ValueError):
                continue
            calls = [{"name": aliases[x["name"]], "arguments": x["arguments"]} for x in raw_calls]
            result = _clean_marker(turn.get("Tool Responses"), "<|Results|>:", "<eor>")
            narration = _clean_marker(turn.get("MOSS"), "<|MOSS|>:", "<eom>")
            units.append(make_unit(unit_id=turn_name, history=copy.deepcopy(history), query=query, state={},
                tools=tools, calls=calls, result=result, narration=narration,
                disposition={"status": "source_mapping_candidate", "action": "execute" if calls else "complete",
                             "reason": "published_tool_call" if calls else "published_no_call"},
                provenance_kind="publisher_semantic_tool_annotation",
                source_validation="Published synthetic tool/result association; inner thoughts excluded; semantics not yet verified.",
                tokenizer=tokenizer))
            history.extend([{"role": "user", "content": query}, {"role": "assistant", "content": narration}])
        groups.append({"group_id": "moss:" + filename + ":" + str(row.get("conversation_id")) + ":" + str(line_number),
            "source": "MOSS", "language": "zh", "source_split": "published_train", "source_generation": "publisher_synthetic",
            "origin": {"path": source_path,
                       "archive_member": member, "line_number": line_number},
            "formal_training_admitted": False, "locked_eval_eligible": False, "units": units})
    return groups


def _leading_json_objects(value):
    result = []
    for line in value.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            break
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            break
        if isinstance(item, dict) and item.get("apiCode"):
            result.append(item)
    return result


def _api_tool(item):
    properties = {}
    required = []
    type_map = {"str": "string", "float": "number", "int": "integer", "bool": "boolean",
                "list": "array", "dict": "object"}
    for name, spec in (item.get("parameters") or {}).items():
        if not isinstance(spec, dict):
            spec = {"type": "string", "description": str(spec)}
        entry = copy.deepcopy(spec)
        entry["type"] = type_map.get(entry.get("type"), entry.get("type", "string"))
        is_required = bool(entry.pop("required", False))
        properties[name] = entry
        if is_required:
            required.append(name)
    alias = "public_apibank_" + re.sub(r"[^a-zA-Z0-9_]+", "_", item["apiCode"]).strip("_").lower()
    return _tool(alias, item.get("description", ""), properties, required)


def _api_bank_groups(config, tokenizer):
    source = config["sources"]["api_bank"]
    per_level = [34, 33, 33]
    groups = []
    for level, wanted in zip((1, 2, 3), per_level):
        api_path = resolve_path(ROOT / source["levels"][str(level)]["api_path"])
        response_path = resolve_path(ROOT / source["levels"][str(level)]["response_path"])
        if digest(api_path) != source["levels"][str(level)]["api_sha256"] or digest(response_path) != source["levels"][str(level)]["response_sha256"]:
            raise ValueError("API-Bank source changed")
        api_rows = json.loads(api_path.read_text())
        response_rows = json.loads(response_path.read_text())
        response_index = defaultdict(list)
        for row in response_rows:
            response_index[row["input"].rsplit("\nAPI-Request:", 1)[0]].append(row)
        eligible = []
        for row_index, row in enumerate(api_rows):
            try:
                raw_calls = parse_python_calls(row.get("output", ""))
            except (SyntaxError, ValueError):
                continue
            definitions = _leading_json_objects(row.get("input", ""))
            declared = {item["apiCode"] for item in definitions}
            if not raw_calls or len(raw_calls) > 5 or any(call["name"] not in declared for call in raw_calls):
                continue
            prefix = row["input"].rsplit("\nGenerate API Request:", 1)[0]
            response = response_index[prefix][0] if response_index.get(prefix) else None
            eligible.append((row_index, row, response, definitions, raw_calls))
        chosen = stable_sample(eligible, wanted, lambda x: f"api-bank:{level}:{x[0]}:" + x[1]["output"])
        for row_index, row, response, definitions, raw_calls in chosen:
            alias = {item["apiCode"]: _api_tool(item)["name"] for item in definitions}
            positives = [item for item in definitions if item["apiCode"] in {x["name"] for x in raw_calls}]
            negatives = [item for item in definitions if item["apiCode"] not in {x["name"] for x in raw_calls}]
            tools = [_api_tool(item) for item in (positives + negatives)[:5]]
            calls = [{"name": alias[x["name"]], "arguments": x["arguments"]} for x in raw_calls]
            query = row["input"].rsplit("\nUser:", 1)[-1].rsplit("\nGenerate API Request:", 1)[0].strip()
            dialogue_context = row["input"]
            for definition in definitions:
                encoded = json.dumps(definition, ensure_ascii=False)
                dialogue_context = dialogue_context.replace(encoded + "\n", "", 1)
            dialogue_context = dialogue_context.rsplit("\nUser:", 1)[0].strip()
            result = ""
            narration = ""
            if response is not None:
                result_suffix = response["input"].rsplit("\nAPI-Request:", 1)[-1]
                result = result_suffix.split("->", 1)[1] if "->" in result_suffix else result_suffix
                narration = response.get("output", "")
            unit = make_unit(unit_id="api_call", history=[], query=query, state={"source_dialogue_context": dialogue_context},
                tools=tools, calls=calls, result=result, narration=narration,
                disposition={"status": "source_mapping_candidate", "action": "execute", "reason": "published_api_call"},
                provenance_kind="publisher_api_annotation",
                source_validation=("API and response rows paired by exact pre-call dialogue prefix; semantics not yet verified."
                                   if response is not None else "Published API-call view has no exact response-row pair; narration supervision unavailable."),
                tokenizer=tokenizer)
            if any(call["name"] == "ToolSearcher" for call in raw_calls):
                discovered = None
                match = re.search(r"API:\s*([^|\"\n]+)", result)
                if match:
                    discovered = match.group(1).strip()
                unit["heads"]["lm"]["status"] = "architecture_mismatch_retrieval_only"
                unit["heads"]["retrieval"] = {
                    "status": "source_tool_discovery_query_candidate",
                    "query_rewrite": raw_calls[0]["arguments"].get("keywords"),
                    "discovered_tool": discovered,
                    "positive_tool_status": "present_in_paired_result" if discovered else "pending_response_pair"
                }
                unit["heads"]["disposition"] = {"status": "unlabelled", "action": None,
                                                   "reason": "ToolSearcher is internal retrieval in Mei, not an executable LM tool."}
                unit["heads"]["narration"]["status"] = "tool_discovery_result_not_narration"
            groups.append({"group_id": f"api-bank:lv{level}:view:{row_index}", "source": "API-Bank", "language": "en",
                "source_split": "published_train", "source_generation": "publisher_synthetic",
                "independent_task_status": "not_certified_release_has_no_dialogue_id",
                "origin": {"api_path": source["levels"][str(level)]["api_path"], "api_row": row_index,
                           "response_path": source["levels"][str(level)]["response_path"] if response is not None else None},
                "formal_training_admitted": False, "locked_eval_eligible": False, "units": [unit]})
    return groups


def _candidate_group_ids(path, sha256, count, prefix):
    if digest(path) != sha256:
        raise ValueError(prefix + " candidate changed")
    rows = [json.loads(line) for line in path.open()]
    selected = stable_sample(rows, count, lambda x: prefix + ":" + str(x.get("group_id")) + ":" + x["id"])
    return [str(row["group_id"]) for row in selected]


def _domain_catalog(prefix, domains):
    result = []
    for domain in sorted(domains):
        name = prefix + DOMAIN_SLUGS.get(domain, re.sub(r"[^a-zA-Z0-9]+", "_", domain).strip("_"))
        if name == prefix:
            name += _sha_text(domain)[:8]
        result.append(_tool(name, f"查询{domain}数据库；约束与请求字段来自当前已验证状态。",
            {"constraints": {"type": "object"},
             "requested_fields": {"type": "array", "items": {"type": "string"}},
             "selected_entities": {"type": "array", "items": {"type": "string"}}},
            ["constraints", "requested_fields"]))
    return result


def _crosswoz_groups(config, tokenizer):
    source = config["sources"]["crosswoz"]
    candidate = resolve_path(ROOT / source["candidate_path"])
    ids = _candidate_group_ids(candidate, source["candidate_sha256"], config["count_per_source"], "crosswoz")
    archive = resolve_path(ROOT / source["archive_path"])
    if digest(archive) != source["archive_sha256"]:
        raise ValueError("CrossWOZ archive changed")
    with zipfile.ZipFile(archive) as z:
        raw = json.load(z.open(source["archive_member"]))
    domains = ["景点", "餐馆", "酒店", "地铁", "出租"]
    catalog = _domain_catalog("public_crosswoz_", domains)
    tool_by_domain = dict(zip(sorted(domains), catalog))
    groups = []
    for group_id in ids:
        row = raw[group_id]
        history = []
        previous_state = {}
        units = []
        messages = row.get("messages", [])
        for index in range(0, len(messages) - 1, 2):
            user, system = messages[index], messages[index + 1]
            if user.get("role") != "usr" or system.get("role") != "sys":
                continue
            acts = user.get("dialog_act", [])
            active = []
            for act in acts:
                if len(act) >= 2 and act[1] in domains and act[1] not in active:
                    active.append(act[1])
            calls = []
            state = system.get("sys_state_init", {})
            for domain in active:
                dstate = state.get(domain, {})
                constraints = {k: v for k, v in dstate.items() if k != "selectedResults" and v not in (None, "", [])}
                for act in acts:
                    if len(act) >= 4 and act[0] == "Inform" and act[1] == domain and act[2] and act[3] not in (None, "", "none"):
                        constraints[act[2]] = act[3]
                requested = [a[2] for a in acts if len(a) >= 4 and a[0] == "Request" and a[1] == domain and a[2]]
                arguments = {"constraints": constraints, "requested_fields": requested}
                if dstate.get("selectedResults"):
                    arguments["selected_entities"] = dstate["selectedResults"]
                calls.append({"name": tool_by_domain[domain]["name"],
                              "arguments": arguments})
            selected = {d: system.get("sys_state", {}).get(d, {}).get("selectedResults", []) for d in active}
            selected = {k: v for k, v in selected.items() if v}
            relaxed = False
            for domain in active:
                before = {k: v for k, v in state.get(domain, {}).items() if k != "selectedResults" and v not in (None, "", [])}
                after = {k: v for k, v in system.get("sys_state", {}).get(domain, {}).items() if k != "selectedResults" and v not in (None, "", [])}
                if before != after:
                    relaxed = True
            is_bye = any(len(a) > 0 and a[0] == "General" and len(a) > 1 and a[1] == "bye" for a in acts)
            disposition = ({"status": "source_mapping_candidate", "action": "review", "reason": "source_relaxed_constraints"}
                           if calls and relaxed else
                           {"status": "source_mapping_candidate", "action": "execute", "reason": "annotated_database_query"}
                           if calls else {"status": "source_mapping_candidate" if is_bye else "unlabelled",
                                          "action": "complete" if is_bye else None, "reason": "annotated_bye" if is_bye else None})
            unit = make_unit(unit_id=f"turn_{index//2}", history=copy.deepcopy(history), query=user["content"], state=previous_state,
                tools=catalog, calls=calls, result=selected, narration=system.get("content", ""), disposition=disposition,
                provenance_kind="crosswoz_dialogue_state_annotation",
                source_validation="Call reconstructed from published user act and pre-query state; selected result names only, not a full host execution result.",
                tokenizer=tokenizer)
            if relaxed:
                unit["heads"]["lm"]["status"] = "source_policy_relaxation_requires_review"
                unit["heads"]["narration"]["status"] = "source_result_pair_requires_policy_review"
                unit["source_validation"] = "Published system changed database constraints before selecting results; do not admit as direct call gold without a policy decision."
            elif calls:
                unit["heads"]["narration"]["status"] = "source_partial_result_not_grounded"
            units.append(unit)
            history.extend([{"role": "user", "content": user["content"]}, {"role": "assistant", "content": system.get("content", "")}])
            previous_state = system.get("sys_state", {})
        groups.append({"group_id": "crosswoz:" + group_id, "source": "CrossWOZ", "language": "zh",
            "source_split": "published_train_cpt_shared", "source_generation": "human_dialogue_with_annotations",
            "origin": {"archive_path": source["archive_path"], "archive_member": source["archive_member"], "source_group_id": group_id},
            "formal_training_admitted": False, "locked_eval_eligible": False, "units": units})
    return groups


def _risawoz_groups(config, tokenizer):
    source = config["sources"]["risawoz"]
    candidate = resolve_path(ROOT / source["candidate_path"])
    ids = _candidate_group_ids(candidate, source["candidate_sha256"], config["count_per_source"], "risawoz")
    archive = resolve_path(ROOT / source["archive_path"])
    if digest(archive) != source["archive_sha256"]:
        raise ValueError("RiSAWOZ archive changed")
    with zipfile.ZipFile(archive) as z:
        rows = json.load(z.open(source["archive_member"]))
    raw = {row["dialogue_id"]: row for row in rows}
    domains = sorted({domain for row in raw.values() for domain in row.get("domains", [])})
    catalog = _domain_catalog("public_risawoz_", domains)
    tool_by_domain = dict(zip(sorted(domains), catalog))
    groups = []
    for group_id in ids:
        row = raw[group_id]
        history = []
        previous_belief = {}
        units = []
        for turn in row.get("dialogue", []):
            active = [d for d in turn.get("turn_domain", []) if d in tool_by_domain]
            visible_domains = active + [d for d in sorted(domains) if d not in active]
            visible_tools = [tool_by_domain[d] for d in visible_domains[:5]]
            belief = turn.get("belief_state", {})
            constraints = belief.get("inform slot-values", {})
            requested = belief.get("turn request", [])
            results = turn.get("db_results", [])
            calls = []
            if results:
                for domain in active:
                    prefix = domain + "-"
                    domain_constraints = {k[len(prefix):]: v for k, v in constraints.items() if k.startswith(prefix)}
                    calls.append({"name": tool_by_domain[domain]["name"],
                                  "arguments": {"constraints": domain_constraints, "requested_fields": list(requested)}})
            actions = turn.get("system_actions", [])
            no_offer = any(a and a[0] == "NoOffer" for a in actions)
            asks = any(a and a[0] in ("Request", "Select") for a in actions)
            bye = any(a and a[0] == "Bye" for a in actions)
            if calls:
                action, reason, status = ("review", "published_no_offer", "source_mapping_candidate") if no_offer else ("execute", "annotated_database_query", "source_mapping_candidate")
            elif asks:
                action, reason, status = "clarify", "published_system_request", "source_mapping_candidate"
            elif bye:
                action, reason, status = "complete", "published_bye", "source_mapping_candidate"
            else:
                action, reason, status = None, None, "unlabelled"
            unit = make_unit(unit_id=f"turn_{turn.get('turn_id')}", history=copy.deepcopy(history),
                query=turn.get("user_utterance", ""), state=previous_belief, tools=visible_tools, calls=calls, result=results,
                narration=turn.get("system_utterance", ""),
                disposition={"status": status, "action": action, "reason": reason},
                provenance_kind="risawoz_belief_state_annotation",
                source_validation="Call reconstructed from published belief state and DB-result association; source annotations retained.",
                tokenizer=tokenizer)
            units.append(unit)
            history.extend([{"role": "user", "content": turn.get("user_utterance", "")},
                            {"role": "assistant", "content": turn.get("system_utterance", "")}])
            previous_belief = belief
        groups.append({"group_id": "risawoz:" + group_id, "source": "RiSAWOZ", "language": "zh",
            "source_split": "published_train_cpt_shared", "source_generation": "human_dialogue_with_annotations",
            "origin": {"archive_path": source["archive_path"], "archive_member": source["archive_member"], "source_group_id": group_id},
            "formal_training_admitted": False, "locked_eval_eligible": False, "units": units})
    return groups


def _summarize(groups):
    source_groups = Counter()
    source_units = Counter()
    head_counts = {name: Counter() for name in ("lm", "retrieval", "disposition", "narration", "confidence")}
    budgets = Counter()
    provenance = Counter()
    for group in groups:
        source_groups[group["source"]] += 1
        for unit in group["units"]:
            source_units[group["source"]] += 1
            for head in head_counts:
                head_counts[head][unit["heads"][head]["status"]] += 1
            budgets["fits_2048" if unit["budget"]["fits_2048"] else "over_2048"] += 1
            provenance["literal_values"] += unit["argument_provenance"]["literal_values"]
            provenance["source_annotation_values"] += unit["argument_provenance"]["source_annotation_values"]
            provenance["units_requiring_semantic_review"] += int(unit["argument_provenance"]["semantic_review_required"])
    return {"source_groups": dict(source_groups), "source_units": dict(source_units),
            "head_status_counts": {k: dict(v) for k, v in head_counts.items()},
            "budget_counts": dict(budgets), "argument_provenance_counts": dict(provenance)}


def run(config, out):
    out = resolve_path(ROOT / out)
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise ValueError("disk reserve")
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, out / "implementation.py.snapshot")
    write_new(out / "config.json", config)
    tokenizer = load_tokenizer(resolve_path(ROOT / config["tokenizer_manifest"]))
    groups = []
    groups.extend(_moss_groups(config, tokenizer))
    groups.extend(_api_bank_groups(config, tokenizer))
    groups.extend(_crosswoz_groups(config, tokenizer))
    groups.extend(_risawoz_groups(config, tokenizer))
    if len(groups) != config["count_per_source"] * 4:
        raise ValueError("source group count mismatch")
    with (out / "task-groups.jsonl").open("x") as handle:
        for group in groups:
            handle.write(json.dumps(group, ensure_ascii=False) + "\n")
    with (out / "units.jsonl").open("x") as handle:
        for group in groups:
            for unit in group["units"]:
                handle.write(json.dumps({"group_id": group["group_id"], "source": group["source"], **unit}, ensure_ascii=False) + "\n")
    summary = _summarize(groups)
    report = {
        "schema": "mei-public-sft-multihead-trial-v1",
        "status": "reviewable_conversion_trial",
        "supersedes_trial": "2026-09-14-public-sft-multihead-conversion-r04",
        "model_generation": "v1.3",
        "task_groups_actual": len(groups),
        "task_groups_target": config["count_per_source"] * 4,
        "source_native_group_ids_present": 300,
        "independent_task_groups_certified": 0,
        "independence_note": "MOSS, CrossWOZ and RiSAWOZ retain 300 source-native conversation IDs, but cross-source family isolation is not complete. API-Bank has no released dialogue ID. No task group is yet certified independent.",
        **summary,
        "formal_training_admitted": 0,
        "locked_eval_created": 0,
        "confidence_results": 0,
        "teacher_calls": 0,
        "external_tool_executions": 0,
        "tokenizer_manifest": config["tokenizer_manifest"],
        "tokenizer_model_sha256": tokenizer.model_sha256,
        "assessment_contract": {
            "lm": "Usable only when the source call is visible and its parameters survive semantic provenance review.",
            "retrieval": "Positive tools are source-backed; other visible tools remain candidate negatives until audited.",
            "disposition": "Only source-observable execute/review/clarify/complete mappings are emitted; missing actions stay unlabelled.",
            "narration": "A published result/response pair is a candidate, not a host-verified result.",
            "confidence": "Always pending_model until a frozen model produces an evaluated call."
        },
        "known_source_findings": {
            "MOSS": "Strong Chinese call and multi-turn coverage; most search arguments are semantic rewrites and all narration needs result grounding review.",
            "API-Bank": "Direct business calls can support LM; ToolSearcher rows are retrieval-query candidates because Mei retrieval is an internal head. Released files lack stable dialogue IDs and many call views have no exact response partner.",
            "CrossWOZ": "Strong literal Chinese state-to-query supervision; selectedResults contain names rather than full records, and constraint relaxation must not become silent execution gold.",
            "RiSAWOZ": "Strong human dialogue plus DB-result association; some belief-state normalization is not literally supported and needs semantic review.",
            "confidence": "No public static source supplies frozen-model correctness outcomes."
        },
        "evidence_files": {}
    }
    review = [
        "# 公开 SFT 四来源多头转换试批", "",
        "本批各取100个来源任务组或视图。MOSS、CrossWOZ、RiSAWOZ保留公开会话ID；API-Bank没有发布对话ID，因此100个视图不计入已证明独立任务组。", "",
        "输出保留原始语言和公开标注，不翻译、不补写问题、不执行外部工具、不生成confidence标签。所有内容仍为审核候选，正式准入为0。", "",
        "## 实际数量", "",
    ]
    for source in sorted(summary["source_groups"]):
        review.append(f"- {source}：{summary['source_groups'][source]}组，{summary['source_units'][source]}个监督单元。")
    review.extend(["", "## 使用边界", "",
        "- LM：公开调用或公开对话状态可以形成候选，但语义派生参数仍需审核。",
        "- Retrieval：当前只确认来源正工具；其他可见工具不能自动当硬负例。",
        "- Disposition：只映射来源明确支持的动作，未覆盖的动作保持未标注。",
        "- Narration：公开结果与回复保留为配对候选，不宣称真实工具执行通过。",
        "- Confidence：全部保持pending_model。", "",
        "逐条内容见 task-groups.jsonl；扁平监督单元见 units.jsonl。"])
    (out / "REVIEW.md").write_text("\n".join(review) + "\n")
    for name in ("task-groups.jsonl", "units.jsonl", "REVIEW.md", "implementation.py.snapshot", "config.json"):
        report["evidence_files"][name] = digest(out / name)
    write_new(out / "REPORT.json", report)
    return report
