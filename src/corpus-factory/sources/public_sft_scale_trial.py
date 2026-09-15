"""Bounded offline conversion of whole published conversations, never production gold."""
import ast
import copy
import hashlib
import heapq
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from profiling import ROOT, digest, resolve_path
from source_manager import load_tokenizer
from local_diagnostics import map_schema_types, parse_declared_tool_calls
from public_sft_multihead_trial import MOSS_POSITIONAL, moss_tools, _clean_marker
from qualified_sft_specimens import compact, fingerprint, leaves, schema_check


def sample_records(records, count, seed):
    """Bounded memory, stable hash sampling of raw records, before call validation."""
    heap, seen = [], 0
    for origin, row in records:
        seen += 1
        key = fingerprint([seed, origin])
        entry = (-int(key, 16), seen, origin, row)
        if len(heap) < count:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    return [(x[2], x[3]) for x in sorted(heap, reverse=True)], seen


def moss_calls(text):
    value = _clean_marker(text, '<|Commands|>:', '<eoc>').strip()
    if not value or value in ('None', 'null', '[]'):
        return []
    tree = ast.parse(value, mode='eval').body
    nodes = tree.elts if isinstance(tree, (ast.Tuple, ast.List)) else [tree]
    calls = []
    for node in nodes:
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError('non_literal_call')
        name = node.func.id
        if name not in MOSS_POSITIONAL:
            raise ValueError('unknown_function')
        keys = [k.arg for k in node.keywords]
        pos = MOSS_POSITIONAL[name]
        if len(node.args) > len(pos) or None in keys or len(keys) != len(set(keys)):
            raise ValueError('spread_or_duplicate_argument')
        args = dict(zip(pos, map(ast.literal_eval, node.args)))
        if set(args) & set(keys):
            raise ValueError('duplicate_argument')
        args.update({k.arg: ast.literal_eval(k.value) for k in node.keywords})
        aliases = {'Search': 'public_search', 'Calculate': 'public_calculate',
                   'Solve': 'public_solve', 'Text2Image': 'public_text_to_image'}
        calls.append({'name': aliases[name], 'arguments': args})
    return calls


def toolace_tools(row):
    system = row['system']
    marker = 'Here is a list of functions in JSON format that you can invoke:'
    if marker not in system:
        raise ValueError('alternate_catalog_format_needs_adapter')
    start = system.index('[', system.index(marker)+len(marker))
    raw, _ = json.JSONDecoder().raw_decode(system, start)
    tools = [{'name': t['name'], 'description': t.get('description', ''),
              'parameters': map_schema_types(t['parameters'])} for t in raw]
    if len({t['name'] for t in tools}) != len(tools):
        raise ValueError('duplicate_tool_name')
    return tools


def call_checks(calls, tools, history):
    by_name = {t['name']: t for t in tools}
    errors, evidence = [], []
    for ci, call in enumerate(calls):
        tool = by_name.get(call['name'])
        if tool is None:
            errors.append({'call': ci, 'reason': 'tool_not_visible'})
            continue
        try:
            schema_check(call['arguments'], tool['parameters'])
        except ValueError as exc:
            errors.append({'call': ci, 'reason': str(exc)})
        for path, value in leaves(call['arguments']):
            # Exact spans are only evidence locations, NOT proof of semantic slot correctness.
            matches = []
            if isinstance(value, str) and len(value) >= 2:
                for event in history:
                    if value in event['content']:
                        matches.append({'event_id': event['event_id'], 'role': event['role']})
            evidence.append({'call': ci, 'path': list(path), 'value': value,
                             'prior_exact_spans': matches,
                             'status': 'literal_location_only' if matches else 'semantic_or_normalization_review'})
    return errors, evidence


def make_view(gid, index, calls, tools, history, after, raw_call, tokenizer):
    errors, evidence = call_checks(calls, tools, history)
    active = {c['name'] for c in calls}
    # Explicit oracle compilation preview, never presented as retrieval/scan success.
    visible = [t for t in tools if t['name'] in active]
    visible += [t for t in tools if t['name'] not in active][:max(0, 5-len(visible))]
    prompt_obj = {'messages': copy.deepcopy(history), 'tools': visible}
    prompt = compact(prompt_obj)
    target = compact(calls)
    pi, ti = tokenizer.encode(prompt), tokenizer.encode(target)
    if tokenizer.decode(pi) != prompt or tokenizer.decode(ti) != target:
        raise ValueError('tokenizer_roundtrip_failure')
    required = 1 + len(pi) + max(128, len(ti)+1)
    prior_result = any(e['role'] == 'tool' for e in history)
    result_refs = [x for x in evidence if any(m['role'] == 'tool' for m in x['prior_exact_spans'])]
    return {'id': f'{gid}:event-{index}', 'group_id': gid, 'input': prompt_obj,
        'source_catalog': tools, 'source_target': raw_call, 'source_after_call': after,
        'calls': calls, 'schema_errors': errors, 'parameter_evidence_locations': evidence,
        'behavior': {'calls': len(calls), 'has_prior_tool_result': prior_result,
                     'arguments_with_prior_result_literal': len(result_refs),
                     'verified_result_dependency': False,
                     'parallel_safety': 'not_verified' if len(calls)>1 else 'not_applicable'},
        'encoding': {'serializer': 'json_preview_not_production_packer', 'prompt_tokens': len(pi),
                     'target_tokens': len(ti), 'joint_tokens_with_reserve': required,
                     'fits_2048': required <= 2048, 'python_roundtrip': True,
                     'browser_wasm': 'not_executed_for_this_trial'},
        'oracle_catalog_projection': True,
        'fits_five_tool_catalog': len(visible)<=5,
        'runtime_alias_mapping': 'pending' if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*',t['name']) for t in visible) else 'not_required_by_name_syntax',
        'heads': {
            'lm': {'status': 'schema_checked_semantics_pending' if calls and not errors else
                   'schema_or_no_call_review_pending', 'target': calls},
            'retrieval': {'positive_tools': sorted(active), 'non_gold_tools': 'unknown',
                          'learned_negatives': 'pending_model_or_review'},
            'disposition': {'status': 'source_action_candidate' if calls else 'unlabelled',
                            'action': 'execute' if calls else None,
                            'per_call_authorization': 'pending_semantic_review'},
            'narration': {'status': 'published_pair_pending_grounding' if after else 'unavailable',
                          'target': None},
            'confidence': {'status': 'pending_model', 'target': None}},
        'semantic_label_masks': {h: 0 for h in ('lm','retrieval','disposition','narration','confidence')},
        'production_label_masks': {h: 0 for h in ('lm','retrieval','disposition','narration','confidence')},
        'use': 'training_or_development_candidate_not_independent_eval'}


def events_for(source, row):
    if source == 'ToolACE':
        tools = toolace_tools(row)
        events = [{'event_id': i, 'role': t['from'], 'content': t['value']}
                  for i,t in enumerate(row['conversations'])]
        return tools, events
    tools = moss_tools(row.get('meta_instruction',''))
    events = []
    # JSON object key order is not a chronology contract (turn_10 sorts before turn_2).
    def turn_number(name):
        match=re.fullmatch(r'turn_(\d+)',name)
        if not match:
            raise ValueError('unknown_turn_order')
        return int(match.group(1))
    for turn_name in sorted(row['chat'],key=turn_number):
        turn=row['chat'][turn_name]
        for field, role, begin, end in [('Human','user','<|Human|>:','<eoh>'),
              ('Commands','call','<|Commands|>:','<eoc>'),
              ('Tool Responses','tool','<|Results|>:','<eor>'),
              ('MOSS','assistant','<|MOSS|>:','<eom>')]:
            content = _clean_marker(turn.get(field,''), begin, end)
            if content.strip():
                events.append({'event_id': len(events), 'role': role, 'content': content,
                               'source_turn': turn_name, 'source_field': field})
    return tools, events


def convert_group(source, origin, row, tokenizer):
    gid = source.lower()+':'+fingerprint(origin)[:20]
    group = {'group_id': gid, 'source': source, 'origin': origin, 'raw_sha256': fingerprint(row),
             'source_native_id': row.get('conversation_id'), 'views': [], 'issues': [],
             'source_generation': 'publisher_synthetic', 'independent_eval_eligible': False}
    try:
        tools, events = events_for(source, row)
    except (ValueError, KeyError, TypeError) as exc:
        group['issues'].append({'event': None, 'reason': str(exc)})
        return group
    group['user_text_sha256'] = fingerprint([e['content'] for e in events if e['role']=='user'])
    for i, event in enumerate(events):
        candidate = event['role']=='call' or (event['role']=='assistant' and event['content'].lstrip().startswith('['))
        if not candidate:
            continue
        try:
            calls = moss_calls(event['content']) if source=='MOSS' else parse_declared_tool_calls(event['content'], [t['name'] for t in tools])
            after = []
            for later in events[i+1:]:
                if later['role'] in ('user','call') or (later['role']=='assistant' and later['content'].lstrip().startswith('[')):
                    break
                after.append(later)
            view = make_view(gid, i, calls, tools, events[:i], after, event['content'], tokenizer)
            view['source'] = source
            query = next((e['content'] for e in reversed(events[:i]) if e['role']=='user'),'')
            view['language'] = 'zh_or_mixed' if re.search(r'[\u3400-\u9fff]',query) else 'en_or_other'
            group['views'].append(view)
        except (SyntaxError, ValueError, KeyError, TypeError) as exc:
            group['issues'].append({'event': i, 'reason': str(exc), 'raw': event['content']})
    return group


def run(config, out):
    out = Path(out)
    if out.exists():
        raise ValueError('immutable_output_exists')
    if shutil.disk_usage(ROOT).free < 100 * 2**30:
        raise ValueError('disk_reserve')
    tokenizer = load_tokenizer(resolve_path(config['tokenizer_manifest']))
    selected, inventory = [], {}
    for source, ref in config['sources'].items():
        path = resolve_path(ref['path'])
        if digest(path) != ref['sha256']:
            raise ValueError('source_hash_mismatch:'+source)
        if source=='ToolACE':
            data=json.loads(path.read_text())
            records=(({'path':ref['path'],'row_index':i,'index_base':0},r) for i,r in enumerate(data))
            chosen, count = sample_records(records,ref['count'],config['seed'])
        else:
            with zipfile.ZipFile(path) as archive:
                member = archive.namelist()[0]
                def records():
                    with archive.open(member) as stream:
                        for i, line in enumerate(stream):
                            r=json.loads(line)
                            if any(re.search(r'[\u3400-\u9fff]',t.get('Human','')) for t in r.get('chat',{}).values()):
                                yield {'path':ref['path'],'member':member,'line_number':i,'index_base':0},r
                chosen,count=sample_records(records(),ref['count'],config['seed'])
        inventory[source]={'eligible_raw_groups':count,'selected':len(chosen),'target':ref['count'],
                           'inclusion_probability':min(1,ref['count']/count),
                           'scope':'Chinese-user-containing conversations in this archive' if source=='MOSS' else 'all records in pinned local file'}
        selected.extend((source,o,r) for o,r in chosen)
    groups=[convert_group(s,o,r,tokenizer) for s,o,r in selected]
    views=[v for g in groups for v in g['views']]
    review=[]
    for source in config['sources']:
        # Audit first parsed call of 30 hash-selected groups. NOT a rate estimate for all turns.
        first=[next(v for v in g['views'] if v['calls']) for g in groups
               if g['source']==source and any(v['calls'] for v in g['views'])]
        review.extend(sorted(first,key=lambda v:fingerprint([config['seed'],'review',v['id']]))[:30])
    report={'schema':'mei-public-sft-scale-trial-v1','generation':'v1.3','status':'conversion_complete_review_pending',
            'source_inventory':inventory,'source_groups_actual':len(groups),'source_groups_target':sum(r['count'] for r in config['sources'].values()),
            'groups_with_views':sum(bool(g['views']) for g in groups),'call_views':len(views),
            'source_views':dict(Counter(v['source'] for v in views)),
            'languages':dict(Counter(v['language'] for v in views)),
            'call_batch_sizes':dict(Counter(len(v['calls']) for v in views)),
            'schema_checked_nonempty_views':sum(bool(v['calls']) and not v['schema_errors'] for v in views),
            'schema_issue_views':sum(bool(v['schema_errors']) for v in views),
            'parse_or_group_issues':sum(len(g['issues']) for g in groups),
            'issue_reason_counts':dict(Counter(i['reason'] for g in groups for i in g['issues'])),
            'schema_issue_reason_counts':dict(Counter(e['reason'] for v in views for e in v['schema_errors'])),
            'over_five_distinct_tool_views':sum(not v['fits_five_tool_catalog'] for v in views),
            'fits_preview_2048':sum(v['encoding']['fits_2048'] for v in views),
            'prior_result_context_views':sum(v['behavior']['has_prior_tool_result'] for v in views),
            'prior_result_literal_views':sum(v['behavior']['arguments_with_prior_result_literal']>0 for v in views),
            'result_dependency_semantically_verified':0,
            'duplicate_selected_user_text_groups':sum(n-1 for n in Counter(g.get('user_text_sha256',g['group_id']) for g in groups).values()),
            'head_candidate_views':{'lm':sum(bool(v['calls']) and not v['schema_errors'] for v in views),
                 'retrieval_positive_only':sum(bool(v['calls']) and not v['schema_errors'] for v in views),
                 'disposition_execute_only':sum(bool(v['calls']) and not v['schema_errors'] for v in views),
                 'narration_result_reply_pair':sum(any(e['role']=='tool' for e in v['source_after_call']) and any(e['role']=='assistant' for e in v['source_after_call']) for v in views),
                 'confidence':0},
            'review_queue_groups':len(review),'formal_admitted_groups':0,'locked_eval_groups':0,
            'tokenizer_manifest':config['tokenizer_manifest'],'tokenizer_model_sha256':tokenizer.model_sha256,
            'teacher_calls':0,'downloads':0,'model_runs':0,
            'limitations':['Counts are raw conversation groups, not certified independent task families.',
                'Public synthetic data is not automatically trustworthy; semantic labels remain masked.',
                'No-call or prose is never automatically labelled refusal/completion.',
                'Full history retained. Oversize data needs evidence-preserving projection, not deletion.',
                'Oracle catalog projection is only a tokenizer preview, not learned retrieval or runtime acceptance.',
                'Exact argument spans do not establish semantic correctness or true result dependency.',
                'Existing CPT overlap unknown per record; none of this trial is independent Eval.',
                'Source license provenance retained from local intake; distribution clearance not asserted.']}
    out.mkdir(parents=True)
    files={'groups.jsonl':groups,'views.jsonl':views,'review-queue.jsonl':review,
           'raw-snapshots.jsonl':[{'source':s,'origin':o,'record':r,'sha256':fingerprint(r)} for s,o,r in selected]}
    for name,rows in files.items():
        (out/name).write_text(''.join(compact(r)+'\n' for r in rows))
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    # Keep helper bytes as well as primary builder for reproducibility.
    for module in ('public_sft_scale_trial','public_sft_multihead_trial','qualified_sft_specimens','local_diagnostics'):
        shutil.copyfile(Path(__file__).with_name(module+'.py'),out/(module+'.py.snapshot'))
    report['artifacts']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    (out/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report


def review_run(config, out):
    """Apply explicit assistant decisions to immutable views; no automatic semantic scoring."""
    out=Path(out)
    if out.exists():
        raise ValueError('immutable_output_exists')
    source=resolve_path(config['views_path'])
    if digest(source)!=config['views_sha256']:
        raise ValueError('review_input_changed')
    views={v['id']:v for v in map(json.loads,source.open())}
    reviewed, usable=[],[]
    for decision in config['decisions']:
        view=copy.deepcopy(views[decision['id']])
        if fingerprint(view)!=decision['view_sha256']:
            raise ValueError('reviewed_view_changed')
        if decision['lm'] not in ('accept_call_semantics','hold','reject_source_gold') or not decision['reason']:
            raise ValueError('invalid_review')
        view['assistant_review']=decision
        ready=(decision['lm']=='accept_call_semantics' and not view['schema_errors']
               and bool(view['calls']) and view['encoding']['fits_2048'] and view['fits_five_tool_catalog'])
        if ready:
            view['semantic_label_masks']['lm']=1
            view['semantic_label_masks']['retrieval']=1
            usable.append(view)
        if decision.get('verified_result_dependency'):
            if not view['behavior']['arguments_with_prior_result_literal']:
                raise ValueError('dependency_evidence_missing')
            view['behavior']['verified_result_dependency']=True
        reviewed.append(view)
    report={'schema':'mei-public-sft-scale-review-v1','generation':'v1.3',
        'status':'bounded_assistant_review_complete_production_pending',
        'review_scope':'30 first parsed nonempty calls per source plus explicitly targeted dependency/result examples; not population error estimates',
        'reviewed_views':len(reviewed),'reviewed_source_groups':len({v['group_id'] for v in reviewed}),
        'first_call_reviewed':dict(Counter(v['source'] for v in reviewed if v['assistant_review']['sample']=='first_call')),
        'decisions':dict(Counter(v['assistant_review']['lm'] for v in reviewed)),
        'semantic_and_preview_eligible_views':len(usable),
        'usable_source_groups':len({v['group_id'] for v in usable}),
        'usable_source_views':dict(Counter(v['source'] for v in usable)),
        'usable_call_batch_sizes':dict(Counter(len(v['calls']) for v in usable)),
        'semantic_head_views':{h:sum(v['semantic_label_masks'][h] for v in usable) for h in ('lm','retrieval','disposition','narration','confidence')},
        'verified_dependency_views':sum(v['behavior']['verified_result_dependency'] for v in reviewed),
        'formal_training_admitted':0,'locked_eval':0,'model_or_live_tool_runs':0,
        'source_views_sha256':digest(source),'tokenizer_changed':False,
        'limitations':['Assistant review is not independent human confirmation.',
          'Eligible views are semantic specimens; runtime aliases, production serializer, masks and execution contracts still need binding.',
          'Unreviewed views remain candidates. Rejected source gold is preserved, not rewritten.',
          'No disposition, narration or confidence labels are fabricated to fill heads.']}
    out.mkdir(parents=True)
    for name,rows in [('reviewed-views.jsonl',reviewed),('semantic-specimens.jsonl',usable)]:
        (out/name).write_text(''.join(compact(v)+'\n' for v in rows))
    (out/'review-decisions.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    shutil.copyfile(__file__,out/'review-builder.py.snapshot')
    md=['# 公开 SFT 千组试批：原文抽查与可用样板','',
        '600组MOSS＋400组ToolACE的转换包另存于同级 public-sft-1000-r02。本目录是审核衍生包，不重复计新增原文。',
        '每来源30组首个可解析非空调用，另定向检查结果依赖。这里的审核由助手完成，用户复核待进行；不代表总体正确率。','',
        f"已复核{len(reviewed)}个视图，{len(usable)}个同时满足调用语义、当前schema检查和2K预览预算。正式训练准入仍为0。",'',
        '## 逐例原文与判断','']
    for view in reviewed:
        d=view['assistant_review']
        md.extend([f"### {view['id']}",'',
                   '原始调用前输入：','','```json',json.dumps(view['input']['messages'],ensure_ascii=False,indent=2),'```','',
                   '原始标注转换的调用：','','```json',json.dumps(view['calls'],ensure_ascii=False,indent=2),'```','',
                   f"判断：{d['lm']}。{d['reason']}",'',
                   f"结构检查：{compact(view['schema_errors'])}；预览预算：{view['encoding']['joint_tokens_with_reserve']} token。",''])
    (out/'REVIEW.md').write_text('\n'.join(md)+'\n')
    report['artifacts']={p.name:digest(p) for p in out.iterdir() if p.is_file()}
    (out/'REPORT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    return report
