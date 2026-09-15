"""Build a reviewed public-source semantic specimen, not a production SFT release.

Review decisions are explicit recipe data. Integrity/schema checks do not perform
semantic review, certify a source pool, or simulate successful remote execution.
"""
import ast
import copy
import hashlib
import json
import random
import zipfile
from collections import Counter
from pathlib import Path

from profiling import ROOT, digest, resolve_path
from source_manager import load_tokenizer

HEADS = ('lm', 'retrieval', 'disposition', 'confidence', 'narration')


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), sort_keys=True)


def fingerprint(value):
    return hashlib.sha256(compact(value).encode()).hexdigest()


def schema_check(value, schema):
    """Deliberately small, declared JSON Schema subset; unknown assertions fail."""
    supported = {'type', 'properties', 'required', 'items', 'additionalProperties',
                 'enum', 'minimum', 'maximum', 'description', 'default', 'title'}
    if set(schema) - supported:
        raise ValueError('unsupported schema assertion')
    kind = schema.get('type')
    types = {'object': lambda v: isinstance(v, dict), 'array': lambda v: isinstance(v, list),
             'string': lambda v: isinstance(v, str), 'integer': lambda v: type(v) is int,
             'number': lambda v: type(v) in (int, float), 'boolean': lambda v: type(v) is bool}
    if kind not in types or not types[kind](value):
        raise ValueError('schema type mismatch')
    if 'enum' in schema and value not in schema['enum']:
        raise ValueError('schema enum mismatch')
    if kind in ('integer', 'number'):
        if value < schema.get('minimum', float('-inf')) or value > schema.get('maximum', float('inf')):
            raise ValueError('schema range mismatch')
    if kind == 'object':
        if not set(schema.get('required', [])) <= set(value):
            raise ValueError('missing required parameter')
        properties = schema.get('properties', {})
        for key, child in value.items():
            if key in properties:
                schema_check(child, properties[key])
            elif schema.get('additionalProperties') is False:
                raise ValueError('unknown parameter')
    if kind == 'array' and 'items' in schema:
        for child in value:
            schema_check(child, schema['items'])


def leaves(value, path=()):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from leaves(item, path + (key,))
    elif isinstance(value, list) and value:
        for idx, item in enumerate(value):
            yield from leaves(item, path + (idx,))
    else:
        yield list(path), value


def evidence_check(calls, evidence, events):
    expected = {(i, compact(path)): value for i, c in enumerate(calls)
                for path, value in leaves(c['arguments'])}
    observed = {}
    for item in evidence:
        key = (item['call_index'], compact(item['argument_path']))
        same = key in expected and (compact(expected[key]) == compact(item['value']) or
            (type(expected[key]) in (int, float) and type(item['value']) in (int, float)
             and expected[key] == item['value']))
        if key in observed or not same:
            raise ValueError('parameter evidence mismatch')
        if not item.get('reason') or not item.get('quotes'):
            raise ValueError('missing semantic evidence review')
        for quote in item['quotes']:
            event = events.get(quote['event_id'])
            if event is None or event['phase'] != 'before_call' or quote['text'] not in event['content']:
                raise ValueError('absent or future parameter evidence')
        observed[key] = item['value']
    if set(expected) != set(observed):
        raise ValueError('uncovered parameter')


def parse_record(text):
    # RiSAWOZ stores Python-style dict strings with JSON boolean names.
    tree = ast.parse(text, mode='eval')
    class Booleans(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id not in ('true', 'false', 'null'):
                raise ValueError('non-literal result')
            return ast.copy_location(ast.Constant({'true': True, 'false': False, 'null': None}[node.id]), node)
    return ast.literal_eval(Booleans().visit(tree))


def verify_result(call, records):
    """Check returned records against the requested constraints, independently of reply."""
    constraints = call['arguments']['constraints']
    if not records:
        raise ValueError('expected nonempty source result')
    for record in records:
        for field, expected in constraints.items():
            value = record.get(field)
            if not (expected in value if isinstance(value, list) else value == expected):
                raise ValueError(f'source result violates {field}: {value!r} != {expected!r}')
        if not set(call['arguments']['requested_fields']) <= set(record):
            raise ValueError('requested result field missing')


def make_case(spec, row, origin, tokenizer):
    modern = 'heads' in row
    source = {'toolace': 'ToolACE'}.get(row.get('source'), row.get('source', 'ToolACE'))
    query = row.get('query') if modern else next(m['content'] for m in reversed(row['messages']) if m['role'] == 'user')
    history = copy.deepcopy(row.get('history', [])) if spec.get('retain_history') else []
    calls = copy.deepcopy(row['heads']['lm']['target'] if modern else row['gold_lm_target'])
    if 'reviewed_calls' in spec:
        calls = copy.deepcopy(spec['reviewed_calls'])
    if not spec.get('review'):
        raise ValueError('explicit semantic review required')
    events = {f'history_{i}': dict(m, phase='before_call') for i, m in enumerate(history)}
    events['query'] = {'role': 'user', 'content': query, 'phase': 'before_call'}
    # No current belief state, current result or source response in call inputs.
    tools = copy.deepcopy(row['visible_tools'])
    names = {c['name'] for c in calls}
    selected = [t for t in tools if t['name'] in names]
    selected += sorted([t for t in tools if t['name'] not in names], key=lambda t: t['name'])[:5-len(selected)]
    random.Random(int(fingerprint(spec['id'])[:16], 16)).shuffle(selected)
    catalog = {t['name']: t for t in selected}
    if len(selected) > 5 or len(catalog) != len(selected):
        raise ValueError('invalid visible catalog')
    for call in calls:
        if call['name'] not in catalog:
            raise ValueError('call not visible')
        schema_check(call['arguments'], catalog[call['name']]['parameters'])
    evidence_check(calls, spec['evidence'], events)
    action = 'execute' if calls else 'complete'
    if not calls and (spec.get('no_call_reason') != 'user_closes_conversation' or '再会' not in query):
        raise ValueError('unjustified no-call target')
    records = []
    if source == 'RiSAWOZ' and calls:
        records = [parse_record(x) for x in row['source_result'][1:]]
        verify_result(calls[0], records)
    narration = None
    facts = spec.get('narration_facts', [])
    if facts:
        narration = row['source_narration']
        for fact in facts:
            value = records[fact['record_index']][fact['field']]
            if value != fact['value'] or fact['reply_quote'] not in narration:
                raise ValueError('narration fact mismatch')
        if not spec.get('narration_review'):
            raise ValueError('reply needs complete semantic review beyond fact matching')
    family = row.get('group_id', row.get('association_group'))
    inp = {'query': query, 'history': history, 'tools': selected}
    prompt = '只输出当前可执行的工具调用JSON数组；无调用输出[]。\n' + compact(inp)
    target = compact(calls)
    pids, tids = tokenizer.encode(prompt), tokenizer.encode(target)
    if tokenizer.decode(pids) != prompt or tokenizer.decode(tids) != target:
        raise ValueError('tokenizer roundtrip failed')
    reserve = max(128, len(tids) + 1)
    if len(pids) + reserve > 2048:
        raise ValueError(f'{spec["id"]}: specimen input exceeds joint budget')
    # Semantic labels are usable in this specimen scope only. Production masks
    # stay zero until adopted runtime/serializer and family split are bound.
    heads = {h: {'semantic_label_mask': 0, 'production_label_mask': 0} for h in HEADS}
    heads['lm'].update(semantic_label_mask=1, target=calls)
    heads['retrieval'].update(semantic_label_mask=int(bool(calls)), positives=sorted(names),
        negatives=[], unknown=[t['name'] for t in tools if t['name'] not in names],
        supervision_scope='positive_pairs_only', candidate_mode='oracle_not_learned')
    heads['disposition'].update(semantic_label_mask=1, target={'action': action,
        'scope': 'current_request', 'call_indices': list(range(len(calls)))},
        authorization='recommendation_only_not_execution_permission')
    heads['confidence'].update(status='pending_model', target=None)
    heads['narration'].update(semantic_label_mask=int(narration is not None), target=narration,
        status='source_result_grounded' if narration else 'excluded_unverified_or_not_needed',
        facts=facts, review=spec.get('narration_review'))
    result = {'id': spec['id'], 'source': source, 'family': family,
        'language': 'en' if source == 'ToolACE' or spec.get('language') == 'en' else 'zh-Hans',
        'origin': origin, 'source_view': row, 'input': inp, 'events': events,
        'heads': heads, 'parameter_evidence': spec['evidence'],
        'semantic_review': {'reviewer': 'Codex_assistant', 'user_review': 'pending',
            'decision': 'qualified_within_semantic_specimen_scope', 'reason': spec['review'],
            'input_target_sha256': fingerprint({'input': inp, 'target': calls}),
            'edits': spec.get('edits', []), 'history_policy': 'retained_prior_only' if history else 'standalone_request_reviewed'},
        'relations': {'batch': 'independent_parameter_complete' if len(calls)>1 else 'single_or_no_call',
            'dependencies_within_batch': [], 'prior_context_retained': bool(history)},
        'source_results': records, 'result_verification': 'constraints_checked_against_frozen_source_records' if records else 'not_executed',
        'scope': {'development_specimen': True, 'formal_training_admitted': False,
            'locked_eval_eligible': False, 'model_success': False,
            'is_complete_task_trajectory': False,
            'license': 'inherit_source_registry; no_new_distribution_or_training_admission'},
        'encoding': {'serializer': 'semantic-json-specimen-v1_not_production',
            'prompt': prompt, 'target': target, 'input_ids': pids+tids+[1],
            'loss_mask': [0]*len(pids)+[1]*(len(tids)+1),
            'prompt_tokens': len(pids), 'target_tokens': len(tids), 'eos_tokens': 1,
            'output_reserve': reserve, 'joint_tokens': len(pids)+reserve}}
    if narration is not None:
        # Narration is a separate, post-result input. Do not copy these facts
        # into the call prompt. The source-result verifier already ran above.
        # Project by the already-issued call, never by the desired reply text.
        args = calls[0]['arguments']
        fields = set(args['constraints']) | set(args['requested_fields']) | {'名称', '班号'}
        projected = [{k: v for k, v in record.items() if k in fields} for record in records]
        result['heads']['narration']['input'] = {'query': query, 'history': history,
            'calls': calls, 'verified_source_results': projected,
            'scope': 'historical_source_snapshot_not_live_execution'}
        nprompt = '只根据已验证结果简短回答，不执行工具。\n' + compact(result['heads']['narration']['input'])
        ni, nt = tokenizer.encode(nprompt), tokenizer.encode(narration)
        if tokenizer.decode(ni) != nprompt or tokenizer.decode(nt) != narration:
            raise ValueError('narration tokenizer roundtrip failed')
        if len(ni) + max(128, len(nt)+1) > 2048:
            raise ValueError('narration specimen exceeds joint budget')
        result['heads']['narration']['encoding'] = {'prompt': nprompt, 'target': narration,
            'input_ids': ni+nt+[1], 'loss_mask': [0]*len(ni)+[1]*(len(nt)+1),
            'prompt_tokens': len(ni), 'target_tokens': len(nt), 'eos_tokens':1,
            'joint_tokens':len(ni)+max(128,len(nt)+1)}
    validate_case(result)
    return result


def validate_case(case):
    target = case['heads']['lm']['target']
    if fingerprint({'input': case['input'], 'target': target}) != case['semantic_review']['input_target_sha256']:
        raise ValueError('review binding changed')
    evidence_check(target, case['parameter_evidence'], case['events'])
    catalog = {t['name']: t for t in case['input']['tools']}
    for call in target:
        if call['name'] not in catalog:
            raise ValueError('target tool absent')
        schema_check(call['arguments'], catalog[call['name']]['parameters'])
    if case['source_results']:
        verify_result(target[0], case['source_results'])
    narration = case['heads']['narration']
    if narration['semantic_label_mask']:
        for fact in narration['facts']:
            if (case['source_results'][fact['record_index']][fact['field']] != fact['value'] or
                    fact['reply_quote'] not in narration['target']):
                raise ValueError('narration evidence changed')
    if 'source_results' in case['input'] or 'heads' in case['input']:
        raise ValueError('future/gold input leakage')
    enc = case['encoding']
    expected = [0]*enc['prompt_tokens']+[1]*(enc['target_tokens']+1)
    if enc['loss_mask'] != expected or len(expected) != len(enc['input_ids']) or enc['input_ids'][-1] != 1:
        raise ValueError('invalid loss mask or EOS')
    if not all(type(i) is int and 0 <= i < 24000 for i in enc['input_ids']):
        raise ValueError('token outside 24K')
    if case['heads']['confidence']['semantic_label_mask'] != 0:
        raise ValueError('static gold is not confidence evidence')


def source_snapshots(config, inputs, groups):
    """Verify selected raw records, without copying gold into model inputs."""
    raw = config['raw_sources']
    for ref in raw.values():
        if digest(resolve_path(ref['path'])) != ref['sha256']:
            raise ValueError('raw archive hash mismatch')
    with zipfile.ZipFile(resolve_path(raw['risawoz']['path'])) as z:
        risa = {r['dialogue_id']: r for r in json.loads(z.read(raw['risawoz']['member']))}
    toolace = json.loads(resolve_path(raw['toolace']['path']).read_text())
    needed = {}
    for spec in config['specimens']:
        row = inputs[spec['input']][spec['line']-1]
        if row.get('source') == 'MOSS':
            origin = groups[row['group_id']]['origin']
            needed.setdefault(origin['archive_member'], set()).add(origin['line_number'])
    moss = {}
    with zipfile.ZipFile(resolve_path(raw['moss']['path'])) as z:
        for member, lines in needed.items():
            with z.open(member) as stream:
                # Original adapter enumerated from zero despite the field name.
                for idx, line in enumerate(stream):
                    if idx in lines:
                        moss[(member, idx)] = json.loads(line)
                    if idx >= max(lines):
                        break
    snapshots = []
    for spec in config['specimens']:
        row = inputs[spec['input']][spec['line']-1]
        if row.get('source') == 'RiSAWOZ':
            group = risa[groups[row['group_id']]['origin']['source_group_id']]
            turn = group['dialogue'][int(row['unit_id'].split('_')[-1])]
            if (turn['user_utterance'] != row['query'] or turn['system_utterance'] != row['source_narration']
                    or turn['db_results'] != row['source_result']):
                raise ValueError('RiSAWOZ adapter/raw mismatch')
            original = {'dialogue_id': group['dialogue_id'], 'turn': turn}
        elif row.get('source') == 'MOSS':
            origin = groups[row['group_id']]['origin']
            original = moss[(origin['archive_member'], origin['line_number'])]
            if not any(isinstance(v, str) and row['query'] in v for _, v in leaves(original)):
                raise ValueError('MOSS query absent from original')
        else:
            original = toolace[row['origin']['row_index']]
            query = next(m['content'] for m in reversed(row['messages']) if m['role']=='user')
            if not any(isinstance(v, str) and query in v for _, v in leaves(original['conversations'])):
                raise ValueError('ToolACE query absent from original')
        snapshots.append({'id': spec['id'], 'raw_record_sha256': fingerprint(original), 'record': original})
    return snapshots


def run(config, out):
    out = Path(out)
    if out.exists():
        raise ValueError('immutable output already exists')
    inputs, bindings = {}, []
    for key, ref in config['inputs'].items():
        path = resolve_path(ref['path'])
        if digest(path) != ref['sha256']:
            raise ValueError(f'input hash mismatch: {key}')
        inputs[key] = [json.loads(line) for line in path.open()]
        bindings.append(ref)
    tokenizer = load_tokenizer(resolve_path(config['tokenizer_manifest']))
    groups = {r['group_id']: r for r in inputs['groups']}
    originals = source_snapshots(config, inputs, groups)
    cases = []
    for spec in config['specimens']:
        row = inputs[spec['input']][spec['line']-1]
        origin = copy.deepcopy(row.get('origin') or groups[row['group_id']]['origin'])
        if row.get('source') == 'MOSS':
            origin['line_number_base'] = 0
        origin['reviewed_adapter_view'] = dict(config['inputs'][spec['input']], line=spec['line'])
        cases.append(make_case(spec, row, origin, tokenizer))
    if len({c['family'] for c in cases}) != config['target_source_groups']:
        raise ValueError('source group count mismatch')
    report = {'schema': 'mei-public-sft-semantic-specimen-v1', 'generation': 'v1.3',
        'status': 'semantic_specimen_qualified_production_pending',
        'target_source_groups': config['target_source_groups'],
        'actual_source_groups': len({c['family'] for c in cases}), 'views': len(cases),
        'family_independence': 'source_dialogue_distinct; template/schema_independence_not_claimed',
        'source_groups': {s: len({c['family'] for c in cases if c['source']==s}) for s in sorted({c['source'] for c in cases})},
        'languages_by_view': dict(Counter(c['language'] for c in cases)),
        'call_batch_sizes': dict(Counter(len(c['heads']['lm']['target']) for c in cases)),
        'semantic_head_views': {h: sum(c['heads'][h]['semantic_label_mask'] for c in cases) for h in HEADS},
        'production_head_views': {h: 0 for h in HEADS},
        'max_specimen_joint_tokens': max(c['encoding']['joint_tokens'] for c in cases),
        'narration_encoded_views': sum(c['heads']['narration']['semantic_label_mask'] for c in cases),
        'max_narration_joint_tokens': max((c['heads']['narration'].get('encoding', {}).get('joint_tokens', 0) for c in cases)),
        'specimen_python_roundtrip_passed': len(cases), 'current_specimen_browser_audit': 'not_executed',
        'model_runs': 0, 'teacher_calls': 0, 'downloads': 0,
        'all_empty_baseline': {'execute_exact': 0,
            'execute_total': sum(bool(c['heads']['lm']['target']) for c in cases),
            'no_call_exact': sum(not c['heads']['lm']['target'] for c in cases),
            'task_ability_passed': False, 'scope': 'static_scorer_sanity_not_model_eval'},
        'raw_record_checks': len(originals), 'raw_source_bindings': config['raw_sources'],
        'frozen_source_inputs': bindings, 'tokenizer_manifest': config['tokenizer_manifest'],
        'tokenizer_manifest_sha256': digest(resolve_path(config['tokenizer_manifest'])),
        'gaps': config['gaps'], 'review_method': 'all_selected_views_assistant_review; user_confirmation_pending'}
    files = {'specimens.jsonl': ''.join(compact(c)+'\n' for c in cases),
             'source-snapshots.jsonl': ''.join(compact(r)+'\n' for r in originals),
             'selection.json': json.dumps(config, ensure_ascii=False, indent=2)+'\n'}
    # Keep reconstructable source bytes, not only a hash of mutable checkout code.
    files['builder.py.snapshot'] = Path(__file__).read_text()
    md = ['# v1.3 公开SFT合格语义样板包', '',
          '30个来源对话组。已逐例由助手审核；用户复核待进行。不是30个已证明独立的Eval组。',
          '只在语义样板范围准入。正式训练、跨端运行、真实模型效果均未通过或未执行。',
          '原文不翻译、不用教师重写。英文来源本身可能为合成对话；公开不等于人工撰写。', '',
          '## 如何看', '', '每例依次给原始需求、必要历史、正确调用、参数依据和解说。完整schema、各头标签、原始视图和编码见specimens.jsonl。', '',
          '## 数量', '', f'{len(cases)}个调用前视图；来源组数：{compact(report["source_groups"])}。',
          f'各头有效语义视图：{compact(report["semantic_head_views"])}。检索仅正工具对，未确认负例不参与负监督。', '',
          '## 使用限制', '', '样板全部保留为开发诊断；不加入tokenizer训练或锁定测试。调用视图是oracle，不代表检索成功。',
          '数据库结果只按历史来源快照验证，不代表当前营业、价格等事实。外部工具没有实际执行。',
          '多次搜索只标识独立调用，不宣称问题已经解答；等待结果后才能决定解说或继续搜索。', '',
          '## 待补能力', '', *['- '+x for x in config['gaps']], '']
    for case in cases:
        md += [f'## {case["id"]} · {case["source"]}', '',
               f'来源组：`{case["family"]}`', '', '原始用户要求：', '', '> '+case['input']['query'], '']
        if case['input']['history']:
            md += ['必要历史：', '', *[f'- {m["role"]}：{m["content"]}' for m in case['input']['history']], '']
        md += ['调用目标：', '', '```json', json.dumps(case['heads']['lm']['target'], ensure_ascii=False, indent=2), '```', '',
               '语义复核：'+case['semantic_review']['reason'], '']
        md += ['- '+'.'.join(map(str,e['argument_path']))+' = '+compact(e['value'])+'；'+e['reason']+'；依据：'+
               ' / '.join('「'+q['text']+'」' for q in e['quotes']) for e in case['parameter_evidence']]
        md += ['', '处置建议：'+case['heads']['disposition']['target']['action']+'。', '']
        if case['heads']['narration']['semantic_label_mask']:
            md += ['原始解说（保留原句）：', '', '> '+case['heads']['narration']['target'], '',
                   '解说依据：'+case['heads']['narration']['review'], '']
        else:
            md += ['解说不纳入监督：缺少可核实结果、回复含多余断言，或本视图不需要解说。', '']
    files['REVIEW.md'] = '\n'.join(md)+'\n'
    out.mkdir(parents=True)
    for name, content in files.items():
        (out/name).write_text(content)
    report['files'] = {name: digest(out/name) for name in files}
    report['implementation_sha256'] = digest(Path(__file__))
    (out/'REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report
