"""Read-only candidate classification against 0405; never certify gold."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json

from profiling import ROOT, digest, resolve_path

HEADS = ('lm', 'retrieval', 'disposition', 'confidence', 'narration')


def classify(source, calls):
    names = {c['name'] for c in calls}
    if not calls:
        return '无调用：需恢复任务状态'
    if source == 'MOSS':
        kinds = {'public_search': '搜索表达', 'public_calculate': '数值计算',
                 'public_solve': '方程求解', 'public_text_to_image': '文生图描述'}
        return kinds.get(next(iter(names)), '混合工具') if len(names) == 1 else '混合工具'
    if source == 'API-Bank':
        return '工具发现' if any('toolsearcher' in n.lower() for n in names) else '业务调用'
    if source in ('CrossWOZ', 'RiSAWOZ'):
        return '跨领域同批查询' if len(names) > 1 else '单领域数据库查询'
    if source == 'Nemotron':
        if names <= {'authenticate_user', 'verify_guest_order'}:
            return '身份与订单凭据'
        if names == {'transfer_to_human_agents'}:
            return '转人工协作'
        if names == {'get_order_status'}:
            return '订单状态查询'
        if names <= {'get_restaurant_menu', 'check_ingredient_availability'}:
            return '菜单与配料查询'
        if names <= {'cancel_order', 'modify_order', 'process_refund', 'place_order'}:
            return '订单变更与退款'
        return '配送异常'
    values = [v for c in calls for v in c.get('arguments', {}).values()]
    if not values:
        return '无参数调用'
    if any(isinstance(v, (list, dict)) for v in values):
        return '嵌套参数调用'
    if any(type(v) in (int, float, bool) for v in values):
        return '数值或布尔参数调用'
    return '字符串参数调用'


def make_review(row, source, relpath, line):
    modern = 'heads' in row
    calls = row['heads']['lm']['target'] if modern else row['gold_lm_target']
    history = row.get('history', row.get('messages', []))
    query = row.get('query')
    if query is None:
        query = next((m.get('content', '') for m in reversed(history) if m.get('role') == 'user'), '')
    visible = {'query': query, 'history': history, 'state': row.get('visible_state', {})}
    visible_text = json.dumps(visible, ensure_ascii=False)
    tools = row['visible_tools']
    issues = ['parameter_roles_and_source_time_unverified', 'family_split_requires_binding',
              'production_serializer_runtime_pending']
    if len(calls) > 1:
        issues += ['multiple_calls_require_dependency_review', 'legacy_single_call_incompatible']
    if len(tools) > 5:
        issues.append('unprojected_catalog_exceeds_visible_five')
    discovery = source == 'API-Bank' and any('toolsearcher' in c['name'].lower() for c in calls)
    if not calls:
        issues.append('no_call_is_not_automatically_refusal_or_completion')
    if modern and row['heads']['lm']['status'] == 'source_policy_relaxation_requires_review':
        issues.append('source_constraint_change_requires_consent_review')
    if source in ('API-Bank', 'CrossWOZ', 'RiSAWOZ'):
        issues.append('gold_or_annotation_conditioned_catalog_not_learned_retrieval')
    elif not modern:
        issues.append('catalog_selection_recipe_not_certified_learned')
    if source == 'MOSS':
        issues.append('published_command_and_result_not_execution_verified')
    if source == 'RiSAWOZ':
        issues.append('belief_normalization_and_disjunction_review')
    if source == 'Nemotron':
        issues += ['single_association_group_not_independent_cases', 'policy_context_and_credentials_review']
    if row.get('cpt_shared'):
        issues.append('source_declares_cpt_shared_not_independent_eval')
    unsupported = []
    for idx, call in enumerate(calls):
        for entity in call.get('arguments', {}).get('selected_entities', []):
            if str(entity) not in visible_text:
                unsupported.append({'call_index': idx, 'value': entity,
                                    'finding': 'literal_absence_not_final_semantic_judgment'})
    if unsupported:
        issues.append('selected_entity_absent_from_prior_visible_text')
    result = row.get('source_result')
    reply = row.get('source_narration')
    paired = result not in (None, '', [], {}) and bool(reply)
    old_action = row.get('heads', {}).get('disposition', {}).get('action')
    heads = {
        'lm': {'status': 'not_applicable_discovery' if discovery else 'pending_semantic_review', 'label_mask': 0},
        'retrieval': {'status': 'discovery_query_candidate' if discovery else
                      'positive_candidate_negative_unknown' if calls else 'pending_task_state', 'label_mask': 0},
        'disposition': {'status': 'pending_policy_evidence', 'label_mask': 0, 'source_action_not_gold': old_action},
        'confidence': {'status': 'pending_model', 'label_mask': 0, 'target': None},
        'narration': {'status': 'not_applicable_discovery' if discovery else
                      'pending_fact_grounding' if paired else 'pending_result_reconstruction', 'label_mask': 0},
    }
    return {'review_id': hashlib.sha256(f'{relpath}:{line}'.encode()).hexdigest()[:24],
            'source': source, 'subclass': classify(source, calls),
            'origin': {'path': relpath, 'line': line},
            'source_group': row.get('group_id', row.get('association_group')),
            'source_unit': row.get('unit_id', row.get('case_id')),
            'query': query, 'calls': calls, 'call_count': len(calls), 'visible_tool_count': len(tools),
            'language_observation': 'query_contains_han' if any('\u3400' <= c <= '\u9fff' for c in query) else 'query_without_han',
            'heads': heads, 'issues': issues, 'selected_entity_findings': unsupported,
            'source_result_pair_present': paired, 'formal_admitted': False}


def run(config, out):
    inputs = []
    for spec in config['inputs']:
        path = resolve_path(ROOT / spec['path'])
        if digest(path) != spec['sha256']:
            raise ValueError(f'input hash mismatch: {path}')
        inputs.append((spec, path))
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    (out / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    (out / 'implementation.py.snapshot').write_bytes(Path(__file__).read_bytes())
    reviews = []
    for spec, path in inputs:
        for i, text in enumerate(path.open(), 1):
            if text.strip():
                row = json.loads(text)
                reviews.append(make_review(row, spec.get('source', row.get('source')), spec['path'], i))
    with (out / 'review-items.jsonl').open('w') as f:
        for item in reviews:
            f.write(json.dumps(item, ensure_ascii=False)+'\n')
    summaries = []
    for source in sorted({r['source'] for r in reviews}):
        rows = [r for r in reviews if r['source'] == source]
        summaries.append({'source': source, 'views': len(rows),
                          'observed_group_ids': len({r['source_group'] for r in rows if r['source_group']}),
                          'subclasses': dict(Counter(r['subclass'] for r in rows)),
                          'call_counts': dict(Counter(r['call_count'] for r in rows)),
                          'language_observations': dict(Counter(r['language_observation'] for r in rows)),
                          'issues': dict(Counter(v for r in rows for v in r['issues'])),
                          'head_status': {h: dict(Counter(r['heads'][h]['status'] for r in rows)) for h in HEADS}})
    (out / 'subclasses.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2)+'\n')
    # One source-grounded example per subclass; excerpts are not separate task groups.
    examples = []
    seen = set()
    for row in reviews:
        key = (row['source'], row['subclass'])
        if key not in seen:
            examples.append(row)
            seen.add(key)
    with (out / 'examples.jsonl').open('w') as f:
        for row in examples:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
    report = {'schema': 'mei-sft-candidate-triage-v1', 'model_generation': 'v1.3',
              'status': 'candidate_reorganized_not_admitted', 'views_reviewed_structurally': len(reviews),
              'source_counts': {s['source']: s['views'] for s in summaries},
              'subclasses': len(seen), 'formal_admitted': 0, 'independent_cases_certified': 0,
              'review_method': 'deterministic structural triage, not semantic certification',
              'source_snapshots': config['inputs'],
              'limitations': ['input views may share tasks; no cross-source independence certified',
                              'issue counts overlap and do not sum to exclusions',
                              'literal absence is a review clue, not a correctness verdict',
                              'all head label masks disabled until specific evidence approval'],
              'files': {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}}
    (out / 'REPORT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    return report
