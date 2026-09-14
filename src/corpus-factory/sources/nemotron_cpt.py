"""Source-faithful CPT text; deliberately independent of Mei execution grammar."""
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from profiling import ROOT, digest


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def strict_loads(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise ValueError('duplicate_json_key:' + k)
            result[k] = v
        return result
    def constant(value):
        raise ValueError('nonfinite_json:' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def json_equal(left, right):
    if type(left) is bool or type(right) is bool:
        return type(left) is type(right) and left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(json_equal(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(json_equal(a, b) for a, b in zip(left, right))
    return left == right


def check_value(value, schema, path='$'):
    """Conservative partial check. Unknown constraints are reported, not certified."""
    errors, unknown = [], []
    if isinstance(schema, bool):
        return ([path + ':false_schema'] if not schema else []), []
    if not isinstance(schema, dict):
        return [path + ':invalid_schema'], []
    supported = {'type', 'properties', 'required', 'items', 'enum', 'const',
                 'additionalProperties', 'description', 'title', 'default', 'examples',
                 '$schema', '$id', '$comment', 'deprecated', 'readOnly', 'writeOnly'}
    unknown += [path + ':' + k for k in schema if k not in supported]
    kinds = {'object': isinstance(value, dict), 'array': isinstance(value, list),
             'string': isinstance(value, str), 'boolean': type(value) is bool,
             'integer': type(value) is int or (type(value) is float and value.is_integer()),
             'number': type(value) in (int, float),
             'null': value is None}
    kind = schema.get('type')
    if kind is not None:
        allowed = kind if isinstance(kind, list) else [kind]
        if any(not isinstance(t, str) or t not in kinds for t in allowed):
            unknown.append(path + ':unknown_type')
        elif not any(kinds[t] for t in allowed):
            errors.append(path + ':type')
    # JSON boolean and number equality must not collapse (True == 1 in Python).
    if 'enum' in schema and not any(json_equal(value, v) for v in schema['enum']):
        errors.append(path + ':enum')
    if 'const' in schema and not json_equal(value, schema['const']):
        errors.append(path + ':const')
    if isinstance(value, dict):
        props = schema.get('properties', {})
        for key in schema.get('required', []):
            if key not in value:
                errors.append(path + '.' + key + ':missing_required')
        for key, item in value.items():
            sub = props.get(key, schema.get('additionalProperties', True))
            e, u = check_value(item, sub, path + '.' + key)
            errors += e; unknown += u
    if isinstance(value, list) and 'items' in schema:
        if isinstance(schema['items'], list):
            unknown.append(path + ':tuple_items')
        else:
            for i, item in enumerate(value):
                e, u = check_value(item, schema['items'], path + '[' + str(i) + ']')
                errors += e; unknown += u
    return errors, unknown


def convert(row):
    tools = row['tools']
    if not isinstance(tools, list) or not isinstance(row['messages'], list):
        raise ValueError('invalid_envelope')
    by_name = defaultdict(list)
    for t in tools:
        spec = t.get('function', t)
        if not isinstance(spec.get('name'), str):
            raise ValueError('invalid_tool_name')
        if spec not in by_name[spec['name']]:
            by_name[spec['name']].append(spec)
    issues, features, messages, links = [], Counter(), [], []
    pending, all_ids = {}, set()
    calls = 0
    for index, original in enumerate(row['messages']):
        role = original['role']
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('unsupported_role')
        # Retain public conversation fields only; never turn teacher reasoning into targets.
        turn = {k: original[k] for k in ('role', 'content', 'tool_calls', 'tool_call_id', 'name') if k in original}
        content = turn.get('content')
        if content is not None and not isinstance(content, str):
            if role == 'tool':
                features['structured_tool_result'] += 1
            else:
                issues.append('nontext_non_tool_content')
        tc = turn.get('tool_calls') or []
        if tc:
            if role != 'assistant':
                raise ValueError('call_in_nonassistant')
            features['parallel_call_turns'] += int(len(tc) > 1)
            for ci, call in enumerate(tc):
                fn = call['function']; name = fn['name']; args = fn['arguments']
                if isinstance(args, str):
                    args = strict_loads(args)
                if not isinstance(args, dict):
                    issues.append('arguments_not_object')
                specs = by_name.get(name, [])
                if not specs:
                    issues.append('undeclared_tool:' + name)
                elif len(specs) > 1:
                    issues.append('conflicting_called_tool:' + name)
                else:
                    e, u = check_value(args, specs[0].get('parameters', {}))
                    issues += ['argument_error:' + x for x in e]
                    issues += ['unchecked_constraint:' + x for x in u]
                cid = call.get('id')
                key = cid if cid is not None else f'position:{index}:{ci}'
                if key in all_ids:
                    issues.append('duplicate_call_id')
                all_ids.add(key); pending[key] = {'turn': index, 'call_index': ci, 'name': name}
                calls += 1
        elif role == 'tool':
            cid = turn.get('tool_call_id')
            if cid is not None:
                match = cid if cid in pending else None
                basis = 'explicit_id'
            elif len(pending) == 1:
                match = next(iter(pending)); basis = 'single_pending_inferred'
                features[basis] += 1
            else:
                match = None; basis = 'unresolved'
            if match is None:
                issues.append('unresolved_result_association')
            else:
                link = pending.pop(match)
                if turn.get('name') and turn['name'] != link['name']:
                    issues.append('result_name_mismatch')
                links.append({**link, 'result_turn': index, 'basis': basis})
        elif pending:
            # New narration/request before a result: keep evidence but flag incomplete interior.
            issues.append('interior_unresolved_calls')
        messages.append(turn)
    if pending:
        # A source ending in a call remains a useful request-to-call example.
        if not messages or not messages[-1].get('tool_calls'):
            issues.append('unfinished_trajectory')
        else:
            features['open_call_tail'] += 1
    features['conflicting_tool_names'] = sum(len(v)>1 for v in by_name.values())
    features['no_call_context'] = int(calls == 0)
    text = 'Tools: ' + dumps(tools) + '\n' + '\n'.join(dumps(m) for m in messages)
    metadata = {'calls': calls, 'issues': sorted(set(issues)), 'features': dict(features),
                'result_links': links, 'source_uuid': row.get('uuid'), 'license': row.get('license'),
                'used_in': row.get('used_in'), 'semantic_review': 'pending',
                'schema_check': 'partial_types_required_enum_properties_items',
                'executable_gold': False, 'public_synthetic': True,
                'runtime_compatibility': 'not_a_cpt_gate', 'reasoning_excluded': True}
    # Schema hash ignores list order but does not merge genuinely different schemas.
    groups = sorted(dumps(t) for t in tools)
    return text, hashlib.sha256(dumps(groups).encode()).hexdigest(), metadata


def run(config, out):
    from source_manager import load_tokenizer, tokenizer_pointer
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise ValueError('disk_reserve')
    for entry in config['sources']:
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('source_hash_changed')
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (out/'implementation.py.snapshot').write_bytes(Path(__file__).read_bytes())
    tok = load_tokenizer(); pointer = tokenizer_pointer()
    totals = Counter(); reasons = Counter(); features = Counter(); seen = {}; by_source = {}
    files = {k: (out/(k+'.jsonl')).open('x') for k in ('candidate', 'review', 'rejected', 'duplicates')}
    def report():
        return {'schema': 'mei-nemotron-cpt-v1', 'counts': dict(totals),
                'issues': dict(reasons), 'features': dict(features), 'sources': by_source,
                'tokenizer': pointer, 'token_method': 'encode_document; boundaries included',
                'release': False, 'quality_admitted_tokens': None,
                'limitations': ['partial structural checks, not full JSON Schema certification',
                    'semantic and evaluation-isolation review pending',
                    'review reserve excluded from candidate token totals',
                    'pilot is prefix smoke test, not population estimate'] if config.get('max_records_per_source') else
                    ['partial structural checks, not full JSON Schema certification',
                     'semantic and evaluation-isolation review pending',
                     'review reserve excluded from candidate token totals']}
    try:
        for entry in config['sources']:
            counts = Counter(); by_source[entry['id']] = counts
            with (ROOT/entry['path']).open() as stream:
                for index, line in enumerate(stream):
                    if config.get('max_records_per_source') and index >= config['max_records_per_source']:
                        break
                    totals['input_records'] += 1; counts['input_records'] += 1
                    try:
                        row = strict_loads(line)
                        text, group, metadata = convert(row)
                        h = hashlib.sha256(text.encode()).hexdigest()
                        if h in seen:
                            totals['duplicates'] += 1; counts['duplicates'] += 1
                            files['duplicates'].write(dumps({'source':entry['id'], 'row_index':index,
                                'source_uuid':row.get('uuid'), 'text_sha256':h, 'retained_origin':seen[h]})+'\n')
                            continue
                        seen[h] = {'source':entry['id'], 'row_index':index}
                        n = len(tok.encode_document(text))
                        bucket = 'review' if metadata['issues'] else 'candidate'
                        reasons.update(metadata['issues']); features.update(metadata['features'])
                        record = {'text': text, 'text_sha256': h, 'source_id': entry['id'],
                                  'group_id': group, 'split': 'candidate-unassigned',
                                  'candidate_kind': 'cpt_raw', 'metadata': {**metadata, 'tokens': n},
                                  'origin': {'path': entry['path'], 'sha256': entry['sha256'], 'row_index': index}}
                        files[bucket].write(dumps(record)+'\n')
                        for target in (totals, counts):
                            target[bucket+'_records'] += 1; target[bucket+'_tokens'] += n
                            target[bucket+'_calls'] += metadata['calls']
                    except (ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
                        totals['rejected_records'] += 1; counts['rejected_records'] += 1
                        files['rejected'].write(dumps({'source':entry['id'], 'row_index':index, 'error':str(exc)})+'\n')
                    if index % 1000 == 0:
                        if shutil.disk_usage(ROOT).free < 100 * 1024**3:
                            raise ValueError('disk_reserve')
                        for f in files.values(): f.flush()
                        (out/'progress.json').write_text(json.dumps(report(), ensure_ascii=False, indent=2))
                        print(dumps({'source': entry['id'], 'row': index, **totals}), flush=True)
    finally:
        for f in files.values(): f.close()
    result = report()
    result['files'] = {k: {'sha256':digest(out/(k+'.jsonl')), 'bytes':(out/(k+'.jsonl')).stat().st_size} for k in files}
    (out/'manifest.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    return result
