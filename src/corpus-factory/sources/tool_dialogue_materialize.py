"""Prepare whole ToolACE conversations, with checked call syntax; never execute tools."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import shutil
import sys

from profiling import resolve_path, ROOT, digest
from local_diagnostics import map_schema_types, parse_declared_tool_calls


def convert(row, validator):
    system = row['system']
    start = system.index('[{') if '[{' in system else system.index('[\n')
    original, _ = json.JSONDecoder().raw_decode(system, start)
    tools = [{**t, 'parameters': map_schema_types(t.get('parameters'))} for t in original]
    for t in tools:
        if t.get('required', 'absent') is None:
            del t['required']
    if not tools or len({t['name'] for t in tools}) != len(tools):
        raise ValueError('empty or duplicate tool names')
    parts = ['Tools: ' + json.dumps(tools, ensure_ascii=False, separators=(',', ':'))]
    call_count = 0
    for turn in row['conversations']:
        role, content = turn['from'], turn['value']
        if role not in ('user', 'assistant', 'tool') or not isinstance(content, str):
            raise ValueError('unsupported turn')
        if role == 'assistant' and content.lstrip().startswith('['):
            calls = parse_declared_tool_calls(content, [t['name'] for t in tools])
            if len(calls) != 1:
                raise ValueError('empty/multiple calls need semantic review')
            content = json.dumps(calls, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
            if not validator(content, tools)['ok']:
                raise ValueError('runtime wire/schema rejected')
            call_count += 1
        parts.append(role + ': ' + content)
    if not call_count:
        raise ValueError('no checked calls')
    return '\n'.join(parts), tools, call_count


def convert_nemotron(row, validator):
    tools = [t['function'] if t.get('type') == 'function' else t for t in row['tools']]
    names = [t['name'] for t in tools]
    if len(names) != len(set(names)):
        raise ValueError('duplicate tool names')
    parts = ['Tools: ' + json.dumps(tools, ensure_ascii=False, separators=(',', ':'))]
    call_count = 0
    pending = set()
    for turn in row['messages']:
        role = turn['role']
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('unsupported role')
        content = turn.get('content', '')
        if content is None:
            content = ''
        if not isinstance(content, str):
            raise ValueError('multimodal content')
        if turn.get('tool_calls'):
            if role != 'assistant' or len(turn['tool_calls']) != 1 or pending:
                raise ValueError('parallel/pending call needs review')
            call = turn['tool_calls'][0]
            args = call['function']['arguments']
            if isinstance(args, str):
                args = json.loads(args)
            wire = json.dumps([{'name': call['function']['name'], 'arguments': args}], ensure_ascii=False,
                              separators=(',', ':'), allow_nan=False)
            if not validator(wire, tools)['ok']:
                raise ValueError('runtime wire/schema rejected')
            if content.strip():
                parts.append('assistant: ' + content)
            parts.append('assistant: ' + wire)
            pending.add(call['id']); call_count += 1
        elif role == 'tool':
            if turn.get('tool_call_id') not in pending:
                raise ValueError('orphan tool result')
            pending.remove(turn['tool_call_id'])
            parts.append('tool: ' + content)
        elif content.strip():
            parts.append(role + ': ' + content)
        # reasoning_content is intentionally excluded; original bytes remain immutable.
    if not call_count or pending:
        raise ValueError('no checked calls or unfinished call')
    return '\n'.join(parts), tools, call_count


def run(config, out):
    source = resolve_path(ROOT / config['path'])
    if digest(source) != config['sha256']:
        raise ValueError('source hash changed')
    if shutil.disk_usage(ROOT).free < 100 * 1024**3:
        raise ValueError('disk reserve')
    sys.path.insert(0, str(ROOT / 'src/platform/_shared/runtime'))
    from byte_grammar import parse_call_text
    from source_manager import load_tokenizer, tokenizer_pointer
    tokenizer = load_tokenizer()
    out.mkdir(parents=True, exist_ok=False)
    (out/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2))
    (out/'implementation.py.snapshot').write_bytes(Path(__file__).read_bytes())
    (out/'parser.py.snapshot').write_bytes((ROOT/'src/corpus-factory/sources/local_diagnostics.py').read_bytes())
    counts = Counter(); seen = set(); schema_seen = set()
    sid = config.get('source_id', 'toolace')
    destination = sid + '.jsonl'
    converter = convert_nemotron if config.get('format') == 'nemotron' else convert
    def records():
        if config.get('format') == 'nemotron':
            with source.open() as stream:
                for line in stream:
                    if line.strip(): yield json.loads(line)
        else:
            yield from json.loads(source.read_text())
    with (out/destination).open('x') as f, (out/'rejections.jsonl').open('x') as rejected:
        for i, row in enumerate(records()):
            if i % 10000 == 0:
                if shutil.disk_usage(out).free < 100 * 1024**3:
                    raise ValueError('disk reserve')
                print(json.dumps({'source_id': sid, 'input_row': i, **counts}), flush=True)
            try:
                text, tools, calls = converter(row, parse_call_text)
            except (ValueError, KeyError, TypeError, SyntaxError, RecursionError) as e:
                counts['rejected'] += 1
                rejected.write(json.dumps({'row': i, 'reason': str(e)}, ensure_ascii=False)+'\n')
                continue
            h = hashlib.sha256(text.encode()).hexdigest()
            if h in seen:
                counts['duplicates'] += 1
                continue
            seen.add(h)
            schema_hash = hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()
            schema_seen.add(schema_hash)
            tokens = len(tokenizer.encode_document(text))
            f.write(json.dumps({'text': text, 'text_sha256': h, 'source_id': sid,
                'group_id': schema_hash, 'origin': {'path': str(source), 'sha256': config['sha256'], 'row_index': i},
                'split': 'candidate-unassigned', 'candidate_kind': 'cpt_raw',
                'metadata': {'public_synthetic': True, 'semantic_review': 'pending', 'executable_gold': False,
                    'checked_single_calls': calls, 'tokens': tokens, 'schema_once_per_conversation': True}}, ensure_ascii=False)+'\n')
            counts.update(records=1, tokens=tokens, characters=len(text), checked_calls=calls)
    report = {'schema': 'mei-tool-dialogue-candidate-v1', **counts, 'unique_schema_groups': len(schema_seen),
        'file': destination, 'sha256': digest(out/destination), 'tokenizer': tokenizer_pointer(),
        'release': False, 'teacher_calls': 0, 'public_synthetic': True,
        'limitations': ['whole conversation rejected if any detected call fails; no partial trajectory repair',
            'wire/schema checks are not semantic or execution verification',
            'repeated schema groups retained for context, not independent tool capacity; group split still required',
            'free assistant text retained as source narration, not core LM SFT targets']}
    (out/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report
