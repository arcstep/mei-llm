"""Small Pro rubric calibration, under a cumulative input+output token budget."""
from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path
import time
import urllib.request

LIMIT = 20_000_000
MODEL = 'deepseek-v4-pro'


def ledger(root: Path) -> tuple[int, list[dict]]:
    total, entries = 0, []
    for contract in sorted(root.glob('pro-*/contract.json')):
        if json.loads(contract.read_text()).get('model') != MODEL:
            continue
        for reservation in sorted(contract.parent.glob('reservation-*.json')):
            outcome = reservation.with_name(reservation.name.replace('reservation-', 'outcome-'))
            if not outcome.exists():
                raise ValueError('unresolved earlier Pro request; reconcile before spending')
            row = json.loads(outcome.read_text())
            if not row.get('usage_known'):
                raise ValueError('earlier Pro usage unknown; reconcile before spending')
            response = reservation.with_name(reservation.name.replace('reservation-', 'response-'))
            raw = response.read_bytes()
            usage = json.loads(raw)['usage']
            values = [usage['prompt_tokens'], usage['completion_tokens']]
            if not all(type(n) is int and n >= 0 for n in values):
                raise ValueError('invalid prior token usage')
            count = sum(values)
            total += count
            entries.append({'response': str(response), 'sha256': hashlib.sha256(raw).hexdigest(), 'tokens': count})
    return total, entries


def calibrate(parent: Path, out: Path, prompt_path: Path, offsets: list[int]) -> dict:
    root = parent.resolve().parent
    if out.resolve().parent != root or not out.name.startswith('pro-'):
        raise ValueError('Pro runs must share the surveyed run root and pro- prefix for token accounting')
    if not offsets or len(offsets) > 24 or len(set(offsets)) != len(offsets) or any(i not in range(24) for i in offsets):
        raise ValueError('use unique offsets from the original 24-record probe')
    with (root / '.pro-teacher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run(root, parent, out, prompt_path, offsets)


def _run(root, parent, out, prompt_path, offsets):
    prior_tokens, history = ledger(root)
    settings = json.loads(Path('/Users/xuehongwei/.claude/settings.json').read_text())['env']
    endpoint = settings['ANTHROPIC_BASE_URL'].rstrip('/')
    if endpoint != 'https://cf.api.fan':
        raise ValueError('authorized provider changed')
    secret = settings['ANTHROPIC_AUTH_TOKEN']
    selection = json.loads((parent / 'selection.json').read_text())
    prompt = prompt_path.read_text()
    out.mkdir(exist_ok=False)
    def write(name, value):
        with (out / name).open('x') as f: json.dump(value, f, ensure_ascii=False, sort_keys=True)
    for name, path in [('rubric.txt', prompt_path), ('implementation.py.snapshot', Path(__file__)), ('selection.json', parent/'selection.json')]:
        with (out/name).open('xb') as f: f.write(path.read_bytes())
    write('contract.json', {'model': MODEL, 'endpoint': endpoint, 'offsets': offsets,
        'budget_unit': 'prompt_tokens + completion_tokens', 'cumulative_limit': LIMIT,
        'prior_tokens': prior_tokens, 'prior_evidence': history,
        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
        'parent_selection_sha256': hashlib.sha256((parent/'selection.json').read_bytes()).hexdigest(),
        'blind': 'no prior judgments or expected labels; revised general rubric',
        'temperature': 0, 'thinking': 'disabled', 'max_tokens': 2048, 'retries': 0})
    total = prior_tokens
    outcomes = []
    started = time.monotonic()
    for offset in offsets:
        original = json.loads((parent/f'request-{offset:04d}.json').read_text())
        messages = [{'role': 'system', 'content': prompt}, original['messages'][1]]
        visible = json.loads(messages[1]['content'])
        if len(visible) != 1 or visible[0]['text_segments'] != selection[offset]['text_segments']:
            raise ValueError('original input changed')
        input_bound = len(json.dumps(messages, ensure_ascii=False).encode()) + 1024
        reserve = input_bound + 2048
        if total + reserve > LIMIT or time.monotonic() - started > 600: break
        payload = {'model': MODEL, 'messages': messages, 'temperature': 0, 'max_tokens': 2048,
            'stream': False, 'response_format': {'type':'json_object'}, 'thinking': {'type':'disabled'}}
        write(f'request-{offset:04d}.json', payload)
        write(f'reservation-{offset:04d}.json', {'prior_tokens': total, 'reserved_tokens': reserve, 'limit': LIMIT})
        entry = {'offset': offset, 'id': selection[offset]['id'], 'source_id': selection[offset]['source_id'], 'usage_known': False}
        try:
            request = urllib.request.Request(endpoint+'/v1/chat/completions', data=json.dumps(payload, ensure_ascii=False).encode(),
                headers={'Content-Type':'application/json', 'Authorization':'Bearer '+secret})
            with urllib.request.urlopen(request, timeout=60) as response: result = json.load(response)
            write(f'response-{offset:04d}.json', result)
            usage = result['usage']; it, ot = usage['prompt_tokens'], usage['completion_tokens']
            if type(it) is not int or type(ot) is not int or not 0 <= it <= input_bound or not 0 <= ot <= 2048:
                raise ValueError('usage unavailable or outside reservation')
            total += it + ot
            entry.update(usage_known=True, usage=usage, returned_model=result.get('model'))
            choice = result['choices'][0]
            if choice.get('finish_reason') != 'stop': raise ValueError('incomplete response')
            items = json.loads(choice['message']['content'])['items']
            if len(items) != 1 or items[0].get('id') != 'r0': raise ValueError('invalid response IDs')
            item = items[0]; evidence = item.get('evidence')
            if item.get('decision') not in {'keep','reject','uncertain'} or not isinstance(item.get('reason'),str) or not 1 <= len(item['reason']) <= 240:
                raise ValueError('invalid decision')
            if not isinstance(evidence,str) or not 1 <= len(evidence) <= 60 or not any(evidence in s['text'] for s in selection[offset]['text_segments']):
                raise ValueError('invalid literal evidence')
            defects = item.get('defects', [])
            if not isinstance(defects, list): raise ValueError('invalid defects')
            for defect in defects:
                excerpt = defect.get('evidence')
                if not isinstance(defect.get('reason'), str) or not defect['reason']:
                    raise ValueError('invalid defect reason')
                if not isinstance(excerpt, str) or not 1 <= len(excerpt) <= 60 or not any(excerpt in s['text'] for s in selection[offset]['text_segments']):
                    raise ValueError('invalid defect evidence')
            entry.update(status='machine_annotated_pending_review', items=[{**item,'id':selection[offset]['id']}])
        except Exception as exc:
            entry.update(status='failed', error=(type(exc).__name__+': '+str(exc).replace(secret,'<redacted>'))[:200])
        entry['cumulative_pro_tokens'] = total
        write(f'outcome-{offset:04d}.json', entry); outcomes.append(entry)
        print(json.dumps({'offset':offset,'status':entry['status'],'cumulative_pro_tokens':total}),flush=True)
        if not entry['usage_known']: break
    report = {'schema':'mei-pro-rubric-calibration-v1','model':MODEL,'selected':len(offsets),'attempted':len(outcomes),
        'outcomes':outcomes,'prior_pro_tokens':prior_tokens,'new_pro_tokens':total-prior_tokens,'cumulative_pro_tokens':total,
        'limit':LIMIT,'elapsed_seconds':time.monotonic()-started,'m1_passed':False,'human_review_passed':False}
    write('review.json',report)
    return report
