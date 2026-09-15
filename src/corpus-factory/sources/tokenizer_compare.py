"""Bounded offline vocabulary-size pilot; never publishes or modifies pointers."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

from profiling import resolve_path, ROOT, digest


def dump(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def identity(row):
    sid = row['source_id']
    # Explicit group identity first; unknown families remain document-level.
    text_hash = hashlib.sha256(row['text'].encode()).hexdigest()
    group = row.get('group_id') or row.get('metadata', {}).get('repo_name')
    return text_hash, f'{sid}:{group}' if group else text_hash


def split_for(group, seed):
    n = int(hashlib.sha256(f'{seed}:{group}'.encode()).hexdigest()[:12], 16)
    return 'dev' if n % 5 == 0 else 'train'


def fragment(text, rng, width):
    start = rng.randrange(max(1, len(text) - width + 1))
    return start, text[start:start + width]


def sample(config, out):
    plan = json.loads((resolve_path(ROOT / config['inventory'])).read_text())
    buckets = defaultdict(list)
    seen = set()
    weights = config['source_weights']
    seed = config['seed']
    files = []
    for item in plan['candidate_files']:
        path = resolve_path(ROOT / item['file'])
        with path.open() as f:
            first = json.loads(next(f))
        sid = first['source_id']
        if sid not in weights:
            continue
        count = item['records']
        rng = random.Random(f'{seed}:{item["file"]}')
        limit = config.get('source_records_per_file', {}).get(sid, config['records_per_file'])
        n = min(count, config['short_records_per_file'] if sid == 'lccc-dialogue' else limit)
        chosen = set(rng.sample(range(count), n))
        h = hashlib.sha256()
        observed = 0
        with path.open('rb') as f:
            for i, raw in enumerate(f):
                h.update(raw)
                observed += 1
                if i not in chosen:
                    continue
                row = json.loads(raw)
                sid = row['source_id']
                if sid not in weights or row.get('split') in {'test', 'dev', 'valid', 'validation', 'calibration'}:
                    continue
                text = row.get('text', '')
                if not text.strip() or '\x00' in text:
                    continue
                th, group = identity(row)
                if th in seen:
                    continue
                seen.add(th)
                part = split_for(group, seed)
                start, excerpt = fragment(text, rng, config['fragment_chars'])
                buckets[(sid, part)].append({'source_id': sid, 'group': group,
                    'text_sha256': th, 'file': item['file'], 'row_index': i,
                    'start_char': start, 'text': excerpt})
        if h.hexdigest() != item['sha256'] or observed != count:
            raise ValueError(f'candidate bytes/count changed: {path}')
        files.append({'file': item['file'], 'sha256': h.hexdigest(), 'records': count})
        print(json.dumps({'sampled_file': item['file'], 'sampled_records': n}), flush=True)
    results = {}
    for part in ['train', 'dev']:
        selected = []
        counters = Counter()
        target = config['train_chars'] if part == 'train' else config['dev_chars']
        for sid, weight in weights.items():
            rows = buckets[(sid, part)]
            random.Random(f'{seed}:{sid}:{part}').shuffle(rows)
            quota = round(target * weight)
            for row in rows:
                if counters[sid] >= quota:
                    break
                selected.append(row)
                counters[sid] += len(row['text'])
        random.Random(f'{seed}:{part}').shuffle(selected)
        with (out / f'{part}.jsonl').open('x') as f:
            for row in selected:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
        results[part] = {'records': len(selected), 'characters': sum(counters.values()),
                         'source_characters': dict(counters), 'sha256': digest(out / f'{part}.jsonl')}
    train = {r['group'] for r in read_rows(out / 'train.jsonl')}
    dev = {r['group'] for r in read_rows(out / 'dev.jsonl')}
    if train & dev:
        raise ValueError('group split overlap')
    results['input_files'] = files
    results['scope'] = 'diagnostic candidate sampling; unknown family and cross-language links remain unverified'
    dump(out / 'sample-manifest.json', results)
    return results


def read_rows(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def evaluate(model, dev, domains, probes):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(model))
    stats = defaultdict(lambda: {'records': 0, 'characters': 0, 'utf8_bytes': 0,
        'tokens': 0, 'byte_tokens': 0, 'roundtrip_failures': 0, 'lengths': []})
    failures = []
    used = Counter()
    for row in read_rows(dev):
        text = row['text']
        ids = sp.encode(text)
        used.update(ids)
        s = stats[domains[row['source_id']]]
        s['records'] += 1
        s['characters'] += len(text)
        s['utf8_bytes'] += len(text.encode())
        s['tokens'] += len(ids)
        s['byte_tokens'] += sum(sp.is_byte(i) for i in ids)
        s['lengths'].append(len(ids))
        if sp.decode(ids) != text:
            s['roundtrip_failures'] += 1
            if len(failures) < 20:
                failures.append({'file': row['file'], 'row_index': row['row_index'],
                    'start_char': row['start_char'], 'before': text, 'after': sp.decode(ids)})
    for s in stats.values():
        lengths = s.pop('lengths')
        s['tokens_per_1000_characters'] = 1000 * s['tokens'] / s['characters']
        s['tokens_per_utf8_byte'] = s['tokens'] / s['utf8_bytes']
        s['p50_tokens'] = percentile(lengths, .5)
        s['p95_tokens'] = percentile(lengths, .95)
    return {'actual_vocab_size': sp.vocab_size(), 'sha256': digest(model),
        'file_bytes': model.stat().st_size, 'domains': dict(stats),
        'distinct_dev_token_ids': len(used), 'roundtrip_examples': failures,
        'probes': [{'text': t, 'tokens': len(sp.encode(t)), 'pieces': sp.encode(t, out_type=str),
                    'exact': sp.decode(sp.encode(t)) == t, 'decoded': sp.decode(sp.encode(t))} for t in probes]}


def run(config, out):
    import sentencepiece as spm
    if config['vocab_sizes'] != [16000, 24000] or not math.isclose(sum(config['source_weights'].values()), 1):
        raise ValueError('pilot requires explicit 16K/24K and normalized weights')
    if shutil.disk_usage(ROOT).free < 100 * 1024 ** 3:
        raise ValueError('disk reserve')
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    pointer = ROOT / 'models/mei-1.2-51m/tokenizer/TOKENIZER.json'
    current = ROOT / 'CURRENT.json'
    immutable = {str(p): digest(p) for p in [pointer, current]}
    dump(out / 'config.json', config)
    shutil.copyfile(__file__, out / 'tokenizer_compare.py.snapshot')
    shutil.copyfile(Path(__file__).with_name('materialize.py'), out / 'materialize.py.snapshot')
    dump(out / 'source-binding.json', {'inventory_sha256': digest(resolve_path(ROOT / config['inventory'])),
         'sentencepiece_version': spm.__version__, 'pointer_hashes': immutable})
    samples = sample(config, out)
    if samples['train']['characters'] < config['train_chars'] * .8:
        raise ValueError('sample under 80% target; inspect per-source shortfall before training')
    results = {}
    for size in config['vocab_sizes']:
        dest = out / f'vocab-{size}'
        dest.mkdir()
        options = {**config['trainer'], 'vocab_size': size, 'model_prefix': str(dest / 'tokenizer')}
        dump(dest / 'effective-options.json', options)
        print(json.dumps({'training_vocab_size': size}), flush=True)
        start = time.time()
        spm.SentencePieceTrainer.train(sentence_iterator=(r['text'] for r in read_rows(out / 'train.jsonl')), **options)
        result = evaluate(dest / 'tokenizer.model', out / 'dev.jsonl', config['source_domains'], config['probes'])
        result['elapsed_seconds_including_audit'] = time.time() - start
        dump(dest / 'report.json', result)
        results[str(size)] = result
        print(json.dumps({'completed_vocab_size': size, 'seconds': result['elapsed_seconds_including_audit']}), flush=True)
    results['legacy_24k'] = evaluate(resolve_path(ROOT / config['legacy_model']), out / 'dev.jsonl', config['source_domains'], config['probes'])
    for p, h in immutable.items():
        if digest(Path(p)) != h:
            raise ValueError(f'pointer changed during experiment: {p}')
    report = {'schema': 'mei-tokenizer-pair-pilot-v1', 'process_complete': True,
        'production_ready': False, 'sample': samples, 'results': results,
        'elapsed_seconds': time.time() - started,
        'limits': ['excerpt diagnostics, not LM or end-to-end tool quality',
                  'same sample used for both new vocabularies; legacy has different training/normalizer',
                  'group-disjoint pilot dev; unknown families/near duplicates not certified',
                  'no locked Eval was selected; future adoption requires isolation and runtime parity']}
    dump(out / 'report.json', report)
    return {'process_complete': True, 'production_ready': False, 'out': str(out)}
