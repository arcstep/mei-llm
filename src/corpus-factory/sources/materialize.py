"""Export usable raw-text candidates from acquired sources; no teacher scoring."""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil

from profiling import ROOT, digest, safe_id


def text_of(value):
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return '\n'.join(value)
    if isinstance(value, dict) and isinstance(value.get('text'), str):
        return value['text']
    raise ValueError('unsupported text shape; preserve via an explicit adapter')


def lines(path):
    return gzip.open(path, 'rb') if path.suffix == '.gz' else path.open('rb')


def materialize(config, out):
    out.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(out).free < 100 * 1024**3:
        raise ValueError('100 GiB free disk reserve reached')
    def write(name, value):
        with (out/name).open('x') as f: json.dump(value, f, ensure_ascii=False, indent=2)
    write('config.json', config)
    with (out/'implementation.py.snapshot').open('xb') as f: f.write(Path(__file__).read_bytes())
    excluded, exclusions = set(), []
    for rel in config.get('exclude_jsonl', []):
        path = ROOT/rel
        with lines(path) as f:
            for raw in f:
                if raw.strip(): excluded.add(hashlib.sha256(text_of(json.loads(raw)).encode()).hexdigest())
        exclusions.append({'path':rel,'sha256':digest(path)})
    seen, results, evidence = set(), [], []
    tokenizer = None
    if config.get('count_tokens'):
        from source_manager import load_tokenizer, tokenizer_pointer
        tokenizer = load_tokenizer()
    for source in config['sources']:
        sid = safe_id(source['source_id'])
        counts = Counter(); chars = 0
        destination = out/(sid+'.jsonl')
        with destination.open('x') as target:
            def emit(text, origin, group_id=None, metadata=None):
                nonlocal chars
                counts['seen'] += 1
                if not text.strip() or '\x00' in text:
                    counts['empty_or_nul'] += 1; return
                h = hashlib.sha256(text.encode()).hexdigest()
                if h in excluded:
                    counts['excluded_text_overlap'] += 1; return
                if h in seen:
                    counts['exact_duplicate'] += 1; return
                seen.add(h)
                record = {'id':hashlib.sha256((sid+':'+h).encode()).hexdigest(), 'text':text,
                    'text_sha256':h, 'source_id':sid, 'origin':origin, 'group_id':group_id,
                    'metadata':metadata or {}, 'candidate_kind':'cpt_raw', 'split':'candidate-unassigned'}
                target.write(json.dumps(record,ensure_ascii=False)+'\n')
                chars += len(text); counts['written'] += 1
                if tokenizer is not None: counts['tokens'] += len(tokenizer.encode_document(text))
            if source['kind'] == 'survey':
                directory = ROOT/source['survey']
                lock_path = directory/'survey.json'
                manifest = json.loads(lock_path.read_text())
                evidence.append({'path':str(lock_path),'sha256':digest(lock_path)})
                selected = set(source.get('include_sources', []))
                for rel, expected in sorted(manifest['artifacts'].items()):
                    if not Path(rel).name.startswith('sample-'): continue
                    origin_source = Path(rel).parts[0]
                    if selected and origin_source not in selected: continue
                    path = (directory/rel).resolve()
                    if not path.is_relative_to(directory.resolve()) or digest(path) != expected:
                        raise ValueError('acquired source sample changed')
                    sample = json.loads(path.read_text())
                    for row in sample['samples']:
                        if row.get('split') in {'test','validation','valid','dev'}:
                            counts['nontrain_split'] += 1; continue
                        emit(row['text'], {'sample_path':str(path),'sample_sha256':expected,
                            'source_id':origin_source,'shard':sample['shard']['path'],
                            'source_revision':sample.get('source_revision'), 'row_index':row['row_index'],
                            'original_split':row.get('split'), 'inclusion_probability':row.get('inclusion_probability')},
                            row.get('unit_id') or (row.get('original',{}).get('group_id') if isinstance(row.get('original'),dict) else None), row.get('metadata'))
            elif source['kind'] == 'wiki_onlyinclude':
                import re
                import html
                path = ROOT/source['path']
                if digest(path) != source['sha256']:
                    raise ValueError('translation source changed')
                pages = {p['title']:p for p in json.loads(path.read_text())['query']['pages'].values()}
                bodies = []; revisions = []
                for title in source['chapter_titles']:
                    page = pages[title]
                    if 'missing' in page: raise ValueError('translation chapter missing')
                    revision = page['revisions'][0]
                    text = revision['slots']['main']['*']
                    blocks = re.findall(r'<onlyinclude>(.*?)</onlyinclude>',text,re.S)
                    if len(blocks) != 1: raise ValueError('ambiguous translation body')
                    body = re.sub(r'</?u>', '', blocks[0])
                    body = re.sub(r'\[\[([^\[\]|]+)\]\]', r'\1', body)
                    if any(mark in body for mark in ('{{','}}','[[',']]','<','>')):
                        raise ValueError('unhandled translation markup')
                    bodies.append(title+'\n'+html.unescape(body).strip())
                    revisions.append({'title':title,'pageid':page['pageid'],'revid':revision['revid']})
                evidence.append({'path':str(path),'sha256':source['sha256']})
                emit('\n\n'.join(bodies),{'path':str(path),'sha256':source['sha256'],'chapters':revisions},
                     source['work_group'],{'language':'zh','translator':source['translator'],
                     'translation':True,'scope':'all listed narrative chapters; prefaces not included',
                     'semantic_review':'pending','register':'historical translation'})
            elif source['kind'] == 'jsonl_stream':
                path = ROOT/source['path']
                source_hash = digest(path)
                evidence.append({'path':str(path),'sha256':source_hash,'sampling':'all supplied train records'})
                with lines(path) as f:
                    for index, raw in enumerate(f):
                        if not raw.strip(): continue
                        original=json.loads(raw)
                        if isinstance(original,dict) and original.get('split') in {'test','validation','valid','dev'}:
                            counts['nontrain_split'] += 1; continue
                        if index % 100000 == 0:
                            if shutil.disk_usage(out).free < 100*1024**3: raise RuntimeError('disk reserve reached')
                            print(json.dumps({'source_id':sid,'input_row':index,'counts':dict(counts)}),flush=True)
                        emit(text_of(original), {'path':str(path),'sha256':source_hash,'row_index':index,
                            'original_split':'train','source_line':original.get('source_line') if isinstance(original,dict) else None},
                            original.get('group_id') if isinstance(original,dict) else None)
            elif source['kind'] == 'jsonl_reservoir':
                path = ROOT/source['path']; take = int(source['records'])
                if not 1 <= take <= 1_000_000: raise ValueError('bounded raw batch requires 1..1M records')
                rng = random.Random(f"{config.get('seed',20260913)}:{sid}")
                reservoir=[]; total=0
                with lines(path) as f:
                    for index, raw in enumerate(f):
                        if not raw.strip(): continue
                        total+=1
                        slot = total-1 if total <= take else rng.randrange(total)
                        if slot < take:
                            if total <= take: reservoir.append((index, raw))
                            else: reservoir[slot]=(index,raw)
                source_hash = digest(path)
                evidence.append({'path':str(path),'sha256':source_hash,'records_scanned':total,
                    'selected':len(reservoir),'sampling':'uniform whole-record reservoir','seed':config.get('seed',20260913)})
                for index, raw in sorted(reservoir):
                    original=json.loads(raw)
                    emit(text_of(original), {'path':str(path),'sha256':source_hash,'row_index':index,
                        'original_split':'train','source_line':original.get('source_line') if isinstance(original,dict) else None,
                        'inclusion_probability':len(reservoir)/total},
                        original.get('group_id') if isinstance(original,dict) else None)
            else: raise ValueError('unsupported acquisition kind')
        result={'source_id':sid,'file':destination.name,'sha256':digest(destination),
                'bytes':destination.stat().st_size,'characters':chars,'counts':dict(counts)}
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    report={'schema':'mei-raw-candidate-batch-v1','sources':results,'source_evidence':evidence,
        'exclusion_sources':exclusions,'records':sum(r['counts'].get('written',0) for r in results),
        'characters':sum(r['characters'] for r in results),'jsonl_bytes':sum(r['bytes'] for r in results),
        'teacher_calls':0,'generated_training_text':False,'release':False,
        'limitations':['exact dedup within this batch and listed eval texts only; cross-batch/near-duplicate isolation still required',
                       'source sampling and raw candidate preparation; not a frozen training release',
                       'upstream schemas, dialogue context, permissions and future split grouping must be preserved']}
    if tokenizer is not None:
        report.update(tokens=sum(r['counts'].get('tokens',0) for r in results),tokenizer=tokenizer_pointer(),token_count_method='encode_document; includes document boundaries')
    write('manifest.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['materialize'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--allow-network',action='store_true')
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text())
    if config.get('mode') == 'node_task_preparation':
        if args.allow_network:
            raise ValueError('node task preparation is offline only')
        from node_task_preparation import run
        result = run(config, args.out)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if config.get('mode') == 'node_task_v12_preparation':
        if args.allow_network:
            raise ValueError('v1.2 node task preparation is offline only')
        from node_task_v12 import run
        result = run(config, args.out)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if config.get('mode') == 'tokenizer_24k_candidate':
        if args.allow_network:
            raise ValueError('tokenizer candidate training is offline only')
        from tokenizer_candidate import run
        result = run(config, args.out)
        print(json.dumps({k: result[k] for k in ['tokenizer_id', 'status', 'production_ready']}, ensure_ascii=False))
        return 0
    if config.get('mode') == 'tokenizer_runtime_audit':
        if args.allow_network:
            raise ValueError('tokenizer runtime audit is offline only')
        from tokenizer_runtime_audit import run
        result = run(config, args.out)
        print(json.dumps({k: result[k] for k in ['status', 'python_cases', 'real_browser_passed']}, ensure_ascii=False))
        return 0
    if config.get('mode') == 'tokenizer_adopt':
        if args.allow_network:
            raise ValueError('tokenizer adoption is offline only')
        from tokenizer_adopt import run
        result = run(config, args.out)
        print(json.dumps({k: result[k] for k in ['tokenizer_id', 'status', 'current_mutated']}, ensure_ascii=False))
        return 0
    if config.get('mode') == 'v12_inputs_freeze':
        if args.allow_network:
            raise ValueError('v1.2 input freezing is offline only')
        from v12_inputs_freeze import run
        result = run(config, args.out)
        print(json.dumps({'status': result['status'], 'release_id': result['release_id'], 'books': result['books']}, ensure_ascii=False))
        return 0
    if config.get('mode') == 'v12_inputs_rebundle':
        if args.allow_network:
            raise ValueError('v1.2 input rebundling is offline only')
        from v12_inputs_rebundle import run
        result = run(config, args.out)
        print(json.dumps({'status': result['status'], 'release_id': result['release_id'], 'books': result['books']}, ensure_ascii=False))
        return 0
    if config.get('mode') == 'tokenizer_pair_pilot':
        if args.allow_network:
            raise ValueError('tokenizer pair pilot is offline only')
        from tokenizer_compare import run
        print(json.dumps(run(config, args.out), ensure_ascii=False))
        return 0
    if config.get('mode') == 'cpt_windows':
        from cpt_windows import run
        print(json.dumps(run(config, args.out), ensure_ascii=False))
        return 0
    if config.get('mode') == 'nemotron_cpt':
        from nemotron_cpt import run
        print(json.dumps(run(config, args.out), ensure_ascii=False))
        return 0
    if config.get('mode') == 'archive_cpt':
        from archive_materialize import run
        print(json.dumps(run(config, args.out, args.allow_network), ensure_ascii=False))
        return 0
    if config.get('mode') == 'gutenberg_cpt':
        from gutenberg_materialize import run
        print(json.dumps(run(config, args.out, args.allow_network), ensure_ascii=False))
        return 0
    if config.get('mode') == 'tool_dialogue':
        from tool_dialogue_materialize import run
        print(json.dumps(run(config, args.out), ensure_ascii=False))
        return
    if config.get('mode') == 'corpus_matrix':
        from corpus_matrix import matrix
        result=matrix(config,args.out)
        print(json.dumps(result['totals']))
        return 0
    if config.get('mode') == 'registered_download':
        if not args.allow_network: raise ValueError('download requires --allow-network')
        from source_manager import entry_for, collector, download
        entry=entry_for(config['source_id'])
        batch=collector().resolve_batch(entry)
        sizes=[row.get('bytes') for row in batch['files']]
        if any(type(n) is not int for n in sizes) or sum(sizes)>config['max_download_bytes']:
            raise ValueError('unknown or excessive download size')
        if shutil.disk_usage(args.out.parent).free-sum(sizes)<100*1024**3:
            raise ValueError('disk reserve reached')
        result=download(config['source_id'],args.out,subset=None,filenames=None,authorization=batch['batch_token'],dry_run=False,cache_dir=None)
        print(json.dumps(result,ensure_ascii=False))
        return 0
    if config.get('mode') == 'approved_batch':
        from preparation_batch import run
        result=run(config,args.out)
        print(json.dumps({'jobs':len(result['jobs']),'release':False}))
        return 0
    if config.get('mode') == 'subtitle_bulk':
        from subtitle_materialize import subtitle_bulk
        result=subtitle_bulk(config,args.out,args.allow_network)
        print(json.dumps({k:result[k] for k in ['records','tokens','release']}))
        return 0
    if config.get('mode') == 'delivery_inventory':
        from delivery import aggregate
        result=aggregate(config,args.out)
        print(json.dumps(result['counts']))
        return 0
    if config.get('mode') == 'parquet_bulk':
        from bulk_materialize import bulk
        result=bulk(config,args.out,args.allow_network)
        print(json.dumps({k:result[k] for k in ['records','tokens','teacher_calls','release']}))
        return 0
    result=materialize(config,args.out)
    print(json.dumps({k:result[k] for k in ['records','characters','jsonl_bytes','teacher_calls','release']}))
    return 0


if __name__=='__main__': raise SystemExit(main())
