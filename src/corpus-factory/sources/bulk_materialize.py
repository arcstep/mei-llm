"""Stream original Parquet row groups into hash-bound CPT candidates.

Production preparation, not a population estimator. Uses the existing frozen
source selection; stores fetched source text before deduplication/token counting.
"""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor
from collections import Counter

from profiling import ROOT, Budget, RangeReader, digest, seeded
from source_manager import load_tokenizer, tokenizer_pointer


def dump(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def group_selection(metadata, fraction, seed, identity):
    if not 0 < fraction <= 1:
        raise ValueError('row_group_fraction must be in (0,1]')
    groups = [i for i in range(metadata.num_row_groups) if metadata.row_group(i).num_rows]
    chosen = sorted(seeded(seed, identity).sample(groups, math.ceil(len(groups)*fraction)))
    return chosen, len(chosen)/len(groups) if groups else 0


def bulk(config, out, allow_network=False):
    import pyarrow as pa
    import pyarrow.parquet as pq
    remote = any('url' in item for item in config['files'])
    if remote and not allow_network:
        raise ValueError('remote preparation requires --allow-network')
    if not 1 <= config.get('download_workers', 2) <= 2:
        raise ValueError('download concurrency must be 1 or 2')
    out.mkdir(parents=True, exist_ok=False)
    reserve = int(config.get('reserve_disk_bytes', 100*1024**3))
    if shutil.disk_usage(out).free < reserve:
        raise RuntimeError('disk reserve reached')
    dump(out/'config.json', config)
    shutil.copyfile(__file__, out/'implementation.py.snapshot')
    budget = Budget(out, int(config.get('max_network_bytes', 30*1024**3)), reserve)
    rawdir = out/'original-columns'; rawdir.mkdir()
    for evidence in config.get('locks', []):
        if digest(ROOT/evidence['path']) != evidence['sha256']:
            raise ValueError('source selection lock changed')

    def acquire(pair):
        index, item = pair
        raw = rawdir/f'{index:04d}.parquet'
        receipt = rawdir/f'{index:04d}.json'
        try:
            # Adoption only uses immutable, hash-checked original-column artifacts.
            if item.get('adopt_receipt'):
                old_path = ROOT/item['adopt_receipt']
                if digest(old_path) != item['adopt_receipt_sha256']:
                    raise ValueError('adoption receipt changed')
                old = json.loads(old_path.read_text())
                old_raw = old_path.parent/old['raw_file']
                identity_keys = ('path','url','source_revision','shard_probability')
                if any(old['source'].get(k) != item.get(k) for k in identity_keys):
                    raise ValueError('adoption source differs')
                if old['seed'] != config['seed'] or old['row_group_fraction'] != config['row_group_fraction']:
                    raise ValueError('adoption selection differs')
                if digest(old_raw) != old['raw_sha256']:
                    raise ValueError('adoption source bytes changed')
                shutil.copyfile(old_raw, raw)
                old.update(raw_file=raw.name, adopted_from=str(old_path), network_reused=True)
                dump(receipt, old)
                return old
            handle = RangeReader(item['url'], budget) if 'url' in item else ROOT/item['path']
            pf = pq.ParquetFile(handle, pre_buffer=False)
            groups, probability = group_selection(pf.metadata, config['row_group_fraction'], config['seed'], item['path'])
            columns = [c for c in config.get('columns', ['text','id','url','title']) if c in pf.schema_arrow.names]
            text_column=config.get('text_column','text')
            if text_column not in columns:
                raise ValueError('configured original text column missing')
            offsets = [0]
            for g in range(pf.metadata.num_row_groups):
                offsets.append(offsets[-1]+pf.metadata.row_group(g).num_rows)
            count = 0
            schema = pa.schema([('text',pa.string()),('row_index',pa.int64()),('row_group',pa.int32()),('metadata_json',pa.string())])
            with pq.ParquetWriter(raw, schema, compression='zstd') as writer:
                for g in groups:
                    budget.reserve(0)
                    rows = pf.read_row_group(g, columns=columns, use_threads=False).to_pylist()
                    records = [{'text':r[text_column], 'row_index':offsets[g]+i,'row_group':g,
                                'metadata_json':json.dumps({k:v for k,v in r.items() if k!=text_column},ensure_ascii=False)}
                               for i,r in enumerate(rows)]
                    writer.write_table(pa.Table.from_pylist(records, schema=schema))
                    count += len(records)
            source_hash = digest(handle) if isinstance(handle,Path) else None
            if not isinstance(handle,Path): handle.close()
            result = {'source':item,'source_file_sha256':source_hash,'population_rows':pf.metadata.num_rows,
                      'population_groups':pf.metadata.num_row_groups,'selected_groups':groups,'rows':count,
                      'conditional_row_probability':probability,'raw_file':raw.name,'raw_sha256':digest(raw),
                      'raw_bytes':raw.stat().st_size,'seed':config['seed'],'row_group_fraction':config['row_group_fraction']}
            dump(receipt,result)
            print(json.dumps({'acquired':item['path'],'rows':count,'bytes':result['raw_bytes']},ensure_ascii=False),flush=True)
            return result
        except Exception as e:
            failure = {'source':item,'error':f'{type(e).__name__}: {e}'}
            dump(rawdir/f'{index:04d}.failure.json',failure)
            print(json.dumps(failure,ensure_ascii=False),flush=True)
            return failure

    with ThreadPoolExecutor(max_workers=config.get('download_workers',2)) as executor:
        acquired = list(executor.map(acquire,enumerate(config['files'])))
    dump(out/'acquisition.json', {'files':acquired,'network_reserved_bytes':budget.reserved})
    if any('error' in r for r in acquired):
        raise RuntimeError('acquisition incomplete; successful originals retained with per-file receipts for new-ID adoption')
    # Stream text only. Excluded and current hashes are compact; no body reservoir.
    from materialize import lines, text_of
    seen = set(); exclusions=[]
    for rel in config.get('exclude_jsonl',[]):
        path=ROOT/rel
        with lines(path) as f:
            for line in f:
                if line.strip(): seen.add(hashlib.sha256(text_of(json.loads(line)).encode()).digest())
        exclusions.append({'path':rel,'sha256':digest(path)})
    tokenizer = load_tokenizer(); counts=Counter(); tokens=0; chars=0
    outputs=[]
    for index, entry in enumerate(acquired):
        destination=out/f'part-{index:04d}.jsonl'
        part_counts=Counter(); part_tokens=0
        with destination.open('x') as target:
            for batch in pq.ParquetFile(rawdir/entry['raw_file']).iter_batches(batch_size=1024):
                budget.reserve(0)
                for row in batch.to_pylist():
                    text=row['text']; part_counts['seen']+=1
                    if not text or not text.strip() or '\x00' in text:
                        part_counts['empty_or_nul']+=1;continue
                    metadata=json.loads(row['metadata_json'])
                    filters=config.get('file_filters',{})
                    file_path=str(metadata.get('path',''))
                    if filters.get('extensions') and Path(file_path).suffix.lower() not in filters['extensions']:
                        part_counts['other_extension']+=1;continue
                    if filters.get('exclude_path_regex') and re.search(filters['exclude_path_regex'],file_path,re.I):
                        part_counts['dependency_or_build_path']+=1;continue
                    if filters.get('exclude_header_regex') and re.search(filters['exclude_header_regex'],text[:2000],re.I):
                        part_counts['explicit_generated_header']+=1;continue
                    if filters.get('licenses') and metadata.get('license') not in filters['licenses']:
                        part_counts['license_not_selected']+=1;continue
                    key=hashlib.sha256(text.encode()).digest()
                    if key in seen:
                        part_counts['exact_or_excluded_duplicate']+=1;continue
                    seen.add(key)
                    n=len(tokenizer.encode_document(text))
                    result={'text':text,'text_sha256':key.hex(),'source_id':config['source_id'],
                            'tokens':n,'group_id':metadata.get('repo_name'),'split':'candidate-unassigned',
                            'origin':{'source':entry['source'],'row_index':row['row_index'],'row_group':row['row_group'],
                                      'raw_sha256':entry['raw_sha256'],
                                      'inclusion_probability':entry['source'].get('shard_probability',1)*entry['conditional_row_probability']},
                            'metadata':metadata}
                    target.write(json.dumps(result,ensure_ascii=False)+'\n')
                    part_counts['written']+=1;part_tokens+=n;chars+=len(text)
        counts.update(part_counts);tokens+=part_tokens
        output={'file':destination.name,'sha256':digest(destination),'bytes':destination.stat().st_size,
                'counts':dict(part_counts),'tokens':part_tokens}
        outputs.append(output)
        print(json.dumps({'materialized':destination.name,'tokens':part_tokens,'total_tokens':tokens}),flush=True)
    report={'schema':'mei-bulk-raw-candidate-v1','status':'candidate_complete','source_id':config['source_id'],
            'records':counts['written'],'counts':dict(counts),'tokens':tokens,'characters':chars,
            'tokenizer':tokenizer_pointer(),'token_count_method':'encode_document; includes document boundaries',
            'outputs':outputs,'exclusions':exclusions,'acquisition_sha256':digest(out/'acquisition.json'),
            'config_sha256':digest(out/'config.json'),'implementation_sha256':digest(out/'implementation.py.snapshot'),
            'release':False,'teacher_calls':0,'generated_text':False,
            'limitations':['candidate preparation, not population inference',
                           'near duplicate and group split isolation remain pending',
                           'mixed script remains unclassified; source language is not actual script proof']}
    dump(out/'manifest.json',report)
    return report
