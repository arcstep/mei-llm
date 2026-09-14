"""Verify immutable candidate pools and publish a deduplicated inventory receipt."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from profiling import ROOT, digest
from source_manager import load_tokenizer, tokenizer_pointer


def aggregate(config,out):
    out.mkdir(parents=True,exist_ok=False)
    with (out/'config.json').open('x') as f:json.dump(config,f,indent=2)
    seen=set(); results=[]; totals=Counter();tokenizer=None
    pointer=tokenizer_pointer()
    with (out/'duplicate-occurrences.jsonl').open('x') as duplicates:
        for rel in config['pools']:
            directory=ROOT/rel; manifest_path=directory/'manifest.json'
            manifest=json.loads(manifest_path.read_text())
            count_path=directory/'token-count.json'
            token_receipt=json.loads(count_path.read_text()) if count_path.exists() else manifest
            if count_path.exists() and token_receipt['manifest_sha256'] != digest(manifest_path):
                raise ValueError('token count binding changed')
            token_hash=token_receipt.get('tokenizer_model_sha256') or token_receipt.get('tokenizer',{}).get('model_sha256')
            if token_hash != pointer['model_sha256']:raise ValueError('mixed tokenizer inventories require separate accounting')
            output_files=manifest.get('outputs',manifest.get('sources',[]))
            counts=Counter(); pool_tokens=token_receipt['tokens'];duplicate_tokens=0
            for item in output_files:
                path=directory/item['file']
                if digest(path)!=item['sha256']:raise ValueError(f'candidate file changed: {path}')
                file_rows=0
                with path.open() as stream:
                    for i,line in enumerate(stream):
                        row=json.loads(line);text=row['text'];body=text.encode();key=hashlib.sha256(body).digest()
                        if key.hex()!=row['text_sha256']:raise ValueError('row text hash changed')
                        file_rows+=1;counts['records']+=1;counts['characters']+=len(text);counts['body_utf8_bytes']+=len(body)
                        if key in seen:
                            if tokenizer is None:tokenizer=load_tokenizer()
                            duplicate_tokens+=len(tokenizer.encode_document(text));counts['duplicates']+=1
                            duplicates.write(json.dumps({'file':str(path),'row_index':i,'text_sha256':key.hex()})+'\n')
                        else:
                            seen.add(key);counts['unique_records']+=1
                            counts['unique_characters']+=len(text);counts['unique_body_utf8_bytes']+=len(body)
                if file_rows!=item['counts']['written']:raise ValueError('manifest row count mismatch')
            if counts['records']!=manifest['records']:raise ValueError('pool record count mismatch')
            totals.update(counts);totals['gross_tokens']+=pool_tokens;totals['unique_tokens']+=pool_tokens-duplicate_tokens
            results.append({'pool':rel,'manifest_sha256':digest(manifest_path),
                            'token_receipt_sha256':digest(count_path) if count_path.exists() else digest(manifest_path),
                            'counts':dict(counts),'gross_tokens':pool_tokens,'duplicate_tokens':duplicate_tokens})
            print(json.dumps({'pool':rel,'counts':dict(counts),'current_tokenizer_unique_tokens':totals['unique_tokens']}),flush=True)
    report={'schema':'mei-candidate-delivery-inventory-v2','counts':dict(totals),'batches':results,
            'tokenizer':pointer,'count_basis':'current frozen tokenizer encode_document(text), BOS/EOS included; JSON metadata excluded',
            'future_tokenizer_count':'unknown until re-encoding; quotas must be reconciled after adoption',
            'target_tokens':3000000000,'release':False,'teacher_calls':0,
            'duplicate_exclusions_sha256':digest(out/'duplicate-occurrences.jsonl'),
            'limitations':['unique count is the union described by ordered pools and duplicate exclusion list; original files are unchanged',
                          'near duplicates, final language/domain proportions and group isolation remain pending']}
    with (out/'manifest.json').open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    return report
