"""Mutually exclusive primary-use x language/script inventory of completed text."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import shutil
import opencc
from profiling import resolve_path, ROOT,digest
from source_manager import load_tokenizer,tokenizer_pointer

DOMAINS={'web-hq-zh':'基础语言理解','wiki-zh':'基础语言理解','wiki-en':'基础语言理解',
         'lccc-dialogue':'口语理解','naturalconv':'口语理解',
         'crosswoz':'任务与工具语义','risawoz':'任务与工具语义',
         'literature-dialogue':'文学与剧本表达','technical-guides':'代码与结构理解',
         'constraint-tests':'代码与结构理解','github-code':'代码与结构理解'}
LANGS=['简体为主','繁体为主','英文为主','中文共有字／待判','繁简混用','中外文混合／其他']

class ScriptClassifier:
    def __init__(self,directory):
        self.tables=[];keys=[]
        for name in ['STCharacters.txt','TSCharacters.txt']:
            p=Path(opencc.__file__).parent/'dictionary'/name
            shutil.copyfile(p,directory/name)
            self.tables.append({'file':name,'sha256':digest(p)})
            keys.append({line.split('\t')[0] for line in p.read_text().splitlines() if '\t' in line and len(line.split('\t')[0])==1})
        self.simple=re.compile('['+re.escape(''.join(sorted(keys[0]-keys[1])))+']+')
        self.trad=re.compile('['+re.escape(''.join(sorted(keys[1]-keys[0])))+']+')
        self.simple_map=dict.fromkeys(map(ord,keys[0]-keys[1]),None)
        self.trad_map=dict.fromkeys(map(ord,keys[1]-keys[0]),None)
        self.han=re.compile('[\u3400-\u9fff\U00020000-\U0003134f]+')
        self.latin=re.compile('[A-Za-z]+');self.kana=re.compile('[\u3040-\u30ff]+')
    def classify(self,text,english_source=False):
        n=len(text);han=n-len(self.han.sub('',text));latin=n-len(self.latin.sub('',text))
        if han==0 or (latin>=40 and han/(han+latin)<.2):
            return '英文为主' if english_source and latin else '中外文混合／其他'
        kana=n-len(self.kana.sub('',text))
        if kana>=5 and kana>han*.1:return '中外文混合／其他'
        simple=n-len(text.translate(self.simple_map));trad=n-len(text.translate(self.trad_map));total=simple+trad
        if not total:return '中文共有字／待判'
        if simple/total>=.95:return '简体为主'
        if trad/total>=.95:return '繁体为主'
        return '繁简混用'


def matrix(config,out):
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    shutil.copyfile(__file__,out/'implementation.py.snapshot')
    classifier=ScriptClassifier(out);tokenizer=load_tokenizer();seen=set()
    domains={**DOMAINS,**config.get('domain_mapping',{})}
    english_sources=set(config.get('english_source_ids',[]))|{'wiki-en','technical-guides','constraint-tests'}
    cells=defaultdict(Counter);by_source=defaultdict(Counter);totals=Counter();files=[];examples=defaultdict(list)
    with (out/'document-classification.jsonl').open('x') as ledger:
        for rel in config['pools']:
            pool=resolve_path(ROOT/rel);manifest=json.loads((pool/'manifest.json').read_text())
            countfile=pool/'token-count.json'
            count=json.loads(countfile.read_text()) if countfile.exists() else manifest
            expected_hash=count.get('tokenizer_model_sha256') or count.get('tokenizer',{}).get('model_sha256')
            if expected_hash!=tokenizer.model_sha256:raise ValueError('tokenizer mismatch')
            output_rows=config.get('pool_outputs',{}).get(rel)
            output_rows=output_rows or manifest.get('outputs') or manifest.get('sources') or ([manifest['output']] if manifest.get('output') else [])
            if not output_rows and manifest.get('file'):
                output_rows=[{'file':manifest['file'],'sha256':manifest['sha256'],
                              'tokens':manifest.get('tokens'),
                              'counts':{'written':manifest.get('records')}}]
            for file in output_rows:
                path=pool/file['file'];records=0;filehash=hashlib.sha256();token_sum=0
                with path.open('rb') as f:
                    for index,line in enumerate(f):
                        filehash.update(line);row=json.loads(line);text=row['text'];body=text.encode();key=hashlib.sha256(body).digest()
                        if key.hex()!=row['text_sha256']:raise ValueError('row hash mismatch')
                        records+=1;sid=row['source_id'];domain=domains.get(sid,'待归类用途')
                        n=row.get('tokens')
                        if n is None:n=len(tokenizer.encode_document(text))
                        token_sum+=n
                        if key in seen:totals['duplicate_records']+=1;totals['duplicate_tokens']+=n;continue
                        seen.add(key)
                        english=(sid in english_sources or row.get('metadata',{}).get('language')=='en')
                        language=classifier.classify(text,english)
                        v={'records':1,'tokens':n,'characters':len(text),'body_utf8_bytes':len(body)}
                        cells[(domain,language)].update(v);by_source[(sid,language)].update(v);totals.update(v)
                        ledger.write(json.dumps({'file':str(path.relative_to(ROOT)),'row_index':index,'text_sha256':key.hex(),'domain':domain,'language':language,'tokens':n},ensure_ascii=False)+'\n')
                        if len(examples[(domain,language)])<3:examples[(domain,language)].append({'file':str(path.relative_to(ROOT)),'row_index':index,'text':text[:300]})
                expected_records=file.get('counts',{}).get('written')
                if expected_records is None and len(output_rows)==1:expected_records=manifest.get('records')
                if expected_records is not None and records!=expected_records:raise ValueError('file record count mismatch')
                if filehash.hexdigest()!=file['sha256']:raise ValueError('file receipt mismatch')
                expected_tokens=file.get('tokens') or file.get('counts',{}).get('tokens')
                if expected_tokens is None and len(output_rows)==1:expected_tokens=count.get('tokens')
                if expected_tokens is None:expected_tokens=next(v['tokens'] for v in count['sources'] if v['file']==file['file'])
                if token_sum!=expected_tokens:raise ValueError('file token count mismatch')
                files.append({'file':str(path.relative_to(ROOT)),'sha256':filehash.hexdigest(),'records':records,'tokens':token_sum})
                print(json.dumps({'file':str(path.relative_to(ROOT)),'tokens_processed':totals['tokens']}),flush=True)
    report={'schema':'mei-corpus-domain-script-inventory-v1','count_basis':'zh-24k-v3 encode_document, BOS/EOS included; completed candidate text only',
            'tokenizer':tokenizer_pointer(),'script_dictionary':classifier.tables,'totals':dict(totals),
            'cells':[{'domain':d,'language':l,**v} for (d,l),v in sorted(cells.items())],
            'source_cells':[{'source_id':s,'language':l,**v} for (s,l),v in sorted(by_source.items())],
            'files':files,'domain_mapping':DOMAINS,'languages':LANGS,
            'classification_method':['one primary domain per source; not independent capability measurements',
              'whole-document label; not per-token language identification',
              'OpenCC exclusive single-character keys; >=95% of discriminating hits yields dominant script',
              'no discriminating characters stays unclassified Chinese; never forced to simplified',
              'English needs existing source label plus Latin dominance; otherwise non-Chinese is left mixed/other',
              'kana-heavy documents remain mixed/other; some other languages/ambiguous character senses may remain misclassified'],
            'classification_ledger_sha256':digest(out/'document-classification.jsonl'),
            'examples':[{'domain':d,'language':l,'items':v} for (d,l),v in sorted(examples.items())],
            'release':False,'teacher_calls':0,'classification_is_heuristic':True,
            'exclusions':'incomplete downloads, unmaterialized raw files, planned sources, SFT/eval and repeated source versions not added'}
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
