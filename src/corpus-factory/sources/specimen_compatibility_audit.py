"""Bounded, read-only 24K and selected-CPT audit of reviewed specimens."""
import hashlib
import json
import math
import sqlite3
import struct
from collections import Counter, defaultdict
from pathlib import Path

from profiling import digest, resolve_path, ROOT
from source_manager import load_tokenizer
from qualified_sft_specimens import compact, leaves


def metrics(tok, text):
    ids = tok.encode(text)
    return {'characters': len(text), 'tokens': len(ids),
            'byte_tokens': sum(tok.sp.is_byte(i) for i in ids),
            'roundtrip': tok.decode(ids) == text}


def run(config, out):
    out = Path(out)
    if out.exists():
        raise ValueError('audit output already exists')
    pack = resolve_path(config['specimen_pack'])
    if digest(pack/'REPORT.json') != config['specimen_report_sha256']:
        raise ValueError('specimen report binding changed')
    manifest = json.loads((pack/'REPORT.json').read_text())
    if digest(pack/'specimens.jsonl') != manifest['files']['specimens.jsonl']:
        raise ValueError('specimen bytes changed')
    cases = [json.loads(l) for l in (pack/'specimens.jsonl').open()]
    tok = load_tokenizer(resolve_path(config['tokenizer_manifest']))
    corpus_run = resolve_path(config['cpt_run'])
    freeze = json.loads(resolve_path(config['cpt_freeze_config']).read_text())
    if freeze['tokenizer_manifest'] != config['tokenizer_manifest']:
        raise ValueError('different tokenizer in selected CPT')
    out.mkdir(parents=True)
    buckets = defaultdict(Counter)
    texts, details, spellings = [], [], {}
    def add(case, part, text):
        m = metrics(tok, text)
        if not m['roundtrip']:
            raise ValueError('roundtrip mismatch')
        key = case['language'] + '/' + part
        buckets[key].update({k:v for k,v in m.items() if k!='roundtrip'})
        buckets[key]['views'] += 1
        texts.append({'id':case['id']+'/'+part, 'text':text, 'ids':tok.encode(text)})
        return m
    for c in cases:
        values = {'query':c['input']['query'], 'schema':compact(c['input']['tools']),
                  'lm_prompt':c['encoding']['prompt'], 'lm_target':c['encoding']['target']}
        if c['heads']['narration']['semantic_label_mask']:
            values['narration_prompt'] = c['heads']['narration']['encoding']['prompt']
            values['narration_target'] = c['heads']['narration']['target']
        row = {'case':c['id'], 'language':c['language'], 'parts':{p:add(c,p,t) for p,t in values.items()}}
        pretty = json.dumps(c['heads']['lm']['target'], ensure_ascii=False, indent=2)
        row['pretty_target_tokens'] = len(tok.encode(pretty))
        row['joint_tokens'] = c['encoding']['joint_tokens']
        prefix = '只输出当前可执行的工具调用JSON数组；无调用输出[]。\n'
        for remove in ['tools','history']:
            alt = dict(c['input']);alt[remove]=[]
            row['removal_delta_'+remove] = len(tok.encode(c['encoding']['prompt']))-len(tok.encode(prefix+compact(alt)))
        details.append(row)
        for call in c['heads']['lm']['target']:
            terms = [('tool_name',call['name'])]
            terms += [('argument_key',str(path[-1])) for path,_ in leaves(call['arguments']) if path]
            for role,text in terms:
                ids = tok.encode(text)
                spellings[(role,text)] = {'role':role,'text':text,'ids':ids,
                    'pieces':[tok.sp.id_to_piece(i) for i in ids],**metrics(tok,text)}
    for term in ['"name":','"arguments":','"constraints":','"requested_fields":','[{','}]',
                 'public_risawoz_hotel','数量','正整数','单价','整数','小数','minimum','maximum','required','properties',
                 '订单号','srcPath','EC1A 1BB','/documents/my_notes.md','　Ａ001\t\r\n▁e\u0301']:
        ids=tok.encode(term)
        spellings[('diagnostic',term)]={'role':'diagnostic','text':term,'pieces':[tok.sp.id_to_piece(i) for i in ids],**metrics(tok,term)}
        texts.append({'id':'diagnostic/'+term,'text':term,'ids':ids})
    # Token counts for components are diagnostic, not an additive prompt budget.
    tokenizer_report = {'tokenizer_id':tok.tokenizer_id,'model_sha256':tok.model_sha256,
        'vocab_size':tok.vocab_size,'aggregate':dict(buckets),'views':details,
        'spellings':list(spellings.values()),'roundtrip_texts':len(texts),
        'compact_target_tokens':sum(r['parts']['lm_target']['tokens'] for r in details),
        'pretty_target_tokens':sum(r['pretty_target_tokens'] for r in details),
        'max_joint_tokens':max(r['joint_tokens'] for r in details),
        'status':'python_measured_browser_pending','other_tokenizers_compared':False,
        'scope':'specimen_serializer_not_production_packer; fragment_counts_not_additive'}

    dbpath = corpus_run/'index.sqlite'
    db = sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
    db.execute('BEGIN')
    files = {r['file_index']:dict(r) for r in db.execute('select * from input_files')}
    def attach(record, excerpt_limit=1800):
        r = dict(record); f = files[r['file_index']]
        with resolve_path(f['bin_path']).open('rb') as h:
            h.seek(r['token_offset']*2); raw=h.read(r['token_length']*2)
        ids=list(struct.unpack('<'+'H'*(len(raw)//2),raw))
        if len(ids)!=r['token_length'] or ids[0]!=2 or ids[-1]!=1:
            raise ValueError('CPT record token boundary mismatch')
        text=tok.decode(ids)
        if hashlib.sha256(text.encode()).hexdigest()!=r['text_sha']:
            raise ValueError('selected CPT text/hash mismatch')
        return {'record_id':r['id'],'source':r['source'],'phase':r['selected_phase'],
            'source_path':str(resolve_path(f['path'])),'row_index_base0':r['row_number'],
            'text_sha256':r['text_sha'],'token_offset':r['token_offset'],'tokens':r['token_length'],
            'encoded_path':f['bin_path'],'excerpt':text[:excerpt_limit],
            'decoded_text':text,'selected':True,'split':r['split'],'group_key':r['group_key']}
    def lookup_sha(sha):
        return db.execute('select r.*,s.phase as selected_phase from records r left join selected s on s.record_id=r.id where r.text_sha=?',(sha,)).fetchone()
    # Exact full request overlap: small known local source files only, not a
    # semantic scan of billions of tokens. MOSS is absent from the source index.
    overlaps=[]; seen_groups={}
    for c in cases:
        seen_groups.setdefault(c['family'],c)
    for source,file_index in [('RiSAWOZ',7),('ToolACE',87)]:
        wanted=[c for c in seen_groups.values() if c['source']==source]
        found={c['family']:[] for c in wanted}
        for idx,line in enumerate(resolve_path(files[file_index]['path']).open()):
            row=json.loads(line);text=row['text']
            for c in wanted:
                if c['input']['query'] in text:
                    rec=lookup_sha(hashlib.sha256(text.encode()).hexdigest())
                    entry={'source_line_base0':idx,'record_id':rec['id'] if rec else None,
                        'selected_phase':rec['selected_phase'] if rec else None,
                        'split':rec['split'] if rec else None}
                    if rec and rec['selected_phase'] is not None:
                        entry['evidence']=attach(rec)
                    found[c['family']].append(entry)
        overlaps += [{'case':c['id'],'source':source,'family':c['family'],'matches':found[c['family']]} for c in wanted]
    evidence=[]; source_counts={}
    for source in config['evidence_sources']:
        count=db.execute('select count(*),sum(r.token_length) from records r join selected s on s.record_id=r.id where r.source=?',(source,)).fetchone()
        source_counts[source]={'selected_records':count[0],'selected_tokens':count[1] or 0}
        rows=db.execute('select r.*,s.phase as selected_phase from records r join selected s on s.record_id=r.id where r.source=? order by r.priority,r.id limit ?',(source,config['per_source_limit']))
        for r in rows:
            evidence.append(attach(r))
    all_sources=[r[0] for r in db.execute('select distinct source from records')]
    total=db.execute('select count(*),sum(r.token_length) from records r join selected s on s.record_id=r.id').fetchone()
    db.commit();db.close()
    cpt_report={'selection_index':str(dbpath),'selection_index_sha256':digest(dbpath),
        'selection_plan_sha256':digest(corpus_run/'selection.json'),
        'freeze_config':config['cpt_freeze_config'],'freeze_config_sha256':digest(resolve_path(config['cpt_freeze_config'])),
        'selected_records':total[0],'selected_tokens':total[1],
        'sources':source_counts,'all_indexed_sources':all_sources,
        'original_group_query_overlap':overlaps,'bounded_evidence_count':len(evidence),
        'source_appearance_is_not_model_mastery':True,
        'production_release_status':'must_verify_separately; selected_index_is_not_inputs_ready',
        'sample_method':'first_by_frozen_priority_within_selected_source; targeted_not_population_estimate'}
    outputs={'tokenizer.json':tokenizer_report,'cpt.json':cpt_report,
        'golden.json':{'schema':'mei-lossless-tokenizer-golden-v1','model_sha256':tok.model_sha256,'cases':texts},
        'config.json':config}
    for name,value in outputs.items():
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    (out/'cpt-evidence.jsonl').write_text(''.join(compact(e)+'\n' for e in evidence))
    (out/'implementation.py.snapshot').write_text(Path(__file__).read_text())
    receipt={'status':'measured_pending_semantic_interpretation_and_browser',
        'specimen_report_sha256':config['specimen_report_sha256'],
        'outputs':{p.name:digest(p) for p in out.iterdir() if p.is_file()}}
    (out/'MEASUREMENT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    return {'texts_checked':len(texts),'bounded_cpt_evidence':len(evidence),
        'selected_tokens':total[1],'max_joint_tokens':tokenizer_report['max_joint_tokens'],
        'out':str(out)}
