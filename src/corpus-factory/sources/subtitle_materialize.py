"""Prepare complete subtitle documents, conservatively one version per work key."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from profiling import ROOT,digest,Budget
from source_manager import load_tokenizer,tokenizer_pointer


def dump(path,value):
    with path.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def work_key(name,kind):
    p=Path(name)
    # TED stores unrelated talks together. OpenSubtitles directories identify a
    # conservative version group, not a verified cross-language canonical work.
    return p.stem if kind=='ted' else str(p.parent)


def subtitle_bulk(config,out,allow_network=False):
    if not allow_network:raise ValueError('archive acquisition requires --allow-network')
    if not 1 <= config.get('download_workers',2) <= 2:raise ValueError('download concurrency')
    out.mkdir(parents=True,exist_ok=False);dump(out/'config.json',config)
    shutil.copyfile(__file__,out/'implementation.py.snapshot')
    budget=Budget(out,config.get('max_network_bytes',4*1024**3),100*1024**3)
    def acquire(item):
        target=out/(item['source_id']+'.zip')
        expected=item['expected_bytes'];budget.reserve(expected)
        request=urllib.request.Request(item['url'],headers={'Accept-Encoding':'identity','User-Agent':'mei-corpus-preparation/1'})
        h=hashlib.sha256();size=0
        with urllib.request.urlopen(request,timeout=45) as response,target.open('xb') as stream:
            if response.status!=200:raise ValueError('expected full archive HTTP 200')
            for block in iter(lambda:response.read(1024**2),b''):
                size+=len(block)
                if size>expected:raise ValueError('archive exceeds frozen size')
                budget.reserve(0);stream.write(block);h.update(block)
        if size!=expected:raise ValueError('archive size differs from frozen survey evidence')
        receipt={'source':item,'file':target.name,'bytes':size,'sha256':h.hexdigest()}
        dump(out/(item['source_id']+'-download.json'),receipt)
        print(json.dumps({'downloaded':item['source_id'],'bytes':size}),flush=True)
        return receipt
    with ThreadPoolExecutor(max_workers=config.get('download_workers',2)) as pool:
        downloads=list(pool.map(acquire,config['sources']))
    seen=set();tokenizer=load_tokenizer();outputs=[];evidence=[]
    from materialize import lines,text_of
    for rel in config.get('exclude_jsonl',[]):
        p=ROOT/rel
        with lines(p) as f:
            for line in f:
                if line.strip():seen.add(hashlib.sha256(text_of(json.loads(line)).encode()).digest())
        evidence.append({'path':rel,'sha256':digest(p)})
    for download in downloads:
        source=download['source'];sid=source['source_id'];counts=Counter()
        db=sqlite3.connect(out/(sid+'-versions.sqlite'))
        db.execute('CREATE TABLE versions (work TEXT PRIMARY KEY, rank TEXT, member TEXT, hash TEXT, text TEXT, metadata TEXT)')
        with zipfile.ZipFile(out/download['file']) as archive:
            for i,member in enumerate(sorted(archive.infolist(),key=lambda m:m.filename)):
                if not member.filename.endswith('.xml'):continue
                counts['xml_files']+=1
                if member.file_size>16*1024**2:
                    counts['oversized_member']+=1;continue
                raw=archive.read(member) # CRC checked by zipfile
                try:tree=ET.fromstring(raw)
                except ET.ParseError:
                    counts['malformed_xml']+=1;continue
                metadata={}
                for section in tree.iter('meta'):
                    for parent in section:
                        for child in parent:
                            metadata[f'{parent.tag}.{child.tag}']=(child.text or '').strip()
                if metadata.get('subtitle.machine_translated')=='1':
                    counts['declared_machine_translation']+=1;continue
                segments=[''.join(s.itertext()).strip() for s in tree.iter('s')]
                text='\n'.join(s for s in segments if s)
                if not text.strip():counts['empty']+=1;continue
                # Prefer explicitly non-machine translated, then higher source
                # rating, then more complete text, with deterministic member tie.
                try:rating=float(metadata.get('subtitle.rating','0'))
                except ValueError:rating=0
                if not 0<=rating<=10:rating=0
                rank=f"{int(metadata.get('subtitle.machine_translated')=='0')}:{rating:06.2f}:{len(text):012d}:{member.filename}"
                key=work_key(member.filename,source['kind'])
                db.execute('INSERT INTO versions VALUES (?,?,?,?,?,?) ON CONFLICT(work) DO UPDATE SET rank=excluded.rank,member=excluded.member,hash=excluded.hash,text=excluded.text,metadata=excluded.metadata WHERE excluded.rank>versions.rank',
                           (key,rank,member.filename,hashlib.sha256(raw).hexdigest(),text,json.dumps(metadata,ensure_ascii=False)))
                if i%1000==0:db.commit();budget.reserve(0)
        db.commit();target=out/(sid+'.jsonl');tokens=0;chars=0
        with target.open('x') as f:
            for key,member,rawhash,text,metadata in db.execute('SELECT work,member,hash,text,metadata FROM versions ORDER BY work'):
                counts['version_groups']+=1;h=hashlib.sha256(text.encode()).digest()
                if h in seen:counts['exact_or_excluded_duplicate']+=1;continue
                seen.add(h);n=len(tokenizer.encode_document(text));tokens+=n;chars+=len(text);counts['written']+=1
                f.write(json.dumps({'text':text,'text_sha256':h.hex(),'source_id':sid,'tokens':n,
                    'group_id':sid+':'+key,'split':'candidate-unassigned',
                    'origin':{'url':source['url'],'archive_sha256':download['sha256'],'member':member,'member_sha256':rawhash},
                    'metadata':json.loads(metadata),'source_identity':'authored_subtitles' if source['kind']!='ted' else 'speech_transcript',
                    'grouping_scope':'conservative within-source version directory; cross-language work linkage pending'},ensure_ascii=False)+'\n')
        db.close()
        result={'source_id':sid,'file':target.name,'sha256':digest(target),'bytes':target.stat().st_size,'tokens':tokens,'characters':chars,'counts':dict(counts)}
        outputs.append(result);print(json.dumps(result),flush=True)
    report={'schema':'mei-subtitle-candidate-v1','outputs':outputs,'records':sum(x['counts'].get('written',0) for x in outputs),
            'tokens':sum(x['tokens'] for x in outputs),'tokenizer':tokenizer_pointer(),'downloads':downloads,
            'exclusions':evidence,'release':False,'teacher_calls':0,'generated_text':False,
            'limitations':['upstream language tags are not proof of actual script','one conservative version group can contain more than one independent edition',
            'source rating and non-machine flags are not semantic quality certificates','near duplicates and cross-language work grouping pending',
            'public archive access is not a blanket underlying-work license; source use conditions remain attached']}
    dump(out/'manifest.json',report)
    return report
