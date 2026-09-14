"""Sample whole tables/documents through a remote ZIP central directory."""
from __future__ import annotations

import fnmatch
import gzip
import hashlib
import io
import json
import math
import random
import zipfile


def survey_json_value(value):
    """Keep non-finite numbers explicit; original typed bytes stay bound separately."""
    if isinstance(value,float) and not math.isfinite(value):
        return {'__mei_survey_nonfinite_float__':repr(value)}
    if isinstance(value,dict): return {str(k):survey_json_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [survey_json_value(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return {'__mei_survey_typed_value__':type(value).__name__,'display':str(value)}


def sample_zip(handle, shard, source, seed, count, metadata_of, raw_directory=None):
    with zipfile.ZipFile(handle) as archive:
        all_entries = [{"path": i.filename, "bytes": i.file_size, "compressed_bytes": i.compress_size,
                        "crc32": i.CRC} for i in archive.infolist() if not i.is_dir()]
        members = sorted((r for r in all_entries if any(fnmatch.fnmatch(r['path'], p) for p in source['member_patterns'])), key=lambda r:r['path'])
        if not members: raise ValueError('no archive members matched')
        rng = random.Random(hashlib.sha256(f'{seed}:{shard["path"]}'.encode()).hexdigest())
        indices = sorted(rng.sample(range(len(members)), min(count,len(members))))
        samples=[]
        for index in indices:
            member=members[index]
            if member['bytes'] > 16 << 20: raise ValueError('selected complete member exceeds 16 MiB bound; no silent replacement')
            raw=archive.read(member['path'])  # ZIP checks CRC; full archive checksum not claimed.
            raw_name=None
            if raw_directory is not None:
                raw_directory.mkdir(parents=True,exist_ok=True)
                raw_name=f"member-{hashlib.sha256(shard['path'].encode()).hexdigest()[:12]}-{index}.raw"
                with (raw_directory/raw_name).open('xb') as f:f.write(raw)
            kind=source['member_format']
            if kind=='parquet-table':
                import pyarrow.parquet as pq
                table=pq.read_table(io.BytesIO(raw))
                original={'headers':table.column_names,'rows':table.to_pylist(),
                          'arrow_schema':str(table.schema),
                          'source_metadata':{k.decode('utf-8',errors='replace'):v.decode('utf-8',errors='replace') for k,v in (table.schema.metadata or {}).items()}}
                original=survey_json_value(original)
                text=json.dumps(original,ensure_ascii=False,sort_keys=True,allow_nan=False)
                original=json.loads(text)  # Survey JSON view; exact typed values retained in raw Parquet.
            elif kind=='jsonl-table-gzip':
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f: decoded=f.read((32 << 20)+1)
                if len(decoded)>32<<20: raise ValueError('expanded table exceeds bound')
                rows=[json.loads(line) for line in decoded.splitlines() if line.strip()]
                original={'rows':rows,'headers':None,'annotations_joined':False}
                text=json.dumps(original,ensure_ascii=False,sort_keys=True)
            elif kind=='xml-subtitles':
                import xml.etree.ElementTree as ET
                tree=ET.fromstring(raw)
                segments=[{'source_id':s.get('id'),'text':''.join(s.itertext()).strip(),
                           'times':[dict(t.attrib) for t in s.iter('time')]} for s in tree.iter('s')]
                if not segments:raise ValueError('no subtitle sentences; explicit adapter needed')
                original={'segments':segments,'source_document':member['path'],'speaker_identity':'unknown',
                          'work_identity_verified':False,'metadata_xml':[ET.tostring(m,encoding='unicode') for m in tree.iter('meta')]}
                text='\n'.join(s['text'] for s in segments)
            elif kind=='xml-document':
                text=raw.decode('utf-8'); original={'xml':text,'dialogue_reconstructed':False}
            else: raise ValueError('unsupported archive member format')
            p=shard['shard_probability']*len(indices)/len(members)
            samples.append({'row_index':index,'member':member,'member_sha256':hashlib.sha256(raw).hexdigest(),
                            'raw_member_file':raw_name,
                            'inclusion_probability':p,'weight':1/p,'text':text,'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
                            'characters':len(text),'utf8_bytes':len(text.encode()),'metadata':metadata_of({}),
                            'original':original,'purpose':'population','semantic_review':'pending',
                            'unit_kind':'whole-table' if 'table' in kind else 'whole-document'})
    return {'shard':shard,'rows':len(members),'archive_entries':all_entries,'samples':samples,
            'archive_size':handle.size,'archive_sha256':None,
            'integrity_scope':'versioned URL, central directory, selected member CRC and SHA256; full archive not hashed',
            'sampling':'uniform whole members within sampled archive; rows are not independent cases'}
