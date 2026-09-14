"""Read-only source diagnostics; raw byte checks do not certify semantics."""
from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
import re
import unicodedata
import urllib.parse
import sys
from pathlib import Path


def fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            h.update(block)
    return h.hexdigest()


def prefix_check(full: Path, head: Path) -> dict:
    n = head.stat().st_size
    same = True
    compared = 0
    with full.open("rb") as a, head.open("rb") as b:
        while block := b.read(4 << 20):
            other = a.read(len(block))
            if other != block:
                same = False
                break
            compared += len(block)
    body_equal_bytes = 0
    with full.open("rb") as a, head.open("rb") as b:
        aa, bb = a.read(4096), b.read(4096)
        ai, bi = aa.find(b"<dblp>"), bb.find(b"<dblp>")
        if ai >= 0 and bi >= 0:
            a.seek(ai + 6); b.seek(bi + 6)
            while block := b.read(4 << 20):
                other = a.read(len(block))
                if block != other:
                    body_equal_bytes += next((i for i, (x, y) in enumerate(zip(block, other)) if x != y), min(len(block), len(other)))
                    break
                body_equal_bytes += len(block)
    def types(path):
        counts = Counter()
        pattern = re.compile(rb"<(article|inproceedings|incollection|phdthesis|mastersthesis|www|book|proceedings)\b")
        carry = b""
        with path.open("rb") as f:
            while block := f.read(4 << 20):
                data = carry + block
                cut = max(0, len(data) - 64)
                counts.update(m.group(1).decode() for m in pattern.finditer(data) if m.start() < cut)
                carry = data[cut:]
            counts.update(m.group(1).decode() for m in pattern.finditer(carry))
        return dict(counts)
    return {"full_path": str(full), "head_path": str(head), "full_bytes": full.stat().st_size,
            "head_bytes": n, "head_sha256": fingerprint(head),
            "head_is_exact_byte_prefix": same, "verified_identical_prefix_bytes": compared,
            "identical_body_prefix_bytes_after_root_open": body_equal_bytes,
            "full_record_open_tag_counts": types(full), "head_record_open_tag_counts": types(head),
            "head_byte_fraction": n / full.stat().st_size,
            "scope": "byte comparison and opening-tag counts, not XML validity or semantic coverage"}


def diagnostics(root: Path, config: dict) -> dict:
    result = {"schema": "mei-local-source-diagnostics-v1", "training_adoption_eligible": False}
    if config.get("prefix_check"):
        item = config["prefix_check"]
        result["prefix_check"] = prefix_check(root / item["full"], root / item["head"])
    if config.get("dialogue_comparison"):
        result["dialogue_comparison"] = dialogue_comparison(root, config["dialogue_comparison"])
    if config.get("schema_inventory"):
        result["schema_inventory"] = schema_inventory(root, config["schema_inventory"])
    if config.get("parquet_metadata_inventory"):
        result["parquet_metadata_inventory"] = parquet_metadata_inventory(root, config["parquet_metadata_inventory"])
    if config.get("schema_parameter_bridge"):
        result["schema_parameter_bridge"] = schema_parameter_bridge(root, config["schema_parameter_bridge"])
    if config.get("toolace_inventory"):
        result["toolace_inventory"] = toolace_inventory(root, config["toolace_inventory"])
    if config.get('toolace_call_preflight'):
        result['toolace_call_preflight'] = toolace_call_preflight(root, config['toolace_call_preflight'])
    if config.get("survey_token_capacity"):
        result["survey_token_capacity"] = survey_token_capacity(root, config["survey_token_capacity"])
    if config.get('literary_overlap'):
        result['literary_overlap']=literary_overlap(root,config['literary_overlap'])
    if config.get('table_source_inventory'):
        result['table_source_inventory']=table_source_inventory(root,config['table_source_inventory'])
    if config.get('subtitle_inventory'):
        result['subtitle_inventory']=subtitle_inventory(root,config['subtitle_inventory'])
    result["receipts"] = []
    for rel in config.get("receipts", []):
        path = root / rel
        row = json.loads(path.read_text())
        result["receipts"].append({"path": str(path), "sha256": fingerprint(path),
            "facts": {k: row[k] for k in ("status", "reviewer", "reviewed_at", "counts", "documents",
                      "tokens", "training_adoption_eligible", "ordering", "dedup", "exclusion") if k in row}})
    return result


def literary_overlap(root:Path, config:dict)->dict:
    rows=[]; sets=[]
    for rel in config['paths']:
        path=root/rel; text=path.read_text()
        start=re.search(r'^\*\*\* START OF .*?\*\*\*\s*$',text,re.M)
        end=re.search(r'^\*\*\* END OF .*?\*\*\*\s*$',text,re.M)
        if not start or not end or start.end()>=end.start():raise ValueError('Gutenberg body boundaries missing')
        body=text[start.end():end.start()];normalized=re.sub(r'[^\w]','',body)
        shingles={hashlib.sha256(normalized[i:i+80].encode()).hexdigest() for i in range(0,max(0,len(normalized)-79))}
        sets.append(shingles);rows.append({'path':rel,'sha256':fingerprint(path),'body_start_character':start.end(),'body_end_character':end.start(),
                                         'body_characters':len(body),'shingle_count':len(shingles)})
    pairs=[]
    for i in range(len(rows)):
        for j in range(i+1,len(rows)):
            common=len(sets[i]&sets[j]);small=min(len(sets[i]),len(sets[j]))
            pairs.append({'left':rows[i]['path'],'right':rows[j]['path'],'shared_shingles':common,'smaller_set_overlap':common/small if small else None})
    return {'files':rows,'overlap':pairs,'normalization_for_overlap_only':'retain Unicode word characters; 80-char shingles, stride1',
            'limitations':['edition/OCR and shingle alignment can miss overlap; zero is not isolation','subworks/editions need bibliographic grouping beyond lexical hashes','original text unchanged; no independent capacity assigned']}


def subtitle_inventory(root:Path,config:dict)->dict:
    import xml.etree.ElementTree as ET
    directory=root/config['survey'];manifest=json.loads((directory/'survey.json').read_text());sources={}
    for rel,expected in manifest['artifacts'].items():
        if not Path(rel).name.startswith('sample-'):continue
        path=directory/rel
        if fingerprint(path)!=expected:raise ValueError('subtitle sample changed')
        sample=json.loads(path.read_text());sid=Path(rel).parts[0]
        item=sources.setdefault(sid,{'sample_documents':0,'frame_documents':0,'frame_directory_groups':Counter(),'sample_metadata':{},'sample_directory_groups':Counter(),'sample_cjk_characters':0,'sample_characters':0})
        entries=[r for r in sample['archive_entries'] if r['path'].endswith('.xml')]
        item['frame_documents']+=len(entries)
        for entry in entries:item['frame_directory_groups'][str(Path(entry['path']).parent)]+=1
        for row in sample['samples']:
            item['sample_documents']+=1;item['sample_characters']+=len(row['text']);item['sample_cjk_characters']+=len(re.findall(r'[\u3400-\u9fff]',row['text']))
            item['sample_directory_groups'][str(Path(row['member']['path']).parent)]+=1
            metadata={}
            for raw in row['original'].get('metadata_xml',[]):
                tree=ET.fromstring(raw)
                for key in ('source/year','source/original','source/genre','subtitle/machine_translated','subtitle/language','subtitle/rating'):
                    value=tree.findtext(key)
                    if value is not None:metadata[key]=value
            for key in ('source/year','source/original','source/genre','subtitle/machine_translated','subtitle/language','subtitle/rating'):
                item['sample_metadata'].setdefault(key,Counter())[metadata.get(key,'unknown')]+=1
    for item in sources.values():
        groups=item.pop('frame_directory_groups');sample_groups=item.pop('sample_directory_groups')
        item['frame_unique_directory_groups']=len(groups)
        item['frame_extra_files_sharing_directory']=sum(n-1 for n in groups.values())
        item['largest_directory_groups']=dict(groups.most_common(20))
        item['sample_extra_files_sharing_directory']=sum(n-1 for n in sample_groups.values())
    return {'survey':str(directory),'manifest_sha256':fingerprint(directory/'survey.json'),'sources':sources,
            'limitations':['upstream metadata labels are not verified authorship or translation quality',
                          'directory grouping only, never a count of duplicate works/versions; flat directories such as TED have no work-group meaning',
                          'observed sample metadata proportions are descriptive; no net-yield estimate or admission',
                          'CJK character counts do not distinguish Mandarin, other Chinese varieties or Japanese']}


def table_source_inventory(root:Path, config:dict)->dict:
    directory=root/config['survey'];manifest=json.loads((directory/'survey.json').read_text())
    projects=Counter();headers=Counter();licenses=Counter();csv_urls=Counter();count=0;total_cells=null_cells=numeric_cells=0
    for rel,expected in manifest['artifacts'].items():
        if not Path(rel).name.startswith('sample-'):continue
        path=directory/rel
        if fingerprint(path)!=expected:raise ValueError('table sample changed')
        for sample in json.loads(path.read_text())['samples']:
            original=sample.get('original',{});count+=1
            meta=json.loads(original.get('source_metadata',{}).get('gittables','{}'))
            url=meta.get('csv_url',''); parts=urllib.parse.urlsplit(url).path.strip('/').split('/')
            project='/'.join(parts[:2]) if urllib.parse.urlsplit(url).hostname=='github.com' and len(parts)>=2 else 'unknown'
            projects[project]+=1;csv_urls[url or 'unknown']+=1;licenses[str(meta.get('license') or 'unknown')]+=1
            headers[json.dumps(original.get('headers'),ensure_ascii=False)]+=1
            for row in original.get('rows',[]):
                for value in (row.values() if isinstance(row,dict) else row):
                    total_cells+=1;null_cells+=int(value is None);numeric_cells+=int(type(value) in (int,float))
    return {'survey':str(directory),'manifest_sha256':fingerprint(directory/'survey.json'),'observed_tables':count,
            'project_counts':dict(projects.most_common()),'header_signatures':dict(headers.most_common(30)),
            'licenses':dict(licenses),'repeated_csv_urls':sum(v-1 for k,v in csv_urls.items() if k!='unknown' and v>1),
            'observed_cells':total_cells,'null_cells':null_cells,'numeric_cells':numeric_cells,
            'limitations':['unweighted observed sample diagnostics; not population ratios or independent net tokens','repository license metadata does not establish every cell origin','same project/header suggests grouping, not semantic equivalence']}


def map_schema_types(schema):
    from copy import deepcopy
    result=deepcopy(schema)
    if not isinstance(result,dict):return result
    aliases={'dict':'object','float':'number','int':'integer','bool':'boolean','list':'array'}
    t=result.get('type')
    if isinstance(t,str):result['type']=aliases.get(t,t)
    if isinstance(result.get('properties'),dict):result['properties']={k:map_schema_types(v) for k,v in result['properties'].items()}
    if isinstance(result.get('items'),dict):result['items']=map_schema_types(result['items'])
    return result


def parse_literal_tool_calls(text):
    """Restricted syntax reader; never execute source calls or nested expressions."""
    import ast
    tree=ast.parse(text,mode='eval').body
    if not isinstance(tree,ast.List):raise ValueError('not a call list')
    def literal(node):
        if isinstance(node,ast.Dict):
            keys=[literal(k) for k in node.keys]
            if any(not isinstance(k,str) for k in keys) or len(set(keys))!=len(keys):raise ValueError('nonstring or duplicate object keys')
            return dict(zip(keys,(literal(v) for v in node.values)))
        if isinstance(node,ast.List):return [literal(v) for v in node.elts]
        if isinstance(node,ast.Constant) and (node.value is None or type(node.value) in (str,int,float,bool)):return node.value
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)) and isinstance(node.operand,ast.Constant) and type(node.operand.value) in (int,float):
            return -node.operand.value if isinstance(node.op,ast.USub) else node.operand.value
        if isinstance(node,ast.Name) and node.id in ('true','false','null'):return {'true':True,'false':False,'null':None}[node.id]
        raise ValueError('nonliteral argument')
    calls=[]
    for item in tree.elts:
        if not isinstance(item,ast.Call) or not isinstance(item.func,ast.Name) or item.args:raise ValueError('non-simple named call')
        keys=[k.arg for k in item.keywords]
        if None in keys or len(set(keys))!=len(keys):raise ValueError('spread or duplicate keyword')
        calls.append({'name':item.func.id,'arguments':dict(zip(keys,(literal(k.value) for k in item.keywords)))})
    return calls


def toolace_call_preflight(root:Path, config:dict)->dict:
    path=root/config['path']
    if fingerprint(path)!=config['sha256']:raise ValueError('ToolACE original hash changed')
    runtime=root/'src/platform/_shared/runtime';sys.path.insert(0,str(runtime))
    try:from byte_grammar import parse_call_text
    finally:sys.path.pop(0)
    counts=Counter();examples={}; outcomes=[]
    for index,row in enumerate(json.loads(path.read_text())):
        try:
            start=row['system'].index('[{') if '[{' in row['system'] else row['system'].index('[\n')
            original_tools,_=json.JSONDecoder().raw_decode(row['system'],start)
            if not isinstance(original_tools,list) or any(not isinstance(t,dict) or 'name' not in t for t in original_tools):raise ValueError('tool list missing')
            tools=[{**t,'parameters':map_schema_types(t.get('parameters'))} for t in original_tools]
            if config.get('omit_null_outer_required'):
                for tool in tools:
                    if 'required' in tool and tool['required'] is None:del tool['required']
        except (ValueError,KeyError,TypeError):
            counts['records_with_unparsed_definitions']+=1;continue
        for turn_index,turn in enumerate(row['conversations']):
            if turn['from']!='assistant':continue
            value=turn['value'];status='non_call_text';detail=None
            if value.lstrip().startswith('['):
                try:calls=(parse_declared_tool_calls(value,[t['name'] for t in tools]) if config.get('declared_name_parser') else parse_literal_tool_calls(value))
                except (ValueError,SyntaxError,TypeError,RecursionError):status='call_syntax_adapter_gap'
                else:
                    if len(calls)>1:status='multiple_calls_need_dependency_review'
                    else:
                        try:
                            wire=json.dumps(calls,ensure_ascii=False,separators=(',',':'),allow_nan=False)
                            parsed=parse_call_text(wire,tools)
                            status='wire_schema_pass_semantics_pending' if parsed['ok'] else 'wire_schema_rejected'
                            detail=parsed.get('error')
                        except (ValueError,UnicodeError):status='wire_value_not_representable'
            counts[status]+=1
            item={'record_index':index,'turn_index':turn_index,'status':status,'detail':detail}
            outcomes.append(item)
            if len(examples.setdefault(status,[]))<8:examples[status].append(item)
    return {'path':str(path),'sha256':fingerprint(path),'counts':dict(counts),'examples':examples,'outcomes':outcomes,
            'runtime_source_bytes':{p.name:p.read_text() for p in runtime.glob('*.py')},
            'training_adoption_eligible':False,'executable_gold_count':None,
            'adapter_options':{k:bool(config.get(k)) for k in ('omit_null_outer_required','declared_name_parser')},
            'limitations':['single-call syntax/schema diagnostic with full declared toolset; no tool name remapping or dropped constraints',
                'multiple calls are not arbitrarily serialized; free text is not labeled as a missing-slot refusal',
                'no tools executed, no semantic gold approval, no 2048-budget or retrieval quality claim',
                'type aliases mapped at schema positions; optional removal only of null outer required metadata, parameter constraints preserved']}


def parse_declared_tool_calls(text,names):
    """Read declared names containing spaces without editing quoted argument text."""
    value=text.strip()
    if not value.startswith('[') or not value.endswith(']'):raise ValueError('call list required')
    body=value[1:-1];position=0;calls=[]
    while position<len(body):
        while position<len(body) and body[position].isspace():position+=1
        if position==len(body):break
        hits=[name for name in names if isinstance(name,str) and body.startswith(name,position) and body[position+len(name):].lstrip().startswith('(')]
        if not hits:raise ValueError('undeclared function name')
        name=max(hits,key=len);start=position+len(name)
        while body[start].isspace():start+=1
        depth=0;quote=None;escaped=False;end=None
        for i in range(start,len(body)):
            char=body[i]
            if quote:
                if escaped:escaped=False
                elif char=='\\':escaped=True
                elif char==quote:quote=None
                continue
            if char in ('"',"'"):quote=char
            elif char=='(':depth+=1
            elif char==')':
                depth-=1
                if depth==0:end=i+1;break
        if end is None:raise ValueError('unclosed function call')
        parsed=parse_literal_tool_calls('[f'+body[start:end]+']')[0]
        parsed['name']=name;calls.append(parsed);position=end
        while position<len(body) and body[position].isspace():position+=1
        if position<len(body):
            if body[position]!=',':raise ValueError('unexpected call suffix')
            position+=1
            if not body[position:].strip():raise ValueError('trailing call delimiter')
    return calls


def toolace_inventory(root: Path, config: dict) -> dict:
    path=root/config['path']
    if fingerprint(path)!=config['sha256']: raise ValueError('ToolACE original hash changed')
    rows=json.loads(path.read_text()); counts=Counter(); names=Counter(); schemas=Counter(); duplicate=Counter(); examples=[]
    runtime=root/'src/platform/_shared/runtime'
    sys.path.insert(0,str(runtime))
    try: import schema_subset
    finally: sys.path.pop(0)
    schema_results={}; schema_failures=Counter()
    for index,row in enumerate(rows):
        users=[t['value'] for t in row['conversations'] if t['from']=='user']
        counts['records']+=1; counts['user_turns']+=len(users)
        counts['records_with_cjk_user_text']+=int(any(re.search(r'[\u3400-\u9fff]',t) for t in users))
        counts['records_with_cjk_any_text']+=int(bool(re.search(r'[\u3400-\u9fff]',json.dumps(row,ensure_ascii=False))))
        duplicate[hashlib.sha256(json.dumps(row,ensure_ascii=False,sort_keys=True).encode()).hexdigest()]+=1
        try:
            start=row['system'].index('[{') if '[{' in row['system'] else row['system'].index('[\n')
            tools,_=json.JSONDecoder().raw_decode(row['system'],start)
            if not isinstance(tools,list) or any(not isinstance(t,dict) or 'name' not in t for t in tools): raise ValueError('not tool definitions')
            counts['records_with_parsed_tools']+=1
            for tool in tools:
                names[str(tool['name'])]+=1
                key=hashlib.sha256(json.dumps(tool,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                schemas[key]+=1
                if key not in schema_results:
                    try: schema_subset.validate_parameter_schema(map_schema_types(tool.get('parameters')))
                    except schema_subset.UnsupportedSchemaError as exc:
                        schema_results[key]=False;schema_failures[exc.keyword]+=1
                    else:schema_results[key]=True
        except (ValueError,KeyError,json.JSONDecodeError) as exc:
            counts['tool_description_parse_failures']+=1
            if len(examples)<12:examples.append({'index':index,'reason':str(exc)})
    return {'path':str(path),'sha256':fingerprint(path),'counts':dict(counts),
            'distinct_declared_tool_names':len(names),'distinct_exact_tool_definitions':len(schemas),
            'most_repeated_names':names.most_common(20),'exact_duplicate_record_excess':sum(v-1 for v in duplicate.values()),
            'parse_failure_examples':examples,'audited_net_tokens':None,
            'parameter_schema_preflight':{'unique_definitions':len(schema_results),'accepted_after_type_alias_mapping':sum(schema_results.values()),
                                          'unsupported_keywords':dict(schema_failures),'mapping':'dict/object,float/number,int/integer,bool/boolean,list/array; schema positions only',
                                          'scope':'parameter-schema compatibility only; no name/call conversion, execution or gold approval'},
            'runtime_source_bytes':{n:(runtime/n).read_text() for n in ('schema_subset.py','canonical_json.py')},
            'limitations':['CJK character census is not a language classifier','same name may have different schemas; schema hashes do not prove independent APIs',
                           'upstream answers not executed; no SFT gold generated']}


def survey_token_capacity(root: Path, config: dict) -> dict:
    from source_manager import load_tokenizer
    tokenizer=load_tokenizer(); results=[]
    for rel in config['surveys']:
        directory=root/rel; manifest=json.loads((directory/'survey.json').read_text()); per_source={}
        for filename,expected in manifest['artifacts'].items():
            if not Path(filename).name.startswith('sample-'):continue
            path=directory/filename
            if fingerprint(path)!=expected:raise ValueError('survey sample hash changed')
            sample=json.loads(path.read_text()); sid=Path(filename).parts[0]
            item=per_source.setdefault(sid,{'source_id':sid,'survey':rel,'sample_records':0,'sample_tokens':0,'estimated_sampling_frame_tokens':0.0,'by_wave':{},'over_2048_records':0})
            for row in sample['samples']:
                if row.get('purpose')!='population':continue
                length=len(tokenizer.encode(row['text']))
                item['sample_records']+=1;item['sample_tokens']+=length;item['over_2048_records']+=int(length>2048)
                item['estimated_sampling_frame_tokens']+=length/row['inclusion_probability']
                w=str(sample['shard']['wave']);shard=sample['shard']
                marginal=row['inclusion_probability']*shard['wave_probability']/shard['shard_probability']
                item['by_wave'][w]=item['by_wave'].get(w,0)+length/marginal
        results.extend(per_source.values())
    return {'tokenizer_id':tokenizer.tokenizer_id,'tokenizer_model_sha256':tokenizer.model_sha256,'sources':results,
            'audited_net_tokens':None,'is_vocab_adaptation_audit':False,
            'limitations':['baseline token measurement of frozen survey views; no new tokenizer',
                           'estimates refer only to each sampling frame and its completed samples; inspect parent profile completeness',
                           'dialogue/schema JSON wrappers included; not final CPT serialization or independent net yield',
                           'wave estimates are diagnostics, not confidence intervals; sources/versions cannot be added without overlap review']}


def schema_parameter_bridge(root: Path, config: dict) -> dict:
    """Test direct parameter embedding, not the power of an external validator."""
    runtime = root / "src/platform/_shared/runtime"
    sys.path.insert(0, str(runtime))
    try:
        import schema_subset
    finally:
        sys.path.pop(0)
    source = root / config["survey"]
    manifest = json.loads((source / "survey.json").read_text())
    rows = []
    for rel, expected in sorted(manifest["artifacts"].items()):
        if not rel.startswith("schema-tests/sample-"): continue
        path = source / rel
        if fingerprint(path) != expected: raise ValueError("schema sample hash mismatch")
        for sample in json.loads(path.read_text())["samples"]:
            unit = sample["original"]
            original = unit["input"]["schema"]
            entry = {"sample": str(path), "sample_sha256": expected, "unit_id": unit["unit_id"],
                     "schema": original, "training_adoption_eligible": False}
            wrapped = {"type": "object", "properties": {"value": original},
                       "required": ["value"], "additionalProperties": False}
            try:
                schema_subset.validate_parameter_schema(wrapped)
            except schema_subset.UnsupportedSchemaError as exc:
                entry.update({"direct_parameter_embedding": "unsupported", "reason": str(exc)})
            else:
                cases = []
                for case, target in zip(unit["input"]["cases"], unit["annotations"]["expected_valid"], strict=True):
                    issues = schema_subset.validate_arguments({"value": case["data"]}, wrapped)
                    cases.append({"data": case["data"], "upstream_expected_valid": target,
                                  "runtime_valid": not issues, "matches": (not issues) == target,
                                  "issues": [i.as_dict() for i in issues]})
                entry.update({"direct_parameter_embedding": "supported", "executed_cases": cases})
            rows.append(entry)
    return {"runtime_schema_id": schema_subset.SCHEMA_SUBSET_ID,
            "runtime_source_bytes": {name: (runtime/name).read_text() for name in ("schema_subset.py", "canonical_json.py")},
            "source_manifest_sha256": fingerprint(source / "survey.json"), "groups": rows,
            "direct_parameter_supported_groups": sum(r["direct_parameter_embedding"] == "supported" for r in rows),
            "executed_cases": sum(len(r.get("executed_cases", [])) for r in rows),
            "matching_cases": sum(c["matches"] for r in rows for c in r.get("executed_cases", [])),
            "limitations": ["direct tool-argument embedding only; no full JSON Schema implementation claimed",
                            "a separately implemented validator tool can accept schema/data references and support more rules",
                            "public conformance tests are not a newly independent Mei evaluation split",
                            "no CPT/SFT/eval release created"]}


def parquet_metadata_inventory(root: Path, config: dict) -> dict:
    import pyarrow.parquet as pq
    paths = sorted({p for pattern in config["globs"] for p in root.glob(pattern) if p.is_file()})
    files = []
    totals = {k: Counter() for k in ("dump", "host", "language", "project", "extension")}
    for path in paths:
        pf = pq.ParquetFile(path)
        columns = [c for c in ("dump", "url", "language", "repo_name", "repository_name", "path") if c in pf.schema_arrow.names]
        counts = {k: Counter() for k in totals}
        rows = 0
        for batch in pf.iter_batches(columns=columns, batch_size=8192, use_threads=False):
            for row in batch.to_pylist():
                try:
                    host = urllib.parse.urlsplit(str(row.get("url") or "")).hostname or "unknown"
                except ValueError:
                    host = "unknown"
                values = {"dump": row.get("dump") or "unknown", "host": host,
                          "language": row.get("language") or "unknown",
                          "project": row.get("repo_name") or row.get("repository_name") or "unknown",
                          "extension": Path(str(row.get("path") or "")).suffix.lower() or "unknown"}
                for key, value in values.items(): counts[key][str(value)] += 1
                rows += 1
        for key in totals: totals[key].update(counts[key])
        files.append({"path": str(path), "sha256": fingerprint(path), "columns": columns,
                      "rows": rows, "footer_rows": pf.metadata.num_rows,
                      "dump_counts": dict(counts["dump"]), "language_counts": dict(counts["language"])})
    return {"files": files, "total_records": sum(f["rows"] for f in files),
            "census_counts": {k: dict(v) for k,v in totals.items()},
            "scope": "all rows of located local files, projected metadata columns only",
            "audited_net_tokens": None, "semantic_review_passed": False}


def schema_inventory(root: Path, config: dict) -> dict:
    directory = root / config["directory"]
    frame_path = root / config["catalog_frame"]
    frame = json.loads(frame_path.read_text())
    prefix = config["upstream_prefix"]
    remote = {r["path"][len(prefix):]: r for r in frame["files"] if r["path"].startswith(prefix)}
    files, ids, titles, drafts, families = [], Counter(), Counter(), Counter(), Counter()
    for path in sorted(directory.glob("*.json")):
        raw = path.read_bytes()
        item = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        upstream = remote.get(path.name)
        item["matches_pinned_upstream"] = blob == upstream["git_blob_sha"] if upstream else None
        try:
            value = json.loads(raw)
            if not isinstance(value, (dict, bool)):
                raise ValueError("schema root is not object/boolean")
            item["parse_status"] = "parsed"
            if isinstance(value, dict):
                item["id"] = value.get("$id", value.get("id"))
                item["title"] = value.get("title")
                item["dialect"] = value.get("$schema")
                if item["id"]: ids[str(item["id"])] += 1
                if item["title"]: titles[str(item["title"])] += 1
                drafts[str(item["dialect"] or "unknown")] += 1
        except Exception as exc:
            item["parse_status"] = f"{type(exc).__name__}: {exc}"
        family = re.sub(r"[-_.]?v?\d+(?:[._-]\d+)*(?:[-_.].*)?$", "", path.stem)
        families[family] += 1
        files.append(item)
    local_names = {Path(r["path"]).name for r in files}
    return {"catalog_frame_sha256": fingerprint(frame_path), "upstream_revision": frame["revision"],
            "local_files": len(files), "upstream_schema_files": len(remote),
            "same_bytes_as_upstream": sum(r["matches_pinned_upstream"] is True for r in files),
            "different_bytes_from_upstream": sum(r["matches_pinned_upstream"] is False for r in files),
            "local_only_names": sorted(local_names - remote.keys()),
            "upstream_only_names": sorted(remote.keys() - local_names),
            "repeated_declared_ids": {k:v for k,v in ids.items() if v > 1},
            "repeated_titles": {k:v for k,v in titles.items() if v > 1},
            "dialects": dict(drafts), "heuristic_filename_families": dict(families.most_common(20)),
            "files": files, "audited_net_tokens": None,
            "limitations": ["matching a file does not prove complete upstream population coverage",
                            "repeated IDs/titles and filename families suggest grouping, not semantic equivalence",
                            "parsed JSON is not a validated schema or gold verifier"]}


def dialogue_comparison(root: Path, config: dict) -> dict:
    """Validate a known legacy serializer against original source lines.

    This explicitly named legacy normalization is checked against source lines
    and prepared outputs; no code from a dataset or source snapshot is executed.
    """
    snapshot = root / config["serializer_snapshot"]
    if fingerprint(snapshot) != config["serializer_sha256"]:
        raise ValueError("legacy serializer hash changed")
    def clean_turn(text):
        text = unicodedata.normalize("NFKC", text).strip()
        text = re.sub(r"(?<=[\u3400-\u9fff])\s+|\s+(?=[\u3400-\u9fff])", "", text)
        text = re.sub(r"\s+([,.!?;:，。！？；：、])", r"\1", text)
        text = re.sub(r"(?<=\d)\s+(?=\d)", "", text)
        return re.sub(r"\s+", " ", text).strip()
    samples = {}
    input_hashes = {}
    for kind in ("raw", "filtered"):
        path = root / config[f"{kind}_sample"]
        input_hashes[str(path)] = fingerprint(path)
        samples[kind] = json.loads(path.read_text())["samples"]
    wanted = {r["original"]["source_line"]: r for r in samples["filtered"]}
    matched = []
    with gzip.open(root / config["raw_jsonl_gz"], "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if number in wanted:
                original = json.loads(line)
                expected = "\n".join(clean_turn(t) for t in original)
                row = wanted[number]
                matched.append({"source_line": number, "original": original, "prepared_text": row["text"],
                                "serializer_matches": expected == row["text"], "raw_turns": len(original),
                                "derived_turns": len(row["text"].split("\n"))})
            if len(matched) == len(wanted):
                break
    valid = len(matched) == len(wanted) and all(r["serializer_matches"] and r["raw_turns"] == r["derived_turns"] for r in matched)
    def describe(rows, kind):
        turns = [r["original"] if kind == "raw" else r["text"].split("\n") for r in rows]
        lengths = sorted(sum(len(t) for t in dialog) for dialog in turns)
        return {"records": len(rows), "turn_count_histogram": dict(Counter(map(len, turns))),
                "dialogue_characters_median": lengths[len(lengths)//2] if lengths else None,
                "dialogue_characters_mean": sum(lengths)/len(lengths) if lengths else None,
                "records_with_question_mark": sum(any("?" in t or "？" in t for t in d) for d in turns)}
    return {"serializer_snapshot": str(snapshot), "serializer_sha256": fingerprint(snapshot),
            "input_sample_hashes": input_hashes, "matched_records": len(matched),
            "historical_serialization_verified_on_sample": valid,
            "raw_statistics": describe(samples["raw"], "raw"),
            "filtered_statistics": describe(samples["filtered"], "filtered") if valid else None,
            "paired_evidence": matched,
            "limitations": ["two independent record samples, not a census of filtering effects",
                            "raw text has pre-cleaning spaces; character lengths are not normalized-equivalent",
                            "question marks are lexical observations, not semantic intent labels"]}
