"""Lossless source-span CPT windows, not independent executable SFT examples."""
import bisect
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from profiling import resolve_path, ROOT, digest


def structural_breaks(text):
    """Prefer message boundaries, then JSON separators outside string literals."""
    messages, structure = [], []
    quoted = escaped = False
    for i, c in enumerate(text):
        if c == '\n':
            messages.append(i + 1)
        if quoted:
            if escaped: escaped = False
            elif c == '\\': escaped = True
            elif c == '"': quoted = False
        elif c == '"': quoted = True
        elif c in ',}]': structure.append(i + 1)
    return messages, structure


def fitting_end(text, start, end, budget, tok):
    """Return a fitting Unicode-character boundary; never decode/rewrite source bytes."""
    if hasattr(tok, 'sp'):
        fragment = text[start:end]
        encoded = tok.sp.encode(fragment, return_type='proto')
        if len(encoded.pieces) + 2 <= budget:
            return end
        # Native offsets refer to original UTF-8, including normalized whitespace.
        # Leave room for a different dummy-prefix tokenization at a split boundary.
        byte_end = encoded.pieces[max(0, budget - 10)].end
        if byte_end:
            end = start + len(fragment.encode('utf-8')[:byte_end].decode('utf-8'))
    if len(tok.encode_document(text[start:end])) <= budget:
        return end
    lo, hi = start, end
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if len(tok.encode_document(text[start:mid])) <= budget: lo = mid
        else: hi = mid
    if lo == start:
        raise ValueError('one_character_exceeds_budget')
    return lo


def units(text, budget, tok, breaks):
    messages, structure = breaks
    start = 0
    while start < len(text):
        # Bound each tokenizer probe; a long document is never re-encoded per window.
        end = fitting_end(text, start, min(len(text), start + budget * 6), budget, tok)
        if end < len(text):
            for boundaries in (messages, structure):
                pos = bisect.bisect_right(boundaries, end) - 1
                if pos >= 0 and boundaries[pos] > start:
                    end = boundaries[pos]; break
        yield start, end
        start = end


def source_context(text):
    """Exact character offsets for individual tool definitions and whole turns."""
    tools, turns = {}, []
    first_end = text.find('\n')
    if text.startswith('Tools: ['):
        decoder = json.JSONDecoder(); i = len('Tools: [')
        while i < first_end and text[i] != ']':
            obj, end = decoder.raw_decode(text, i)
            name = obj.get('function', obj).get('name')
            tools.setdefault(name, []).append((i, end))
            i = end
            if text[i:i+1] == ',': i += 1
    offset = first_end + 1
    for line in text[offset:].splitlines(keepends=True):
        turn = json.loads(line)
        turns.append((offset, offset + len(line), turn))
        offset += len(line)
    return tools, turns


def split_record(row, tok, limit=2048, overlap_budget=128, context_budget=256):
    if limit < 32 or overlap_budget < 0 or context_budget < 0 or overlap_budget + context_budget >= limit - 16:
        raise ValueError('invalid_window_budget')
    text = row['text']; breaks = structural_breaks(text)
    tools, turns = source_context(text)
    turn_starts = [a for a, _, _ in turns]
    budget = limit - overlap_budget - context_budget
    previous = None
    for number, (start, end) in enumerate(units(text, budget, tok, breaks)):
        overlap = None
        if previous and len(tok.encode_document(text[previous[0]:previous[1]])) <= overlap_budget:
            overlap = previous
        body_start = overlap[0] if overlap else start
        partial = start not in [0, *breaks[0]] or (end != len(text) and end not in breaks[0])
        selected, unresolved = [], []
        nearby = turns[max(0, bisect.bisect_right(turn_starts, start)-1):bisect.bisect_left(turn_starts,end)] if turns else []
        wanted = []
        for link in row.get('metadata',{}).get('result_links',[]):
            ra,rb,_ = turns[link['result_turn']]
            if ra < end and rb > start:
                ca,cb,_ = turns[link['turn']]
                if cb <= body_start:
                    wanted.append(('prior_call', (ca,cb)))
        for a,b,turn in nearby:
            for call in turn.get('tool_calls') or []:
                for span in tools.get(call['function']['name'], []):
                    if span[1] <= body_start: wanted.append(('tool_definition', span))
                # Context is extractive, not a claim that the last user supplied every argument.
                for ua,ub,u in reversed(turns[:bisect.bisect_left(turn_starts,a)]):
                    if u['role'] == 'user':
                        if ub <= body_start: wanted.append(('recent_user', (ua,ub)))
                        break
        prefix = ''
        seen = set()
        for kind, span in wanted:
            if span in seen: continue
            seen.add(span)
            trial = prefix + kind + ': ' + text[span[0]:span[1]] + '\n'
            if len(tok.encode_document(trial)) <= context_budget:
                prefix = trial; selected.append({'kind':kind,'span':list(span)})
            else:
                unresolved.append({'kind':kind,'span':list(span),'reason':'context_budget'})
        # Explicit prose distinguishes broken JSON/text continuation from a complete call envelope.
        header = 'Source continuation (partial message):\n' if partial else ''
        prefix = ('Earlier source context:\n' + prefix + 'Continuation:\n') if prefix else ''
        output = prefix + header + text[body_start:end]
        ids = tok.encode_document(output)
        while len(ids) > limit and selected:
            removed = selected.pop(); unresolved.append({**removed,'reason':'joint_budget'})
            prefix = ''.join(s['kind']+': '+text[s['span'][0]:s['span'][1]]+'\n' for s in selected)
            prefix = ('Earlier source context:\n'+prefix+'Continuation:\n') if prefix else ''
            output = prefix + header + text[body_start:end]
            ids = tok.encode_document(output)
        if len(ids) > limit and overlap:
            overlap = None; body_start = start; output = prefix + header + text[start:end]
            ids = tok.encode_document(output)
        if len(ids) > limit:
            raise ValueError('joint_window_budget_exceeded')
        yield {'text': output, 'source_id':row['source_id'], 'group_id':row['group_id'],
               'origin':row['origin'], 'split':row['split'], 'candidate_kind':'cpt_window',
               'parent_text_sha256':row['text_sha256'], 'window_index':number,
               'metadata': {'tokens':len(ids), 'primary_span':[start,end],
                   'overlap_span':list(overlap) if overlap else None, 'context_spans':selected,
                   'uncovered_context':unresolved, 'partial_message':partial,
                   'dependency_coverage':'not_certified', 'sft_eligible':False,
                   'semantic_review':'pending', 'executable_gold':False,
                   'source_uuid':row.get('metadata',{}).get('source_uuid'),
                   'parent_metadata_inheritance':'resolve parent_text_sha256 in frozen parent file',
                   'repeated_context_characters':sum(s['span'][1]-s['span'][0] for s in selected)+(start-body_start),
                   'repeated_context_tokens_standalone':len(tok.encode(prefix+text[body_start:start]))}}
        previous = (start,end)


def run(config, out):
    from source_manager import load_tokenizer, tokenizer_pointer
    source = resolve_path(ROOT/config['path']); parent = resolve_path(ROOT/config['parent_manifest'])
    if digest(source) != config['sha256'] or digest(parent) != config['parent_manifest_sha256']:
        raise ValueError('parent_hash_changed')
    tok = load_tokenizer(); pointer = tokenizer_pointer()
    if json.loads(parent.read_text())['tokenizer']['model_sha256'] != pointer['model_sha256']:
        raise ValueError('tokenizer_changed_recount_required')
    if shutil.disk_usage(ROOT).free < 100*1024**3: raise ValueError('disk_reserve')
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    (out/'implementation.py.snapshot').write_bytes(Path(__file__).read_bytes())
    counts = Counter()
    with source.open() as src, (out/'windows.jsonl').open('x') as dst:
        for i,line in enumerate(src):
            if config.get('max_records') and i >= config['max_records']: break
            row = json.loads(line); text = row['text']
            if hashlib.sha256(text.encode()).hexdigest()!=row['text_sha256']: raise ValueError('text_hash_changed')
            counts['source_records'] += 1
            counts['source_tokens'] += row['metadata']['tokens']
            counts['source_characters'] += len(text)
            end = 0
            for w in split_record(row,tok,config.get('limit',2048),config.get('overlap_budget',128),config.get('context_budget',256)):
                m=w['metadata'];a,b=m['primary_span']
                if a!=end or b<=a: raise ValueError('source_span_gap_or_overlap')
                end=b
                dst.write(json.dumps(w,ensure_ascii=False,separators=(',',':'))+'\n')
                counts['windows'] += 1;counts['window_exposure_tokens'] += m['tokens']
                counts['max_window_tokens']=max(counts['max_window_tokens'],m['tokens'])
                counts['primary_characters'] += b-a
                counts['partial_message_windows'] += int(m['partial_message'])
                counts['context_windows'] += bool(m['context_spans'])
                counts['uncovered_context_windows'] += bool(m['uncovered_context'])
                counts['overlap_windows'] += bool(m['overlap_span'])
                counts['repeated_context_characters'] += m['repeated_context_characters']
                counts['repeated_context_tokens_standalone'] += m['repeated_context_tokens_standalone']
            if end!=len(text): raise ValueError('source_tail_missing')
            if i%1000==0:
                if shutil.disk_usage(ROOT).free < 100*1024**3: raise ValueError('disk_reserve')
                dst.flush();(out/'progress.json').write_text(json.dumps(dict(counts),indent=2));print(json.dumps(dict(counts)),flush=True)
    if counts['primary_characters']!=counts['source_characters']:raise ValueError('coverage_mismatch')
    report={'schema':'mei-cpt-windows-v1','counts':dict(counts),'tokenizer':pointer,
            'parent':config['path'],'parent_sha256':config['sha256'],
            'file_sha256':digest(out/'windows.jsonl'),'release':False,
            'inventory_policy':'derived view; never add to parent independent capacity',
            'token_accounting':'source_tokens inherited; exposure includes repeated context, wrappers and BOS/EOS; standalone repeat count nonadditive',
            'span_unit':'Python Unicode character offsets, half-open',
            'limitations':['CPT continuation, not independent SFT gold',
                'context extraction is bounded; dependency completeness not certified',
                'oversized messages split at JSON separators then character boundaries; partial fragments labeled',
                'semantic and cross-split checks inherited as pending']}
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));return report
