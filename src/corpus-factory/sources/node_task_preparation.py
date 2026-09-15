"""Offline node-task semantic candidates and QAT reference selection, never admission."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

from profiling import resolve_path

ROOT = Path(__file__).resolve().parents[3]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f: json.dump(obj, f, ensure_ascii=False, indent=2)

def validate_plan(plan, catalog):
    tasks = plan['tasks']
    assert 0 < len(tasks) <= 6, 'task_count'
    ids = [t['id'] for t in tasks]
    assert len(set(ids)) == len(ids), 'duplicate_id'
    by = {t['id']: t for t in tasks}
    for t in tasks:
        assert t['tool'] in catalog, 'unknown_tool'
        assert t['gate'] in ('all_succeeded', 'all_passed'), 'unknown_gate'
        assert len(t['after']) == len(set(t['after'])), 'duplicate_dependency'
        assert all(x in by and x != t['id'] for x in t['after']), 'unknown_dependency'
    visiting, visited = set(), set()
    def visit(i):
        assert i not in visiting, 'cycle'
        if i in visited: return
        visiting.add(i)
        for j in by[i]['after']: visit(j)
        visiting.remove(i); visited.add(i)
    for i in ids: visit(i)
    return True

def ready(plan, states):
    return sorted(t['id'] for t in plan['tasks'] if states.get(t['id'], 'pending') == 'pending'
        and all(states.get(d) in ({'passed'} if t['gate'] == 'all_passed' else {'passed', 'issues'})
                for d in t['after']))

def evaluate_state(c):
    if c.get('reply_revision') is not None and c['reply_revision'] != c['revision']:
        return {'decision': 'ignore_stale_reply', 'ready': []}
    r = ready(c['plan'], c['states'])
    if r: return {'decision': 'execute', 'ready': r}
    states = [c['states'].get(t['id'], 'pending') for t in c['plan']['tasks']]
    if all(s == 'passed' for s in states):
        return {'decision': 'request_approval' if c.get('upload_requested') else 'complete', 'ready': []}
    if c.get('requires_remote') and not c.get('online', True):
        return {'decision': 'queue_assistance', 'ready': []}
    return {'decision': 'incomplete', 'ready': []}

def run(config, out):
    out = Path(out)
    if out.exists(): raise ValueError('use a new output ID')
    if shutil.disk_usage(ROOT).free < 100 * 1024**3: raise ValueError('disk reserve')
    cpt = resolve_path(ROOT / config['cpt_release'])
    if sha(cpt) != config['cpt_sha256']: raise ValueError('CPT binding changed')
    release = json.loads(cpt.read_text())
    out.mkdir(parents=True)
    write(out/'config.json', config)
    fixtures = json.loads((resolve_path(ROOT/config['fixtures'])).read_text())
    demo = ROOT/'src/demos/data-check'
    # Execute the existing JS validators and tools, not a Python reimplementation.
    js = '''import fs from 'node:fs';
import {TOOLS} from './src/demos/data-check/catalog.mjs';
import {validateCall,runCheck} from './src/demos/data-check/checks.mjs';
const cases=JSON.parse(fs.readFileSync(0,'utf8')); const result=[];
for(const c of cases){ for(const t of c.plan.tasks){
validateCall(t.tool,t.args); const r=runCheck(c.sheet,t.tool,t.args);
if(r.issue_count!==c.expected_issues[t.id]) throw Error(c.id+': '+t.id+' issue mismatch');
result.push({case_id:c.id,task_id:t.id,tool:t.tool,issue_count:r.issue_count,checked:r.checked});}}
process.stdout.write(JSON.stringify({catalog:TOOLS,results:result}));'''
    proc = subprocess.run(['node','--input-type=module','-e',js], cwd=ROOT,
        input=json.dumps(fixtures['plans']), text=True, capture_output=True, check=True)
    checks=json.loads(proc.stdout); catalog={x['name']: x for x in checks['catalog']}
    for c in fixtures['plans']: validate_plan(c['plan'],catalog)
    for c in fixtures['states']:
        validate_plan(c['plan'],catalog)
        actual=evaluate_state(c)
        if actual != c['expected']: raise ValueError(f"state gold mismatch {c['id']}: {actual}")
    # Graph equivalence: independent task order is irrelevant; dependency violations are not.
    parallel=next(c['plan'] for c in fixtures['plans'] if c['shape']=='parallel')
    reversed_plan=copy.deepcopy(parallel); reversed_plan['tasks'].reverse()
    assert ready(parallel,{})==ready(reversed_plan,{})
    malformed=copy.deepcopy(parallel)
    malformed['tasks'][0]['after']=[malformed['tasks'][1]['id']]
    malformed['tasks'][1]['after']=[malformed['tasks'][0]['id']]
    try: validate_plan(malformed,catalog)
    except AssertionError: pass
    else: raise ValueError('cycle not rejected')
    for split in ('train','dev'):
        rows=[c for c in fixtures['plans'] if c['split']==split]
        # Semantic records only: the plan target is not yet a runtime LM envelope.
        write(out/f'sft-candidate/{split}-planning.json', {
            'status':'semantic_candidate_not_compiled','records':rows,
            'loss_mask_status':'pending_plan_editor_serializer',
            'independent_holdout':False})
    write(out/'eval-dev/state-cases.json', {'status':'development_fixtures',
        'records':fixtures['states'],'independent_holdout':False})
    write(out/'eval-dev/semantic-edge-cases.json', {'status':'pending_semantic_review_and_executable_gold',
        'records':fixtures['semantic_edge_cases'],'independent_holdout':False})
    write(out/'eval-dev/verification.json', {'status':'passed',
        'real_js_tool_results':checks['results'], 'state_cases':len(fixtures['states']),
        'cycle_rejected':True,'parallel_order_invariance':True,
        'scope':'tool semantics and reference task-state oracle; not production runtime/model validation'})
    write(out/'tool-catalog.json', checks['catalog'])
    # Cheap deterministic record references; no token re-encoding and no new source data.
    db=sqlite3.connect(f'file:{(cpt.parent/"record-index.sqlite").resolve()}?mode=ro',uri=True)
    db.row_factory=sqlite3.Row
    qat=[]
    for f in release['token_files']:
        rows=db.execute('''select r.id,r.source,r.domain,r.language,r.group_key,r.split,
            r.text_sha,o.phase,o.output_offset,o.token_length
            from records r indexed by records_source_phase_order
            join selected_offsets o on r.id=o.record_id where r.source=? and r.split='train'
            order by r.phase,r.priority,r.id limit ?''',
            (f['source'],config['qat_records_per_source'])).fetchall()
        for row in rows:
            d=dict(row)
            if d['split']!='train': raise ValueError('QAT reserve includes nontrain')
            d.update(token_file=str(cpt.parent/f['path']), token_file_sha256=f['sha256'])
            qat.append(d)
    db.close()
    by_domain={}
    for row in qat: by_domain[row['domain']]=by_domain.get(row['domain'],0)+row['token_length']
    write(out/'qat-reserve/manifest.json', {'status':'references_pending_source_quality_and_final_tokenizer',
        'cpt_release_sha256':config['cpt_sha256'], 'records':qat,
        'unique_records':len(qat),'tokens':sum(by_domain.values()),'domain_tokens':by_domain,
        'selection':'first K selected records ordered by original phase/seeded priority/ID per source; engineering probe, not representative training mixture',
        'training_authorized':False, 'token_bytes_rehashed_this_run':False,
        'recount_required_if_tokenizer_changes':True})
    legacy=resolve_path(ROOT/config['legacy_sft_manifest'])
    old=json.loads(legacy.read_text())
    write(out/'legacy-review.json', {'manifest':str(legacy),'sha256':sha(legacy),
        'base_binding':old.get('base_binding'),'families':old.get('families'),
        'decision':'pending_per_binding_review_not_automatically_reused',
        'required_checks':['execute/empty-call counts','gold runtime replay','loss mask',
        'provenance','family isolation','new tokenizer binding'],
        'confidence':'rebuild_from_new_model_real_outcomes',
        'historical_eval':'retain_regression_scores; inspected holdout is not fresh blind test'})
    bindings={
        'planning':'semantic train/dev seeds; plan-editing tools and serializer pending',
        'retrieval':'catalog available; independent queries/hard negatives pending; learned batches generated later',
        'full_call':'existing five check tools verified by JS; compiled supervised rows pending',
        'agent':'DAG/state development fixtures; live runtime and trajectory compilation pending',
        'mw_disposition':'retain current contract; proposed tool-based cooperation is an ablation only',
        'confidence':'recipe only; requires final model real correct/incorrect calls',
        'narration':'verified result evidence available; supervision pending',
        'qat':'train-only source references; quality admission and balanced replay recipe pending',
        'eval':'development graph/state oracle; no new locked test release'}
    write(out/'BINDINGS.json',bindings)
    evidence=[cpt,resolve_path(ROOT/config['fixtures']),legacy,Path(__file__),
        demo/'catalog.mjs',demo/'checks.mjs',ROOT/'src/corpus-factory/sources/materialize.py']
    write(out/'SOURCE-BINDING.json', [{'path':str(p),'sha256':sha(p)} for p in evidence])
    write(out/'implementation-snapshot.json', {str(p.relative_to(ROOT)):p.read_text()
        for p in evidence if p.suffix in ('.py','.mjs')})
    report={'status':'prepared_candidates_not_training_ready','planning_cases':len(fixtures['plans']),
        'train_planning_cases':sum(c['split']=='train' for c in fixtures['plans']),
        'dev_planning_cases':sum(c['split']=='dev' for c in fixtures['plans']),
        'state_cases':len(fixtures['states']),'real_js_checks':len(checks['results']),
        'qat_reference_records':len(qat),'qat_reference_tokens':sum(by_domain.values()),
        'semantic_review':'pending; author-written seeds, not public natural-language diversity evidence',
        'locked_test_cases':0,'model_training_started':False}
    write(out/'REPORT.json',report)
    write(out/'FILES.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    return report
