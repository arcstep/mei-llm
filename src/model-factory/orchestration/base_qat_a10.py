"""Server-resident milestone CPT -> Base QAT -> Float CPT sequencing."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime,timezone

from common.paths import ROOT, ensure_formal_on_path
ensure_formal_on_path()
from common.evidence_stage import write_json, stage, sha256
from common.source_capture import json_digest,manifest
from training.qat.base_replay import bound,load,make_references

PIPELINE='mei-51m-base-qat-cuda-v1'


def read(path):
    p=Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def prepare(request):
    root=Path(request['campaign_dir']);root.mkdir(parents=True,exist_ok=True)
    output=root/'prepared.json'
    if output.exists():
        old=read(output)
        if old['request']!=request:raise ValueError('immutable preparation request differs')
        return old['config']
    snapshot=load(request['snapshot']);corpus=load(request['corpus'])
    if snapshot['tokens_seen']!=request['base_tokens']:raise ValueError('wrong Base exposure')
    for item in [snapshot['checkpoint'],snapshot['weights'],snapshot['tokenizer']]:
        if sha256(item['path'])!=item['sha256']:raise ValueError('Base bytes mismatch')
    side=Path(snapshot['checkpoint']['path']).with_suffix('.json')
    meta=read(side)
    if meta.get('sha256')!=snapshot['checkpoint']['sha256'] or meta.get('tokens_seen')!=snapshot['tokens_seen']:
        raise ValueError('snapshot/sidecar mismatch')
    consumed=meta['sampler']['source_tokens_drawn']
    if sum(consumed.values())!=snapshot['tokens_seen']:raise ValueError('source ledger mismatch')
    cfg=dict(request)
    cfg.update(action='base_qat',precision='bf16_amp',kernel_mode='compile_pointwise',
               seed=20260916,lr=0.00005,batch_size=1,grad_accum=4,
               first_selection_tokens=20000000,rungs=[5000000,10000000,20000000,30000000,50000000,75000000,100000000],
               gates={'max_aggregate_ce_delta':0.15,'max_source_ce_delta':0.40,'minimum_ce_improvement':0.002},
               disk_reserve_bytes=100*1024**3,
               ftc_deferred='user authorization 2026-09-16; task controls and SFT run locally',
               source_manifest_sha256=manifest(ROOT)['manifest_sha256'])
    if cfg['base_tokens'] < 2500000000:
        cfg['rungs']=[5000000,10000000,20000000,30000000]
        cfg['stop_on_first_qualified']=True
    else:
        cfg['stop_on_first_qualified']=False
    refs=make_references(corpus,consumed,cfg['rungs'][-1],cfg['seed'])
    write_json(root/'replay.json',refs)
    cfg['replay']=bound(root/'replay.json')
    cfg['fingerprint']=json_digest(cfg)
    write_json(output,{'request':request,'config':cfg})
    return cfg


def invoke(action,cfg):
    root=Path(cfg['campaign_dir'])
    out=root/(action+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    cfg={**cfg,'out':str(out)}
    if action=='run':
        cfg['readiness']=bound(root/'READY.json')
        previous=read(root/'LATEST.json')
        if previous:cfg['resume']=previous['checkpoint']
    path=root/(out.name+'.json');write_json(path,cfg)
    write_json(root/'ACTIVE.json',{'action':action,'run_dir':str(out),'config':bound(path)})
    code=subprocess.call([sys.executable,'-m','mei_llm','base-qat',action,'--config',str(path),'--confirm-training'])
    terminal=read(out/'terminal.json');receipt=read(out/'receipt.json')
    if code!=0 or not terminal.get('ok') or not receipt.get('ok'):
        raise RuntimeError(f'{action} failed: {out}; {terminal}')
    if action=='probe':write_json(root/'READY.json',{**receipt,'probe_receipt':bound(out/'receipt.json')})
    return receipt


def service_state(name):
    return subprocess.run(['systemctl','show',name,'-p','ActiveState','--value'],capture_output=True,text=True,check=True).stdout.strip()


def pause_cpt(cpt,service,root):
    active=read(cpt/'campaign/ACTIVE.json');run=Path(active['run_dir'])
    if service_state(service)=='active':
        write_json(root/'CPT-PAUSE.json',{'state':'requested','run_dir':str(run),'reason':'stage Base QAT'})
        (run/'STOP_REQUESTED').touch(exist_ok=True)
        deadline=time.monotonic()+600
        while service_state(service) in ('active','activating','deactivating'):
            if time.monotonic()>deadline:raise TimeoutError('CPT did not stop at optimizer boundary')
            time.sleep(2)
    receipt=read(run/'receipt.json');terminal=read(run/'terminal.json')
    if not receipt.get('ok') or not terminal.get('ok'):raise ValueError('CPT stop has no successful complete-state receipt')
    binding=receipt['checkpoint']
    if sha256(binding['path'])!=binding['sha256']:raise ValueError('CPT pause checkpoint corrupted')
    side=read(Path(binding['path']).with_suffix('.json'))
    if side.get('tokens_seen')!=receipt['tokens_seen']:raise ValueError('CPT pause state mismatch')
    write_json(root/'CPT-PAUSE.json',{'state':'checkpoint_verified','run_dir':str(run),
               'receipt':bound(run/'receipt.json'),'checkpoint':binding,'tokens_seen':receipt['tokens_seen'],
               'fingerprint':receipt['fingerprint'],'cpt_complete':receipt['cpt_complete']})
    return receipt


def resume_cpt(cpt,service,paused,root):
    if paused['cpt_complete']:return
    if sha256(paused['checkpoint']['path'])!=paused['checkpoint']['sha256']:
        raise ValueError('paused CPT bytes changed')
    entries=read(cpt/'checkpoints/STORE.json')['checkpoints']
    newest=max(entries,key=lambda x:x['step'])
    if newest['sha256']!=paused['checkpoint']['sha256']:raise ValueError('CPT latest changed during QAT')
    subprocess.run(['systemctl','start',service],check=True)
    write_json(root/'CPT-RESUMED.json',{'checkpoint':paused['checkpoint'],'tokens_seen':paused['tokens_seen'],
               'service':service,'resume_via':'original unchanged CPT service/source/optimizer/RNG/sampler'})


def supervise(cfg):
    root=Path(cfg['root']);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'controller.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpt=Path(cfg['cpt_root']);service=cfg['cpt_service']
    for tokens in cfg['milestones']:
        campaign=root/f'base-{tokens:010d}'
        if read(campaign/'HANDOFF.json').get('done'):continue
        snapshot=cpt/'milestones'/f'tokens-{tokens:010d}'/'SNAPSHOT.json'
        while not snapshot.exists():
            write_json(root/'STATUS.json',{'state':'waiting_cpt_milestone','base_tokens':tokens,
                       'cpt':read(cpt/'campaign/STATUS.json'),'updated_unix':time.time()})
            if service_state(service) not in ('active','activating'):
                raise RuntimeError('CPT stopped before required milestone; no automatic retry')
            time.sleep(2)
        request={'campaign_dir':str(campaign),'snapshot':bound(snapshot),
                 'corpus':bound(cpt/'campaign/prepared/corpus.json'),'base_tokens':tokens,
                 'execution_hostname':cfg['execution_hostname']}
        # Prepare references while the immutable CPT worker remains active.
        prepared=prepare(request)
        paused=pause_cpt(cpt,service,campaign)
        try:
            if not (campaign/'READY.json').exists():
                write_json(root/'STATUS.json',{'state':'qat_preflight','base_tokens':tokens,'cpt_paused_at':paused['tokens_seen']})
                invoke('probe',prepared)
            write_json(root/'STATUS.json',{'state':'qat_running','base_tokens':tokens,'cpt_paused_at':paused['tokens_seen']})
            result=read(campaign/'RESULT.json') or invoke('run',prepared)
            if not result.get('process_complete'):raise RuntimeError('QAT paused; preserve for explicit resume')
            write_json(campaign/'HANDOFF.json',{'done':True,'result':bound(campaign/'RESULT.json'),
                       'quality_passed':result['quality_passed'],'local_transfer':'pending',
                       'cpt_resumes_from':paused['checkpoint']})
        except Exception as exc:
            write_json(root/'STATUS.json',{'state':'qat_failed','base_tokens':tokens,'error':repr(exc)})
            resume_cpt(cpt,service,paused,campaign)
            raise
        resume_cpt(cpt,service,paused,campaign)
    write_json(root/'STATUS.json',{'state':'all_remote_stages_complete','local_transfer':'pending',
               'safe_to_shutdown':False,'reason':'local hash/load/handoff verification still required'})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','probe','run','supervise'])
    p.add_argument('--config',type=Path,required=True);args=p.parse_args();cfg=read(args.config)
    if cfg['execution_hostname']!=socket.gethostname():raise ValueError('incorrect execution host')
    if args.action=='prepare':print(json.dumps(prepare(cfg)));return
    if args.action=='supervise':supervise(cfg);return
    if cfg['source_manifest_sha256']!=manifest(ROOT)['manifest_sha256']:raise ValueError('QAT source changed')
    gpu_lock=Path('/tmp/mei-cpt-gpu0.lock').open('a+')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with stage(cfg,track='research_candidate',pipeline_id=PIPELINE) as out:
        from training.qat.base_campaign import probe,run
        if args.action=='probe':
            # Actual device quantizer/STE parity plus token-weighted loss tests.
            test=subprocess.run([sys.executable,'-m','unittest','tests.test_cq2_torch_51m','tests.test_cuda_qat'],
                                capture_output=True,text=True)
            (out/'numerical-tests.log').write_text(test.stdout+test.stderr)
            if test.returncode:raise ValueError('CUDA QAT numerical tests failed')
            probe(cfg,out)
        else:run(cfg,out)

if __name__=='__main__':main()
