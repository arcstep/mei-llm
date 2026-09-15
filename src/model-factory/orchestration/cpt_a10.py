"""Server-local, explicitly bound CUDA CPT; no CURRENT or automatic promotion."""
import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import fcntl
from datetime import datetime, timezone

from common.evidence_stage import sha256, write_json, stage
from common.paths import ensure_formal_on_path
ensure_formal_on_path()


PIPELINE='mei-51m-cpt-cuda-v1'


def bound(path):
    p=Path(path).resolve()
    return {'path':str(p),'sha256':sha256(p)}


def prepare(request):
    root=Path(request['input_release']).resolve()
    release=json.loads((root/'RELEASE.json').read_text())
    if sha256(root/'RELEASE.json') != request['input_release_sha256'] or release['status']!='inputs_ready':
        raise ValueError('input release changed or not ready')
    tokenizer=json.loads((root/'TOKENIZER.json').read_text())
    if tokenizer['vocab_size']!=24000 or tokenizer['encoding_profile_id']!='mei-lossless-identity-v1':
        raise ValueError('wrong tokenizer profile')
    model=root/tokenizer['model_file']
    if sha256(model)!=tokenizer['model_sha256'] or release['tokenizer']['model_sha256']!=tokenizer['model_sha256']:
        raise ValueError('tokenizer model binding mismatch')
    for item in release['token_files']+release['reserve_files']:
        p=(root/item['path']).resolve()
        if not p.is_relative_to(root) or p.stat().st_size!=item['tokens']*2 or sha256(p)!=item['sha256']:
            raise ValueError('corpus file changed: '+item['path'])
    for key in ('record_index','schedule','sampler_validation'):
        item=release[key]
        if sha256(root/item['path'])!=item['sha256']:raise ValueError('release evidence changed: '+key)
    if json.loads((root/'SAMPLER-VALIDATION.json').read_text())['status']!='passed':
        raise ValueError('sampler validation failed')
    schedule=json.loads((root/'schedule.json').read_text())
    out=Path(request['prepared']).resolve();out.mkdir(parents=True,exist_ok=False)
    schedule.update(kind='scratch',allow_repeat=False,parent_tokens_seen=0,
        lr={'kind':'cosine_tokens','base':request.get('learning_rate',3e-4),
            'final':request.get('final_learning_rate',3e-5),'horizon_tokens':schedule['exposure_tokens']})
    sources={}
    for item in release['token_files']:
        name=item['source']; sources[name]={'train':[bound(root/item['path'])],
            'dev':[bound(root/r['path']) for r in release['reserve_files'] if r['source']==name and r['split']=='dev']}
        schedule['sources'][name]['token_quota']=sum(s['sources'][name]['token_quota'] for s in schedule['curriculum'])
        schedule['sources'][name]['paths']=[str(root/item['path'])]
    for s in schedule['curriculum']:
        s.update(batch_size=int(request.get('batch_size',4)),grad_accum=1)
    corpus={'schema':'mei-cuda-encoded-corpus-v1','ok':True,'sources':sources,'token_dtype':'uint16',
            'vocab_size':24000,'tokenizer':bound(model),'input_release':bound(root/'RELEASE.json')}
    write_json(out/'schedule.json',schedule);write_json(out/'corpus.json',corpus)
    cfg={'action':'cpt','purpose':'scratch_campaign','generation':'v1.3','vocab_size':24000,
         'precision':'bf16_amp','seed':20260914,'schedule':bound(out/'schedule.json'),
         'corpus':bound(out/'corpus.json'),'tokenizer':bound(model),
         'out':request['run_dir'],'checkpoint_store':request['checkpoint_store'],
         'checkpoint_tokens':10000000,'checkpoint_seconds':900,'disk_reserve_bytes':15*1024**3,
         'trace_every_steps':10,'kernel_mode':request.get('kernel_mode','compile_pointwise'),
         'compile':False,'activation_checkpointing':True,
         'input_release':bound(root/'RELEASE.json'),'preparation_request':request,
         'execution_hostname':request['execution_hostname']}
    write_json(out/'train.json',cfg)
    write_json(out/'PREPARED.json',{'input_release':bound(root/'RELEASE.json'),'train_config':bound(out/'train.json'),
        'scheduled_tokens':schedule['exposure_tokens'],'backend_acceptance':'pending_gpu_probe_and_restart',
        'training_started':False})
    return cfg


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run','supervise'])
    parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    if args.action=='prepare':
        print(json.dumps(prepare(cfg),indent=2));return 0
    if cfg['execution_hostname']!=socket.gethostname():raise ValueError('wrong execution host')
    if args.action=='supervise':
        campaign=Path(cfg['out']).parent;campaign.mkdir(parents=True,exist_ok=True)
        lock=(campaign/'supervisor.lock').open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        store=Path(cfg['checkpoint_store'])/'STORE.json'
        if store.exists():
            checkpoints=json.loads(store.read_text())['checkpoints']
            if checkpoints:
                latest=max(checkpoints,key=lambda r:r['step'])
                total=json.loads(Path(cfg['schedule']['path']).read_text())['exposure_tokens']
                if latest['tokens']>=total:
                    write_json(campaign/'STATUS.json',{'state':'complete','tokens_seen':latest['tokens']});return 0
                cfg['resume']={'path':str(store.parent/latest['path']),'sha256':latest['sha256']}
        attempt=campaign/('attempt-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        cfg['out']=str(attempt)
        control=campaign/(attempt.name+'.json');write_json(control,cfg)
        write_json(campaign/'ACTIVE.json',{'run_dir':str(attempt),'config':bound(control)})
        result=subprocess.run([sys.executable,'-m','mei_llm','cpt','a10','run','--config',str(control),'--confirm-training'])
        receipt_path=attempt/'receipt.json'
        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        write_json(campaign/'STATUS.json',{'state':'complete' if receipt.get('cpt_complete') else 'stopped',
            'exit_code':result.returncode,'run_dir':str(attempt),'stop_reason':receipt.get('stop_reason'),
            'tokens_seen':receipt.get('tokens_seen')})
        return result.returncode
    if cfg['purpose']=='scratch_campaign':
        acceptance=cfg.get('backend_acceptance')
        if not acceptance:raise ValueError('campaign needs successful CUDA resource/restart acceptance')
        p=Path(acceptance['path'])
        if sha256(p)!=acceptance['sha256']:raise ValueError('backend acceptance changed')
        report=json.loads(p.read_text())
        if not report.get('ok') or report.get('tokenizer',{}).get('sha256')!=cfg['tokenizer']['sha256']:
            raise ValueError('backend acceptance failed or tokenizer changed')
        from common.source_capture import manifest
        if report.get('source_manifest_sha256')!=manifest(Path(__file__).resolve().parents[3])['manifest_sha256']:
            raise ValueError('backend acceptance uses different source bytes')
        for key in ('precision','kernel_mode','activation_checkpointing','vocab_size'):
            if report.get(key)!=cfg[key]:raise ValueError('backend acceptance config differs: '+key)
        schedule=json.loads(Path(cfg['schedule']['path']).read_text())
        if {s['batch_size'] for s in schedule['curriculum']}!={report.get('batch_size')}:
            raise ValueError('backend batch size differs')
        for item in report.get('receipts',[]):
            if sha256(item['path'])!=item['sha256']:raise ValueError('backend test receipt changed')
        if len(report.get('receipts',[]))<3:raise ValueError('backend restart test evidence incomplete')
    gpu_lock=Path('/tmp/mei-cpt-gpu0.lock').open('a+')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from training.cpt.cuda_cpt import train
    with stage(cfg,track='research_candidate',pipeline_id=PIPELINE):
        train(cfg)
    return 0


if __name__=='__main__':raise SystemExit(main())
