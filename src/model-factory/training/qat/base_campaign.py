"""Base-only CUDA CQ2 campaign. Own anchors, bounded rungs, immutable checkpoints."""
import gc
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from common.evidence_stage import write_json, sha256
from training.qat.base_replay import load, bound, Replay
from training.qat.cuda_qat import update, group_map
from training.qat.cq2_torch_51m import forward_qat
from training.torch_backend.model import NeedleZh, NeedleZhConfig
from training.torch_backend.optim import configure
from training.torch_backend.performance import install_kernels
from training.torch_backend.precision import autocast
from training.torch_backend import checkpoint


def setup(cfg):
    snapshot = load(cfg['snapshot'])
    corpus, refs = load(cfg['corpus']), load(cfg['replay'])
    for b in [snapshot['weights'], snapshot['checkpoint'], snapshot['tokenizer']]:
        if sha256(b['path']) != b['sha256']:
            raise ValueError('snapshot bytes changed')
    for src in corpus['sources'].values():
        for b in src['train'] + src.get('dev', []):
            if sha256(b['path']) != b['sha256']:
                raise ValueError('corpus bytes changed')
    if corpus['tokenizer'] != snapshot['tokenizer']:
        raise ValueError('Base/replay tokenizer mismatch')
    configure(cfg['seed'], deterministic=True)
    install_kernels(cfg['kernel_mode'])
    model = NeedleZh(NeedleZhConfig(**snapshot['model_config']))
    with np.load(snapshot['weights']['path'], allow_pickle=False) as archive:
        model.load_state_dict({k:torch.from_numpy(archive[k].copy()) for k in archive.files}, strict=True)
    model = model.cuda()
    model.checkpoint_blocks = False
    if sum(p.numel() for p in model.parameters()) != 51463797:
        raise ValueError('LM geometry changed')
    return model, Replay(corpus,refs)


def optimizer(model,cfg):
    return torch.optim.Adam(model.parameters(),lr=cfg['lr'],betas=(.9,.999),eps=1e-8,foreach=True)


def step(model,opt,replay,cursor,stop,cfg):
    batches=[]
    for _ in range(cfg['grad_accum']):
        if cursor>=stop: break
        batch,n=replay.batch(cursor,stop,cfg['batch_size'],'cuda')
        batches.append(batch);cursor+=n
    return update(model,opt,batches,cfg['precision'])


@torch.no_grad()
def evaluate(model,replay,cfg,quantized):
    model.eval();model.enable_qat(quantized)
    # Reconstruct once per evaluation, never once per text window.
    from training.qat.cq2_torch_51m import quantized_parameters
    params=quantized_parameters(model,ste=False) if quantized else None
    roles={}
    for row in replay.refs['dev']:
        x,y=replay.window(row)
        xt=torch.tensor(x[None,:],device='cuda');yt=torch.tensor(y,device='cuda')
        with autocast(cfg['precision']):
            output=(torch.func.functional_call(model,params,(xt,),strict=False) if quantized else model(xt))['logits']
        nll=float(F.cross_entropy(output[0].float(),yt,reduction='sum'))
        if not math.isfinite(nll): raise FloatingPointError('nonfinite dev NLL')
        value=roles.setdefault(row['source'],{'nll_sum':0.,'tokens':0})
        value['nll_sum']+=nll;value['tokens']+=len(y)
    if not roles: raise ValueError('empty fixed dev')
    for value in roles.values():value['token_ce']=value['nll_sum']/value['tokens']
    total=sum(v['tokens'] for v in roles.values())
    nll=sum(v['nll_sum'] for v in roles.values())
    model.train();model.enable_qat(True)
    return {'token_ce':nll/total,'nll_sum':nll,'predicted_tokens':total,'roles':roles,
            'quantized':quantized,'precision':cfg['precision'],'dev_uncovered':replay.refs['dev_uncovered'],
            'scope':'same-tokenizer fixed dispersed 32x2048 per available source; LM only, not tool capability'}


def decision(anchor,master,quant,cfg,history,tokens):
    delta=quant['token_ce']-anchor['token_ce']
    source_deltas={n:v['token_ce']-anchor['roles'][n]['token_ce'] for n,v in quant['roles'].items()}
    ok=delta<=cfg['gates']['max_aggregate_ce_delta'] and max(source_deltas.values())<=cfg['gates']['max_source_ce_delta']
    item={'tokens':tokens,'quantized_ce':quant['token_ce'],'master_ce':master['token_ce'],
          'delta_vs_anchor':delta,'source_deltas':source_deltas,'passed':ok}
    history=history+[item]
    eligible=[v for v in history if v['passed'] and v['tokens']>=cfg['first_selection_tokens']]
    best=min(eligible,key=lambda v:v['quantized_ce']) if eligible else None
    improvements=[history[i-1]['quantized_ce']-history[i]['quantized_ce'] for i in range(1,len(history))]
    plateau=(best is not None and len(improvements)>=2 and
             all(d<cfg['gates']['minimum_ce_improvement'] for d in improvements[-2:]))
    return history,best,plateau


def probe(cfg,out):
    model,replay=setup(cfg);model.enable_qat(True);opt=optimizer(model,cfg)
    n=cfg['batch_size']*cfg['grad_accum']*2048
    loss,norm,count=step(model,opt,replay,0,n,cfg)
    checkpoint.save_state(out/'probe-start.pt',model,opt,fingerprint=cfg['fingerprint'],tokens_seen=n)
    # Continue, reload complete state, repeat the exact same next update.
    step(model,opt,replay,n,2*n,cfg)
    checkpoint.save_state(out/'probe-continuous.pt',model,opt,fingerprint=cfg['fingerprint'],tokens_seen=2*n)
    checkpoint.load_state(out/'probe-start.pt',model,opt,fingerprint=cfg['fingerprint'])
    before=time.perf_counter()
    step(model,opt,replay,n,2*n,cfg);torch.cuda.synchronize()
    elapsed=time.perf_counter()-before
    compared=checkpoint.compare_state(out/'probe-continuous.pt',model,opt)
    # Read-only tiny dev forward for both deployment weights and Float master.
    dev=replay.refs['dev'];replay.refs['dev']=dev[:1]
    float_dev=evaluate(model,replay,cfg,False);quant_dev=evaluate(model,replay,cfg,True)
    receipt={'ok':compared['ok'],'restart':compared,'last_loss':loss,'grad_norm':norm,
             'update_tokens':n,'warm_update_seconds':elapsed,'warm_tokens_per_second':n/elapsed,
             'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'float_smoke':float_dev,'cq2_smoke':quant_dev,
             'fingerprint':cfg['fingerprint'],'purpose':'mechanism and resource acceptance; not QAT quality'}
    write_json(out/'receipt.json',receipt)
    return receipt


def run(cfg,out):
    readiness=load(cfg['readiness'])
    if not readiness.get('ok') or readiness['fingerprint']!=cfg['fingerprint']:
        raise ValueError('QAT needs matching real CUDA readiness')
    model,replay=setup(cfg);opt=optimizer(model,cfg)
    root=Path(cfg['campaign_dir']);states=root/'checkpoints';states.mkdir(exist_ok=True)
    mapping=root/'group-map.json'
    if not mapping.exists():write_json(mapping,group_map(model))
    anchor_path=root/'anchor.json'
    if not anchor_path.exists():
        write_json(out/'progress.json',{'state':'evaluating_float_anchor'})
        anchor=evaluate(model,replay,cfg,False)
        anchor.update(base_snapshot=cfg['snapshot'],replay=cfg['replay'])
        write_json(anchor_path,anchor)
    anchor=json.loads(anchor_path.read_text())
    if anchor['base_snapshot']!=cfg['snapshot'] or anchor['replay']!=cfg['replay']:
        raise ValueError('anchor binding mismatch')
    tokens=steps=0;history=[];best=None;last=None;last_loss=None;done=False
    if cfg.get('resume'):
        state=checkpoint.load_state(cfg['resume']['path'],model,opt,fingerprint=cfg['fingerprint'])
        if sha256(cfg['resume']['path'])!=cfg['resume']['sha256']:raise ValueError('resume hash mismatch')
        tokens,steps,history,best=state['tokens_seen'],state['step'],state['history'],state['best']
        last=cfg['resume'];done=state['done']
    model.train();model.enable_qat(True)
    started=time.perf_counter();start_tokens=tokens;last_saved=time.monotonic()
    def save():
        nonlocal last,last_saved
        path=states/f'state-{tokens:010d}.pt'
        if not path.exists():
            checkpoint.save_state(path,model,opt,fingerprint=cfg['fingerprint'],tokens_seen=tokens,step=steps,
                                  history=history,best=best,done=done,replay=cfg['replay'])
        last=bound(path);last_saved=time.monotonic()
        write_json(root/'LATEST.json',{'checkpoint':last,'tokens_seen':tokens,'step':steps,'done':done})
    trace=(out/'updates.jsonl').open('x')
    try:
        for target in cfg['rungs']:
            if done:break
            if target<=tokens:continue
            while tokens<target:
                import shutil
                if shutil.disk_usage(root).free<cfg['disk_reserve_bytes']:
                    raise OSError('QAT disk reserve reached')
                if (out/'STOP_REQUESTED').exists():
                    save()
                    write_json(out/'receipt.json',{'ok':True,'process_complete':False,'state':'paused','tokens_seen':tokens,'checkpoint':last})
                    return
                loss,norm,n=step(model,opt,replay,tokens,target,cfg)
                tokens+=n;steps+=1;last_loss=loss
                row={'tokens_seen':tokens,'target_tokens':target,'max_tokens':cfg['rungs'][-1],
                     'step':steps,'loss':loss,'grad_norm':norm,'seconds':time.perf_counter()-started,
                     'process_tokens_per_second':(tokens-start_tokens)/(time.perf_counter()-started),
                     'base_tokens':cfg['base_tokens']}
                trace.write(json.dumps(row)+'\n');trace.flush()
                if steps%10==0 or steps==1:write_json(out/'heartbeat.json',row)
                if time.monotonic()-last_saved>900 and tokens<target:save()
            write_json(out/'progress.json',{'state':'evaluating_rung','tokens_seen':tokens})
            master=evaluate(model,replay,cfg,False);quant=evaluate(model,replay,cfg,True)
            history,best,plateau=decision(anchor,master,quant,cfg,history,tokens)
            done=plateau or (cfg.get('stop_on_first_qualified',False) and best is not None) or target==cfg['rungs'][-1]
            rung=root/f'rung-{tokens:010d}.json'
            write_json(rung,{'master':master,'cq2':quant,'decision':history[-1],'best':best,'plateau':plateau,
                            'anchor':bound(anchor_path),'fingerprint':cfg['fingerprint']})
            save()
            write_json(root/'STATUS.json',{'state':'complete' if done else 'running','tokens_seen':tokens,'best':best,
                       'latest_rung':bound(rung),'last_checkpoint':last})
        if not done:raise ValueError('unexpected incomplete QAT campaign')
        result={'ok':True,'process_complete':True,'quality_passed':best is not None,'tokens_seen':tokens,
                'base_tokens':cfg['base_tokens'],'best':best,'anchor':bound(anchor_path),'history':history,
                'group_map':bound(mapping),'last_checkpoint':last,'release_eligible':False,
                'fingerprint':cfg['fingerprint'],'last_loss':last_loss}
        if best is not None:
            chosen=states/f"state-{best['tokens']:010d}.pt"
            state=torch.load(chosen,map_location='cpu',weights_only=True)
            dest=root/'selected-master.npz'
            if not dest.exists():
                with dest.with_suffix('.partial').open('xb') as f:
                    np.savez(f,**{k:v.numpy() for k,v in state['model'].items()})
                dest.with_suffix('.partial').rename(dest)
            result.update(selected_checkpoint=bound(chosen),selected_weights=bound(dest))
        write_json(root/'RESULT.json',result);write_json(out/'receipt.json',result)
    finally:
        trace.close()
