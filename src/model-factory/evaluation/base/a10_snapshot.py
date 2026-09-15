"""Read-only MLX validation of a copied A10 milestone on fixed local dev windows."""
import argparse
import json
import math
from pathlib import Path
import time

from common.paths import ensure_formal_on_path
from common.evidence_stage import sha256, write_json, stage


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    if cfg.get('action')!='mlx_snapshot_eval':raise ValueError('wrong evaluation action')
    snapshot=Path(cfg['snapshot']['path']);weights=Path(cfg['weights']['path'])
    root=Path(cfg['input_release']['path'])
    for item in (cfg['snapshot'],cfg['weights']):
        if sha256(item['path'])!=item['sha256']:raise ValueError('snapshot bytes changed')
    if sha256(root/'RELEASE.json')!=cfg['input_release']['sha256']:raise ValueError('input release changed')
    snap=json.loads(snapshot.read_text());release=json.loads((root/'RELEASE.json').read_text())
    if snap['weights']['sha256']!=cfg['weights']['sha256']:raise ValueError('wrong copied weights')
    if snap['tokenizer']['sha256']!=release['tokenizer']['model_sha256']:raise ValueError('tokenizer mismatch')
    with stage(cfg,track='diagnostic',pipeline_id='mei-51m-a10-mlx-snapshot-eval-v1') as out:
        ensure_formal_on_path()
        import mlx.core as mx
        from architecture import NeedleZh
        from config import NeedleZhConfig
        from common.checkpoint import load_params
        from common.data import PackedTokenSource
        from common.train_common import masked_lm_loss, stack_windows
        mx.set_default_device(mx.cpu if cfg.get('device')=='cpu' else mx.gpu)
        model=NeedleZh(NeedleZhConfig(**snap['model_config']));model.eval()
        loaded=load_params(model,weights,strict=True,return_report=True)
        if loaded['unexpected'] or loaded['missing']:raise ValueError('tensor contract mismatch')
        limit=int(cfg.get('windows_per_source',32))
        if not 1<=limit<=128:raise ValueError('bounded dev audit requires 1..128 windows per source')
        started=time.monotonic();roles=[]
        for item in sorted(release['reserve_files'],key=lambda x:(x['source'],x['path'])):
            if item['split']!='dev':continue
            path=root/item['path']
            if sha256(path)!=item['sha256']:raise ValueError('dev bytes changed')
            source=PackedTokenSource([path],2048)
            count=min(limit,len(source));indices=[i*len(source)//count for i in range(count)]
            nll=0.;tokens=0
            for index in indices:
                batch=stack_windows([source[index]])
                loss=float(masked_lm_loss(model(batch['x'])['logits'],batch['y'],batch['mask']))
                if not math.isfinite(loss):raise ValueError('nonfinite dev loss')
                n=int(mx.sum(batch['mask']).item());nll+=loss*n;tokens+=n
            row={'source':item['source'],'path':item['path'],'sha256':item['sha256'],
                 'indices':indices,'predicted_tokens':tokens,'nll_sum':nll,'loss':nll/max(tokens,1)}
            roles.append(row);write_json(out/'progress.json',{'roles':roles,'elapsed_seconds':time.monotonic()-started})
        total=sum(r['predicted_tokens'] for r in roles)
        if not total:raise ValueError('no dev targets')
        write_json(out/'receipt.json',{'ok':True,'scope':'fixed stratified dev LM loss; not SFT tool capability or release approval',
            'tokens_seen':snap['tokens_seen'],'roles':roles,'predicted_tokens':total,
            'loss':sum(r['nll_sum'] for r in roles)/total,'load_report':loaded,
            'elapsed_seconds':time.monotonic()-started,'remote_training_dependency':False})
    return 0


if __name__=='__main__':raise SystemExit(main())
