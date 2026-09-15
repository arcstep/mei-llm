"""Small synthetic resource/restart fixture, never counted as CPT exposure."""
import argparse
import json
from pathlib import Path
import socket


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','resume','accept'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--tokenizer',type=Path,required=True)
    args=parser.parse_args()
    from orchestration.cpt_a10 import bound
    from common.evidence_stage import write_json
    root=args.root.resolve();prep=root/'prepared'
    if args.action=='prepare':
        import numpy as np
        prep.mkdir(parents=True,exist_ok=False)
        names=['language','oral','code','tools'];sources={}
        rng=np.random.default_rng(20260914)
        for name in names:
            path=prep/(name+'.bin');rng.integers(4,24000,size=32769,dtype=np.uint16).tofile(path)
            sources[name]={'train':[bound(path)],'dev':[]}
        schedule={'kind':'scratch','allow_repeat':False,'parent_tokens_seen':0,'sampler':'quota_plan',
            'sampler_seed':20260914,'sources':{name:{'token_quota':24576} for name in names},
            'exposure_tokens':98304,'lr':{'kind':'cosine_tokens','base':3e-4,'final':3e-5,'horizon_tokens':98304},
            'curriculum':[{'id':f'probe-{i+1}','index':i,'seq_len':2048,'batch_size':4,'grad_accum':1,
                'stage_tokens':32768,'stop_at_tokens':32768*(i+1),
                'sources':{name:{'token_quota':8192} for name in names}} for i in range(3)]}
        corpus={'schema':'mei-cuda-encoded-corpus-v1','ok':True,'vocab_size':24000,'token_dtype':'uint16',
            'tokenizer':bound(args.tokenizer),'sources':sources,'scope':'synthetic kernel and restart fixture only'}
        write_json(prep/'schedule.json',schedule);write_json(prep/'corpus.json',corpus)
        cfg={'action':'cpt','purpose':'backend_probe','vocab_size':24000,'precision':'bf16_amp','seed':20260914,
            'schedule':bound(prep/'schedule.json'),'corpus':bound(prep/'corpus.json'),
            'tokenizer':bound(args.tokenizer),'out':str(root/'baseline'),'checkpoint_tokens':1000000,
            'kernel_mode':'compile_pointwise','compile':False,'activation_checkpointing':True,
            'execution_hostname':socket.gethostname()}
        write_json(prep/'baseline.json',cfg)
        write_json(prep/'prefix.json',{**cfg,'out':str(root/'prefix'),'stop_after_steps':5})
    elif args.action=='resume':
        cfg=json.loads((prep/'baseline.json').read_text())
        prefix=json.loads((root/'prefix/receipt.json').read_text());baseline=json.loads((root/'baseline/receipt.json').read_text())
        write_json(prep/'resume.json',{**cfg,'out':str(root/'resume'),'resume':prefix['checkpoint'],
            'compare_to':baseline['checkpoint']})
    else:
        from common.source_capture import manifest
        cfg=json.loads((prep/'baseline.json').read_text())
        receipts=[root/name/'receipt.json' for name in ('baseline','prefix','resume')]
        reports=[json.loads(p.read_text()) for p in receipts];comparison=reports[-1].get('restart_comparison',{})
        ok=(all(r['ok'] for r in reports) and reports[0]['cpt_complete'] and reports[-1]['cpt_complete']
            and reports[1]['step']==5 and comparison.get('ok') and comparison.get('sampler_and_exposure_equal'))
        value={'ok':bool(ok),'scope':'synthetic 2048/batch4 BF16 resource and bit-identical model/Adam/RNG/cursor restart; not model quality',
            'tokenizer':cfg['tokenizer'],'receipts':[bound(p) for p in receipts],'batch_size':4,
            'source_manifest_sha256':manifest(Path(__file__).resolve().parents[3])['manifest_sha256'],
            'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in reports),
            **{k:cfg[k] for k in ('precision','kernel_mode','activation_checkpointing','vocab_size')}}
        write_json(root/'ACCEPTANCE.json',value);print(json.dumps(value,indent=2))
        if not ok:raise SystemExit(2)


if __name__=='__main__':main()
