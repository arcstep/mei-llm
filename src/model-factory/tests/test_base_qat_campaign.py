import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from training.qat.base_replay import make_references, Replay
from training.qat.base_campaign import decision

class BaseQatTests(unittest.TestCase):
    def test_replay_consumed_prefix_unique_and_partial_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'tokens.bin';np.arange(100,900,dtype=np.uint16).tofile(p)
            corpus={'sources':{'one':{'train':[{'path':str(p)}],'dev':[]}}}
            refs=make_references(corpus,{'one':400},99,17,seq_len=8)
            self.assertEqual(refs,make_references(corpus,{'one':400},99,17,seq_len=8))
            starts=[r['start'] for r in refs['train']]
            self.assertEqual(len(starts),len(set(starts)))
            self.assertTrue(all(s+8<400 for s in starts))
            r=Replay(corpus,refs)
            first,n=r.batch(0,13,4,'cpu');second,m=r.batch(13,24,4,'cpu')
            self.assertEqual((n,m),(13,11))
            self.assertEqual(float(first[2].sum()+second[2].sum()),24)
            self.assertTrue(torch.equal(first[2][-1]+second[2][0],torch.ones(8)))
            with self.assertRaises(ValueError):make_references(corpus,{'one':10},100,1,seq_len=8)

    def test_dev_selection_never_selects_smoke_or_failed_candidate(self):
        cfg={'first_selection_tokens':20000000,'gates':{'max_aggregate_ce_delta':.15,'max_source_ce_delta':.4,'minimum_ce_improvement':.002}}
        def metric(n):return {'token_ce':n,'roles':{'a':{'token_ce':n}}}
        anchor=metric(2.)
        history,best,stop=decision(anchor,metric(2),metric(2.1),cfg,[],5000000)
        self.assertIsNone(best)
        history,best,stop=decision(anchor,metric(2),metric(2.099),cfg,history,10000000)
        self.assertIsNone(best)
        history,best,stop=decision(anchor,metric(2),metric(2.098),cfg,history,20000000)
        self.assertEqual(best['tokens'],20000000);self.assertTrue(stop)
        history,best,stop=decision(anchor,metric(2),metric(2.3),cfg,history,30000000)
        self.assertEqual(best['tokens'],20000000)
        self.assertFalse(history[-1]['passed'])

if __name__=='__main__':unittest.main()
