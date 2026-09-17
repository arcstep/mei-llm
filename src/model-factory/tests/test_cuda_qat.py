import copy
import unittest

import torch

from training.qat.cuda_qat import update,group_map
from training.torch_backend.model import NeedleZh,NeedleZhConfig


class CudaQatTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(51)
        cfg=NeedleZhConfig(vocab_size=64,d_model=32,n_layers=3,n_heads=4,n_kv_heads=2,
                          engram_layers=(0,2),engram_slots=32)
        self.model=NeedleZh(cfg);self.model.enable_qat(True)

    def test_token_weighted_accumulation_matches_joint_masked_batch(self):
        x=torch.tensor([[2,4,5,6],[2,14,15,16]])
        y=torch.tensor([[4,5,6,1],[14,15,16,1]])
        mask=torch.tensor([[1.,1,1,1],[1.,0,0,0]])
        other=copy.deepcopy(self.model)
        a=torch.optim.Adam(self.model.parameters(),lr=.001)
        b=torch.optim.Adam(other.parameters(),lr=.001)
        loss_a,norm_a,count_a=update(self.model,a,[(x,y,mask)],"fp32")
        loss_b,norm_b,count_b=update(other,b,[(x[:1],y[:1],mask[:1]),(x[1:],y[1:],mask[1:])],"fp32")
        self.assertEqual((count_a,count_b),(5,5));self.assertAlmostEqual(loss_a,loss_b,places=5)
        for (name,pa),(_,pb) in zip(self.model.named_parameters(),other.named_parameters()):
            if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=2e-4,atol=3e-6,msg=name)

    def test_checkpoint_recomputation_is_rejected_until_qat_bound(self):
        self.model.checkpoint_blocks=True
        with self.assertRaises(ValueError):update(self.model,None,[],"fp32")

    def test_group_map_uses_portable_mixed_storage_and_covers_every_parameter(self):
        mapping=group_map(self.model)
        self.assertEqual(set(mapping["tensors"]),set(dict(self.model.named_parameters())))
        self.assertEqual(mapping["tensors"]["embed.weight"]["storage"],"cq4")
        self.assertEqual(mapping["tensors"]["blocks.0.attn.q_proj.weight"]["storage"],"cq2")
        self.assertEqual(mapping["group_size"],128)


if __name__=="__main__":unittest.main()
