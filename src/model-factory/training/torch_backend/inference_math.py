"""Shape-stable FP32 projections for the quantized CUDA research reference.

Each row uses an eight-row GEMM, including prefill. No global Torch functions
are patched, and training/Float inference retain their original operations.
This is an inference correctness path, not a packed deployment kernel.
"""
import torch
from torch import nn
from torch.nn import functional as F


def linear(x, weight, bias=None, *, block_rows=0):
    if not block_rows:
        return F.linear(x, weight, bias)
    if block_rows != 8 or torch.is_grad_enabled():
        raise ValueError("fixed-row projections require no-grad inference and eight rows")
    if x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("fixed-row reference projections require FP32 operands")
    rows = x.reshape(-1,x.shape[-1])
    outputs = []
    for start in range(0, rows.shape[0], block_rows):
        block = rows[start:start+block_rows]
        padded = F.pad(block,(0,0,0,block_rows-block.shape[0]))
        outputs.append(F.linear(padded,weight,bias)[:block.shape[0]])
    return torch.cat(outputs,dim=0).reshape(*x.shape[:-1],weight.shape[0])


class InferenceLinear(nn.Linear):
    inference_block_rows = 0

    def forward(self,x):
        block_rows = self.inference_block_rows if not self.training and not torch.is_grad_enabled() else 0
        return linear(x,self.weight,self.bias,block_rows=block_rows)
