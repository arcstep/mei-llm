"""Bounded pointwise fusion; precision is controlled by precision.py and FP32 islands."""
import torch


def install_kernels(mode):
    if mode == "eager":
        return
    if mode != "compile_pointwise":
        raise ValueError(mode)
    from . import model
    model.walsh_hadamard = torch.compile(model.walsh_hadamard, fullgraph=True, dynamic=False)
    model.sinkhorn = torch.compile(model.sinkhorn, fullgraph=True, dynamic=False)
