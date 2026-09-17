"""CUDA tensor loop for the registered CQ2 replay objective.

Uses the shared quota sampler and full-state checkpoint format. The enclosing
phase executor must validate Float anchors, same-data control and source/data
bindings before calling this loop; this module is not a command-line entry.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from common.evidence_stage import sha256,write_json
from training.torch_backend import checkpoint
from training.torch_backend.precision import autocast
from training.qat.cq2_torch_51m import forward_qat
from training.qat.cq2_policy_51m import GROUP_SIZE,QUANT_MATH_ID,lm_storage_dtype


def update(model,opt,batches,precision):
    if not model.qat_activation_ste:raise ValueError("CQ2 replay requires activation and KV STE")
    if getattr(model,"checkpoint_blocks",False):
        raise ValueError("QAT functional weights need a separately validated recomputation adapter")
    total = sum(int(mask.sum()) for x,y,mask in batches)
    if total<=0:raise ValueError("empty replay update")
    opt.zero_grad(set_to_none=True);loss_sum = 0.0
    for x,y,mask in batches:
        with autocast(precision):logits = forward_qat(model,x)["logits"]
        terms = F.cross_entropy(logits.float().flatten(0,1),y.flatten(),reduction="none")
        nll = (terms*mask.flatten()).sum()
        if not torch.isfinite(nll):raise FloatingPointError("nonfinite CQ2 replay NLL")
        (nll/total).backward();loss_sum += float(nll.detach())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
    opt.step()
    return loss_sum/total,float(norm),total


def group_map(model):
    tensors = {}
    for name,value in model.named_parameters():
        storage = lm_storage_dtype(name,value.shape)
        tensors[name] = {"shape":list(value.shape),"elements":value.numel(),"storage":storage,
            "group_bits":[] if storage=="f16" else [4 if storage=="cq4" else 2]*((value.numel()+GROUP_SIZE-1)//GROUP_SIZE)}
    return {"schema":"mei-cuda-cq2-group-map-v1","group_size":GROUP_SIZE,
            "quant_math_id":QUANT_MATH_ID,"tensors":tensors}

