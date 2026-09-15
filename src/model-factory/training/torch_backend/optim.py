"""Explicit initialization, precision, loss and optimizer shared across CUDA runs."""
import os
import math
import hashlib
import torch
from torch.nn import functional as F
from .model import NeedleZh, NeedleZhConfig

def configure(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    if not torch.cuda.is_available():
        raise RuntimeError("vocabulary training requires CUDA; never fall back to local CPU")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_model(config):
    cfg = NeedleZhConfig(vocab_size=int(config["vocab_size"]),
                         architecture_id=f"mei-vocab-study-pytorch-v1-v{int(config['vocab_size'])}")
    model = NeedleZh(cfg)
    # A larger embedding must not consume a different number of random draws
    # before initializing the shared backbone. Independent per-name streams
    # give all vocabulary variants byte-identical non-vocabulary initialization.
    seed = int(config.get("seed", 1729))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".weight") or name.endswith(".tables") or name.startswith("mhc_phi_") or name == "conf_probes":
                generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:8], "little") % (2**63))
                std = 0.02 / math.sqrt(2 * cfg.n_layers) if name.endswith(("o_proj.weight", "value_proj.weight")) else 0.02
                p.normal_(0, std, generator=generator)
    backbone = hashlib.sha256()
    initial = hashlib.sha256()
    for name, p in model.named_parameters():
        initial.update(name.encode())
        initial.update(p.detach().numpy().tobytes())
        if name != "embed.weight":
            backbone.update(name.encode())
            backbone.update(p.detach().numpy().tobytes())
    model.initial_backbone_sha256 = backbone.hexdigest()
    model.initial_parameters_sha256 = initial.hexdigest()
    model = model.cuda()
    model.checkpoint_blocks = bool(config.get("activation_checkpointing", False))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0, foreach=True)
    forward = torch.compile(model, dynamic=False) if config.get("compile", False) else model
    return model, forward, optimizer


def update(forward, model, optimizer, x, y, mask, precision="fp32"):
    from .precision import autocast
    optimizer.zero_grad(set_to_none=True)
    with autocast(precision):
        logits = forward(x)["logits"]
    with torch.autocast("cuda", enabled=False):
        nll = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="none")
        loss = (nll * mask.flatten()).sum() / mask.sum().clamp_min(1)
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(grad_norm)
