"""Explicit precision policy for bounded CUDA diagnostics, not product adoption."""
from contextlib import contextmanager

import torch

POLICIES = ("fp32", "bf16_amp")


def policy(name):
    if name not in POLICIES:
        raise ValueError("unsupported diagnostic precision: " + str(name))
    return {
        "name": name,
        "version": "mei-vocab-study-precision-v1",
        "autocast": "bfloat16" if name == "bf16_amp" else "disabled",
        "parameters": "float32", "optimizer": "float32", "gradient_scaler": False,
        "fp32_regions": ["mHC projections and residual mixing", "Sinkhorn", "normalization statistics",
                         "WHT on FP32 residual stream", "loss", "gradient norm", "Adam updates"],
    }


@contextmanager
def autocast(name):
    policy(name)
    enabled = name == "bf16_amp"
    if enabled and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is not supported on this CUDA device")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
        yield


class DtypeAudit:
    """Observe real first-forward module outputs; detach hooks before timing warmup ends."""
    def __init__(self, model):
        self.outputs = {}
        self.handles = []
        for name, module in model.named_modules():
            if name.endswith(("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "mlp")):
                def observe(_module, _inputs, output, key=name):
                    self.outputs[key] = str(output.dtype)
                self.handles.append(module.register_forward_hook(observe))
        def observe_model(_module, _inputs, output):
            self.outputs["logits"] = str(output["logits"].dtype)
            self.outputs["hidden"] = str(output["hidden"].dtype)
        self.handles.append(model.register_forward_hook(observe_model))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def verify(self, name, model, optimizer):
        expected = "torch.bfloat16" if name == "bf16_amp" else "torch.float32"
        failures = []
        for key, dtype in self.outputs.items():
            target = "torch.float32" if key.endswith("mlp") or key == "hidden" else expected
            if dtype != target:
                failures.append(f"{key}: {dtype}, expected {target}")
        if "logits" not in self.outputs or not any(k.endswith("q_proj") for k in self.outputs):
            failures.append("missing forward dtype observations")
        parameter_dtypes = sorted({str(p.dtype) for p in model.parameters()})
        gradient_dtypes = sorted({str(p.grad.dtype) for p in model.parameters() if p.grad is not None})
        optimizer_dtypes = sorted({str(v.dtype) for s in optimizer.state.values() for v in s.values()
                                   if isinstance(v, torch.Tensor) and v.is_floating_point()})
        for key, dtypes in (("parameters", parameter_dtypes), ("gradients", gradient_dtypes), ("optimizer", optimizer_dtypes)):
            if dtypes != ["torch.float32"]:
                failures.append(f"{key}: {dtypes}")
        return {"ok": not failures, "outputs": self.outputs, "failures": failures,
                "parameter_dtypes": parameter_dtypes, "gradient_dtypes": gradient_dtypes,
                "optimizer_dtypes": optimizer_dtypes}
