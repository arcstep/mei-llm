"""Immutable full-state checkpoint adapter for CUDA research runs."""
from pathlib import Path
import os

import torch

from common.evidence_stage import sha256, write_json


def save_state(path, model, optimizer, **bindings):
    path = Path(path)
    if path.exists() or path.with_suffix('.partial').exists():
        raise FileExistsError(path)
    temp = path.with_suffix('.partial')
    with temp.open('xb') as stream:
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
                    **bindings}, stream)
        stream.flush();os.fsync(stream.fileno())
    temp.rename(path)
    write_json(path.with_suffix('.json'), {'sha256': sha256(path), 'path': path.name, **bindings})


def load_state(path, model, optimizer, **expected):
    import json
    path = Path(path)
    if sha256(path) != json.loads(path.with_suffix('.json').read_text())['sha256']:
        raise ValueError('checkpoint hash mismatch')
    state = torch.load(path, map_location='cpu', weights_only=True)
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError('checkpoint binding mismatch: ' + key)
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    return {k: v for k, v in state.items() if k not in ('model', 'optimizer', 'torch_rng', 'cuda_rng')}


def compare_state(path, model, optimizer):
    state = torch.load(path, map_location='cpu', weights_only=True)
    failures = []
    maximum = 0.0
    def walk(a, b, key):
        nonlocal maximum
        if isinstance(a, torch.Tensor):
            b = b.detach().cpu()
            if a.shape != b.shape or a.dtype != b.dtype:
                failures.append(key + ':shape/dtype')
                return
            if not torch.equal(a, b):
                failures.append(key)
                maximum = max(maximum, float((a - b).abs().max()))
        elif isinstance(a, dict):
            if set(a) != set(b):
                failures.append(key + ':keys')
                return
            for k in a:
                walk(a[k], b[k], key + ':' + str(k))
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                failures.append(key + ':length')
            else:
                for i, (aa, bb) in enumerate(zip(a, b)):
                    walk(aa, bb, key + ':' + str(i))
        elif a != b:
            failures.append(key)
    walk(state['model'], model.state_dict(), 'model')
    walk(state['optimizer'], optimizer.state_dict(), 'optimizer')
    walk(state['torch_rng'], torch.get_rng_state(), 'torch_rng')
    walk(state['cuda_rng'], torch.cuda.get_rng_state_all(), 'cuda_rng')
    return {'ok': not failures, 'comparison': 'bit-identical full parameters, optimizer and Torch/CUDA RNG',
            'failures': failures, 'max_abs_error': maximum}
