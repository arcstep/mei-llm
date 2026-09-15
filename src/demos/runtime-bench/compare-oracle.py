"""Describe raw GPU/native logits differences; does not qualify model accuracy."""
import argparse
import hashlib
import json
import math
from pathlib import Path

p = argparse.ArgumentParser(__doc__)
p.add_argument('native', type=Path)
p.add_argument('gpu', type=Path)
a = p.parse_args()
x, y = (json.loads(f.read_text()) for f in (a.native, a.gpu))
results = {}
for key in ('first_logits', 'second_logits', 'two_token_logits', 'varied128_logits', 'next128_logits'):
    u, v = x[key], y[key]
    if len(u) != 24000 or len(v) != 24000 or not all(math.isfinite(z) for z in u + v):
        raise ValueError('Invalid logits: ' + key)
    results[key] = {
        'cosine_similarity': sum(i*j for i,j in zip(u,v)) / math.sqrt(sum(i*i for i in u)*sum(j*j for j in v)),
        'max_absolute_difference': max(abs(i-j) for i,j in zip(u,v)),
        'native_argmax': max(range(len(u)), key=u.__getitem__),
        'gpu_argmax': max(range(len(v)), key=v.__getitem__),
    }
print(json.dumps({
    'kind': 'descriptive-numeric-comparison-not-quality-gate',
    'sources': {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (a.native,a.gpu)},
    'results': results,
    'gpu_repeat_max_abs': y['repeat_max_abs'],
    'gpu_batch_vs_incremental_max_abs': max(abs(i-j) for i,j in zip(y['second_logits'],y['two_token_logits'])),
    'canonical_numeric_parity_claimed': False,
}, indent=2))
