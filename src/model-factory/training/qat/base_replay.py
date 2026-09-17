"""Immutable replay references constrained to a milestone's consumed train prefix."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def bound(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p)}


def load(binding):
    p = Path(binding['path'])
    if not p.is_absolute() or sha(p) != binding['sha256']:
        raise ValueError('binding mismatch: ' + str(p))
    return json.loads(p.read_text())


def make_references(corpus, consumed, budget, seed, seq_len=2048):
    # One aligned original window per selection, no sampling with replacement.
    rng = np.random.default_rng(seed)
    names = sorted(corpus['sources'])
    count = math.ceil(budget / seq_len)
    capacities = {n: max(0, (int(consumed[n]) - 1) // seq_len) for n in names}
    if sum(capacities.values()) < count:
        raise ValueError('insufficient consumed prefix for QAT replay')
    weights = np.array([capacities[n] for n in names], dtype=np.float64)
    raw = weights / weights.sum() * count
    quotas = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - quotas), kind='stable')[:count-int(quotas.sum())]:
        quotas[i] += 1
    rows = []
    for n, q in zip(names, quotas):
        files = corpus['sources'][n]['train']
        if len(files) != 1:
            raise ValueError('replay requires one frozen token file per source')
        if int(consumed[n]) > Path(files[0]['path']).stat().st_size // 2:
            raise ValueError('consumed prefix exceeds source')
        ids = rng.choice(capacities[n], int(q), replace=False)
        rows.extend({'source': n, 'start': int(i)*seq_len} for i in ids)
    rng.shuffle(rows)
    dev = []
    uncovered = []
    for n in names:
        bindings = corpus['sources'][n].get('dev', [])
        if not bindings:
            uncovered.append(n)
        for f in bindings:
            capacity = (Path(f['path']).stat().st_size//2 - 1)//seq_len
            if capacity <= 0:
                uncovered.append(n)
                continue
            for i in np.linspace(0, capacity-1, min(32, capacity), dtype=int):
                dev.append({'source': n, 'file': f, 'start': int(i)*seq_len})
    return {'schema': 'mei-milestone-qat-replay-v1', 'seed': seed, 'seq_len': seq_len,
            'budget': budget, 'consumed_train_prefix': consumed, 'train': rows, 'dev': dev,
            'dev_uncovered': sorted(set(uncovered)), 'source_quotas_windows': dict(zip(names, map(int, quotas)))}


class Replay:
    def __init__(self, corpus, refs):
        self.corpus, self.refs = corpus, refs
        self.seq_len = refs['seq_len']
        self.arrays = {}

    def window(self, row):
        binding = row.get('file') or self.corpus['sources'][row['source']]['train'][0]
        path = binding['path']
        if path not in self.arrays:
            self.arrays[path] = np.memmap(path, dtype=np.uint16, mode='r')
        start = row['start']
        chunk = self.arrays[path][start:start+self.seq_len+1].astype(np.int64)
        if len(chunk) != self.seq_len+1 or chunk.min() <= 0 or chunk.max() >= 24000:
            raise ValueError('invalid/unmasked replay tokens')
        return chunk[:-1], chunk[1:]

    def batch(self, cursor, stop, batch_size, device):
        import torch
        if not 0 <= cursor < stop <= self.refs['budget']:
            raise ValueError('invalid exposure interval')
        xs, ys, masks = [], [], []
        start_cursor = cursor
        for _ in range(batch_size):
            if cursor >= stop:
                break
            index, offset = divmod(cursor, self.seq_len)
            take = min(self.seq_len-offset, stop-cursor)
            x, y = self.window(self.refs['train'][index])
            mask = np.zeros(self.seq_len, np.float32)
            mask[offset:offset+take] = 1
            xs.append(x); ys.append(y); masks.append(mask)
            cursor += take
        tensors = tuple(torch.tensor(np.stack(v), device=device, dtype=dt)
                        for v, dt in [(xs,torch.long),(ys,torch.long),(masks,torch.float32)])
        return tensors, cursor-start_cursor
