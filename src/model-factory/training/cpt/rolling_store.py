"""Bounded checkpoint ownership for a single new training campaign."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write(path, value):
    path=Path(path); temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,sort_keys=True)+'\n');temp.replace(path)


class RollingStore:
    def __init__(self, root, fingerprint, keep=2):
        self.root=Path(root).resolve(); self.keep=int(keep)
        if self.keep < 2 or self.root == Path('/'):
            raise ValueError('at least two recovery checkpoints required')
        self.root.mkdir(parents=True,exist_ok=True)
        self.ledger=self.root/'STORE.json'
        if self.ledger.exists():
            self.state=json.loads(self.ledger.read_text())
            if self.state['fingerprint'] != fingerprint or self.state['keep'] != self.keep:
                raise ValueError('checkpoint store ownership mismatch')
        else:
            if any(self.root.iterdir()):
                raise ValueError('refuse to adopt a nonempty unmanaged checkpoint directory')
            self.state={'fingerprint':fingerprint,'keep':self.keep,'checkpoints':[]}
            write(self.ledger,self.state)

    def committed(self, path, *, step, tokens, milestone=False):
        path=Path(path).resolve()
        if path.parent != self.root or path.is_symlink():
            raise ValueError('checkpoint outside owned store')
        meta=path.with_suffix('.json')
        receipt=json.loads(meta.read_text())
        actual=sha(path)
        if receipt['sha256'] != actual or receipt.get('fingerprint') != self.state['fingerprint']:
            raise ValueError('checkpoint receipt mismatch')
        entries=self.state['checkpoints']
        item=next((r for r in entries if r['path']==path.name),None)
        if item is None:
            item={'path':path.name,'sha256':actual,'bytes':path.stat().st_size,'step':step,'tokens':tokens,'milestone':False}
            entries.append(item)
        elif item['sha256'] != actual:
            raise ValueError('committed checkpoint changed')
        item['milestone'] = item['milestone'] or milestone
        entries.sort(key=lambda r:r['step'])
        protected={r['path'] for r in entries[-self.keep:]} | {r['path'] for r in entries if r['milestone']}
        # Persist the new recoverable checkpoint before retiring anything.
        write(self.ledger,self.state)
        for r in list(entries):
            if r['path'] in protected:continue
            old=self.root/r['path']
            if old.parent != self.root or old.is_symlink() or sha(old) != r['sha256']:
                raise ValueError('retirement target changed')
            with (self.root/'RETIREMENTS.jsonl').open('a') as log:
                log.write(json.dumps({**r,'replaced_by':path.name})+'\n')
            old.unlink(); old.with_suffix('.json').unlink(); entries.remove(r)
            write(self.ledger,self.state)


def enough_space(directory, *, reserve_bytes, checkpoint_bytes=700*1024**2):
    # Includes atomic-write overlap and emergency save headroom.
    return shutil.disk_usage(directory).free >= reserve_bytes + 2*checkpoint_bytes
