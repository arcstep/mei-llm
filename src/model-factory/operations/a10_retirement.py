"""Inventory explicitly retired A10 experiments. No deletion in this helper.

Run on the server with Python's standard library; stream its JSON to the local
evidence directory. Final weights, latest task optimizer states, small evidence
and source bytes are retained. Historical intermediate probes/raw caches are
listed separately under the user's explicit experiment-retirement instruction.
"""
import hashlib
import json
from pathlib import Path
import re
import os
import shutil
import sys


def main():
    root = Path('/root/mei-vocab-study')
    targets = sorted(p for p in root.iterdir() if
                     re.fullmatch(r'workspace-v\d+(?:\.tar\.gz)?',p.name) or
                     p.name.startswith('assets-'))
    records = []
    newest = {}
    for target in targets:
        paths = target.rglob('*') if target.is_dir() else [target]
        for p in paths:
            if not p.is_file() or p.is_symlink():
                continue
            stat = p.stat()
            rel = str(p.relative_to(root))
            row = {'path':rel,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
            keep = ((stat.st_size <= 2*1024**2 and '__pycache__' not in p.parts)
                    or p.suffix in ('.npz','.safetensors','.model','.vocab')
                    or '/source/' in rel or '/src/' in rel)
            row['retain'] = keep
            records.append(row)
            if p.suffix == '.pt' and any(s in rel for s in
                    ('/cpt-audited','/qat-cq2-audited','/sft-bootstrap-audited','/float-task-control-audited')):
                previous = newest.get(str(p.parent))
                if previous is None or p.name > Path(previous['path']).name:
                    newest[str(p.parent)] = row
    for row in newest.values():
        row['retain'] = True
    for row in records:
        if row['retain']:
            with (root/row['path']).open('rb') as f:
                row['sha256'] = hashlib.file_digest(f,'sha256').hexdigest()
    print(json.dumps({'schema':'mei-a10-retirement-plan-v1','root':str(root),
          'targets':[p.name for p in targets], 'files':records,
          'bytes':sum(r['bytes'] for r in records),
          'retained_bytes':sum(r['bytes'] for r in records if r['retain']),
          'deletion_authorized_by':'2026-09-14 user: previous 300-base experiments all cleared',
          'deletion_executed':False},ensure_ascii=False))


def retire(plan_path, proof_path, receipt_path):
    plan_path=Path(plan_path);plan=json.loads(plan_path.read_text())
    proof=json.loads(Path(proof_path).read_text())
    if not proof.get('ok') or hashlib.sha256(plan_path.read_bytes()).hexdigest()!=proof['plan_sha256']:
        raise ValueError('verified backup does not bind this deletion plan')
    expected={(r['path'],r['sha256']) for r in plan['files'] if r['retain']}
    if expected!={(r['remote_path'],r['sha256']) for r in proof['backups']}:
        raise ValueError('retained backup coverage incomplete')
    root=Path(plan['root'])
    if root!=Path('/root/mei-vocab-study'):raise ValueError('unexpected retirement root')
    targets=[root/n for n in plan['targets']]
    for target in targets:
        if target.parent!=root or target.is_symlink() or not (
            re.fullmatch(r'workspace-v\d+(?:\.tar\.gz)?',target.name) or target.name.startswith('assets-')):
            raise ValueError('unapproved retirement target')
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            cwd=(proc/'cwd').resolve(strict=True)
            if any(cwd==t or cwd.is_relative_to(t) for t in targets):
                raise ValueError('live process uses retired workspace: '+proc.name)
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    actual=set()
    for t in targets:
        for p in (t.rglob('*') if t.is_dir() else [t]):
            if p.is_file() and not p.is_symlink():actual.add(str(p.relative_to(root)))
    if actual!={r['path'] for r in plan['files']}:raise ValueError('retirement inventory changed')
    for r in plan['files']:
        st=(root/r['path']).stat()
        if st.st_size!=r['bytes'] or st.st_mtime_ns!=r['mtime_ns']:
            raise ValueError('retirement bytes changed: '+r['path'])
    before=shutil.disk_usage(root)._asdict()
    for t in targets:
        if t.is_dir():shutil.rmtree(t)
        else:t.unlink()
    after=shutil.disk_usage(root)._asdict()
    receipt={'ok':True,'deleted_targets':plan['targets'],'logical_bytes_removed':plan['bytes'],
             'before':before,'after':after,'backup_plan_sha256':proof['plan_sha256'],
             'all_targets_absent':all(not p.exists() for p in targets)}
    Path(receipt_path).write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='deleted_targets'}))


if __name__ == '__main__':
    if len(sys.argv)==5 and sys.argv[1]=='retire':retire(*sys.argv[2:])
    elif len(sys.argv)==1:main()
    else:raise SystemExit('usage: inventory on stdin, or retire PLAN BACKUP_PROOF RECEIPT')
