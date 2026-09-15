"""Server-local health snapshots and hourly reports; no local-host dependency."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import time


def read(path, default=None):
    try:return json.loads(Path(path).read_text())
    except FileNotFoundError:return {} if default is None else default


def atomic(path, value):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    cfg=read(args.config);reports=Path(cfg['reports']);reports.mkdir(parents=True,exist_ok=True)
    campaign=Path(cfg['campaign']);active=read(campaign/'ACTIVE.json');status=read(campaign/'STATUS.json')
    run=Path(active['run_dir']) if active else None
    heartbeat=read(run/'heartbeat.json') if run else {}
    terminal=read(run/'terminal.json') if run else {}
    free=shutil.disk_usage(campaign if campaign.exists() else campaign.parent).free
    alerts=[]
    age=time.time()-(run/'heartbeat.json').stat().st_mtime if run and (run/'heartbeat.json').exists() else None
    if free<cfg.get('minimum_free_bytes',15*1024**3):alerts.append('disk_reserve_reached')
    if active and not terminal and age is not None and age>900:alerts.append('training_heartbeat_stale')
    if terminal and not terminal.get('ok'):alerts.append('training_process_failed')
    if run and not terminal and 'disk_reserve_reached' in alerts:
        (run/'STOP_REQUESTED').touch(exist_ok=True)
    store=read(Path(cfg['checkpoint_store'])/'STORE.json')
    checkpoints=store.get('checkpoints',[])
    last=max(checkpoints,key=lambda r:r['step']) if checkpoints else None
    sampler=read(Path(cfg['checkpoint_store'])/Path(last['path']).with_suffix('.json')) if last else {}
    try:
        gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader,nounits'],
                           capture_output=True,text=True,timeout=5).stdout.strip()
    except (OSError,subprocess.TimeoutExpired):gpu='unavailable'
    now=datetime.now(timezone.utc)
    tokens=(status.get('tokens_seen',0) if status.get('state')=='complete' else heartbeat.get('tokens_seen',status.get('tokens_seen',0))) or 0
    snapshot={'timestamp_utc':now.isoformat(),'state':terminal.get('state',status.get('state','preparing')),
        'run_dir':str(run) if run else None,'actual_tokens':tokens,'planned_tokens':cfg['planned_tokens'],
        'stage':heartbeat.get('stage'),'loss':heartbeat.get('loss'),
        'tokens_per_second':heartbeat.get('process_tokens_per_second'),'heartbeat_age_seconds':age,
        'disk_free_bytes':free,'gpu_util_memory_mib_temperature':gpu,'alerts':alerts,
        'latest_checkpoint':last,'source_tokens_at_latest_checkpoint':sampler.get('sampler',{}).get('source_tokens_drawn',{}),
        'local_computer_required':False}
    if active and not terminal:snapshot['state']='running' if heartbeat else 'starting'
    atomic(reports/'latest.json',snapshot)
    hour=now.strftime('%Y-%m-%dT%H');destination=reports/(hour+'.json')
    if not destination.exists():
        atomic(destination,snapshot)
        text=f"CPT {now.isoformat()}\n\n状态：{snapshot['state']}\n曝光：{tokens:,} / {cfg['planned_tokens']:,} token\n阶段：{snapshot['stage']}\nLoss：{snapshot['loss']}\n速度：{snapshot['tokens_per_second']} tok/s\n磁盘可用：{free/1024**3:.2f} GiB\n告警：{', '.join(alerts) or '无'}\n\n细来源曝光（最近有效断点，可能滞后于当前进度）：\n"
        text+=''.join(f"- {s}: {n:,}\n" for s,n in sorted(snapshot['source_tokens_at_latest_checkpoint'].items()))
        (reports/(hour+'.md')).write_text(text)
    # One week of hourly reports; latest remains available.
    for old in sorted(reports.glob('20??-??-??T??.json'))[:-168]:
        old.unlink();old.with_suffix('.md').unlink(missing_ok=True)
    print(json.dumps({k:snapshot[k] for k in ('state','actual_tokens','planned_tokens','disk_free_bytes','alerts')}))


if __name__=='__main__':main()
