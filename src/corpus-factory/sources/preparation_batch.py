"""Sequential approved preparation jobs, with persistent logs and receipts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from profiling import ROOT,digest


def run(config,out):
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    (out/'implementation.py.snapshot').write_bytes(Path(__file__).read_bytes())
    prerequisite=config.get('wait_for_process')
    if prerequisite:
        while True:
            p=subprocess.run(['ps','-p',str(prerequisite['pid']),'-o','command='],capture_output=True,text=True)
            if p.returncode or prerequisite['command_contains'] not in p.stdout:break
            time.sleep(5)
    jobs=[]
    for i,job in enumerate(config['jobs']):
        if job.get('requires') and not (ROOT/job['requires']).is_file():
            jobs.append({'job':job,'status':'blocked_missing_input'});continue
        recipe=ROOT/job['config'];log=out/f'{i:02d}-{job["name"]}.log'
        args=[sys.executable,'-m','mei_llm','corpus','source',job['action'],'--config',str(recipe),'--out',str(ROOT/job['out'])]
        if job.get('network'):args.append('--allow-network')
        if job.get('resume_from'):args+=['--resume-from',str(ROOT/job['resume_from'])]
        start=time.time()
        with log.open('x') as f:
            process=subprocess.run(args,cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT/'src')},stdout=f,stderr=subprocess.STDOUT)
        result={'job':job,'exit_code':process.returncode,'seconds':time.time()-start,'log':str(log),'config_sha256':digest(recipe),'status':'process_complete' if process.returncode==0 else 'failed'}
        (out/f'{i:02d}-receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));jobs.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    report={'schema':'mei-approved-preparation-batch-v1','jobs':jobs,'release':False}
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    return report
