"""Serial, resumable queue with one deadline; no submission publication."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '.v3_deps'))
import psutil

OUT = ROOT / 'artifacts/v3/bounded_experiments'
QUEUE = [(c, f) for f in ('Dev1','Dev2','Dev3')
         for c in ('tabm_base','tabm_path','temporal_path','native_tree')]


def write(path, obj):
    tmp=path.with_suffix('.tmp.json')
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8');os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--hours',type=float,default=6.)
    args=parser.parse_args()
    if not 0 < args.hours <= 8: raise ValueError('Invalid budget')
    OUT.mkdir(parents=True,exist_ok=True)
    status_path=OUT/'queue_status.json'
    if status_path.exists():
        previous=json.loads(status_path.read_text())
        pid=previous.get('pid')
        if pid and psutil.pid_exists(pid):
            process=psutil.Process(pid)
            if process.create_time()==previous.get('process_created'):
                raise RuntimeError('Queue is already live')
    started=time.time();deadline=started+3600*args.hours
    state={'pid':os.getpid(),'process_created':psutil.Process().create_time(),
           'status':'running','started_unix':started,'deadline_unix':deadline,'jobs':[]}
    write(status_path,state)
    for candidate,fold in QUEUE:
        dest=OUT/f'{candidate}_{fold}'; summary=dest/'summary.json'
        if summary.exists():
            recorded=json.loads(summary.read_text())
            if recorded.get('status')=='complete':
                state['jobs'].append({'candidate':candidate,'fold':fold,'status':'already_complete'})
                continue
        child_status=dest/'status.json'
        if child_status.exists():
            old=json.loads(child_status.read_text())
            if old.get('status')=='running' and psutil.pid_exists(old.get('pid',-1)):
                process=psutil.Process(old['pid'])
                cmd=' '.join(process.cmdline())
                if 'run_bounded_experiments.py' in cmd and candidate in cmd and fold in cmd:
                    raise RuntimeError('An identical experiment is live')
        remaining=deadline-time.time()
        if remaining<=0:state['status']='budget_exhausted';break
        dest.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(ROOT/'analysis/v3/run_bounded_experiments.py'),
                 '--candidate',candidate,'--fold',fold,'--hours',str(min(8.,remaining/3600))]
        state['current']={'candidate':candidate,'fold':fold}
        write(status_path,state)
        print('START',candidate,fold,flush=True)
        with (dest/'run.log').open('w',encoding='utf-8') as log:
            try:
                with subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT) as process:
                    state['child_pid']=process.pid;write(status_path,state)
                    try:code=process.wait(timeout=max(1,deadline-time.time()))
                    except subprocess.TimeoutExpired:
                        process.terminate();process.wait(timeout=30)
                        state['status']='budget_exhausted';break
            except BaseException:
                state['status']='failed';write(status_path,state);raise
        state['jobs'].append({'candidate':candidate,'fold':fold,'status':'complete' if code==0 else 'failed','exit_code':code})
        print('END',candidate,fold,code,flush=True)
        if code != 0:
            state['status']='failed';break
    else:state['status']='complete'
    state['finished_unix']=time.time();state.pop('child_pid',None);state.pop('current',None)
    write(status_path,state)


if __name__=='__main__':main()
