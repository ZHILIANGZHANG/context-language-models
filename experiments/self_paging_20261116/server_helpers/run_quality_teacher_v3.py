import fcntl,hashlib,json,os,pathlib,signal,subprocess,time,traceback
H=pathlib.Path('/home/zhangcheng/self-paging-20261116')
C=pathlib.Path('/data/zhangcheng/context-language-models-797680f-quality')
P='/data/zhangcheng/self-paging-20261116/venv/bin/python'
R=H/'quality-teacher-v3'
def write(p,x):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def ident(pid):
    p=pathlib.Path('/proc')/str(pid)
    return dict(pid=pid,uid=p.stat().st_uid,start_ticks=int((p/'stat').read_text().rsplit(')',1)[1].split()[19]),cwd=str((p/'cwd').resolve()))
def free():
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
    return {int(i) for i,u,m,v in (r.split(',') for r in rows) if int(m)<500 and int(v)==0 and u.strip() not in apps}
def main():
    assert not (H/'STOP').exists() and not (H/'postpilot/STOP').exists()
    assert (H/'mask-recovery/user-stopped-quality-review.json').exists()
    assert (H/'quality-review/sample-summary.json').exists()
    campaign=open(H/'campaign.lock','a');fcntl.flock(campaign,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=open(H/'quality-teacher-v3.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not R.exists()
    import xml.etree.ElementTree as ET
    suites=ET.parse(H/"quality-cpu.xml").getroot().findall("testsuite")
    assert sum(int(s.get("tests")) for s in suites)==73
    assert all(int(s.get(k,0))==0 for s in suites for k in ["failures","errors","skipped"])
    m=json.loads((H/'quality-source-manifest.json').read_text())
    assert all(hashlib.sha256((C/k).read_bytes()).hexdigest()==v for k,v in m.items())
    a=free();time.sleep(3);g=sorted(a&free());assert g,'No idle card; nothing launched'
    R.mkdir();write(R/'identity.json',ident(os.getpid()))
    env=os.environ|{'PYTHONPATH':str(C)+':'+str(H),'CUDA_VISIBLE_DEVICES':str(g[0]),'OMP_NUM_THREADS':'4'}
    with (R/'stdout.log').open('w') as f:
        p=subprocess.Popen([P,str(H/'quality_teacher_v3.py')],cwd=C,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        identity=ident(p.pid);write(R/'launch.json',dict(identity=identity,gpu=g[0],time=time.time()))
        try:
            while p.poll() is None:
                if (H/'STOP').exists() or (R/'STOP').exists() or (H/'quality-review/STOP').exists():raise RuntimeError('User STOP')
                time.sleep(3)
        except BaseException:
            if p.poll() is None and ident(p.pid)==identity:os.killpg(p.pid,signal.SIGTERM)
            p.wait();raise
        write(R/'exit.json',dict(returncode=p.returncode,time=time.time(),identity=identity))
        if p.returncode:raise RuntimeError('Witness failed; preserve evidence, no retry')
if __name__=='__main__':
    try:main()
    except BaseException:
        if R.exists():write(R/'error.json',dict(time=time.time(),traceback=traceback.format_exc()))
        raise
