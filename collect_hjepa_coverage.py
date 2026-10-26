"""Collect a fresh seed-42 cohort, seal tests, and prepare small/expanded corpora."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from structural_scene import diverse_structural_scenes
from prepare_hjepa_splits import assert_scene_disjoint,partition
from prepare_hjepa_coverage import compose
from run_hjepa_lora_experiment import hash_file,merge_shards,write_json
from experiment_reproducibility import worker_environment


def run(args):
    root=Path(__file__).resolve().parent;os.chdir(root)
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Fresh coverage cohort required.')
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.time();specs=json.loads(json.dumps(diverse_structural_scenes(args.seed,32,8,8)))
    prior=[];historical=[]
    for path in Path('runs/hjepa').glob('*/manifest.json'):
        m=json.loads(path.read_text(encoding='utf-8'))
        if m.get('scenes'):prior.extend(m['scenes']);historical.append(str(path))
    assert_scene_disjoint(prior,[dict(s,split='test') for s in specs])
    tests=[s for s in specs if s['split']=='test'];development=[s for s in specs if s['split']!='test']
    assert_scene_disjoint(development,tests)
    protocol=dict(seed=args.seed,counts=dict(train=384,val=96,test=96),jobs=args.jobs,queue_seconds=0,
        endpoint_only=True,historical_manifest_paths=historical,prior_unique_scenes=len({s['scene_id'] for s in prior}),
        test_used_for_fitting=False,old_source=str(args.old),old_source_sha256=hash_file(args.old/'manifest.json'))
    write_json(args.output/'protocol.json',protocol);write_json(args.output/'scene_specs.json',dict(scenes=specs))
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    names=['AGENTS.md','collect_hjepa_coverage.py','prepare_hjepa_coverage.py','experiment_reproducibility.py',
        'collect_hjepa.py','structural_scene.py','simulate.py','perception.py','hjepa_scene.py',
        'prepare_hjepa_views.py','position_observation.py','prepare_hjepa_splits.py']
    for n in names:shutil.copyfile(root/n,snapshot/n)
    write_json(args.output/'source_provenance.json',{n:hash_file(root/n) for n in names})
    children=[];status=dict(pid=os.getpid(),stage='starting',started_at=started,seed=args.seed,queue_seconds=0)
    def launch(name,command):
        with (args.output/(name+'.log')).open('wb') as f:
            p=subprocess.Popen([sys.executable,'-u',*command],stdout=f,stderr=subprocess.STDOUT,env=worker_environment(args.seed),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        children.append((name,p));return p
    def state(stage):
        status.update(stage=stage,updated_at=time.time(),workers=[dict(name=n,pid=p.pid,exit_code=p.poll()) for n,p in children])
        write_json(args.output/'status.json',status)
    try:
        workers=[launch('collection_'+str(i),['collect_hjepa.py','--diverse','--endpoint-only','--seed',str(args.seed),
            '--train-per-family','32','--val-per-family','8','--test-per-family','8',
            '--shard-index',str(i),'--shard-count',str(args.jobs),'--output',str(args.output/f'shard_{i}')]) for i in range(args.jobs)]
        state('physical_collection')
        for p in workers:
            if p.wait():raise RuntimeError('Collection failed; inspect worker log. Do not discard completed shards.')
        merged=merge_shards(args.output,specs,args.jobs)
        merged['historical_manifest_paths']=historical;write_json(args.output/'manifest.json',merged)
        partition(args.output,args.output/'development',('train','val'))
        sealed=partition(args.output,args.output/'sealed_test',('test',),evaluation_only=True)
        write_json(args.output/'sealed_test/test_lock.json',dict(seed=args.seed,test_scenes=96,
            manifest_sha256=hash_file(args.output/'sealed_test/manifest.json'),sealed_at=time.time(),never_evaluated=True,
            historical_manifest_paths=historical,disjoint_from_historical_and_development=True))
        prep=launch('development_views',['prepare_hjepa_views.py','--source',str(args.output/'development'),
            '--output',str(args.output/'development_views')]);state('development_sensor_preparation')
        if prep.wait():raise RuntimeError('Sensor preparation failed; preserve raw cohort.')
        new=args.output/'development_views'
        small=compose([(args.old,('train',)),(new,('val',))],args.output/'small_policy',args.seed)
        expanded=compose([(args.old,('train',)),(new,('train','val'))],args.output/'expanded_policy',args.seed)
        assert small['split_counts']==dict(train=192,val=96)
        assert expanded['split_counts']==dict(train=576,val=96)
        assert_scene_disjoint(expanded['scenes'],sealed['scenes'])
        assert hash_file(args.old/'manifest.json')==protocol['old_source_sha256']
        status.update(finished_at=time.time(),elapsed_seconds=time.time()-started,
            small_counts=small['split_counts'],expanded_counts=expanded['split_counts'])
        state('complete');print(json.dumps(status),flush=True)
    except BaseException as e:
        for _,p in children:
            if p.poll() is None:p.terminate()
        status['error']=repr(e);state('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('runs/hjepa/coverage_corpus_seed42'))
    p.add_argument('--old',type=Path,default=Path('runs/hjepa/paired_view_trainval_v3_20261026'))
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--jobs',type=int,default=3)
    a=p.parse_args()
    if a.jobs<1 or a.jobs>6:p.error('Use 1..6 local workers.')
    run(a)
