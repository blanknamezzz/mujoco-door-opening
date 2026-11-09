"""Collect real inspection RGB-D prefixes for train/validation map ablations."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timedelta
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from run_mobile_reward_experiment import ROOT,ZONE,write_json,file_hash,worker_init


PHYSICS=('scene.xml','simulate.py','mobile_door.py','mobile_crossed_v2.py','mobile_active_v3.py',
         'door_stress_scenes.py','perception.py','grasp_plan.py','prm.py','hjepa_scene.py')


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def collect_prefix(spec,development,output):
    from mobile_crossed_v2 import restore_scene,MotionDoor
    from mobile_door import reached_view
    from mobile_active_v3 import inspect
    from mobile_partial_map_v5 import PartialDoorMap,cad_prior
    begin=time.perf_counter();folder=Path(output)/'data'/spec['scene_id'];folder.mkdir(parents=True)
    old=Path(development)/'data'/spec['scene_id'];protected={}
    cached={}
    for branch in (0,1):
        path=old/str(branch)/'observations.npz';protected[str(path)]=file_hash(path)
        with np.load(path) as arrays:cached[branch]={k:arrays[k].copy() for k in (
            'initial_rgb','initial_proprio','post_rgb','post_proprio','post_view_available')}
    old_record_path=old/'1'/'records.json';protected[str(old_record_path)]=file_hash(old_record_path)
    old_scan=read(old_record_path)[0]['inspection']
    exp=MotionDoor(restore_scene(spec),spec['start_pose'],planner='fixed')
    q0,v0=exp.d.qpos.copy(),exp.d.qvel.copy();initial,rgb0,state0=reached_view(exp)
    np.testing.assert_array_equal(q0,exp.d.qpos);np.testing.assert_array_equal(v0,exp.d.qvel)
    np.testing.assert_allclose(state0,cached[0]['initial_proprio'],rtol=0,atol=1e-7)
    scan,rgb1,state1=inspect(exp)
    q1,v1=exp.d.qpos.copy(),exp.d.qvel.copy();post,post_rgb,post_state=reached_view(exp)
    np.testing.assert_array_equal(q1,exp.d.qpos);np.testing.assert_array_equal(v1,exp.d.qvel)
    np.testing.assert_allclose(state1,post_state,rtol=0,atol=0)
    np.testing.assert_allclose(state1,cached[1]['post_proprio'],rtol=0,atol=1e-7)
    rgb_diagnostics=[]
    for actual,previous in ((rgb0,cached[0]['initial_rgb']),(post_rgb,cached[1]['post_rgb'])):
        error=np.abs(actual.astype(int)-previous.astype(int))
        item=dict(max_pixel_error=int(error.max()),mean_channel_error=float(error.mean()),changed_channels=int((error>0).sum()))
        rgb_diagnostics.append(item)
        if item['mean_channel_error']>.01 or item['max_pixel_error']>3:
            raise AssertionError('Reached image differs from archived candidate prefix: '+str(item))
    for key in ('success','collision','unsafe_abort'):
        if scan[key]!=old_scan[key]:raise AssertionError('Inspection prefix status mismatch: '+key)
    for key in ('path_length_m','duration_s'):
        if abs(scan[key]-old_scan[key])>1e-7:raise AssertionError('Inspection prefix physical cost mismatch: '+key)
    np.savez_compressed(folder/'reached_rgbd.npz',
        **{'initial_'+k:v for k,v in initial.items()},**{'post_'+k:v for k,v in post.items()},
        initial_proprio=state0,post_proprio=state1,initial_qpos=q0,initial_qvel=v0,post_qpos=q1,post_qvel=v1)
    mapper=PartialDoorMap(cad_prior(restore_scene(spec)))
    map_start=time.perf_counter();mapper.add_frame(initial,state0);before=mapper.query()
    # A failed inspection cannot provide a newly available policy observation.
    if scan['success']:mapper.add_frame(post,state1)
    after=mapper.query()
    np.testing.assert_array_equal(q1,exp.d.qpos);np.testing.assert_array_equal(v1,exp.d.qvel)
    assert exp.reset_count==1
    result=dict(scene_id=spec['scene_id'],group_id=spec['group_id'],split=spec['split'],seed=42,
        pid=os.getpid(),initial=before,fused=after,frame_statistics=mapper.frame_stats,
        inspection=scan,prefix_rgb_comparison=rgb_diagnostics,base_reset_count=exp.reset_count,
        original_candidate_prefix_states_match=True,queries_preserve_live_state=True,
        raw_rgbd_sha256=file_hash(folder/'reached_rgbd.npz'),protected_input_sha256=protected,
        map_query_seconds=time.perf_counter()-map_start,seconds=time.perf_counter()-begin)
    write_json(folder/'maps.json',result);write_json(folder/'trace.json',exp.trace)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--development',type=Path,default=ROOT/'runs/mobile_active_seed42_20261109')
    p.add_argument('--groups-per-split',type=int)
    p.add_argument('--splits',nargs='+',default=['train','val'],choices=['train','val'])
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--estimate-seconds',type=int,required=True)
    args=p.parse_args();args.output=args.output.resolve();args.development=args.development.resolve()
    if (args.output/'receipt.json').exists():raise ValueError('Use a new ignored run directory.')
    begin=time.perf_counter();now=datetime.now(ZONE)
    args.output.mkdir(parents=True,exist_ok=True)
    from experiment_reproducibility import seed_everything
    reproducibility=seed_everything(42)
    previous=read(args.development/'source_hashes.json')
    for name in PHYSICS:
        if file_hash(ROOT/name)!=previous[name]:raise ValueError('Archived development physics differs: '+name)
    specs=[s for s in read(args.development/'manifest.json')['scenes'] if s['split'] in args.splits]
    if args.groups_per_split is not None:
        keep=set()
        for split in args.splits:
            groups=sorted({s['group_id'] for s in specs if s['split']==split})
            if args.groups_per_split>len(groups):raise ValueError('Insufficient development groups.')
            keep.update(groups[:args.groups_per_split])
        specs=[s for s in specs if s['group_id'] in keep]
    if not specs:raise ValueError('Empty collection.')
    write_json(args.output/'manifest.json',dict(seed=42,scenes=specs,scope='train/validation only; no test outcomes'))
    snapshot=args.output/'source';snapshot.mkdir();hashes={}
    for name in (*PHYSICS,'mobile_sweep_v4.py','mobile_partial_map_v5.py','collect_mobile_partial_maps.py','test_mobile_partial_map_v5.py'):
        hashes[name]=file_hash(ROOT/name);shutil.copy2(ROOT/name,snapshot/name)
    write_json(args.output/'source_hashes.json',hashes)
    receipt=dict(pid=os.getpid(),seed=42,status='running',started_at=now.isoformat(),
        runtime_estimate_seconds=args.estimate_seconds,queue_estimate_seconds=0,
        expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat(),
        scenes=len(specs),workers=args.workers,collection_only=True,gpu_training_concurrent=False,
        reproducibility=reproducibility)
    write_json(args.output/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
    results=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=worker_init) as pool:
        futures=[pool.submit(collect_prefix,s,args.development,args.output) for s in specs]
        for f in as_completed(futures):
            result=f.result();results.append(result)
            progress=dict(pid=os.getpid(),phase='reached_rgbd_prefix_collection',completed_scenes=len(results),
                total_scenes=len(specs),recent_seconds=result['seconds'],recent_map_seconds=result['map_query_seconds'],
                elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json',progress);print(json.dumps(progress),flush=True)
    protected={k:v for r in results for k,v in r['protected_input_sha256'].items()}
    if any(file_hash(path)!=h for path,h in protected.items()):raise AssertionError('Frozen development input changed.')
    report=dict(seed=42,scenes=len(results),results=results,original_inputs_unchanged=True,
        test_outcome_files_opened=0,seconds=time.perf_counter()-begin,
        map_query_seconds=sum(r['map_query_seconds'] for r in results),
        collection_only=True,gpu_training_concurrent=False)
    write_json(args.output/'report.json',report)
    receipt.update(status='complete',exit_code=0,actual_seconds=report['seconds'],completed_at=datetime.now(ZONE).isoformat())
    write_json(args.output/'receipt.json',receipt)
    print(json.dumps(dict(status='complete',seconds=report['seconds'],scenes=len(results))),flush=True)


if __name__=='__main__':
    try:main()
    except Exception:
        import traceback
        import sys
        if '--output' in sys.argv:
            receipt_path=Path(sys.argv[sys.argv.index('--output')+1])/'receipt.json'
            if receipt_path.exists():
                failed=read(receipt_path);failed.update(status='failed',exit_code=1,traceback=traceback.format_exc())
                write_json(receipt_path,failed)
        traceback.print_exc();raise
