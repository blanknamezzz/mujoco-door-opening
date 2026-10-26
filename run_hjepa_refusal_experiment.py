"""Frozen calibration vs matched rank/BCE LoRA, with a new physical test."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from experiment_reproducibility import worker_environment,seed_everything
from structural_scene import independent_test_scenes
from prepare_hjepa_splits import assert_scene_disjoint
from door_side_reward import validation_key
from run_hjepa_lora_experiment import hash_file,write_json,merge_shards


def run(args):
    os.chdir(Path(__file__).resolve().parent)
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Fresh experiment required.')
    reproducibility=seed_everything(args.seed)
    development=json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    if development['split_counts']!=dict(train=576,val=96):raise ValueError('Fixed matched development corpus required.')
    specs=json.loads(json.dumps(independent_test_scenes(args.seed,8,stream=args.scene_stream)))
    prior=[];history=[]
    for path in Path('runs/hjepa').rglob('manifest.json'):
        manifest=json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('scenes'):prior.extend(manifest['scenes']);history.append(str(path))
    assert_scene_disjoint(prior,specs);assert_scene_disjoint(development['scenes'],specs)
    args.output.mkdir(parents=True);started=time.time()
    preserved={str(p):hash_file(p) for p in (args.initial,args.original,args.previous)}
    protocol=dict(seed=args.seed,reproducibility=reproducibility,scene_stream=args.scene_stream,train_scenes=576,validation_scenes=96,test_scenes=96,
        steps_per_trained_arm=args.steps,train_arms=dict(rank_raw='door-side',bce_raw='side-feasibility'),
        threshold_choices=['zero','one global validation-fitted refusal offset'],position_views='paired',position_encoder='backbone_patch_mean',
        original_core_frozen=True,lora_only=True,teacher_kl=.02,queue_seconds=0,
        preserved_checkpoints_sha256=preserved,development_manifest_sha256=hash_file(args.data/'manifest.json'),
        historical_manifests=history,prior_unique_scenes=len({s['scene_id'] for s in prior}),
        selection='same validation side/task/openings/utility tuple across six methods, ties prefer existing frozen method',
        test_used_for_selection=False,initial_checkpoint=str(args.initial),initial_checkpoint_sha256=hash_file(args.initial))
    write_json(args.output/'protocol.json',protocol);write_json(args.output/'pending_test_scenes.json',dict(scenes=specs))
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    files=[*Path('.').glob('*.py'),Path('AGENTS.md')]
    for file in files:shutil.copyfile(file,snapshot/file.name)
    write_json(args.output/'source_provenance.json',{p.name:hash_file(p) for p in files})
    children=[];phases={};status=dict(pid=os.getpid(),started_at=started,seed=args.seed,queue_seconds=0)
    def stage(name):
        status.update(stage=name,updated_at=time.time(),workers=[dict(name=n,pid=p.pid,exit_code=p.poll()) for n,p,_ in children])
        write_json(args.output/'status.json',status)
    def launch(name,command):
        tick=time.monotonic()
        with (args.output/(name+'.log')).open('wb') as log:
            p=subprocess.Popen([sys.executable,'-u',*command],stdout=log,stderr=subprocess.STDOUT,
                env=worker_environment(args.seed),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        children.append((name,p,tick));return p
    def wait(p,name):
        stage(name)
        if p.wait():raise RuntimeError(name+' failed; preserve logs and completed work.')
        phases[name]=time.monotonic()-next(t for _,q,t in children if q==p)
    def calibrate(name,initial):
        p=launch(name,['calibrate_hjepa_refusal.py','--data',str(args.data),'--initial',str(initial),
            '--output',str(args.output/name),'--seed',str(args.seed)])
        wait(p,name)
        return json.loads((args.output/name/'training.json').read_text())
    try:
        reports={};checkpoints={}
        reports['frozen_calibrated']=calibrate('frozen_calibrated',args.initial)
        reports['frozen_raw']=json.loads((args.initial.parent/'training.json').read_text())
        reports['frozen_raw']['selected_validation']=reports['frozen_calibrated']['decision_calibration']['baseline_validation']
        checkpoints['frozen_raw']=args.initial;checkpoints['frozen_calibrated']=args.output/'frozen_calibrated/selected.pt'
        for arm,objective in protocol['train_arms'].items():
            p=launch(arm,['train_hjepa_listwise.py','--data',str(args.data),'--initial',str(args.initial),
                '--output',str(args.output/arm),'--objective',objective,'--position-views','paired',
                '--position-encoder','backbone_patch_mean','--steps',str(args.steps),'--seed',str(args.seed)])
            wait(p,arm+'_training');reports[arm]=json.loads((args.output/arm/'training.json').read_text())
            checkpoints[arm]=args.output/arm/'selected.pt'
            calibrated=arm.replace('_raw','_calibrated')
            reports[calibrated]=calibrate(calibrated,checkpoints[arm]);checkpoints[calibrated]=args.output/calibrated/'selected.pt'
        order=('frozen_raw','frozen_calibrated','rank_raw','rank_calibrated','bce_raw','bce_calibrated')
        winner=max(order,key=lambda n:validation_key(reports[n]['selected_validation'],'door-side'))
        selected=checkpoints[winner]
        for name in ('best.pt','selected.pt'):shutil.copyfile(selected,args.output/name)
        shutil.copyfile(selected.parent/'adapter.pt',args.output/'adapter.pt')
        report=deepcopy(reports[winner]);report.update(refusal_study=protocol,selected_candidate=winner,
            validation_methods={n:reports[n]['selected_validation'] for n in order},method_order=list(order),
            methods_checkpoint_paths={n:str(checkpoints[n]) for n in order},
            newly_executed_updates=0 if winner.startswith('frozen') else args.steps,
            test_arrays_opened=0,selected_by_validation_only=True)
        write_json(args.output/'training.json',report)
        frozen=dict(selected_candidate=winner,frozen_at=time.time(),selected_sha256=hash_file(args.output/'selected.pt'),
            adapter_sha256=hash_file(args.output/'adapter.pt'),candidate_sha256={n:hash_file(checkpoints[n]) for n in order})
        write_json(args.output/'frozen_for_test.json',frozen)
        corpus=args.output/'test_corpus';corpus.mkdir();tick=time.monotonic()
        workers=[launch('collection_'+str(i),['collect_hjepa.py','--test-only','--endpoint-only',
            '--test-cases-per-family','8','--seed',str(args.seed),'--scene-stream',args.scene_stream,
            '--shard-index',str(i),'--shard-count',str(args.jobs),'--output',str(corpus/f'shard_{i}')]) for i in range(args.jobs)]
        stage('standalone_test_collection')
        for p in workers:
            if p.wait():raise RuntimeError('Collection failed; retain completed shards.')
        phases['standalone_test_collection']=time.monotonic()-tick
        manifest=merge_shards(corpus,specs,args.jobs)
        assert manifest['evaluation_only'] and len(manifest['scenes'])==96
        write_json(corpus/'test_lock.json',dict(manifest_sha256=hash_file(corpus/'manifest.json'),
            model_frozen_before_collection=frozen,seed=args.seed,scene_stream=args.scene_stream,never_evaluated=True))
        p=launch('test_views',['prepare_hjepa_views.py','--source',str(corpus),'--output',str(args.output/'test_views')]);wait(p,'test_sensor_preparation')
        receipt=dict(evaluation_started_at=time.time(),**frozen);write_json(corpus/'evaluation_receipt.json',receipt)
        reference_names=['original','decomposed','rank_raw','bce_raw','frozen_calibrated','rank_calibrated','bce_calibrated']
        references=[args.original,args.previous,*[checkpoints[n] for n in reference_names[2:]]]
        command=['evaluate_structural_training.py','--data',str(args.output/'test_views'),'--model',str(args.output),
            '--baseline',str(args.initial),'--output',str(args.output/'independent_test'),'--skip-canonical','--device','cuda']
        for path in references:command+=['--reference',str(path)]
        p=launch('evaluation',command);wait(p,'test_evaluation_and_selected_replay')
        # Verify two other frozen decision rules live. Never fit to this test.
        key_for={'frozen_raw':'old_checkpoint',**{name:f'reference_{i}' for i,name in enumerate(reference_names)}}
        controls=[]
        for name in ('frozen_calibrated','bce_raw','rank_raw','bce_calibrated'):
            if name!=winner and len(controls)<2:controls.append(name)
        for name in controls:
            p=launch(name+'_replay',['replay_hjepa_reference.py','--checkpoint',str(checkpoints[name]),
                '--report',str(args.output/'independent_test/heldout.json'),'--key',key_for[name],
                '--output',str(args.output/(name+'_live_replay'))]);wait(p,name+'_live_replay')
        p=launch('report',['report_hjepa_refusal.py','--experiment',str(args.output),'--output',str(args.figure)]);wait(p,'report')
        assert all(hash_file(Path(p))==h for p,h in preserved.items())
        assert hash_file(args.data/'manifest.json')==protocol['development_manifest_sha256']
        assert all(hash_file(checkpoints[n])==frozen['candidate_sha256'][n] for n in order)
        assert hash_file(args.output/'selected.pt')==frozen['selected_sha256'] and hash_file(args.output/'adapter.pt')==frozen['adapter_sha256']
        replay=json.loads((args.output/'independent_test/replay.json').read_text())
        assert len(replay)==96 and all(r['selection_matches'] and r['outcome_matches'] for r in replay)
        for n in controls:
            r=json.loads((args.output/(n+'_live_replay')/'verification.json').read_text());assert r['cases']==96 and r['all_actions_outcomes_rewards_match']
        receipt.update(evaluation_completed_at=time.time(),cases=96,physical_replays=288);write_json(corpus/'evaluation_receipt.json',receipt)
        write_json(args.output/'phase_timings.json',dict(queue_seconds=0,phase_seconds=phases,elapsed_seconds=time.time()-started))
        status.update(finished_at=time.time(),elapsed_seconds=time.time()-started,selected_candidate=winner,control_replays=controls);stage('complete')
        print(json.dumps(status),flush=True)
    except BaseException as e:
        for _,p,_ in children:
            if p.poll() is None:p.terminate()
        status['error']=repr(e);stage('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('runs/hjepa/coverage_corpus_seed42/expanded_policy'))
    p.add_argument('--output',type=Path,default=Path('runs/hjepa/refusal_model_seed42'))
    p.add_argument('--initial',type=Path,default=Path('runs/hjepa/pooling_model_seed42/selected.pt'))
    p.add_argument('--original',type=Path,default=Path('runs/hjepa/structural_model/selected.pt'))
    p.add_argument('--previous',type=Path,default=Path('runs/hjepa/reward_v2_calibrated20261026/selected.pt'))
    p.add_argument('--figure',type=Path,default=Path('docs/hjepa_refusal_seed42_results.png'))
    p.add_argument('--scene-stream',default='refusal_v1');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=240);p.add_argument('--jobs',type=int,default=3)
    a=p.parse_args()
    if a.steps<1 or not 1<=a.jobs<=6:p.error('Positive steps and1..6 workers required.')
    run(a)
