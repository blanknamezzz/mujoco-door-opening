"""Run decomposed-reward LoRA and a newly frozen independent physical test."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from prepare_hjepa_splits import assert_scene_disjoint
from run_hjepa_lora_experiment import hash_file, merge_shards, write_json


def run(args):
    from structural_scene import independent_test_scenes
    root=Path(__file__).resolve().parent
    os.chdir(root)
    if (args.output/'status.json').exists():
        raise FileExistsError('Preserve existing experiment; choose a new output directory.')
    args.output.mkdir(parents=True,exist_ok=True)
    args.test.mkdir(parents=True,exist_ok=True)
    specs=json.loads(json.dumps(independent_test_scenes(args.test_seed,4)))
    manifest=json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    if any(s['split']=='test' for s in manifest['scenes']):
        raise ValueError('Dedicated train/validation manifest required.')
    prior_specs=list(manifest['scenes'])
    for path in (Path('runs/hjepa/diverse_lora_data/manifest.json'),Path('runs/hjepa/independent_test_data/manifest.json'),
                 Path('runs/hjepa/reward_v2_test20261026/manifest.json')):
        if path.exists():
            prior_specs.extend(json.loads(path.read_text(encoding='utf-8'))['scenes'])
    assert_scene_disjoint(prior_specs,specs)
    initial_hash,original_hash=hash_file(args.initial),hash_file(args.original)
    baseline_path=args.baseline or args.initial
    baseline_hash=hash_file(baseline_path)
    status=dict(pid=os.getpid(),stage='training_and_test_collection',started_at=time.time(),updated_at=time.time(),
        steps=args.steps,lora_rank=8,original_core_frozen=True,reward_version='outcome_components_v2',
        trainval=str(args.data),evaluation_only_dataset=str(args.test),independent_test_scenes=48,
        independent_test_seed=args.test_seed,splits=dict(train=192,val=24),
        initial_checkpoint=str(args.initial),original_checkpoint=str(args.original),
        initial_sha256=initial_hash,original_sha256=original_hash,
        baseline_checkpoint=str(baseline_path),baseline_sha256=baseline_hash,
        phase='decision_calibration' if args.calibrate else 'joint_lora',
        joint_training_steps_this_phase=0 if args.calibrate else args.steps)
    write_json(args.output/'protocol.json',dict(status,scene_batch=args.scene_batch,batch_size=128,
        test_outcomes_used_for_selection=False,selection='validation correct decisions, then terminal utility, then fixed temporal loss',
        terminal_utility='execute: 2*success-1-collision; refuse: 0',
        baseline_step_zero=True,extra_policy_updates=0,
        prior_test_scenes='historical only; new seed and geometry for final testing',
        paper_basis=['https://arxiv.org/abs/2609.18651','https://arxiv.org/abs/2608.09762',
                     'https://arxiv.org/abs/2606.10305','https://arxiv.org/abs/2603.02115','https://arxiv.org/abs/2607.13033']))
    write_json(args.output/'test_scene_specs.json',dict(evaluation_only=True,scenes=specs))
    snapshot=args.output/'source_snapshot'
    snapshot.mkdir()
    sources={}
    for name in ('AGENTS.md','decomposed_reward.py','train_hjepa_decomposed.py','run_decomposed_experiment.py',
                 'hjepa_model.py','hjepa_lora.py','train_hjepa.py','collect_hjepa.py','structural_scene.py',
                 'evaluate_structural_training.py','run_hjepa.py','report_hjepa_holdout.py','calibrate_hjepa_decision.py'):
        shutil.copyfile(root/name,snapshot/name)
        sources[name]=hash_file(root/name)
    write_json(args.output/'source_provenance.json',sources)
    children=[]
    def launch(command,name):
        with (args.output/(name+'.log')).open('wb') as log:
            proc=subprocess.Popen([sys.executable,'-u',*command],stdout=log,stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        children.append((name,proc))
        return proc
    def heartbeat(stage=None):
        if stage: status['stage']=stage
        status.update(updated_at=time.time(),workers=[dict(name=n,pid=p.pid,exit_code=p.poll()) for n,p in children])
        write_json(args.output/'status.json',status)
    try:
        command=(['calibrate_hjepa_decision.py','--data',str(args.data),'--initial',str(args.initial),
                  '--baseline',str(baseline_path),'--output',str(args.output)] if args.calibrate else
                 ['train_hjepa_decomposed.py','--data',str(args.data),'--initial',str(args.initial),
                  '--output',str(args.output),'--steps',str(args.steps),'--scene-batch',str(args.scene_batch)])
        trainer=launch(command,'training')
        for i in range(args.jobs):
            shard=args.test/f'shard_{i}'
            if not (shard/'manifest.json').exists():
                launch(['collect_hjepa.py','--test-only','--seed',str(args.test_seed),'--output',str(shard),
                    '--shard-index',str(i),'--shard-count',str(args.jobs),'--resume'],'test_collection_'+str(i))
        frozen=False
        while any(p.poll() is None for _,p in children):
            failed=[(n,p.returncode) for n,p in children if p.poll() not in (None,0)]
            if failed: raise RuntimeError(f'Worker failed: {failed}')
            if trainer.poll()==0 and not frozen:
                write_json(args.output/'frozen_for_test.json',dict(selected_sha256=hash_file(args.output/'selected.pt'),
                    adapter_sha256=hash_file(args.output/'adapter.pt'),frozen_at=time.time()))
                frozen=True
            heartbeat('test_collection' if trainer.poll()==0 else 'training_and_test_collection')
            time.sleep(5)
        if any(p.returncode for _,p in children): raise RuntimeError('A worker failed.')
        if not frozen:
            write_json(args.output/'frozen_for_test.json',dict(selected_sha256=hash_file(args.output/'selected.pt'),
                adapter_sha256=hash_file(args.output/'adapter.pt'),frozen_at=time.time()))
        if hash_file(args.initial)!=initial_hash or hash_file(args.original)!=original_hash:
            raise AssertionError('A preserved checkpoint changed.')
        test=merge_shards(args.test,specs,args.jobs)
        assert_scene_disjoint(prior_specs,test['scenes'])
        if not test['evaluation_only'] or any(e['split']!='test' for e in test['episodes']):
            raise AssertionError('New test contains a training record.')
        write_json(args.test/'test_lock.json',dict(evaluation_only=True,scenes=48,records=len(test['episodes']),
            manifest_sha256=hash_file(args.test/'manifest.json'),disjoint_from_all_previous_scene_ids_and_geometry=True,
            test_outcomes_used_for_selection=False,locked_at=time.time()))
        evaluator=launch(['evaluate_structural_training.py','--data',str(args.test),'--model',str(args.output),
            '--baseline',str(baseline_path),'--reference',str(args.original),'--output',str(args.output/'independent_test'),
            '--skip-canonical','--device','cuda'],'evaluation')
        while evaluator.poll() is None:
            heartbeat('independent_test_evaluation')
            time.sleep(5)
        if evaluator.returncode: raise RuntimeError('Evaluation failed; inspect evaluation.log.')
        plotter=launch(['report_hjepa_holdout.py','--evaluation',str(args.output/'independent_test'),
            '--training',str(args.output/'training.json'),'--output',str(args.figure)],'report')
        if plotter.wait(): raise RuntimeError('Report rendering failed.')
        frozen_hash=json.loads((args.output/'frozen_for_test.json').read_text())
        replay=json.loads((args.output/'independent_test/replay.json').read_text())
        assert len(replay)==48 and all(x['selection_matches'] and x['outcome_matches'] for x in replay)
        if hash_file(args.output/'selected.pt')!=frozen_hash['selected_sha256'] or hash_file(args.output/'adapter.pt')!=frozen_hash['adapter_sha256']:
            raise AssertionError('Frozen model changed during test.')
        if hash_file(args.initial)!=initial_hash or hash_file(args.original)!=original_hash:
            raise AssertionError('A preserved checkpoint changed during test.')
        if hash_file(baseline_path)!=baseline_hash:
            raise AssertionError('Baseline changed during test.')
        status['finished_at']=time.time()
        heartbeat('complete')
    except BaseException as error:
        for _,p in children:
            if p.poll() is None: p.terminate()
        status['error']=repr(error)
        heartbeat('failed')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('runs/hjepa/diverse_lora_trainval'))
    p.add_argument('--initial',type=Path,default=Path('runs/hjepa/lora_continued_model/selected.pt'))
    p.add_argument('--original',type=Path,default=Path('runs/hjepa/structural_model/selected.pt'))
    p.add_argument('--output',type=Path,default=Path('runs/hjepa/reward_v2_model20261026'))
    p.add_argument('--test',type=Path,default=Path('runs/hjepa/reward_v2_test20261026'))
    p.add_argument('--figure',type=Path,default=Path('docs/hjepa_reward_decomposed.png'))
    p.add_argument('--test-seed',type=int,default=20261026)
    p.add_argument('--steps',type=int,default=600)
    p.add_argument('--scene-batch',type=int,default=8)
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--calibrate',action='store_true',help='Fit one validation threshold on a trained decomposed checkpoint')
    p.add_argument('--baseline',type=Path,help='Frozen scalar-policy baseline for calibrated evaluation')
    args=p.parse_args()
    if args.steps<1 or args.jobs<1 or args.jobs>8 or args.scene_batch<2 or args.scene_batch%2:
        p.error('Use positive steps, 1..8 collectors and an even scene batch >=2.')
    run(args)
