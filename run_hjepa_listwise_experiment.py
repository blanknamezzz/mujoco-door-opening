"""Two controlled P-policy arms and a fresh, frozen physical test suite."""
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
from door_side_reward import reward_contract,validation_key
from experiment_reproducibility import worker_environment


def run(args):
    from structural_scene import independent_test_scenes
    root=Path(__file__).resolve().parent
    os.chdir(root)
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Fresh experiment directory required.')
    if args.test.exists() and any(args.test.iterdir()):
        raise FileExistsError('Fresh test directory required.')
    if args.view_ablation:
        development=json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
        if set(development.get('position_rear_files',{}))!={s['scene_id'] for s in development['scenes']}:
            raise ValueError('Prepare complete current rear-camera observations before a sensor ablation.')
    args.output.mkdir(parents=True,exist_ok=True); args.test.mkdir(parents=True,exist_ok=True)
    arms=('single_view','paired_view') if args.view_ablation else ('head_only','position_lora')
    specs=json.loads(json.dumps(independent_test_scenes(args.seed,4)))
    prior=[]; prior_manifests=[]
    # Check every previous top-level dataset, including the most recent test.
    for path in sorted(Path('runs/hjepa').glob('*/manifest.json')):
        manifest=json.loads(path.read_text(encoding='utf-8'))
        if manifest.get('scenes'):
            prior.extend(manifest['scenes']); prior_manifests.append(str(path))
    assert_scene_disjoint(prior,specs)
    preserved={str(p):hash_file(p) for p in (args.initial,args.original,args.previous)}
    write_json(args.output/'protocol.json',dict(seed=args.seed,test_scenes=48,steps_per_arm=args.steps,
        official_native_architecture=True,full_model_training=False,original_core_frozen=True,
        arms=list(arms),objective=args.objective,view_ablation=args.view_ablation,endpoint_only_test=args.endpoint_only,
        selection=('validation correct side decisions, correct task decisions, openings, terminal utility; ties prefer '+('single-view' if args.view_ablation else 'head control')
            if args.objective=='door-side' else 'validation correct decisions, openings, terminal utility; ties prefer head control'),
        door_side_reward=reward_contract() if args.objective=='door-side' else None,
        trainval=str(args.data),target=('measured side feasibility plus within-side base quality' if args.objective=='door-side' else
            'any successful physical base under initial fixed reachable grasp selector; else refusal'),
        prior_manifest_paths=prior_manifests,test_used_for_selection=False,preserved_file_sha256=preserved))
    write_json(args.output/'test_scene_specs.json',dict(evaluation_only=True,scenes=specs))
    snapshot=args.output/'source_snapshot'; snapshot.mkdir()
    source_names=['AGENTS.md','train_hjepa_listwise.py','run_hjepa_listwise_experiment.py','door_side_reward.py',
        'position_observation.py','prepare_hjepa_views.py',
        'hjepa_model.py','hjepa_lora.py','train_hjepa.py','evaluate_structural_training.py',
        'structural_scene.py','collect_hjepa.py','run_hjepa.py','report_hjepa_holdout.py']
    for name in source_names: shutil.copyfile(root/name,snapshot/name)
    write_json(args.output/'source_provenance.json',{n:hash_file(root/n) for n in source_names})
    status=dict(pid=os.getpid(),stage='starting',started_at=time.time(),steps_per_arm=args.steps,
        test_seed=args.seed,original_core_frozen=True,objective=args.objective)
    children=[]
    def launch(name,command):
        with (args.output/(name+'.log')).open('wb') as log:
            proc=subprocess.Popen([sys.executable,'-u',*command],stdout=log,stderr=subprocess.STDOUT,
                env=worker_environment(args.seed),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        children.append((name,proc)); return proc
    def heartbeat(stage):
        status.update(stage=stage,updated_at=time.time(),workers=[dict(name=n,pid=p.pid,exit_code=p.poll()) for n,p in children])
        write_json(args.output/'status.json',status)
    def wait(proc,stage):
        while proc.poll() is None:
            failed=[n for n,p in children if p.poll() not in (None,0)]
            if failed:raise RuntimeError('Failed workers: '+str(failed))
            heartbeat(stage); time.sleep(5)
        if proc.returncode:raise RuntimeError(stage+' failed')
    try:
        control=launch(arms[0],['train_hjepa_listwise.py','--data',str(args.data),'--initial',str(args.initial),
            '--output',str(args.output/arms[0]),'--steps',str(args.steps),'--objective',args.objective,
            '--seed',str(args.seed),
            *(['--position-views','repeat-front'] if args.view_ablation else ['--head-only'])])
        for i in range(3):
            launch('test_collection_'+str(i),['collect_hjepa.py','--test-only','--seed',str(args.seed),
                '--output',str(args.test/f'shard_{i}'),'--shard-index',str(i),'--shard-count','3',
                *(['--endpoint-only'] if args.endpoint_only else [])])
        wait(control,'single_view_lora_and_collection' if args.view_ablation else 'head_control_and_collection')
        lora=launch(arms[1],['train_hjepa_listwise.py','--data',str(args.data),'--initial',str(args.initial),
            '--output',str(args.output/arms[1]),'--steps',str(args.steps),'--objective',args.objective,
            '--seed',str(args.seed),
            '--position-views','paired' if args.view_ablation else 'single'])
        wait(lora,'paired_view_lora_and_collection' if args.view_ablation else 'position_lora_and_collection')
        reports={n:json.loads((args.output/n/'training.json').read_text()) for n in arms}
        def key(n):
            s=reports[n]['selected_validation']
            return validation_key(s,args.objective)
        winner=max(arms,key=key)
        for file in ('best.pt','selected.pt','adapter.pt'):
            shutil.copyfile(args.output/winner/file,args.output/file)
        report=reports[winner]
        report.update(selected_arm=winner,arm_summaries={n:r['selected_validation'] for n,r in reports.items()},
            joint_training_steps_this_phase=args.steps,selection_across_arms=True)
        if args.view_ablation:
            report['comparison_arms']=[dict(name=arms[0],label='Front x2',reference='reference_2'),
                dict(name=arms[1],label='Front+rear',reference='reference_3')]
            report['sensor_ablation']=dict(control='repeated front RGB, two encoder passes',treatment='front and rear RGB, two encoder passes',
                same_update_budget=True,same_initialization=True,no_extra_parameters=True)
        write_json(args.output/'training.json',report)
        write_json(args.output/'frozen_for_test.json',dict(selected_sha256=hash_file(args.output/'selected.pt'),
            adapter_sha256=hash_file(args.output/'adapter.pt'),selected_arm=winner,frozen_at=time.time()))
        for name,p in list(children):
            if name.startswith('test_collection'):wait(p,'test_collection')
        manifest=merge_shards(args.test,specs,3)
        assert_scene_disjoint(prior,manifest['scenes'])
        assert manifest['evaluation_only'] and all(e['split']=='test' for e in manifest['episodes'])
        evaluation_data=args.test
        if args.view_ablation:
            prep=launch('test_sensor_preparation',['prepare_hjepa_views.py','--source',str(args.test),'--output',str(args.test/'views')])
            wait(prep,'test_sensor_preparation')
            evaluation_data=args.test/'views'
        write_json(args.test/'test_lock.json',dict(manifest_sha256=hash_file(args.test/'manifest.json'),
            evaluation_manifest=str(evaluation_data/'manifest.json'),evaluation_manifest_sha256=hash_file(evaluation_data/'manifest.json'),
            test_used_for_selection=False,disjoint_from_all_previous_scene_ids_and_geometry=True,
            prior_unique_scene_count=len({s['scene_id'] for s in prior}),locked_at=time.time()))
        evaluation=launch('evaluation',['evaluate_structural_training.py','--data',str(evaluation_data),
            '--model',str(args.output),'--baseline',str(args.initial),'--reference',str(args.original),
            '--reference',str(args.previous),'--reference',str(args.output/arms[0]/'selected.pt'),
            '--reference',str(args.output/arms[1]/'selected.pt'),
            *(['--reference',str(args.output/arms[1]/'initial_adapter.pt')] if args.view_ablation else []),
            '--output',str(args.output/'independent_test'),'--skip-canonical','--device','cuda'])
        wait(evaluation,'independent_test_evaluation')
        plot=launch('report',['report_hjepa_holdout.py','--evaluation',str(args.output/'independent_test'),
            '--training',str(args.output/'training.json'),'--output',str(args.figure)])
        wait(plot,'report')
        replay=json.loads((args.output/'independent_test/replay.json').read_text())
        assert len(replay)==48 and all(x['selection_matches'] and x['outcome_matches'] for x in replay)
        frozen=json.loads((args.output/'frozen_for_test.json').read_text())
        assert hash_file(args.output/'selected.pt')==frozen['selected_sha256']
        assert hash_file(args.output/'adapter.pt')==frozen['adapter_sha256']
        assert all(hash_file(Path(p))==sha for p,sha in preserved.items())
        status.update(finished_at=time.time(),selected_arm=winner,elapsed_seconds=time.time()-status['started_at'])
        heartbeat('complete')
    except BaseException as e:
        for _,p in children:
            if p.poll() is None:p.terminate()
        status['error']=repr(e);heartbeat('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('runs/hjepa/diverse_lora_trainval'))
    p.add_argument('--initial',type=Path,default=Path('runs/hjepa/lora_continued_model/selected.pt'))
    p.add_argument('--original',type=Path,default=Path('runs/hjepa/structural_model/selected.pt'))
    p.add_argument('--previous',type=Path,default=Path('runs/hjepa/reward_v2_calibrated20261026/selected.pt'))
    p.add_argument('--output',type=Path)
    p.add_argument('--test',type=Path)
    p.add_argument('--figure',type=Path)
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--seed',type=int)
    p.add_argument('--objective',choices=('door-side','successful-set'),default='door-side')
    p.add_argument('--view-ablation',action='store_true',help='Both arms use LoRA; single vs paired fixed sensors')
    p.add_argument('--endpoint-only',action='store_true',help='Collect physical test results without temporal video arrays')
    args=p.parse_args()
    side=args.objective=='door-side'
    args.output=args.output or Path('runs/hjepa/paired_view_model20261026' if args.view_ablation else 'runs/hjepa/side_reward_model20261026' if side else 'runs/hjepa/listwise_model20261026')
    args.seed=args.seed if args.seed is not None else 42
    args.test=args.test or Path(f'runs/hjepa/side_reward_test{args.seed}' if side else f'runs/hjepa/listwise_test{args.seed}')
    args.figure=args.figure or Path('docs/hjepa_paired_view_results.png' if args.view_ablation else 'docs/hjepa_side_reward_results.png' if side else 'docs/hjepa_listwise_results.png')
    if args.steps<1:p.error('Positive steps required.')
    run(args)
