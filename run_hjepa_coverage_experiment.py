"""Fixed-seed paired-view LoRA: 192 vs 576 training scenes, new 96/96 val/test."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from experiment_reproducibility import worker_environment
from prepare_hjepa_splits import assert_scene_disjoint
from run_hjepa_lora_experiment import hash_file,write_json


def selection_key(summary):
    return (summary['correct_side_decisions'],summary['correct_task_decisions'],
        summary['opening_successes'],summary['mean_terminal_utility_v2'])


def run(args):
    root=Path(__file__).resolve().parent;os.chdir(root)
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Fresh model experiment required.')
    collection=json.loads((args.corpus/'status.json').read_text())
    if collection['stage']!='complete':raise ValueError('Finish and verify corpus collection first.')
    sealed=args.corpus/'sealed_test';receipt=sealed/'evaluation_receipt.json'
    if receipt.exists():raise ValueError('This test has already been exposed; preserve it and collect another cohort.')
    lock=json.loads((sealed/'test_lock.json').read_text())
    if hash_file(sealed/'manifest.json')!=lock['manifest_sha256']:raise ValueError('Sealed test hash changed.')
    small=json.loads((args.corpus/'small_policy/manifest.json').read_text())
    expanded=json.loads((args.corpus/'expanded_policy/manifest.json').read_text())
    tests=json.loads((sealed/'manifest.json').read_text())
    assert small['split_counts']==dict(train=192,val=96) and expanded['split_counts']==dict(train=576,val=96)
    assert [s for s in small['scenes'] if s['split']=='val']==[s for s in expanded['scenes'] if s['split']=='val']
    assert_scene_disjoint(expanded['scenes'],tests['scenes'])
    args.output.mkdir(parents=True,exist_ok=True)
    started=time.time();arms=('small_data','expanded_data')
    preserved={str(p):hash_file(p) for p in (args.initial,args.original,args.previous,args.previous_paired)}
    data_hashes={str(args.corpus/n/'manifest.json'):hash_file(args.corpus/n/'manifest.json') for n in ('small_policy','expanded_policy','sealed_test')}
    protocol=dict(seed=args.seed,steps_per_arm=args.steps,arms=list(arms),queue_seconds=0,
        training_scenes=dict(small_data=192,expanded_data=576),validation_scenes=96,test_scenes=96,
        objective='door-side',position_views='paired',lora_rank=8,original_core_frozen=True,
        selection='validation correct side decisions, task decisions, openings, terminal utility; ties prefer small-data control',
        test_used_for_selection=False,preserved_file_sha256=preserved,data_manifest_sha256=data_hashes,
        corpus=str(args.corpus),historical_manifest_paths=lock['historical_manifest_paths'])
    write_json(args.output/'protocol.json',protocol)
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    names=['AGENTS.md','run_hjepa_coverage_experiment.py','train_hjepa_listwise.py','experiment_reproducibility.py',
        'door_side_reward.py','position_observation.py','prepare_hjepa_views.py','train_hjepa.py',
        'hjepa_model.py','hjepa_lora.py','evaluate_structural_training.py','run_hjepa.py','report_hjepa_holdout.py',
        'structural_scene.py','collect_hjepa.py','simulate.py','perception.py','hjepa_scene.py']
    for n in names:shutil.copyfile(root/n,snapshot/n)
    write_json(args.output/'source_provenance.json',{n:hash_file(root/n) for n in names})
    status=dict(pid=os.getpid(),stage='starting',started_at=started,seed=args.seed,queue_seconds=0)
    children=[]
    def launch(name,command):
        with (args.output/(name+'.log')).open('wb') as log:
            p=subprocess.Popen([sys.executable,'-u',*command],stdout=log,stderr=subprocess.STDOUT,
                env=worker_environment(args.seed),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        children.append((name,p));return p
    def stage(name):
        status.update(stage=name,updated_at=time.time(),workers=[dict(name=n,pid=p.pid,exit_code=p.poll()) for n,p in children])
        write_json(args.output/'status.json',status)
    def wait(p,name):
        stage(name)
        if p.wait():raise RuntimeError(name+' failed; inspect its log and preserve artifacts.')
    try:
        reports={}
        for arm,corpus_name in zip(arms,('small_policy','expanded_policy')):
            p=launch(arm,['train_hjepa_listwise.py','--data',str(args.corpus/corpus_name),'--initial',str(args.initial),
                '--output',str(args.output/arm),'--objective','door-side','--position-views','paired',
                '--steps',str(args.steps),'--seed',str(args.seed)])
            wait(p,arm+'_training');reports[arm]=json.loads((args.output/arm/'training.json').read_text())
        winner=max(arms,key=lambda n:selection_key(reports[n]['selected_validation']))
        for filename in ('best.pt','selected.pt','adapter.pt'):
            shutil.copyfile(args.output/winner/filename,args.output/filename)
        report=reports[winner]
        report.update(selected_arm=winner,arm_summaries={n:r['selected_validation'] for n,r in reports.items()},
            selection_across_arms=True,comparison_arms=[dict(name=arms[0],label='192 scenes',reference='reference_2'),
                dict(name=arms[1],label='576 scenes',reference='reference_3')],
            coverage_ablation=dict(seed=args.seed,new_validation_scenes=96,sealed_test_scenes=96,
                same_update_budget=True,same_initialization=True,same_sensor_mode='paired'))
        write_json(args.output/'training.json',report)
        frozen=dict(selected_arm=winner,selected_sha256=hash_file(args.output/'selected.pt'),
            adapter_sha256=hash_file(args.output/'adapter.pt'),frozen_at=time.time())
        write_json(args.output/'frozen_for_test.json',frozen)
        prep=launch('test_sensor_preparation',['prepare_hjepa_views.py','--source',str(sealed),'--output',str(args.output/'test_views')])
        wait(prep,'test_sensor_preparation')
        write_json(receipt,dict(evaluation_started_at=time.time(),model_directory=str(args.output),**frozen))
        evaluator=launch('evaluation',['evaluate_structural_training.py','--data',str(args.output/'test_views'),
            '--model',str(args.output),'--baseline',str(args.initial),'--reference',str(args.original),
            '--reference',str(args.previous),'--reference',str(args.output/'small_data/selected.pt'),
            '--reference',str(args.output/'expanded_data/selected.pt'),
            '--reference',str(args.output/'expanded_data/initial_adapter.pt'),'--reference',str(args.previous_paired),
            '--output',str(args.output/'independent_test'),'--skip-canonical','--device','cuda'])
        wait(evaluator,'independent_test_evaluation')
        plot=launch('report',['report_hjepa_holdout.py','--evaluation',str(args.output/'independent_test'),
            '--training',str(args.output/'training.json'),'--output',str(args.figure)])
        wait(plot,'report')
        replay=json.loads((args.output/'independent_test/replay.json').read_text())
        assert len(replay)==96 and all(x['selection_matches'] and x['outcome_matches'] for x in replay)
        assert all(hash_file(Path(p))==h for p,h in preserved.items())
        assert all(hash_file(Path(p))==h for p,h in data_hashes.items())
        assert hash_file(args.output/'selected.pt')==frozen['selected_sha256'] and hash_file(args.output/'adapter.pt')==frozen['adapter_sha256']
        done=json.loads(receipt.read_text());done.update(evaluation_completed_at=time.time(),cases=96)
        write_json(receipt,done)
        status.update(finished_at=time.time(),elapsed_seconds=time.time()-started,selected_arm=winner)
        stage('complete');print(json.dumps(status),flush=True)
    except BaseException as e:
        for _,p in children:
            if p.poll() is None:p.terminate()
        status['error']=repr(e);stage('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,default=Path('runs/hjepa/coverage_corpus_seed42'))
    p.add_argument('--output',type=Path,default=Path('runs/hjepa/coverage_model_seed42'))
    p.add_argument('--initial',type=Path,default=Path('runs/hjepa/lora_continued_model/selected.pt'))
    p.add_argument('--original',type=Path,default=Path('runs/hjepa/structural_model/selected.pt'))
    p.add_argument('--previous',type=Path,default=Path('runs/hjepa/reward_v2_calibrated20261026/selected.pt'))
    p.add_argument('--previous-paired',type=Path,default=Path('runs/hjepa/paired_view_model20261026/paired_view/selected.pt'))
    p.add_argument('--figure',type=Path,default=Path('docs/hjepa_coverage_seed42_results.png'))
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args()
    if a.steps<1:p.error('Positive training steps required.')
    run(a)
