"""Matched CLS/patch-mean LoRA, then a fresh named-stream physical test."""
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
from structural_scene import independent_test_scenes
from run_hjepa_lora_experiment import hash_file,merge_shards,write_json
from door_side_reward import validation_key


def run(args):
    os.chdir(Path(__file__).resolve().parent)
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Fresh experiment required.')
    preflight=Path('runs/hjepa/pooling_preflight_seed42/native_parity.json')
    check=json.loads(preflight.read_text())
    assert check['original_core_unchanged'] and all(r['gradient_tensors']>0 for r in check['results'].values())
    development=json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    assert development['split_counts']==dict(train=576,val=96)
    specs=json.loads(json.dumps(independent_test_scenes(args.seed,8,stream=args.scene_stream)))
    prior=[];paths=[]
    for path in Path('runs/hjepa').rglob('manifest.json'):
        m=json.loads(path.read_text(encoding='utf-8'))
        if m.get('scenes'):prior.extend(m['scenes']);paths.append(str(path))
    assert_scene_disjoint(prior,specs);assert_scene_disjoint(development['scenes'],specs)
    args.output.mkdir(parents=True)
    started=time.time();arms=dict(cls='backbone_cls',patch_mean='backbone_patch_mean')
    preserved={str(p):hash_file(p) for p in (args.initial,args.original,args.previous,args.coverage)}
    protocol=dict(seed=args.seed,scene_stream=args.scene_stream,scene_stream_algorithm='SeedSequence(seed, spawn_key=first128bitsSHA256(stream), little-endian32bit words)',
        train_scenes=576,validation_scenes=96,test_scenes=96,steps_per_arm=args.steps,
        arms=arms,position_views='paired',official_model=True,lora_only=True,original_core_frozen=True,
        matched_position_head_input_dim=196,queue_seconds=0,preflight_sha256=hash_file(preflight),
        prior_unique_scenes=len({s['scene_id'] for s in prior}),historical_manifests=paths,
        preserved_checkpoints_sha256=preserved,development_manifest_sha256=hash_file(args.data/'manifest.json'),
        selection='Validation side-correct, task-correct, openings, terminal utility; ties prefer CLS; no test reselection',
        teacher='Each arm uses its own unadapted pooling scores for the same0.02 KL anchor')
    write_json(args.output/'protocol.json',protocol);write_json(args.output/'pending_test_scenes.json',dict(scenes=specs))
    names=['AGENTS.md','run_hjepa_pooling_experiment.py','train_hjepa_listwise.py','hjepa_model.py',
        'collect_hjepa.py','structural_scene.py','experiment_reproducibility.py','position_observation.py',
        'door_side_reward.py','train_hjepa.py','hjepa_lora.py','evaluate_structural_training.py',
        'run_hjepa.py','report_hjepa_holdout.py','prepare_hjepa_views.py','verify_hjepa_pooling.py',
        'probe_hjepa_spatial.py']
    snapshot=args.output/'source_snapshot';snapshot.mkdir()
    for name in names:shutil.copyfile(name,snapshot/name)
    write_json(args.output/'source_provenance.json',{n:hash_file(Path(n)) for n in names})
    status=dict(pid=os.getpid(),started_at=started,seed=args.seed,stage='starting',queue_seconds=0)
    children=[];phase_seconds={}
    def state(stage):
        status.update(stage=stage,updated_at=time.time(),workers=[dict(name=n,pid=p.pid,exit_code=p.poll()) for n,p,_ in children])
        write_json(args.output/'status.json',status)
    def launch(name,command):
        tick=time.monotonic()
        with (args.output/(name+'.log')).open('wb') as f:
            p=subprocess.Popen([sys.executable,'-u',*command],stdout=f,stderr=subprocess.STDOUT,
                env=worker_environment(args.seed),creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        children.append((name,p,tick));return p
    def wait(p,name):
        state(name)
        if p.wait():raise RuntimeError(name+' failed; inspect its preserved log.')
        tick=next(t for _,q,t in children if q==p);phase_seconds[name]=time.monotonic()-tick
    try:
        reports={}
        for arm,encoder in arms.items():
            p=launch(arm,['train_hjepa_listwise.py','--data',str(args.data),'--initial',str(args.initial),
                '--output',str(args.output/arm),'--steps',str(args.steps),'--seed',str(args.seed),
                '--position-views','paired','--position-encoder',encoder,'--objective','door-side'])
            wait(p,arm+'_training');reports[arm]=json.loads((args.output/arm/'training.json').read_text())
        winner=max(arms,key=lambda n:validation_key(reports[n]['selected_validation'],'door-side'))
        for name in ('best.pt','selected.pt','adapter.pt'):shutil.copyfile(args.output/winner/name,args.output/name)
        report=reports[winner]
        report.update(selected_arm=winner,arm_summaries={n:r['selected_validation'] for n,r in reports.items()},
            comparison_arms=[dict(name='cls',label='CLS',reference='reference_2'),
                dict(name='patch_mean',label='Patch mean',reference='reference_3')],
            pooling_ablation=protocol,selection_across_arms=True)
        write_json(args.output/'training.json',report)
        frozen=dict(selected_arm=winner,selected_sha256=hash_file(args.output/'selected.pt'),
            adapter_sha256=hash_file(args.output/'adapter.pt'),frozen_at=time.time())
        write_json(args.output/'frozen_for_test.json',frozen)
        # Generate outcomes only after selection. No GPU-training contention
        # is hidden inside this standalone collection timing.
        corpus=args.output/'test_corpus';corpus.mkdir()
        tick=time.monotonic()
        collectors=[launch('collection_'+str(i),['collect_hjepa.py','--test-only','--endpoint-only',
            '--test-cases-per-family','8','--seed',str(args.seed),'--scene-stream',args.scene_stream,
            '--shard-index',str(i),'--shard-count',str(args.jobs),'--output',str(corpus/f'shard_{i}')]) for i in range(args.jobs)]
        state('test_collection')
        for p in collectors:
            if p.wait():raise RuntimeError('Test collection failed; preserve completed shards.')
        phase_seconds['standalone_test_collection']=time.monotonic()-tick
        merged=merge_shards(corpus,specs,args.jobs)
        assert merged['evaluation_only'] and len(merged['scenes'])==96
        write_json(corpus/'test_lock.json',dict(manifest_sha256=hash_file(corpus/'manifest.json'),
            seed=args.seed,scene_stream=args.scene_stream,disjoint_from_historical_and_development=True,
            model_frozen_before_collection=frozen,never_evaluated=True))
        p=launch('test_views',['prepare_hjepa_views.py','--source',str(corpus),'--output',str(args.output/'test_views')])
        wait(p,'test_sensor_preparation')
        receipt=dict(evaluation_started_at=time.time(),**frozen)
        write_json(corpus/'evaluation_receipt.json',receipt)
        p=launch('evaluation',['evaluate_structural_training.py','--data',str(args.output/'test_views'),
            '--model',str(args.output),'--baseline',str(args.initial),'--reference',str(args.original),
            '--reference',str(args.previous),'--reference',str(args.output/'cls/selected.pt'),
            '--reference',str(args.output/'patch_mean/selected.pt'),'--reference',str(args.coverage),
            '--reference',str(args.output/'cls/initial_adapter.pt'),'--reference',str(args.output/'patch_mean/initial_adapter.pt'),
            '--output',str(args.output/'independent_test'),'--skip-canonical','--device','cuda'])
        wait(p,'test_evaluation_and_replay')
        p=launch('report',['report_hjepa_holdout.py','--evaluation',str(args.output/'independent_test'),
            '--training',str(args.output/'training.json'),'--output',str(args.figure)])
        wait(p,'report')
        replay=json.loads((args.output/'independent_test/replay.json').read_text())
        assert len(replay)==96 and all(r['selection_matches'] and r['outcome_matches'] for r in replay)
        assert all(hash_file(Path(p))==h for p,h in preserved.items())
        assert hash_file(args.data/'manifest.json')==protocol['development_manifest_sha256']
        assert hash_file(args.output/'selected.pt')==frozen['selected_sha256']
        assert hash_file(args.output/'adapter.pt')==frozen['adapter_sha256']
        receipt.update(evaluation_completed_at=time.time(),cases=96);write_json(corpus/'evaluation_receipt.json',receipt)
        write_json(args.output/'phase_timings.json',dict(queue_seconds=0,phase_seconds=phase_seconds,elapsed_seconds=time.time()-started))
        status.update(finished_at=time.time(),elapsed_seconds=time.time()-started,selected_arm=winner);state('complete')
        print(json.dumps(status),flush=True)
    except BaseException as e:
        for _,p,_ in children:
            if p.poll() is None:p.terminate()
        status['error']=repr(e);state('failed');raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('runs/hjepa/coverage_corpus_seed42/expanded_policy'))
    p.add_argument('--output',type=Path,default=Path('runs/hjepa/pooling_model_seed42'))
    p.add_argument('--initial',type=Path,default=Path('runs/hjepa/lora_continued_model/selected.pt'))
    p.add_argument('--original',type=Path,default=Path('runs/hjepa/structural_model/selected.pt'))
    p.add_argument('--previous',type=Path,default=Path('runs/hjepa/reward_v2_calibrated20261026/selected.pt'))
    p.add_argument('--coverage',type=Path,default=Path('runs/hjepa/coverage_model_seed42/selected.pt'))
    p.add_argument('--figure',type=Path,default=Path('docs/hjepa_pooling_seed42_results.png'))
    p.add_argument('--scene-stream',default='pooling_v1')
    p.add_argument('--steps',type=int,default=240)
    p.add_argument('--jobs',type=int,default=3)
    p.add_argument('--seed',type=int,default=42)
    args=p.parse_args()
    if args.steps<1 or not 1<=args.jobs<=6:p.error('Positive steps and1..6 local workers required.')
    run(args)
