"""Read-only frozen tensors, grouped splits, sweep constraints and replay audit."""
from collections import Counter,defaultdict
import gc
import json
from pathlib import Path
import time
import numpy as np


def audit_outcome_run(output,source_root=None):
    import hjepa_runtime
    import torch
    from run_mobile_reward_experiment import file_hash,write_json
    output=Path(output);source_root=Path(source_root) if source_root is not None else Path(__file__).resolve().parent
    def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
    begin=time.perf_counter();torch.set_num_threads(4)
    protocol=read(output/'protocol.json');frozen=read(output/'frozen_for_test.json');manifest=read(output/'manifest.json')
    original=torch.load(protocol['initial'],map_location='cpu',weights_only=True)
    fixed=[n for n in original['model'] if 'lora_' not in n and not n.startswith(('position_policy.','grasp_policy.'))]
    audits={}
    for name in protocol['models']:
        pg=torch.load(output/name/'selected_pg.pt',map_location='cpu',weights_only=True);audits[name]={}
        for label in ('selected_pg','last_pg','selected','last'):
            cp=pg if label=='selected_pg' else torch.load(output/name/(label+'.pt'),map_location='cpu',weights_only=True)
            exact=all(torch.equal(original['model'][n],cp['model'][n]) for n in fixed)
            norms=all(torch.equal(original[k],cp[k]) for k in ('state_mean','state_std'))
            assert exact and norms,'Original frozen tensors or statistics changed.'
            same_pg=all(torch.equal(v,cp['model'][n]) for n,v in pg['model'].items()
                        if not n.startswith('inspection_policy.')) if label in ('selected','last') else None
            if same_pg is not None:assert same_pg,'Continuation changed during independent gate fitting.'
            changed_lora=sum(not torch.equal(v,cp['model'][n]) for n,v in original['model'].items() if 'lora_' in n)
            if label=='last_pg':assert changed_lora>0,'LoRA updates were not applied.'
            audits[name][label]=dict(step=cp['step'],frozen_tensor_count=len(fixed),original_tensors_exact=exact,
                normalization_exact=norms,changed_lora_tensors_from_v3_initial=changed_lora,
                selected_pg_exact_outside_inspection_head=same_pg,
                changed_inspection_tensors_from_selected_pg=sum(not torch.equal(v,cp['model'][n])
                    for n,v in pg['model'].items() if n.startswith('inspection_policy.')))
            if label=='selected':
                assert file_hash(output/name/'selected.pt')==frozen['checkpoint_sha256'][name]
                assert cp['pg_selected_sha256']==file_hash(output/name/'selected_pg.pt')
            if cp is not pg:del cp
            gc.collect()
        del pg;gc.collect()
    del original;gc.collect()
    assert file_hash(protocol['initial'])==frozen['baseline_sha256']
    assert file_hash(output/'manifest.json')==frozen['manifest_sha256']
    sources=read(output/'source_hashes.json')
    assert all(file_hash(source_root/n)==h and file_hash(output/'source'/n)==h for n,h in sources.items())
    saved_source_only=source_root.resolve()==(output/'source').resolve()
    for filename in ('preserved_hashes.json','reused_development_hashes.json'):
        assert all(file_hash(p)==h for p,h in read(output/filename).items())
    groups=defaultdict(set);layouts=defaultdict(set)
    for s in manifest['scenes']:
        groups[s['group_id']].add(s['split']);layouts[s['group_id']].add(s['layout'])
    assert all(len(v)==1 for v in groups.values())
    assert all(v=={'front_blocked','rear_blocked','both_blocked','free'} for v in layouts.values())
    assert len({s['geometry_sha256'] for s in manifest['scenes']})==len(manifest['scenes'])
    tests=[s for s in manifest['scenes'] if s['split']=='test']
    test_files=[output/'data'/s['scene_id']/str(b)/'observations.npz' for s in tests for b in (0,1)]
    assert (output/'frozen_for_test.json').stat().st_mtime<min(p.stat().st_mtime for p in test_files)
    count=entries=scans=forced=0;max_speed=max_step=0.;collisions=[]
    masks=read(output/'test_sweep_queries.json');constraint=Counter()
    for spec in tests:
        mask=masks[spec['scene_id']]['mask']
        for branch in ('0','1'):
            records=read(output/'data'/spec['scene_id']/branch/'records.json')
            for r in records:
                outcome='success' if r['success'] else 'collision' if r['collision'] else 'other_failure'
                constraint[('allowed' if mask[r['base_id']] else 'rejected')+'_'+outcome]+=1
    for name,config in protocol['policies'].items():
        done=read(output/'policy_results'/name/'replay_verification.json')
        assert done['all_match'] and done['count']==len(tests)
        decisions={d['scene_id']:d for d in read(output/'policy_results'/name/'test.json')['decisions']}
        for spec in tests:
            folder=output/'policy_results'/name/'replays'/spec['scene_id']
            item=read(folder/'result.json');r=item['record'];trace=read(folder/'trace.json')
            assert item['match'] and r['base_reset_count']==1
            assert all(b['time_s']>=a['time_s']-1e-8 for a,b in zip(trace,trace[1:]))
            if r['opening_tested']:assert r['result']['continuous_state_at_skill_entry'];entries+=1
            if r['inspection_attempted']:
                scans+=1
                if r['inspection_success']:
                    assert not r['inspection_unsafe_abort']
                    assert all(abs(t['door_angle_deg'])<.5 for t in trace if t['phase']=='inspection')
            if config['guard']:
                mask=masks[spec['scene_id']]['mask']
                if not any(mask):
                    assert r['refused'] and not r['inspection_attempted'] and r['sweep_forced_refusal'];forced+=1
                elif not r['refused'] and not r['inspection_unsafe_abort']:assert mask[r['base_id']]
            expected=decisions[spec['scene_id']]
            assert r['success']==expected['success'] and r['collision']==expected['collision']
            max_speed=max(max_speed,max((np.linalg.norm(t['base_velocity'][:2]) for t in trace),default=0.))
            max_step=max(max_step,r['max_trace_step_m'])
            if r['collision']:collisions.append(dict(policy=name,scene_id=spec['scene_id'],stage=r['failure_stage'],
                first_contact=r['result'].get('first_unexpected_contact'),stop_reason=r['result'].get('stop_reason')))
            count+=1
    assert count==len(tests)*len(protocol['policies']) and max_speed<.31 and max_step<.02
    result=dict(seed=42,original_frozen_tensor_count=len(fixed),checkpoint_audits=audits,
        all_original_tensors_and_normalizations_exact=True,all_selected_pg_frozen_during_gate_fitting=True,
        preserved_files_and_reused_data_exact=True,archived_sources_exact=True,
        current_and_archived_sources_exact=True if not saved_source_only else None,
        verified_source_root=str(source_root),saved_launch_snapshot_mode=saved_source_only,
        source_snapshot_files=len(sources),selected_models_frozen_before_test_collection=True,
        group_count=len(groups),test_group_count=len(tests)//4,geometry_unique=len(manifest['scenes']),
        replay_count=count,all_replays_match=True,all_episode_reset_counts_one=True,
        opening_entry_continuity_checks=entries,actual_inspections=scans,
        forced_initial_refusal_checks=forced,max_sampled_base_speed_m_s=float(max_speed),
        max_recorded_step_m=max_step,test_sweep_candidate_counts=dict(constraint),collisions=collisions,
        audit_seconds=time.perf_counter()-begin)
    write_json(output/'tensor_and_physics_audit.json',result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archived-sources',action='store_true',help='Audit a completed older run against its immutable launch source, not a subsequently edited worktree.')
    args=parser.parse_args();r=audit_outcome_run(args.output,args.output/'source' if args.archived_sources else None)
    print(json.dumps({k:r[k] for k in ('original_frozen_tensor_count','replay_count',
        'opening_entry_continuity_checks','actual_inspections','test_sweep_candidate_counts','audit_seconds')}))
