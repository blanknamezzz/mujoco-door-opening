"""Post-test descriptive train/validation/test audit; never selects a policy."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correct(row):
    return bool(row['success'] or (not row['feasible'] and row['refused']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--development',type=Path,required=True)
    parser.add_argument('--evaluation',type=Path,required=True)
    args=parser.parse_args()
    training,development,evaluation=(p.resolve() for p in (args.training,args.development,args.evaluation))
    frozen=read(training/'frozen_for_test.json')
    training_report=read(training/'report.json')
    dev=read(development/'report.json')
    test=read(evaluation/'report.json')
    evaluation_frozen=read(evaluation/'evaluation_frozen.json')
    if frozen['selected_policy']!='standard' or evaluation_frozen['primary_policy']!='standard':
        raise AssertionError('Pretest primary policy changed.')
    if sha256(training/'manifest.json')!=frozen['manifest_sha256'] or sha256(training/'manifest.json')!=evaluation_frozen['evaluation_manifest_sha256']:
        raise AssertionError('Frozen manifest changed.')
    if sha256(training/'frozen_for_test.json')!=evaluation_frozen['training_selection_sha256']:
        raise AssertionError('Training selection changed after test.')
    if not read(evaluation/'tensor_and_physics_audit.json')['all_replays_match']:
        raise AssertionError('Live replay verification failed.')
    summaries={}
    for name in ('standard','retention'):
        summaries[name]={split:(dev['results'][name]['summary'][split] if split!='test' else test['test'][name])
                         for split in ('train','val','test')}
        if summaries[name]['val']!=training_report['validation'][name]:
            raise AssertionError('Development audit disagrees with selected validation.')
    a=read(evaluation/'policy_results'/'standard'/'test.json')['decisions']
    b=read(evaluation/'policy_results'/'retention'/'test.json')['decisions']
    old={row['scene_id']:row for row in a};new={row['scene_id']:row for row in b}
    if len(old)!=72 or set(old)!=set(new):
        raise AssertionError('Test pair identities differ.')
    improved=[];regressed=[];action_changes=[];inspection_changes=[]
    for scene_id in old:
        x,y=old[scene_id],new[scene_id]
        if (x['feasible'],x['group_id'],x['layout'])!=(y['feasible'],y['group_id'],y['layout']):
            raise AssertionError('Different physical scene in pair.')
        if not correct(x) and correct(y):improved.append(scene_id)
        if correct(x) and not correct(y):regressed.append(scene_id)
        if (x['branch'],x['base_id'],x['grasp_id'])!=(y['branch'],y['base_id'],y['grasp_id']):
            action_changes.append(scene_id)
        if x['inspection_attempted']!=y['inspection_attempted']:
            inspection_changes.append(scene_id)
    layouts=('front_blocked','rear_blocked','both_blocked','free')
    by_layout={layout:{name:dict(scenes=sum(r['layout']==layout for r in rows),
                                  correct=sum(correct(r) for r in rows if r['layout']==layout),
                                  inspections=sum(r['inspection_attempted'] for r in rows if r['layout']==layout))
                       for name,rows in (('standard',a),('retention',b))}
               for layout in layouts}
    result=dict(seed=42,scope='exploratory post-test analysis; no policy reselection',
        primary_policy_frozen_before_test='standard',retention_new_weights_not_promoted=True,
        training_selection_sha256=sha256(training/'frozen_for_test.json'),
        test_manifest_sha256=sha256(training/'manifest.json'),
        test_evaluation_frozen_sha256=sha256(evaluation/'evaluation_frozen.json'),
        development_geometry_overlap=dev['geometry_overlap'],
        development_group_overlap=dev['group_overlap'],
        train_val_test={name:{split:dict(correct=s['correct_decisions'],scenes=s['scenes'],
            feasible=s['feasible_scenes'],openings=s['openings'],correct_refusals=s['correct_refusals'],
            false_refusals=s['false_refusals'],collisions=s['collisions'],inspections=s['inspections'],
            mean_net_utility=s['mean_net_utility']) for split,s in summary.items()}
            for name,summary in summaries.items()},
        paired_test=dict(improved_scenes=improved,regressed_scenes=regressed,
            improved_count=len(improved),regressed_count=len(regressed),
            changed_actions=action_changes,changed_inspections=inspection_changes,
            mean_net_utility_advantage=summaries['retention']['test']['mean_net_utility']-
                                        summaries['standard']['test']['mean_net_utility'],
            inspection_reduction=summaries['standard']['test']['inspections']-
                                 summaries['retention']['test']['inspections'],
            mean_path_reduction_m=summaries['standard']['test']['mean_path_length_m']-
                                  summaries['retention']['test']['mean_path_length_m'],
            mean_duration_reduction_s=summaries['standard']['test']['mean_duration_s']-
                                      summaries['retention']['test']['mean_duration_s']),
        errors={name:[dict(scene_id=r['scene_id'],group_id=r['group_id'],layout=r['layout'],
            feasible=r['feasible'],refused=r['refused'],inspection_attempted=r['inspection_attempted'])
            for r in rows if not correct(r)] for name,rows in (('standard',a),('retention',b))},
        by_layout=by_layout,
        limitations=['single training seed','nine known scene templates','validation reused across development',
                     'candidate-limited feasibility','known-map navigation and ideal omnidirectional base'])
    output=evaluation/'post_analysis.json'
    if output.exists():
        raise ValueError('Preserve existing post-analysis file.')
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),paired=result['paired_test'],errors=result['errors']),ensure_ascii=False))


if __name__=='__main__':main()
