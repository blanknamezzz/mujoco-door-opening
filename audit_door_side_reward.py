"""Check door-side reward labels using train/validation metadata only."""
import argparse
from collections import defaultdict,Counter
import json
from pathlib import Path
from door_side_reward import side_reward_targets,reward_contract


def audit(data,output):
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('evaluation_only') or any(s['split'] not in ('train','val') for s in manifest['scenes']):
        raise ValueError('Audit requires dedicated train/validation data, not held-out tests.')
    specs={s['scene_id']:s for s in manifest['scenes']}
    tables=defaultdict(lambda:defaultdict(list))
    for e in manifest['episodes']:
        if e['scene_id'] not in specs or e['split']!=specs[e['scene_id']]['split']:
            raise ValueError('Episode split mismatch.')
        tables[e['scene_id']][e['base_id']].append(e)
    if set(tables)!=set(specs):raise ValueError('Missing scene records.')
    counts=defaultdict(Counter); cases=[]
    for scene_id,table in tables.items():
        groups=[g for _,g in sorted(table.items())]
        target=side_reward_targets(groups,manifest['grasp_budget'])
        pattern=('both' if target['can_open'].all() else 'front' if target['can_open'][0] else
                 'back' if target['can_open'][1] else 'neither')
        counts[specs[scene_id]['split']][pattern]+=1
        cases.append(dict(scene_id=scene_id,split=specs[scene_id]['split'],pattern=pattern,
            **{k:v.tolist() for k,v in target.items()}))
    report=dict(reward=reward_contract(),scenes=len(cases),counts={k:dict(v) for k,v in counts.items()},
        cases=cases,test_records_opened=0,pixel_arrays_opened=0,source_manifest=str(data/'manifest.json'))
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('runs/hjepa/diverse_lora_trainval'))
    p.add_argument('--output',type=Path,default=Path('runs/hjepa/side_reward_audit20261026/labels.json'))
    a=p.parse_args();audit(a.data,a.output)
