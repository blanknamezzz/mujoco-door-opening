"""Print collection/training progress without opening large trajectory arrays."""
import argparse
import json
from pathlib import Path
import time


def status(data, model):
    path = model/'status.json'
    if not path.exists():
        return dict(stage='not_started')
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('evaluation_only_dataset'):
        data = Path(report['evaluation_only_dataset'])
    shards, completed_scenes, episodes = [], set(), 0
    for directory in sorted(data.glob('shard_*')):
        manifest = directory/'manifest.json'
        if not manifest.exists():
            manifest = directory/'manifest.partial.json'
        if not manifest.exists():
            continue
        try:
            records = json.loads(manifest.read_text(encoding='utf-8'))['episodes']
        except json.JSONDecodeError:
            continue  # Collector can be writing its scene-level progress file.
        ids = {e['scene_id'] for e in records}
        completed_scenes.update(ids)
        episodes += len(records)
        shards.append(dict(shard=directory.name, completed_scenes=len(ids), episodes=len(records),
                           complete=(directory/'manifest.json').exists()))
    age = time.time()-report.get('updated_at', report['started_at'])
    result = {key: report.get(key) for key in ('stage', 'pid', 'scenes', 'splits', 'steps', 'lora_rank', 'error')}
    if report.get('independent_test_scenes'):
        result['independent_test_scenes'] = report['independent_test_scenes']
        result['independent_test_dataset'] = str(data)
    result.update(elapsed_minutes=round((time.time()-report['started_at'])/60, 1),
                  heartbeat_age_seconds=round(age, 1),
                  collection_completed_scenes=len(completed_scenes), collection_completed_records=episodes, shards=shards)
    for filename in ('training.log', 'evaluation.log'):
        log = model/filename
        if log.exists():
            result[filename] = log.read_text(encoding='utf-8', errors='replace').splitlines()[-2:]
            if filename == 'training.log':
                for line in reversed(log.read_text(encoding='utf-8', errors='replace').splitlines()):
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if 'step' in entry:
                        result['last_logged_training_step'] = entry['step']
                        break
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/diverse_lora_data'))
    latest = Path('runs/hjepa/lora_continued_model')
    p.add_argument('--model', type=Path, default=latest if (latest/'status.json').exists() else Path('runs/hjepa/diverse_lora_model'))
    args = p.parse_args()
    print(json.dumps(status(args.data, args.model), indent=2))
