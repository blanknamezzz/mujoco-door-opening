"""Separate train/validation and historical test manifests without copying videos."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path


def scene_fingerprint(scene):
    geometry = {k: v for k, v in scene.items() if k not in ('scene_id', 'split', 'family')}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def assert_scene_disjoint(training_scenes, testing_scenes):
    seen = {s['scene_id'] for s in training_scenes}
    tests = {s['scene_id'] for s in testing_scenes}
    fingerprints = {scene_fingerprint(s) for s in training_scenes}
    if seen & tests or fingerprints & {scene_fingerprint(s) for s in testing_scenes}:
        raise ValueError('Training/validation and test scene IDs or geometry overlap.')
    if any(s['split'] != 'test' for s in testing_scenes):
        raise ValueError('Dedicated test dataset contains a non-test scene.')


def partition(source, output, splits, evaluation_only=False):
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    result = deepcopy(manifest)
    result.update(scenes=[s for s in manifest['scenes'] if s['split'] in splits],
                  episodes=[], evaluation_only=evaluation_only, source_manifest=str(source/'manifest.json'))
    for ep in manifest['episodes']:
        if ep['split'] not in splits:
            continue
        record = dict(ep)
        for field in ('file', 'pixels_file'):
            if record.get(field):
                record[field] = Path(os.path.relpath((source/record[field]).resolve(), output.resolve())).as_posix()
        result['episodes'].append(record)
    output.mkdir(parents=True, exist_ok=True)
    path = output/'manifest.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != result:
            raise FileExistsError('Existing partition differs; choose a new output directory.')
    else:
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=Path('runs/hjepa/diverse_lora_data'))
    p.add_argument('--trainval', type=Path, default=Path('runs/hjepa/diverse_lora_trainval'))
    p.add_argument('--test', type=Path, default=Path('runs/hjepa/historical_test_24'))
    a = p.parse_args()
    trainval = partition(a.source, a.trainval, ('train', 'val'))
    test = partition(a.source, a.test, ('test',), evaluation_only=True)
    assert_scene_disjoint(trainval['scenes'], test['scenes'])
    print(json.dumps(dict(trainval_scenes=len(trainval['scenes']), historical_test_scenes=len(test['scenes']),
                         trainval_has_test_records=any(e['split'] == 'test' for e in trainval['episodes'])), indent=2))
