"""Collect a broad suite, train reversible LoRA, then evaluate frozen policies.

Long local run. Each collection worker resumes completed scenes. Logs, source
snapshots, status, provenance and checkpoints stay under ignored runs/.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def migrate_initial_collection(data, specs, jobs):
    """Preserve completed cases from the initial serial collector, without reruns."""
    partial = data/'manifest.partial.json'
    if not partial.exists() or any((data/f'shard_{i}').exists() for i in range(jobs)):
        return
    previous = json.loads(partial.read_text(encoding='utf-8'))
    if previous['scenes'] != specs:
        raise ValueError('Initial serial collection does not match requested scene specs.')
    for i in range(jobs):
        directory = data/f'shard_{i}'
        directory.mkdir()
        manifest = deepcopy(previous)
        manifest['scenes'] = specs[i::jobs]
        ids = {s['scene_id'] for s in manifest['scenes']}
        manifest['episodes'] = [e for e in previous['episodes'] if e['scene_id'] in ids]
        for episode in manifest['episodes']:
            for field in ('file', 'pixels_file'):
                name = episode.get(field)
                if name:
                    source, target = data/name, directory/name
                    if not source.resolve().is_relative_to(data.resolve()) or not target.resolve().is_relative_to(data.resolve()):
                        raise ValueError('Dataset migration path escapes the dataset directory.')
                    if target.exists():
                        raise FileExistsError(target)
                    source.rename(target)
        write_json(directory/'manifest.partial.json', manifest)
    partial.rename(data/'initial_collection.partial.json')


def merge_shards(data, specs, jobs):
    manifests = [json.loads((data/f'shard_{i}'/'manifest.json').read_text(encoding='utf-8')) for i in range(jobs)]
    combined = deepcopy(manifests[0])
    combined.update(scenes=specs, episodes=[], collection_shards=jobs)
    for i, manifest in enumerate(manifests):
        for ep in manifest['episodes']:
            episode = dict(ep)
            for field in ('file', 'pixels_file'):
                if episode.get(field):
                    episode[field] = f'shard_{i}/'+episode[field]
            combined['episodes'].append(episode)
    by_id = {}
    for ep in combined['episodes']:
        by_id.setdefault(ep['scene_id'], []).append(ep)
        if not (data/ep['file']).is_file() or (ep.get('pixels_file') and not (data/ep['pixels_file']).is_file()):
            raise FileNotFoundError(ep['file'])
    if set(by_id) != {s['scene_id'] for s in specs}:
        raise ValueError('Missing or extra scene IDs in collected shards.')
    for episodes in by_id.values():
        if {(e['base_id'], e['grasp_id']) for e in episodes} != {(b, g) for b in range(4) for g in range(4)} | {(4, 0)}:
            raise ValueError('Incomplete or duplicate action table.')
        if len(episodes) != 17:
            raise ValueError('Duplicate episodes.')
    write_json(data/'manifest.json', combined)
    return combined


def run(args):
    from structural_scene import diverse_structural_scenes
    os.chdir(ROOT)
    args.data.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    status_path = args.output/'status.json'
    if (args.output/'selected.pt').exists():
        raise FileExistsError('Completed model already exists; choose a fresh output.')
    specs = json.loads(json.dumps(diverse_structural_scenes(args.seed)))
    base_hash = hash_file(args.base)
    sources = ['hjepa_lora.py', 'hjepa_model.py', 'train_hjepa.py', 'structural_scene.py', 'collect_hjepa.py',
               'run_hjepa_lora_experiment.py', 'evaluate_structural_training.py', 'run_hjepa.py', 'hjepa_scene.py']
    snapshot = args.output/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    provenance = dict(base_checkpoint=str(args.base), base_sha256=base_hash, seed=args.seed,
        scenes=len(specs), splits={s: sum(x['split'] == s for x in specs) for s in ('train', 'val', 'test')},
        jobs=args.jobs, steps=args.steps, lora_rank=8, lora_alpha=16., lora_dropout=.05,
        full_core_training=False, original_core_frozen=True, reward_version='unchanged slide-3 v1', sources={})
    for filename in sources:
        shutil.copyfile(ROOT/filename, snapshot/filename)
        provenance['sources'][filename] = hash_file(ROOT/filename)
    write_json(args.output/'provenance.json', provenance)
    status = dict(pid=os.getpid(), stage='collection', started_at=time.time(), **provenance)
    children = []
    try:
        if not (args.data/'manifest.json').exists():
            migrate_initial_collection(args.data, specs, args.jobs)
            for i in range(args.jobs):
                directory = args.data/f'shard_{i}'
                if (directory/'manifest.json').exists():
                    continue
                log = (args.output/f'collection_{i}.log').open('ab')
                command = [sys.executable, '-u', 'collect_hjepa.py', '--diverse', '--seed', str(args.seed),
                           '--output', str(directory), '--shard-index', str(i), '--shard-count', str(args.jobs), '--resume']
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                log.close()
                children.append((proc, f'collection_{i}'))
            while any(p.poll() is None for p, _ in children):
                failed = [(label, p.returncode) for p, label in children if p.poll() not in (None, 0)]
                if failed:
                    raise RuntimeError(f'Collection failed: {failed}')
                status.update(updated_at=time.time(), workers=[dict(name=n, pid=p.pid, exit_code=p.poll()) for p, n in children])
                write_json(status_path, status)
                time.sleep(5)
            if any(p.returncode for p, _ in children):
                raise RuntimeError('A collector failed; inspect collection logs.')
            manifest = merge_shards(args.data, specs, args.jobs)
        else:
            manifest = json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
            if manifest['scenes'] != specs:
                raise ValueError('Existing combined dataset uses different scenes.')
        status.update(stage='training', episodes=len(manifest['episodes']), frames=sum(e['frames'] for e in manifest['episodes']))
        write_json(status_path, status)
        command = [sys.executable, '-u', 'train_hjepa.py', '--data', str(args.data), '--output', str(args.output),
                   '--init-checkpoint', str(args.base), '--lora-rank', '8', '--steps', str(args.steps),
                   '--policy-updates', str(args.policy_updates), '--device', 'cuda']
        with (args.output/'training.log').open('ab') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        children = [(proc, 'training')]
        status['training_pid'] = proc.pid
        write_json(status_path, status)
        if proc.wait():
            raise RuntimeError('LoRA training failed; inspect training.log. Original checkpoint is preserved.')
        if hash_file(args.base) != base_hash:
            raise AssertionError('Original checkpoint was modified.')
        freeze = dict(selected_sha256=hash_file(args.output/'selected.pt'), adapter_sha256=hash_file(args.output/'adapter.pt'),
                      base_unchanged=True, frozen_at=time.time())
        write_json(args.output/'frozen_for_test.json', freeze)
        status.update(stage='evaluation', updated_at=time.time(), **freeze)
        write_json(status_path, status)
        command = [sys.executable, '-u', 'evaluate_structural_training.py', '--data', str(args.data),
                   '--model', str(args.output), '--baseline', str(args.base), '--output', str(args.output/'evaluation'), '--device', 'cuda']
        with (args.output/'evaluation.log').open('ab') as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        children = [(proc, 'evaluation')]
        if proc.wait():
            raise RuntimeError('Evaluation failed; trained adapter is preserved. Inspect evaluation.log.')
        if hash_file(args.output/'selected.pt') != freeze['selected_sha256'] or hash_file(args.base) != base_hash:
            raise AssertionError('Frozen checkpoint changed during evaluation.')
        status.update(stage='complete', finished_at=time.time(), original_checkpoint_unchanged=True)
        write_json(status_path, status)
    except BaseException as error:
        for proc, _ in children:
            if proc.poll() is None:
                proc.terminate()
        for proc, _ in children:
            if proc.poll() is None:
                proc.wait(timeout=30)
        status.update(stage='failed', error=repr(error), updated_at=time.time())
        write_json(status_path, status)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/diverse_lora_data'))
    p.add_argument('--output', type=Path, default=Path('runs/hjepa/diverse_lora_model'))
    p.add_argument('--base', type=Path, default=Path('runs/hjepa/structural_model/selected.pt'))
    p.add_argument('--seed', type=int, default=20261018)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--steps', type=int, default=1200)
    p.add_argument('--policy-updates', type=int, default=400)
    a = p.parse_args()
    if not 1 <= a.jobs <= 8 or a.steps < 1:
        p.error('Use 1..8 collection workers and positive training steps.')
    run(a)
