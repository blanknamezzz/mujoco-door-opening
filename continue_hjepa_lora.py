"""Continue LoRA while collecting a physically separate, evaluation-only suite."""
import argparse
import hashlib
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
    root = Path(__file__).resolve().parent
    os.chdir(root)
    if (args.output/'status.json').exists():
        raise FileExistsError('Continuation already exists; choose a fresh output.')
    args.output.mkdir(parents=True, exist_ok=True)
    args.test.mkdir(parents=True, exist_ok=True)
    specs = json.loads(json.dumps(independent_test_scenes(args.test_seed, args.test_per_family)))
    manifest = json.loads((args.trainval/'manifest.json').read_text(encoding='utf-8'))
    if any(s['split'] not in ('train', 'val') for s in manifest['scenes']) or any(e['split'] == 'test' for e in manifest['episodes']):
        raise ValueError('Continuation requires a train/validation-only manifest.')
    assert_scene_disjoint(manifest['scenes'], specs)
    hashes = {'initial_checkpoint': hash_file(args.initial), 'original_checkpoint': hash_file(args.original)}
    status = dict(pid=os.getpid(), stage='training_and_test_collection', started_at=time.time(),
                  updated_at=time.time(), steps=args.steps, lora_rank=8, original_core_frozen=True,
                  trainval=str(args.trainval), evaluation_only_dataset=str(args.test),
                  independent_test_seed=args.test_seed, independent_test_scenes=len(specs),
                  splits={s: sum(x['split'] == s for x in manifest['scenes']) for s in ('train', 'val')},
                  initial_checkpoint=str(args.initial), original_checkpoint=str(args.original), checkpoint_hashes=hashes)
    write_json(args.output/'protocol.json', dict(status, test_outcomes_used_for_selection=False,
        adapter_lr=5e-5, head_lr=1e-4, policy_updates=200, reward_unchanged=True))
    write_json(args.output/'test_scene_specs.json', dict(evaluation_only=True, scenes=specs))
    snapshot = args.output/'source_snapshot'
    snapshot.mkdir()
    sources = {}
    for name in ('train_hjepa.py', 'hjepa_model.py', 'hjepa_lora.py', 'collect_hjepa.py', 'structural_scene.py',
                 'prepare_hjepa_splits.py', 'evaluate_structural_training.py', 'continue_hjepa_lora.py'):
        shutil.copyfile(root/name, snapshot/name)
        sources[name] = hash_file(root/name)
    write_json(args.output/'source_provenance.json', sources)
    children = []
    def launch(command, label):
        with (args.output/(label+'.log')).open('wb') as log:
            proc = subprocess.Popen([sys.executable, '-u', *command], stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        children.append((label, proc))
        return proc
    try:
        trainer = launch(['train_hjepa.py', '--data', str(args.trainval), '--output', str(args.output),
            '--init-checkpoint', str(args.initial), '--lora-rank', '8', '--steps', str(args.steps),
            '--adapter-learning-rate', '5e-5', '--head-learning-rate', '1e-4', '--policy-updates', '200',
            '--device', 'cuda'], 'training')
        for index in range(args.jobs):
            shard = args.test/f'shard_{index}'
            if not (shard/'manifest.json').exists():
                launch(['collect_hjepa.py', '--test-only', '--seed', str(args.test_seed), '--output', str(shard),
                    '--test-cases-per-family', str(args.test_per_family), '--shard-index', str(index),
                    '--shard-count', str(args.jobs), '--resume'], f'test_collection_{index}')
        frozen = False
        while any(p.poll() is None for _, p in children):
            failed = [(n, p.returncode) for n, p in children if p.poll() not in (None, 0)]
            if failed:
                raise RuntimeError(f'Continuation subprocess failed: {failed}; inspect corresponding log.')
            if trainer.poll() == 0 and not frozen:
                # Freeze the adapter before any independent-test evaluation.
                write_json(args.output/'frozen_for_test.json', dict(
                    selected_sha256=hash_file(args.output/'selected.pt'), adapter_sha256=hash_file(args.output/'adapter.pt'),
                    original_base_unchanged=hash_file(args.original) == hashes['original_checkpoint'], frozen_at=time.time()))
                frozen = True
            status.update(updated_at=time.time(), workers=[dict(name=n, pid=p.pid, exit_code=p.poll()) for n, p in children])
            write_json(args.output/'status.json', status)
            time.sleep(5)
        if any(p.returncode for _, p in children):
            raise RuntimeError('Continuation subprocess failed.')
        if hash_file(args.initial) != hashes['initial_checkpoint'] or hash_file(args.original) != hashes['original_checkpoint']:
            raise AssertionError('A preserved checkpoint changed.')
        if not frozen:
            write_json(args.output/'frozen_for_test.json', dict(
                selected_sha256=hash_file(args.output/'selected.pt'), adapter_sha256=hash_file(args.output/'adapter.pt'),
                original_base_unchanged=True, frozen_at=time.time()))
        test = merge_shards(args.test, specs, args.jobs)
        if not test['evaluation_only'] or any(e['split'] != 'test' for e in test['episodes']):
            raise AssertionError('Independent test suite is not evaluation-only.')
        status.update(stage='independent_test_evaluation', updated_at=time.time())
        write_json(args.output/'status.json', status)
        evaluator = launch(['evaluate_structural_training.py', '--data', str(args.test), '--model', str(args.output),
            '--baseline', str(args.initial), '--reference', str(args.original), '--output', str(args.output/'independent_test'),
            '--skip-canonical', '--device', 'cuda'], 'evaluation')
        while evaluator.poll() is None:
            status['updated_at'] = time.time()
            write_json(args.output/'status.json', status)
            time.sleep(5)
        if evaluator.returncode:
            raise RuntimeError('Independent evaluation failed; inspect evaluation.log.')
        frozen_hashes = json.loads((args.output/'frozen_for_test.json').read_text(encoding='utf-8'))
        if hash_file(args.output/'selected.pt') != frozen_hashes['selected_sha256']:
            raise AssertionError('Test evaluation changed the frozen model.')
        status.update(stage='complete', updated_at=time.time(), finished_at=time.time())
        write_json(args.output/'status.json', status)
    except BaseException as error:
        for _, p in children:
            if p.poll() is None:
                p.terminate()
        for _, p in children:
            if p.poll() is None:
                p.wait(timeout=30)
        status.update(stage='failed', error=repr(error), updated_at=time.time())
        write_json(args.output/'status.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trainval', type=Path, default=Path('runs/hjepa/diverse_lora_trainval'))
    parser.add_argument('--test', type=Path, default=Path('runs/hjepa/independent_test_data'))
    parser.add_argument('--output', type=Path, default=Path('runs/hjepa/lora_continued_model'))
    parser.add_argument('--initial', type=Path, default=Path('runs/hjepa/diverse_lora_model/selected.pt'))
    parser.add_argument('--original', type=Path, default=Path('runs/hjepa/structural_model/selected.pt'))
    parser.add_argument('--test-seed', type=int, default=20261019)
    parser.add_argument('--test-per-family', type=int, default=4)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 8 or args.steps < 1:
        parser.error('Use 1..8 collectors and positive training steps.')
    run(args)
