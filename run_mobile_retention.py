"""Matched learning-rate and frozen-output retention LoRA continuations."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from run_mobile_reward_experiment import ROOT, ZONE, write_json, file_hash
from mobile_branch_balance_v8 import ScenePairSampler, pg_selection_key
from mobile_retention_v10 import ARMS, retention_components, pg_rates


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, model, source, **metadata):
    from hjepa_model import save_checkpoint
    save_checkpoint(path, model, source['state_mean'], source['state_std'], seed=42,
                    experiment='retention_v10', **metadata)


def validate_pg(model, source, validation):
    from run_mobile_partial_training import predict, evaluate
    return evaluate(model, validation, predict(model, source, validation),
                    {mode: dict(mode=mode, risk=1.) for mode in ('never', 'always')})[0]


def fixed_teacher(args, model, source, training):
    import torch
    from run_mobile_partial_training import forward
    if any(scene['spec']['split'] != 'train' for scene in training):
        raise ValueError('Frozen teacher may only encode training scenes.')
    path = args.output/'teacher_logits.pt'
    identities = [scene['spec']['scene_id'] for scene in training]
    initial_hash = file_hash(args.initial)
    if path.exists():
        cache = torch.load(path, map_location='cpu', weights_only=True)
        if cache['initial_sha256'] != initial_hash or cache['scene_ids'] != identities:
            raise ValueError('Teacher cache identity changed.')
        return cache['predictions']
    model.eval()
    predictions = {}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for scene in training:
            prediction = forward(model, source, scene, False)
            predictions[scene['spec']['scene_id']] = {
                key: prediction[key].detach().float().cpu() for key in ('p0', 'p1', 'g0', 'g1')}
    torch.save(dict(seed=42, initial_sha256=initial_hash, scene_ids=identities,
                    predictions=predictions), path)
    write_json(args.output/'teacher_cache.json', dict(seed=42, split='train',
        initial_sha256=initial_hash, scene_ids=identities, cache_sha256=file_hash(path),
        mode='frozen initial model eval, bf16 autocast, batched candidates as in training',
        test_outcome_reads=0))
    return predictions


def train_pg(args, arm, scenes):
    import torch
    from experiment_reproducibility import seed_everything
    from mobile_partial_policy_v5 import load_partial_checkpoint
    from hjepa_lora import resume_lora, freeze_core_statistics, frozen_core_digest
    from run_mobile_outcome_experiment import pg_targets
    from run_mobile_partial_training import forward, predict
    seed_everything(42)
    begin = time.perf_counter()
    folder = args.output/arm
    folder.mkdir()
    model, source = load_partial_checkpoint(args.initial, 'cuda')
    lora = resume_lora(model, 8, 16., .05)
    model.requires_grad_(False)
    for name, parameter in model.model.named_parameters():
        parameter.requires_grad_('lora_' in name)
    model.position_policy.requires_grad_(True)
    model.grasp_policy.requires_grad_(True)
    rates = pg_rates(ARMS[arm]['lr_scale'])
    model.config['retention_v10'] = dict(arm=arm, **ARMS[arm], pg_rates=rates,
        teacher='fixed initial PG; eval outputs on training inputs only',
        objective='same P + .5G CE plus weight*KL(teacher||student), temperature1',
        visibility='original unsafe-post P and unobserved G masks retained',
        inference='unchanged current-depth v5 execution and terminal utility')
    frozen = frozen_core_digest(model)
    training = [s for s in scenes if s['spec']['split'] == 'train']
    validation = [s for s in scenes if s['spec']['split'] == 'val']
    teacher_begin = time.perf_counter()
    teacher = fixed_teacher(args, model, source, training)
    teacher_seconds = time.perf_counter()-teacher_begin
    teacher_hash = file_hash(args.output/'teacher_logits.pt')
    sampler = ScenePairSampler(training, 'paired')
    rng = random.Random(42)
    summary = validate_pg(model, source, validation)
    best = pg_selection_key(summary['never'], summary['always'])
    history = [dict(step=0, summary=summary)]
    selected = 0
    timings, refreshes, diagnostics, batches = [], [], [], []
    metadata = dict(phase='pg', initial_sha256=file_hash(args.initial),
                    manifest_sha256=file_hash(args.output/'manifest.json'))
    save(folder/'selected_pg.pt', model, source, step=0, **metadata)
    model.enable_activation_checkpointing(16)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        dict(params=[*model.position_policy.parameters(), *model.grasp_policy.parameters()], lr=rates['head']),
        dict(params=[p for p in model.model.parameters() if p.requires_grad], lr=rates['lora'])], weight_decay=.01)
    for step in range(1, args.pg_steps+1):
        if (step-1) % 40 == 0:
            targets = pg_targets(training, predict(model, source, training), 'cuda')
            digest = hashlib.sha256(b''.join(targets[s['spec']['scene_id']][key].tobytes()
                for s in training for key in ('p0', 'p1', 'g0', 'g1'))).hexdigest()
            refreshes.append(dict(step=step-1, training_scenes=len(training),
                                  targets_sha256=digest, test_outcome_reads=0))
        tick = time.perf_counter()
        model.eval()
        model.model.get_level(1).encoder.encoder.train()
        freeze_core_statistics(model)
        optimizer.zero_grad(set_to_none=True)
        batch = sampler.sample(rng)
        batches.append([scene['spec']['scene_id'] for scene in batch])
        values = []
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for scene in batch:
                identity = scene['spec']['scene_id']
                values.append(retention_components(forward(model, source, scene, False),
                    teacher[identity], targets[identity], scene, ARMS[arm]['retention_weight']))
            parts = {key: torch.stack([value[key] for value in values]).mean() for key in values[0]}
        if not all(bool(torch.isfinite(value)) for value in parts.values()):
            raise FloatingPointError('Nonfinite retention loss.')
        parts['loss'].backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.)
        optimizer.step()
        torch.cuda.synchronize()
        timings.append(time.perf_counter()-tick)
        row = dict(step=step, **{key: float(value.detach()) for key, value in parts.items()})
        diagnostics.append(row)
        if step == 1 or step % 10 == 0:
            progress = dict(pid=os.getpid(), phase='pg_'+arm, step=step, total_steps=args.pg_steps,
                seconds_per_update=float(np.mean(timings[-10:])), elapsed_seconds=time.perf_counter()-begin,
                loss=row['loss'], ce=row['ce'], kl=row['kl'])
            write_json(args.output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
        if step == 1 or step % 40 == 0 or step == args.pg_steps:
            summary = validate_pg(model, source, validation)
            history.append(dict(step=step, summary=summary))
            key = pg_selection_key(summary['never'], summary['always'])
            if key > best:
                best, selected = key, step
                save(folder/'selected_pg.pt', model, source, step=step, **metadata)
    save(folder/'last_pg.pt', model, source, step=args.pg_steps, **metadata)
    if frozen_core_digest(model) != frozen:
        raise AssertionError('Frozen original H-JEPA changed.')
    if file_hash(args.output/'teacher_logits.pt') != teacher_hash:
        raise AssertionError('Frozen teacher predictions changed.')
    write_json(folder/'retention_diagnostics.json', dict(seed=42, rows=diagnostics,
        split='train only', retention_weight=ARMS[arm]['retention_weight'],
        teacher_cache_sha256=teacher_hash, teacher_stays_fixed=True))
    write_json(folder/'training_batches.json', dict(seed=42, batches=batches))
    result = dict(seed=42, steps=args.pg_steps, selected_step=selected, history=history,
        teacher_refreshes=refreshes, training_step_seconds=timings, lora=lora, rates=rates,
        teacher_cache_sha256=teacher_hash, teacher_build_or_load_seconds=teacher_seconds,
        batches_sha256=file_hash(folder/'training_batches.json'),
        original_core_exact=True, test_outcome_reads=0, seconds=time.perf_counter()-begin)
    write_json(folder/'pg_training.json', result)
    del model, source, optimizer, parameters, parts, values, teacher
    gc.collect()
    torch.cuda.empty_cache()
    return result


def train_gate(args, arm, scenes):
    import torch
    from experiment_reproducibility import seed_everything
    from mobile_partial_policy_v5 import load_partial_checkpoint
    from run_mobile_inspection_regret import prepare, evaluate
    from mobile_inspection_regret_v6 import inspection_loss
    from run_mobile_active_experiment import validation_key
    seed_everything(42)
    begin = time.perf_counter()
    folder = args.output/arm
    model, source = load_partial_checkpoint(folder/'selected_pg.pt', 'cuda')
    model.requires_grad_(False)
    fixed = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
             if not name.startswith('inspection_policy.')}
    data = prepare(SimpleNamespace(output=folder), model, source, scenes)
    validation = [s for s in scenes if s['spec']['split'] == 'val']
    training = [i for i, row in enumerate(data['rows']) if row['split'] == 'train']
    model.config['inspection_regret_v6'] = dict(arm='joint_regret', temperature=.2,
        continuation='selected PG and LoRA frozen; actual fixed selected-action counterfactual targets',
        input='initial RGB, relative goal, declared risk, initial depth only')
    model.inspection_policy.requires_grad_(True)
    rng = random.Random(42)
    summary, decisions = evaluate(model, validation, data, arm)
    best, selected = validation_key(summary), 0
    history = [dict(step=0, summary=summary)]
    metadata = dict(phase='inspection', pg_selected_sha256=file_hash(folder/'selected_pg.pt'),
                    manifest_sha256=file_hash(args.output/'manifest.json'))
    save(folder/'selected.pt', model, source, step=0, **metadata)
    optimizer = torch.optim.AdamW(model.inspection_policy.parameters(), lr=3e-4, weight_decay=.01)
    for step in range(1, args.gate_steps+1):
        sample = torch.tensor(rng.sample(training, min(32, len(training))), device='cuda')
        model.inspection_policy.train()
        optimizer.zero_grad(set_to_none=True)
        raw = model.inspection_policy(data['features'][sample]).float()
        result = inspection_loss(raw, data['labels'][sample], data['cost'][sample], data['risk'][sample],
            data['direct_predicted_value'][sample], data['advantage'][sample], data['eligible'][sample], 'joint_regret')
        if not bool(torch.isfinite(result['loss'])):
            raise FloatingPointError('Nonfinite I loss.')
        result['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.inspection_policy.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 40 == 0 or step == args.gate_steps:
            summary, decisions = evaluate(model, validation, data, arm)
            history.append(dict(step=step, summary=summary, loss=float(result['loss'].detach())))
            if validation_key(summary) > best:
                best, selected = validation_key(summary), step
                save(folder/'selected.pt', model, source, step=step, **metadata)
            progress = dict(pid=os.getpid(), phase='gate_'+arm, step=step, total_steps=args.gate_steps,
                            elapsed_seconds=time.perf_counter()-begin)
            write_json(args.output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    save(folder/'last.pt', model, source, step=args.gate_steps, **metadata)
    if any(not torch.equal(value, model.state_dict()[name].detach().cpu()) for name, value in fixed.items()):
        raise AssertionError('Non-I continuation changed during I training.')
    cp = torch.load(folder/'selected.pt', map_location='cpu', weights_only=True)
    model.inspection_policy.load_state_dict({name.removeprefix('inspection_policy.'): value
        for name, value in cp['model'].items() if name.startswith('inspection_policy.')})
    summary, decisions = evaluate(model, validation, data, arm)
    write_json(folder/'validation.json', dict(summary=summary, decisions=decisions))
    result = dict(seed=42, steps=args.gate_steps, selected_step=selected, history=history,
                  validation=summary, all_non_I_exact=True, test_outcome_reads=0, seconds=time.perf_counter()-begin)
    write_json(folder/'gate_training.json', result)
    del model, source, optimizer, fixed, cp, data
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prior', type=Path, default=ROOT/'runs/p9_train_seed42_20261111')
    parser.add_argument('--history-root', type=Path, default=ROOT/'runs')
    parser.add_argument('--development-groups', type=int)
    parser.add_argument('--test-groups', type=int, default=18)
    parser.add_argument('--pg-steps', type=int, default=320)
    parser.add_argument('--gate-steps', type=int, default=1600)
    parser.add_argument('--stream', default='mobile_retention_v10_20261111')
    parser.add_argument('--estimate-seconds', type=int, required=True)
    args = parser.parse_args()
    args.output, args.prior, args.history_root = (p.resolve() for p in (args.output, args.prior, args.history_root))
    if ROOT/'runs' not in args.output.parents or (args.output.exists() and any(args.output.iterdir())):
        raise ValueError('A new ignored run directory is required; preserve old runs.')
    if min(args.pg_steps, args.gate_steps, args.test_groups, args.estimate_seconds) < 1:
        raise ValueError('Positive budgets required.')
    args.output.mkdir(parents=True, exist_ok=True)
    begin, now = time.perf_counter(), datetime.now(ZONE)
    receipt = dict(pid=os.getpid(), seed=42, status='starting', started_at=now.isoformat(),
        queue_estimate_seconds=0, runtime_estimate_seconds=args.estimate_seconds,
        expected_completion=(now+timedelta(seconds=args.estimate_seconds)).isoformat())
    write_json(args.output/'receipt.json', receipt)
    from experiment_reproducibility import seed_everything
    receipt['reproducibility'] = seed_everything(42)
    import torch
    torch.set_num_threads(4)
    previous = read(args.prior/'protocol.json')
    args.initial, args.development, args.maps = (Path(previous[k]) for k in ('initial', 'development', 'maps'))
    if file_hash(args.initial) != previous['initial_sha256']:
        raise ValueError('Initial frozen PG hash changed.')
    protected = read(args.prior/'preserved_hashes.json').copy()
    if not all(file_hash(path) == digest for path, digest in protected.items()):
        raise ValueError('A previously protected input changed.')
    for name, digest in read(args.prior/'source_hashes.json').items():
        if file_hash(ROOT/name) != digest:
            raise ValueError('Frozen source changed: '+name)
    old_selection = read(args.prior/'frozen_for_test.json')
    reference_path = args.prior/old_selection['selected_policy']/'selected.pt'
    if file_hash(reference_path) != old_selection['checkpoint_sha256'][old_selection['selected_policy']]:
        raise ValueError('Frozen reference checkpoint changed.')
    for path in [args.initial, *args.prior.glob('*.json'), *args.prior.glob('*/*.pt')]:
        protected[str(path)] = file_hash(path)
    from run_mobile_outcome_experiment import make_run_manifest, attach_development
    from mobile_crossed_v2 import geometry_identity
    manifest = make_run_manifest(args.development, args.test_groups, args.stream, args.output, args.development_groups)
    history = set()
    for path in args.history_root.glob('*/manifest.json'):
        if path.parent.resolve() != args.output:
            history.update(geometry_identity(s) for s in read(path).get('scenes', [])
                           if 'start_pose' in s and 'obstacles' in s)
    if history & {geometry_identity(s) for s in manifest['scenes'] if s['split'] == 'test'}:
        raise ValueError('Fresh test overlaps an older geometry.')
    manifest.update(schema_version=10, historical_geometries_excluded_from_test=len(history))
    write_json(args.output/'manifest.json', manifest)
    scenes = attach_development(args, [s for s in manifest['scenes'] if s['split'] in ('train', 'val')])
    protected.update(read(args.output/'reused_development_hashes.json'))
    for scene in scenes:
        folder = args.maps/'data'/scene['spec']['scene_id']
        scene['maps'] = read(folder/'maps.json')
        for name in ('maps.json', 'reached_rgbd.npz'):
            protected[str(folder/name)] = file_hash(folder/name)
    source = args.output/'source'
    source.mkdir()
    for name in ['AGENTS.md', 'scene.xml', '.gitignore', *sorted(p.name for p in ROOT.glob('*.py'))]:
        shutil.copy2(ROOT/name, source/name)
    hashes = {path.name: file_hash(path) for path in source.iterdir()}
    write_json(args.output/'source_hashes.json', hashes)
    protocol = dict(seed=42, arms=ARMS, prior=str(args.prior), initial=str(args.initial),
        initial_sha256=file_hash(args.initial), development=str(args.development), maps=str(args.maps), counts=manifest['counts'],
        pg_steps=args.pg_steps, gate_steps=args.gate_steps, batch_scenes=2,
        encoder='same rank8 LoRA only; original core and normalization frozen',
        losses='same masked P+.5G CE plus lambda0or1 KL(fixedInitialPG||currentPG), temperature1',
        optimizer='one backward per step in all4arms; PG rate scales1or.1; same AdamW decay.01 and clip1; I rates unchanged',
        pg_selection='same safety-first both-branch key; earliest tie retained',
        gate='same1600 joint-regret I steps after freezing selected PG and LoRA',
        retention_diagnostics='train-only CE/KL per branch; one frozen initial teacher cache shared by all4arms; not new classes or capacity',
        literature=['https://arxiv.org/abs/1606.09282','https://arxiv.org/abs/1503.02531'],
        evaluation_utility='unchanged success1 ordinaryFailure-1 collision-2 refusal0 and actual observation cost',
        map='unchanged historical current-depth v5', reference_path=str(reference_path), reference_sha256=file_hash(reference_path),
        test_references=['frozen_v9_learned', 'frozen_v9_always', 'frozen_v9_never'],
        no_new_test_if_all_selected_tensors_identical_to_reference=True, test_outcome_reads=0)
    write_json(args.output/'protocol.json', protocol)
    write_json(args.output/'preserved_hashes.json', protected)
    training, validation, phases = {}, {}, {}
    receipt.update(status='training', startup_seconds=time.perf_counter()-begin)
    write_json(args.output/'receipt.json', receipt)
    for arm in ARMS:
        pg = train_pg(args, arm, scenes)
        gate = train_gate(args, arm, scenes)
        training[arm], validation[arm] = dict(pg=pg, gate=gate), gate['validation']
        phases['pg_'+arm], phases['gate_'+arm] = pg['seconds'], gate['seconds']
    if len({row['pg']['batches_sha256'] for row in training.values()}) != 1:
        raise AssertionError('Matched arms did not use identical scene batches.')
    if len({row['pg']['teacher_cache_sha256'] for row in training.values()}) != 1:
        raise AssertionError('Matched arms changed the fixed teacher.')
    baseline = torch.load(args.initial, map_location='cpu', weights_only=True)
    original = [name for name in baseline['model'] if 'lora_' not in name
        and not name.startswith(('position_policy.', 'grasp_policy.', 'inspection_policy.'))]
    reference = torch.load(reference_path, map_location='cpu', weights_only=True)
    audits, identity = {}, {}
    for arm in ARMS:
        pg = torch.load(args.output/arm/'selected_pg.pt', map_location='cpu', weights_only=True)
        audits[arm] = {}
        for label in ('selected_pg', 'last_pg', 'selected', 'last'):
            cp = pg if label == 'selected_pg' else torch.load(args.output/arm/(label+'.pt'), map_location='cpu', weights_only=True)
            if any(not torch.equal(baseline['model'][n], cp['model'][n]) for n in original):
                raise AssertionError('Original core changed in '+arm+'/'+label)
            if any(not torch.equal(baseline[key], cp[key]) for key in ('state_mean', 'state_std')):
                raise AssertionError('Normalization changed.')
            fixed = (all(torch.equal(value, cp['model'][name]) for name, value in pg['model'].items()
                         if not name.startswith('inspection_policy.')) if label in ('selected', 'last') else None)
            if fixed is False:
                raise AssertionError('Non-I continuation changed.')
            changed = sum(not torch.equal(value, cp['model'][name]) for name, value in baseline['model'].items() if 'lora_' in name)
            if label == 'last_pg' and not changed:
                raise AssertionError('LoRA did not actually update.')
            if label == 'selected':
                identity[arm] = all(torch.equal(value, cp['model'][name]) for name, value in reference['model'].items())
            audits[arm][label] = dict(step=cp['step'], original_tensor_count=len(original),
                original_and_norms_exact=True, changed_lora_tensors=changed, selected_pg_exact_outside_I=fixed)
            if cp is not pg:
                del cp
                gc.collect()
        del pg
        gc.collect()
    if not all(file_hash(path) == digest for path, digest in protected.items()):
        raise AssertionError('Protected prior input changed.')
    if not all(file_hash(ROOT/name) == file_hash(source/name) == digest for name, digest in hashes.items()):
        raise AssertionError('Source identity changed during run.')
    write_json(args.output/'tensor_audit.json', dict(seed=42, checkpoints=audits,
        all_original422_and_norms_exact=True, all_non_I_frozen_during_gate=True,
        selected_tensor_identity_to_reference=identity, test_outcome_reads=0))
    from run_mobile_active_experiment import best_policy
    selected = best_policy(validation)
    frozen = dict(seed=42, selected_policy=selected, selected_before_test=True,
        manifest_sha256=file_hash(args.output/'manifest.json'),
        checkpoint_sha256={arm: file_hash(args.output/arm/'selected.pt') for arm in ARMS},
        reference_path=str(reference_path), reference_sha256=file_hash(reference_path), validation_metrics=validation,
        selected_tensor_identity_to_reference=identity, new_test_required=not all(identity.values()),
        test_references=protocol['test_references'], test_outcomes_collected=False, frozen_at=datetime.now(ZONE).isoformat())
    write_json(args.output/'frozen_for_test.json', frozen)
    phases['startup_io_selection_audit'] = time.perf_counter()-begin-sum(phases.values())
    report = dict(seed=42, selected_policy=selected, training=training, validation=validation, phase_seconds=phases,
        seconds=time.perf_counter()-begin, test_outcome_reads=0, all_original422_and_norms_exact=True,
        selected_tensor_identity_to_reference=identity, new_test_required=frozen['new_test_required'],
        status='training_complete_fresh_test_pending' if frozen['new_test_required'] else 'training_complete_same_policy_no_new_test')
    write_json(args.output/'report.json', report)
    receipt.update(status='complete', exit_code=0, actual_seconds=report['seconds'],
                   completed_at=datetime.now(ZONE).isoformat(), phase_seconds=phases)
    write_json(args.output/'receipt.json', receipt)
    print(json.dumps(dict(status=report['status'], seconds=report['seconds'], selected=selected,
                          validation=validation, identity=identity)), flush=True)


if __name__ == '__main__':
    if os.environ.get('PYTHONHASHSEED') != '42':
        raise SystemExit(subprocess.run([sys.executable, *sys.argv], cwd=ROOT,
            env=dict(os.environ, PYTHONHASHSEED='42', CUBLAS_WORKSPACE_CONFIG=':4096:8')).returncode)
    try:
        main()
    except Exception as error:
        if '--output' in sys.argv:
            path = Path(sys.argv[sys.argv.index('--output')+1])/'receipt.json'
            if path.exists():
                receipt = read(path)
                if receipt.get('pid') == os.getpid():
                    receipt.update(status='failed', exit_code=1, error=repr(error))
                    write_json(path, receipt)
        raise
