"""Frozen train/validation-only visual probes; no task-policy deployment.

Measure whether current observed RGB features separate measured side feasibility.
Labels never enter feature extraction. A probe predicting an openable side does
not prove that a base/grasp policy can execute it successfully.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
import hjepa_runtime
import numpy as np
import torch
from torch.nn import functional as F
from door_side_reward import side_reward_targets
from experiment_reproducibility import seed_everything
from hjepa_model import load_checkpoint
from hjepa_lora import frozen_core_digest
from run_hjepa_lora_experiment import hash_file, write_json
from train_hjepa import rgb_tensor


FEATURES=('projected_cls','backbone_cls','patch_mean','grid2x2_random192','grid2x2_full')
L2_VALUES=(.0001,.001,.01,.1,1.)


def spatial_features(hidden, projector, projection):
    """Same native token sequence; row-major quadrants preserve image space."""
    if hidden.ndim!=3 or hidden.shape[1]!=257 or hidden.shape[2]!=192:
        raise ValueError('Expected official 224px / patch14 ViT tokens, B,257,192.')
    patches=hidden[:,1:].float().reshape(-1,16,16,192)
    grid=patches.reshape(-1,2,8,2,8,192).mean((2,4)).flatten(1)
    return dict(projected_cls=projector(hidden[:,0]).float(),
        backbone_cls=hidden[:,0].float(),patch_mean=patches.mean((1,2)),
        grid2x2_random192=grid@projection,grid2x2_full=grid)


def standardize(train, validation):
    """Fit normalization to training rows only, including constant features."""
    mean=train.mean(0);std=train.std(0,unbiased=False).clamp_min(1e-5)
    return (train-mean)/std,(validation-mean)/std,mean,std


def fit_logistic(features, targets, l2):
    """Convex shared binary probe, deterministic zero initialization on CPU."""
    x=features.double();y=targets.double()
    weight=torch.zeros(x.shape[-1],dtype=torch.float64,requires_grad=True)
    bias=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([weight,bias],max_iter=150,tolerance_grad=1e-8,
        tolerance_change=1e-11,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(x@weight+bias,y)+l2*weight.square().sum()/2
        loss.backward();return loss
    opt.step(closure)
    return weight.detach(),bias.detach()


def probe_metrics(logits, labels):
    if logits.shape!=labels.shape or logits.ndim!=2 or logits.shape[1]!=2:
        raise ValueError('Expected one front and back observation per scene.')
    prediction=logits>=0;truth=labels.bool()
    recall=[float((prediction[truth==v]==truth[truth==v]).double().mean()) for v in (False,True)]
    selected=logits.argmax(1);refused=logits.max(1).values<0
    feasible=truth.any(1);chosen=truth.gather(1,selected[:,None]).flatten()
    correct=torch.where(refused,~feasible,chosen)
    return dict(scenes=len(labels),binary_accuracy=float((prediction==truth).double().mean()),
        balanced_binary_accuracy=sum(recall)/2,bce=float(F.binary_cross_entropy_with_logits(logits.double(),labels.double())),
        correct_side_decisions=int(correct.sum()),false_refusals=int((refused&feasible).sum()),
        correct_refusals=int((refused&~feasible).sum()),wrong_side=int((~refused&feasible&~chosen).sum()),
        chosen_sides=torch.where(refused,2,selected).tolist(),correct_by_scene=correct.tolist())


def run(args):
    if args.output.exists() and any(args.output.iterdir()):raise FileExistsError('Use a fresh probe output.')
    started=time.monotonic();args.output.mkdir(parents=True)
    reproducibility=seed_everything(args.seed);torch.set_num_threads(4)
    manifest=json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('evaluation_only') or any(s['split'] not in ('train','val') for s in manifest['scenes']):
        raise ValueError('Dedicated train/validation manifest required; never open test observations.')
    initial_hash=hash_file(args.model);data_hash=hash_file(args.data/'manifest.json')
    model,source=load_checkpoint(args.model,'cuda');model.eval();model.requires_grad_(False)
    if not model.is_official:raise ValueError('Official model required.')
    frozen=frozen_core_digest(model)
    vit=model.model.get_level(1).encoder.encoder
    projector=model.model.get_level(1).encoder.projector
    rng=torch.Generator().manual_seed(args.seed)
    projection=torch.linalg.qr(torch.randn(768,192,generator=rng),mode='reduced').Q.cuda()
    groups=defaultdict(lambda:defaultdict(list))
    for e in manifest['episodes']:
        if e['split'] not in ('train','val'):raise ValueError('Unexpected test episode.')
        groups[e['scene_id']][e['base_id']].append(e)
    rows=[];extracted={k:[] for k in FEATURES}
    snapshots=('probe_hjepa_spatial.py','hjepa_model.py','experiment_reproducibility.py','door_side_reward.py','AGENTS.md')
    snap=args.output/'source_snapshot';snap.mkdir()
    for name in snapshots:shutil.copyfile(name,snap/name)
    protocol=dict(seed=args.seed,reproducibility=reproducibility,data=str(args.data),data_sha256=data_hash,
        checkpoint=str(args.model),checkpoint_sha256=initial_hash,features=list(FEATURES),l2_values=L2_VALUES,
        source_sha256={n:hash_file(Path(n)) for n in snapshots},model_updates=0,test_arrays_opened=0,
        projection='seed42 Gaussian QR, fixed 768->192, shared across both camera views',
        task='same shared linear binary side-feasibility probe; side command appended to RGB features',
        threshold='fixed zero logit; refuse when neither side predicts feasible; otherwise argmax',
        selection='validation correct side decisions, balanced binary accuracy, negative BCE',
        limitation='Frozen diagnostic only: side success labels marginalize grasps; no base/grasp execution tested.')
    write_json(args.output/'protocol.json',protocol)
    specs={s['scene_id']:s for s in manifest['scenes']}
    ids=sorted(groups)
    # Read initial observations once per scene. Arrival/temporal data never opened.
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for offset in range(0,len(ids),args.scene_batch):
            images=[]
            for sid in ids[offset:offset+args.scene_batch]:
                table=[g for _,g in sorted(groups[sid].items())]
                targets=side_reward_targets(table,source['grasp_budget'])
                if sid not in manifest.get('position_rear_files',{}):raise ValueError('Rear observation missing.')
                with np.load(args.data/table[0][0]['file'],allow_pickle=False) as arrays:
                    front=arrays['initial_rgb'].copy()
                rear=np.load(args.data/manifest['position_rear_files'][sid],allow_pickle=False)
                images.extend((front,rear))
                rows.append(dict(scene_id=sid,split=specs[sid]['split'],family=specs[sid]['family'],
                    can_open=targets['can_open'].tolist()))
            hidden=vit(model.normalize_pixels(rgb_tensor(np.stack(images),'cuda'))).last_hidden_state
            values=spatial_features(hidden,projector,projection)
            for k,v in values.items():extracted[k].append(v.float().cpu().numpy().reshape(-1,2,v.shape[-1]))
            write_json(args.output/'progress.json',dict(stage='extract',scenes=len(rows),total=len(ids),elapsed_seconds=time.monotonic()-started))
    extract_seconds=time.monotonic()-started
    assert frozen_core_digest(model)==frozen
    del hidden,values,model;torch.cuda.empty_cache()
    arrays={k:np.concatenate(v) for k,v in extracted.items()}
    np.savez(args.output/'features.npz',**arrays)
    write_json(args.output/'scenes.json',rows)
    train=torch.tensor([r['split']=='train' for r in rows]);val=~train
    labels=torch.tensor([r['can_open'] for r in rows],dtype=torch.float64)
    results={}
    for name in FEATURES:
        # The commanded side is available at inference, not the allowed side.
        side=torch.tensor([-1.,1.])[None,:,None].expand(len(rows),-1,-1)
        all_x=torch.cat([torch.from_numpy(arrays[name]),side],-1).double()
        tx,vx,mean,std=standardize(all_x[train].flatten(0,1),all_x[val].flatten(0,1))
        trials=[];best=None
        for l2 in L2_VALUES:
            weight,bias=fit_logistic(tx,labels[train].flatten(),l2)
            tm=probe_metrics((tx@weight+bias).reshape(-1,2),labels[train])
            vm=probe_metrics((vx@weight+bias).reshape(-1,2),labels[val])
            key=(vm['correct_side_decisions'],vm['balanced_binary_accuracy'],-vm['bce'])
            trials.append(dict(l2=l2,train=tm,validation=vm))
            if best is None or key>best[0]:best=(key,l2,weight,bias,tm,vm)
        _,l2,weight,bias,tm,vm=best
        torch.save(dict(weight=weight,bias=bias,mean=mean,std=std,l2=l2,seed=args.seed,feature=name),args.output/(name+'.pt'))
        families={}
        val_logits=(((all_x[val]-mean)/std)@weight+bias)
        val_families=[r['family'] for r in rows if r['split']=='val']
        for family in sorted(set(val_families)):
            mask=torch.tensor([f==family for f in val_families])
            families[family]=probe_metrics(val_logits[mask],labels[val][mask])
        results[name]=dict(dimensions=all_x.shape[-1],trainable_probe_parameters=weight.numel()+1,
            selected_l2=l2,train=tm,validation=vm,validation_by_family=families,trials=trials)
        write_json(args.output/'progress.json',dict(stage='fit',feature=name,validation=vm,elapsed_seconds=time.monotonic()-started))
    assert hash_file(args.model)==initial_hash and hash_file(args.data/'manifest.json')==data_hash
    report=dict(protocol=protocol,results=results,train_scenes=int(train.sum()),validation_scenes=int(val.sum()),
        extract_seconds=extract_seconds,elapsed_seconds=time.monotonic()-started,original_checkpoint_unchanged=True,
        frozen_core_unchanged=True,test_arrays_opened=0,seed=args.seed,reproducibility=reproducibility)
    write_json(args.output/'report.json',report)
    print(json.dumps(dict(complete=True,elapsed_seconds=report['elapsed_seconds'],
        results={k:dict(validation_side_correct=v['validation']['correct_side_decisions'],
            train_side_correct=v['train']['correct_side_decisions'],dim=v['dimensions']) for k,v in results.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('runs/hjepa/coverage_corpus_seed42/expanded_policy'))
    p.add_argument('--model',type=Path,default=Path('runs/hjepa/lora_continued_model/selected.pt'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--scene-batch',type=int,default=16)
    run(p.parse_args())
