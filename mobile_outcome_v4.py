"""Matched outcome heads and a separately fitted observation decision.

Both losses have identical parameter capacity. Only existing encoder LoRA and
P/G heads train in phase one; phase two freezes them and fits the independent
inspection head using train-only frozen continuation choices.
"""
from copy import deepcopy
import math
import torch
from torch import nn
from torch.nn import functional as F
from mobile_active_v3 import base_commands,relative_command,pose_from_state,INSPECT_POSE

SUCCESS,OTHER,COLLISION,REFUSAL=range(4)


def record_outcome(record):
    if record['success'] and record['collision']:
        raise ValueError('Successful opening cannot simultaneously be a collision.')
    if record.get('refused'):return REFUSAL
    return SUCCESS if record['success'] else COLLISION if record['collision'] else OTHER


def outcome_values(logits,risk=1.):
    if logits.shape[-1] not in (3,4) or not math.isfinite(risk) or risk<1:
        raise ValueError('Three/four outcome logits and risk>=1 required.')
    values=logits.new_tensor([1.,-1.,-1.-risk,0.][:logits.shape[-1]],dtype=torch.float32)
    return (logits.float().softmax(-1)*values).sum(-1)


def outcome_loss(logits,labels,kind,risk=1.):
    if kind=='factorized':return F.cross_entropy(logits.float(),labels.long())
    if kind!='scalar':raise ValueError('Unknown outcome supervision.')
    values=logits.new_tensor([1.,-1.,-1.-risk,0.][:logits.shape[-1]],dtype=torch.float32)
    return F.mse_loss(outcome_values(logits,risk),values[labels.long()])


def convert_head(head,outputs,extra_inputs=0,cost_bias=.18):
    if head.net[-1].out_features!=1:raise ValueError('Scalar-head initialization required.')
    new=deepcopy(head);first=head.net[0];last=head.net[-1]
    if extra_inputs:
        new.net[0]=nn.Linear(first.in_features+extra_inputs,first.out_features,
                             device=first.weight.device,dtype=first.weight.dtype)
        with torch.no_grad():
            new.net[0].weight.zero_();new.net[0].weight[:,:first.in_features].copy_(first.weight)
            new.net[0].bias.copy_(first.bias)
    new.net[-1]=nn.Linear(last.in_features,outputs,device=last.weight.device,dtype=last.weight.dtype)
    with torch.no_grad():
        new.net[-1].weight.zero_();new.net[-1].bias.zero_()
        new.net[-1].weight[0].copy_(last.weight[0]);new.net[-1].bias[0].copy_(last.bias[0])
        new.net[-1].weight[2].copy_(-last.weight[0]);new.net[-1].bias[2].copy_(-last.bias[0]-2.)
        if outputs==5:new.net[-1].bias[4]=math.log(math.expm1(cost_bias))
    return new


def configure_outcomes(model):
    old_position=model.position_policy
    model.position_policy=convert_head(old_position,3)
    model.grasp_policy=convert_head(model.grasp_policy,3)
    # Current goal command + declared risk and planner mode are available at
    # decision time. Layout labels, post images and actual outcomes are absent.
    model.inspection_policy=convert_head(old_position,5,extra_inputs=2)
    model.config['mobile_outcome_v4']=dict(outcomes=['success','other_failure','collision'],
        inspection_outcomes=['success','other_failure','collision','refusal'],
        inspection_cost='softplus separate scalar; actual measured path/time target',
        training='matched scalar expected-return MSE versus factorized outcome CE',
        encoder='continue existing rank8 LoRA; frozen original core',
        gate='freeze selected P/G and LoRA, then fit independent inspection head')


def load_outcome_checkpoint(path,device='cpu'):
    from hjepa_model import DoorHJEPA,UPSTREAM_COMMIT
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    if checkpoint.get('upstream_commit')!=UPSTREAM_COMMIT or not checkpoint['config'].get('mobile_outcome_v4'):
        raise ValueError('Expected a pinned-upstream v4 outcome checkpoint.')
    model=DoorHJEPA(checkpoint['config'])
    configure_outcomes(model);model.load_state_dict(checkpoint['model']);model.to(device)
    # State normalization participates in GPU inference just like the model.
    # Keep the archived CPU state dictionary intact for independent audits.
    for key in ('state_mean','state_std'):checkpoint[key]=checkpoint[key].to(device)
    return model.eval(),checkpoint


def position_logits(model,front,other,state):
    latents=torch.cat([front.expand(2,-1),other.expand(2,-1)],0)
    return model.position_policy(torch.cat([latents,torch.as_tensor(base_commands(state)[:4],device=front.device)],-1)).float()


def gate_features(front,state,risk,guard):
    command=relative_command(INSPECT_POSE,pose_from_state(state))
    extra=torch.as_tensor([*command,risk/3.,float(guard)],device=front.device,dtype=front.dtype)
    return torch.cat([front,extra[None]],-1)


def inspection_value(model,features,risk=1.):
    raw=model.inspection_policy(features).float()
    return outcome_values(raw[...,:4],risk)-F.softplus(raw[...,4]),raw


def station_values(logits,risk=1.,mask=None):
    values=outcome_values(logits,risk)
    if mask is not None:
        mask=torch.as_tensor(mask,device=values.device,dtype=torch.bool)
        if mask.shape!=values.shape:raise ValueError('One constraint per station required.')
        values=values.masked_fill(~mask,-1e6)
    return torch.cat([values,values.new_zeros(1)])


def continuation_class(scene,p1,g1,risk=1.,mask=None):
    scan=scene['branches'][1]['records'][0]['inspection']
    if scan['unsafe_abort']:return COLLISION if scan['collision'] else OTHER
    b=int(station_values(p1,risk,mask).argmax())
    if b==4:return REFUSAL
    g=int(outcome_values(g1[b],risk).argmax())
    return record_outcome(scene['branches'][1]['records'][2*b+g])
