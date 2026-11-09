"""Matched image-only and current-depth conditioned P/I outcome heads."""
from copy import deepcopy
import torch
from torch import nn
from mobile_outcome_v4 import (configure_outcomes,base_commands,gate_features,
                               outcome_values,station_values,inspection_value)


MAP_FEATURE_NAMES=('known_base_collision','observed_obstruction','sampled_free_fraction',
                   'normalized_observed_clearance')


def extend_input(head,extra):
    head=deepcopy(head);old=head.net[0]
    layer=nn.Linear(old.in_features+extra,old.out_features,device=old.weight.device,dtype=old.weight.dtype)
    with torch.no_grad():
        layer.weight.zero_();layer.weight[:,:old.in_features].copy_(old.weight);layer.bias.copy_(old.bias)
    head.net[0]=layer;return head


def configure_partial_policy(model,use_map):
    model.position_policy=extend_input(model.position_policy,4)
    model.inspection_policy=extend_input(model.inspection_policy,16)
    model.config['mobile_partial_v5']=dict(use_map_features=bool(use_map),
        map_feature_names=list(MAP_FEATURE_NAMES),capacity='matched+4Pinputs,+16Iinputs',
        images_only_control='zero map feature slots; identical hard depth constraints',
        input_scope='P current reached map; I initial map only; no hidden obstacle geometry',
        initialization='v4 selected outcome checkpoint; existing rank8 LoRA continued')


def load_partial_checkpoint(path,device='cpu'):
    from hjepa_model import DoorHJEPA,UPSTREAM_COMMIT
    source=torch.load(path,map_location='cpu',weights_only=True)
    if source.get('upstream_commit')!=UPSTREAM_COMMIT or not source['config'].get('mobile_partial_v5'):
        raise ValueError('Expected a pinned-upstream v5 partial-map checkpoint.')
    use_map=source['config']['mobile_partial_v5']['use_map_features']
    model=DoorHJEPA(source['config']);configure_outcomes(model);configure_partial_policy(model,use_map)
    model.load_state_dict(source['model']);model.to(device)
    for key in ('state_mean','state_std'):source[key]=source[key].to(device)
    return model.eval(),source


def current_map_features(model,features,device):
    values=torch.as_tensor(features,device=device,dtype=torch.float32)
    if values.shape!=(4,4) or not torch.isfinite(values).all():raise ValueError('Four finite current features per station required.')
    if not model.config['mobile_partial_v5']['use_map_features']:values=values*0.
    return values


def partial_position_logits(model,front,other,state,features):
    latents=torch.cat([front.expand(2,-1),other.expand(2,-1)],0)
    commands=torch.as_tensor(base_commands(state)[:4],device=front.device)
    values=current_map_features(model,features,front.device)
    return model.position_policy(torch.cat([latents,commands,values],-1)).float()


def partial_gate_features(model,front,state,initial_features,risk=1.):
    # Only the initial map is supplied. Fused/post measurements are absent.
    features=current_map_features(model,initial_features,front.device).reshape(1,16)
    return torch.cat([gate_features(front,state,risk,True),features],-1)
