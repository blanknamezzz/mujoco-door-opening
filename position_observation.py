"""Candidate-aligned fixed-camera observations, with no feasibility inputs."""
import numpy as np
import torch
import math


def position_decision_logits(model, features):
    """A validation-fitted offset affects only the explicit refusal command.

    Physical candidates keep their learned ordering. With no refusal command
    (forced execution diagnostics), the offset has no effect.
    """
    logits=model.position_policy(features).squeeze(-1)
    offset=float(getattr(model,'config',{}).get('position_refusal_offset',0.))
    if not math.isfinite(offset):raise ValueError('Finite refusal offset required.')
    if offset==0:return logits
    refusal=features[...,-4:].abs().sum(-1)==0
    if (refusal.sum(-1)>1).any():raise ValueError('Duplicate refusal commands.')
    return logits+refusal.to(logits.dtype)*offset


def observe_rear_camera(exp):
    """A second fixed sensor looking at the CLOSED door, without moving robot.

    Mirror the existing sensor through the door plane, render, then restore
    calibration. Neither the robot state nor any constraint is queried/changed.
    """
    import mujoco
    from perception import observe
    cid=exp.m.camera('perception').id
    pos=exp.m.cam_pos[cid].copy(); quat=exp.m.cam_quat[cid].copy()
    try:
        rear=pos.copy();rear[1]=-rear[1]
        target=np.array([.8,0.,.85])
        backward=rear-target;backward/=np.linalg.norm(backward)
        right=np.cross([0.,0.,1.],backward);right/=np.linalg.norm(right)
        up=np.cross(backward,right)
        exp.m.cam_pos[cid]=rear
        mujoco.mju_mat2Quat(exp.m.cam_quat[cid],np.column_stack([right,up,backward]).ravel())
        mujoco.mj_forward(exp.m,exp.d)
        return observe(exp)
    finally:
        exp.m.cam_pos[cid]=pos;exp.m.cam_quat[cid]=quat
        mujoco.mj_forward(exp.m,exp.d)


def encode_position_views(model,pixels,proprio,rear_pixels=None):
    encoder=getattr(model,'encode_position',model.encode)
    front=encoder(pixels,proprio)
    mode=getattr(model,'config',{}).get('position_views','single')
    if mode=='single':
        return front
    if mode not in ('paired','repeat-front'):raise ValueError('Unknown position sensor mode.')
    if mode=='repeat-front':rear_pixels=pixels  # matched encoder-compute control, no extra observation
    if rear_pixels is None or rear_pixels.shape!=pixels.shape:
        raise ValueError('Paired sensor policy requires an actual matching rear observation.')
    return torch.stack([front,encoder(rear_pixels,proprio)],1)


def position_features(latents,conditions):
    """Batched B,N,4 commands; negative/positive Y selects front/rear sensor.

    Refusal receives the mean of both current observations. Candidate-side
    association is known command geometry; openable-side labels never enter.
    Max over candidate logits remains the same deployed selection rule.
    """
    if conditions.ndim!=3 or conditions.size(-1)!=4:raise ValueError('Expected B,N,4 base commands.')
    if latents.ndim==2:
        expanded=latents[:,None].expand(-1,conditions.size(1),-1)
    elif latents.ndim==3 and latents.size(1)==2:
        front=latents[:,0:1].expand(-1,conditions.size(1),-1)
        rear=latents[:,1:2].expand_as(front)
        expanded=torch.where(conditions[:,:,1:2]>0,rear,front)
        refusal=conditions.abs().sum(-1,keepdim=True)==0
        expanded=torch.where(refusal,latents.mean(1,keepdim=True).expand_as(front),expanded)
    else:raise ValueError('Expected one or two native position latents.')
    return torch.cat([expanded,conditions],-1)
