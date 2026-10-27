"""Counterfactual reward/credit for P's door-side choice, training labels only.

Labels require complete measured action tables. No pillar, handle-face, hinge
limit, scene-family or future-outcome feature is supplied to inference.
"""
import numpy as np
import torch
from torch.nn import functional as F

VERSION = 'door_side_v1'
SIDES = ('front', 'back', 'refuse')
BASE_WEIGHT = .25


def reward_contract(objective='door-side'):
    contract=dict(version=VERSION,side_names=list(SIDES),openable_side=2.,
        wrong_side_when_other_opens=-2.,neither_side_opens=-1.,refusal=0.,
        successful_base=1.,failed_base_on_openable_side=-1.,base_weight=BASE_WEIGHT,
        source='complete measured counterfactual tables, independent of current G ranking',
        policy_inputs='initial RGB, proprioception and base commands only',
        loss='reward-gap weighted side ranking plus 0.25 within-side base ranking',
        side_score_pool='max, matching the deployed base argmax')
    if objective=='side-feasibility':
        contract.update(supervision_version='side_feasibility_v1',
            loss='BCE of each physical side max minus refusal logit, plus0.25 within-side base ranking',
            binary_targets='side has at least one measured successful base/grasp; both positive allowed',
            probability_calibration_claim=False)
    return contract


def side_reward_targets(groups, expected_grasps=4):
    """Groups are candidate-order lists of episode dicts from ONE full scene."""
    if not groups or any(not g for g in groups):
        raise ValueError('Complete nonempty action groups required.')
    scene_ids={e['scene_id'] for g in groups for e in g}
    if len(scene_ids)!=1:
        raise ValueError('Counterfactuals must belong to the same scene.')
    sides=[]; base_success=[]; base_ids=[]
    for group in groups:
        if len({e['base_id'] for e in group})!=1:
            raise ValueError('Each group must represent one base.')
        base_ids.append(group[0]['base_id'])
        if any(e.get('positioning_only') for e in group):
            if len(group)!=1 or not group[0].get('positioning_only') or group[0]['success']:
                raise ValueError('Refusal is one no-execution action.')
            sides.append(2); base_success.append(False); continue
        if len(group)!=expected_grasps or {e['grasp_id'] for e in group}!=set(range(expected_grasps)):
            raise ValueError('Missing or duplicated physical grasp outcomes.')
        labels={e.get('operating_side') for e in group}
        if len(labels)!=1 or next(iter(labels)) not in SIDES[:2]:
            raise ValueError('Physical actions must have a consistent commanded side.')
        side=next(iter(labels))
        # The side is an attribute of the candidate command, not a hidden
        # permitted-direction label. Cross-check its recorded action geometry.
        for e in group:
            if 'base_pose' in e:
                y=e['base_pose'][1]
                if not np.isfinite(y) or y==0 or ('front' if y<0 else 'back')!=side:
                    raise ValueError('Commanded side disagrees with base pose.')
        sides.append(SIDES.index(side)); base_success.append(any(e['success'] for e in group))
    if len(set(base_ids))!=len(base_ids) or sides.count(2)!=1 or set(sides)!={0,1,2}:
        raise ValueError('Need distinct bases, both physical sides and exactly one refusal.')
    sides=np.asarray(sides,dtype=np.int64)
    base_success=np.asarray(base_success,dtype=bool)
    can_open=np.asarray([bool(base_success[sides==i].any()) for i in (0,1)])
    side_rewards=np.asarray([2. if possible else -2. if can_open.any() else -1.
                             for possible in can_open]+[0.],dtype=np.float32)
    base_rewards=np.asarray([0. if s==2 or not can_open[s] else 1. if ok else -1.
                             for s,ok in zip(sides,base_success)],dtype=np.float32)
    return dict(sides=sides,side_rewards=side_rewards,base_rewards=base_rewards,
        position_rewards=side_rewards[sides]+BASE_WEIGHT*base_rewards,
        base_success=base_success,can_open=can_open)


def _rank_per_scene(scores, rewards, allowed=None, temperature=.2, margin=.25):
    gap=(rewards.detach()[:,:,None]-rewards.detach()[:,None,:]).clamp_min(0)
    if allowed is not None:
        gap=gap*allowed
    differences=scores[:,:,None]-scores[:,None,:]
    return (gap*F.softplus((margin-differences)/temperature)).sum((-1,-2))/gap.sum((-1,-2)).clamp_min(1)


def door_side_loss(scores, sides, side_rewards, base_rewards):
    """Exact finite-candidate reward ranking; never sample an oracle action."""
    if scores.ndim!=2 or not scores.size(0) or scores.shape!=sides.shape or scores.shape!=base_rewards.shape:
        raise ValueError('Expected matching batch x candidate scores, sides and base rewards.')
    if side_rewards.shape!=(scores.size(0),3) or not torch.isfinite(scores).all():
        raise ValueError('Three finite side targets required per scene.')
    if not torch.isfinite(side_rewards).all() or not torch.isfinite(base_rewards).all():
        raise ValueError('Rewards must be finite.')
    membership=sides[:,:,None]==torch.arange(3,device=sides.device)[None,None,:]
    counts=membership.sum(1)
    if not ((counts[:,:2]>0).all() and (counts[:,2]==1).all() and ((sides>=0)&(sides<=2)).all()):
        raise ValueError('Both sides and one refusal required.')
    pooled=scores[:,:,None].expand(-1,-1,3).masked_fill(~membership,-torch.inf).amax(1)
    side=_rank_per_scene(pooled,side_rewards).mean()
    within=[]
    for s in (0,1):
        eligible=(sides==s)&(side_rewards[:,s:s+1]>0)
        within.append(_rank_per_scene(scores,base_rewards,eligible[:,:,None]&eligible[:,None,:]))
    base=(torch.stack(within,-1).sum(-1)/(side_rewards[:,:2]>0).sum(-1).clamp_min(1)).mean()
    return side+BASE_WEIGHT*base, dict(side=side,base=base)


def side_feasibility_loss(scores, sides, side_rewards, base_rewards):
    """Two independent measured feasibility labels, relative to refusal.

    Keeps the same scalar P head, candidate max and deployed argmax. Unlike
    a categorical side target, both sides can be feasible simultaneously.
    This is a training objective, not a probability-calibration guarantee.
    """
    _,parts=door_side_loss(scores,sides,side_rewards,base_rewards)
    membership=sides[:,:,None]==torch.arange(3,device=sides.device)[None,None,:]
    pooled=scores[:,:,None].expand(-1,-1,3).masked_fill(~membership,-torch.inf).amax(1)
    margins=pooled[:,:2]-pooled[:,2:3]
    targets=(side_rewards[:,:2]>0).to(scores.dtype)
    binary=F.binary_cross_entropy_with_logits(margins,targets)
    return binary+BASE_WEIGHT*parts['base'],dict(side_bce=binary,base=parts['base'])


def decision_side_metrics(selected, episodes):
    """Evaluate credit AFTER choice; missing legacy metadata is not a label."""
    physical=[e for e in episodes if not e.get('positioning_only')]
    available=bool(physical) and {e.get('operating_side') for e in physical}=={'front','back'}
    if not available:
        return dict(side_metrics_available=False)
    successful_sides={e['operating_side'] for e in physical if e['success']}
    successful_bases={e['base_id'] for e in physical if e['success']}
    refused=selected.get('positioning_only',False)
    good_side=not refused and selected['operating_side'] in successful_sides
    reward=0. if refused else 2. if good_side else -2. if successful_sides else -1.
    return dict(side_metrics_available=True,chosen_side_can_open=bool(good_side),
        side_decision_correct=bool(good_side or (refused and not successful_sides)),
        wrong_side_error=bool(successful_sides and not refused and not good_side),
        within_side_base_error=bool(good_side and selected['base_id'] not in successful_bases),
        position_side_reward=reward)


def validation_key(summary, objective):
    overall=(summary['correct_task_decisions'],summary['opening_successes'],summary['mean_terminal_utility_v2'])
    if objective in ('door-side','side-feasibility'):
        if summary.get('side_evaluated_scenes')!=summary['scenes']:
            raise ValueError('Side-based selection requires complete side evaluation.')
        return (summary['correct_side_decisions'],*overall)
    return overall
