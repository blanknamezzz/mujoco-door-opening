"""Matched scene-pair sampling and safety-first two-branch PG selection."""
from collections import defaultdict

ARMS={
    'random_ce':dict(sampler='random',collision_weight=1.),
    'paired_ce':dict(sampler='paired',collision_weight=1.),
    'random_collision4':dict(sampler='random',collision_weight=4.),
    'paired_collision4':dict(sampler='paired',collision_weight=4.),
}


class ScenePairSampler:
    def __init__(self,scenes,kind):
        if kind not in ('random','paired'):raise ValueError('Unknown sampler.')
        if len(scenes)<2 or any(s['spec']['split']!='train' for s in scenes):
            raise ValueError('At least two training scenes required; no validation/test sampling.')
        self.scenes=list(scenes);self.kind=kind;groups=defaultdict(list)
        for scene in scenes:groups[scene['spec']['group_id']].append(scene)
        self.groups=[groups[key] for key in sorted(groups)]
        if kind=='paired' and any(len(group)!=4 or len({s['spec']['layout'] for s in group})!=4 for group in self.groups):
            raise ValueError('Paired sampling needs intact four-layout groups.')

    def sample(self,rng):
        return rng.sample(self.scenes if self.kind=='random' else rng.choice(self.groups),2)


def weighted_outcome_loss(logits,labels,collision_weight):
    import torch
    from torch.nn import functional as F
    if collision_weight not in (1.,4.):raise ValueError('Predeclared collision weights 1 or 4 required.')
    if logits.ndim!=2 or logits.shape[-1]!=3:raise ValueError('Three terminal outcome logits required.')
    weights=None if collision_weight==1. else logits.new_tensor([1.,1.,collision_weight],dtype=torch.float32)
    return F.cross_entropy(logits.float(),labels.long(),weight=weights)


def pg_selection_key(direct,always):
    # A gain in the direct branch must not hide new collisions. Within equal
    # safety, preserve post-inspection accuracy before improving direct use.
    return (-direct['collisions']-always['collisions'],always['correct_decisions'],direct['correct_decisions'],
            direct['openings']+always['openings'],direct['mean_net_utility']+always['mean_net_utility'],
            -direct['mean_path_length_m']-always['mean_path_length_m'])
