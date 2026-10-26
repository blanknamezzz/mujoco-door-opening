"""Task-specific outcome decomposition inspired by 2026 reward-model papers.

Component labels and preferences are training targets, never policy inputs.
Progress/milestones are auxiliary predictions, not repeatable terminal bonuses.
"""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

COMPONENTS = ('success', 'collision', 'positioned', 'aligned', 'grasped', 'safe_progress', 'shortfall')
VERSION = 'outcome_components_v2'


def terminal_utility(success, collision, abstained=False):
    return 0. if abstained else 2.*float(success)-1.-float(collision)


def outcome_target(episode):
    if episode.get('positioning_only'):
        raise ValueError('Refusal has no invented physical outcome-component targets.')
    positioned = bool(episode.get('current_reachable', False) and episode.get('grasp_observed', True))
    milestones = episode['milestones']
    aligned = positioned and bool(milestones[1])
    grasped = aligned and bool(milestones[2])
    return np.asarray([episode['success'], episode['collision'], positioned, aligned, grasped,
                       episode['safe_opening_fraction'], episode['shortfall']], dtype=np.float32)


class DecomposedPolicy(nn.Module):
    def __init__(self, scalar_head, feature_dim, allow_refusal=False, blend=1.):
        super().__init__()
        self.net = deepcopy(scalar_head.net)
        last = self.net[-1]
        self.legacy_head = deepcopy(last)
        self.net[-1] = nn.Linear(last.in_features, len(COMPONENTS)).to(last.weight.device)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.feature_dim = feature_dim
        self.allow_refusal = allow_refusal
        self.blend = float(blend)
        self.score_offset = 0.

    def component_logits(self, inputs):
        return self.net(inputs)

    def forward(self, inputs):
        hidden = self.net[:-1](inputs)
        probabilities = self.net[-1](hidden).sigmoid()
        # Raw score is expected independently verified terminal task utility.
        # Position/alignment/grasp/progress cannot compensate for failed tasks.
        score = 2*probabilities[..., :1]-1-probabilities[..., 1:2]
        if self.allow_refusal:
            score = score+self.score_offset  # optional validation-fitted decision threshold
            is_refusal = inputs[..., -self.feature_dim:].abs().sum(-1, keepdim=True) == 0
            score = torch.where(is_refusal, torch.zeros_like(score), score)
        if self.blend < 1:
            score = (1-self.blend)*self.legacy_head(hidden)+self.blend*score
        return score


def configure_decomposition(model, config=None):
    config = dict(config or dict(version=VERSION, blend=1., component_names=list(COMPONENTS)))
    if config['version'] != VERSION:
        raise ValueError('Unknown decomposed reward version.')
    if not isinstance(model.position_policy, DecomposedPolicy):
        model.position_policy = DecomposedPolicy(model.position_policy, 4, True, config.get('blend', 1.))
        model.grasp_policy = DecomposedPolicy(model.grasp_policy, 5, False, config.get('blend', 1.))
        from hjepa_model import MLP
        device = next(model.parameters()).device
        model.progress_low = MLP(int(model.config['level1']['wm']['embed_dim']), 128, 1).to(device)
        model.progress_high = MLP(int(model.config['level2']['wm']['embed_dim']), 128, 1).to(device)
    model.position_policy.blend = model.grasp_policy.blend = float(config.get('blend', 1.))
    model.position_policy.score_offset = float(config.get('position_score_offset', 0.))
    model.config['reward_decomposition'] = config
    return model


def component_loss(logits, targets):
    binary = F.binary_cross_entropy_with_logits(logits[..., :5], targets[..., :5])
    continuous = F.mse_loss(logits[..., 5:].sigmoid(), targets[..., 5:])
    return binary+continuous


def preference_loss(scores, utilities, margin=.25, temperature=.25):
    scores, utilities = scores.flatten(), utilities.detach().flatten()
    preferences = utilities[:, None] > utilities[None, :]+margin
    differences = (scores[:, None]-scores[None, :])/temperature
    if not preferences.any():
        return scores.sum()*0.
    return F.softplus(-differences[preferences]).mean()
