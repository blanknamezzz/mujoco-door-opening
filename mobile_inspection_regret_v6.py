"""Initial-view observation training with fixed-policy counterfactual regret.

The architecture and deployed v5 value comparison stay unchanged. Future
observations and execution outcomes are labels only, never gate features.
"""
import numpy as np

ARMS = ('outcome', 'joint_regret', 'decision_regret')
TEMPERATURE = .2


def paired_teacher(scene, prediction, risk=1.):
    """Query actual outcomes of frozen greedy choices, not oracle actions."""
    import torch
    from mobile_outcome_v4 import station_values, outcome_values, continuation_class
    from run_mobile_active_experiment import choose
    with torch.no_grad():
        p0 = station_values(prediction['p0'], risk, scene['maps']['initial']['mask'])
        p1 = station_values(prediction['p1'], risk, scene['maps']['fused']['mask'])
        g0 = outcome_values(prediction['g0'], risk)
        g1 = outcome_values(prediction['g1'], risk)
        scores = tuple(t.detach().cpu().numpy() for t in (p0, p0.new_tensor(-1e6), p1, g0, g1))
        direct = choose(scene, scores, 'never', 'fixed_direct_teacher')
        inspect = choose(scene, scores, 'always', 'fixed_inspect_teacher')
        direct_utility = direct['terminal_utility'] - (risk - 1.) * direct['collision']
        inspect_utility = inspect['terminal_utility'] - (risk - 1.) * inspect['collision']
        cost = inspect['observation_cost']
        label = continuation_class(scene, prediction['p1'], prediction['g1'], risk,
                                   scene['maps']['fused']['mask'])
    return dict(outcome_class=int(label), cost=float(cost), risk=float(risk),
                direct_predicted_value=float(p0.max()),
                direct_actual_utility=float(direct_utility),
                inspect_actual_utility=float(inspect_utility),
                advantage=float(inspect_utility - cost - direct_utility),
                eligible=bool(any(scene['maps']['initial']['mask'])),
                direct_choice=[direct['base_id'], direct['grasp_id']],
                inspect_choice=[inspect['base_id'], inspect['grasp_id']],
                targets_from_frozen_selected_actions=True)


def inspection_loss(raw, labels, costs, risks, direct_values, advantages, eligible, arm):
    """Equal architecture/batches, outcome auxiliary vs weighted policy regret."""
    import torch
    from torch.nn import functional as F
    from mobile_outcome_v4 import outcome_loss, outcome_values
    if arm not in ARMS:
        raise ValueError('Unknown observation objective.')
    terms = []
    predicted = torch.empty_like(costs)
    for risk in (1., 3.):
        mask = risks == risk
        if mask.any():
            terms.append(outcome_loss(raw[mask, :4], labels[mask], 'factorized', risk))
            predicted[mask] = outcome_values(raw[mask, :4], risk) - F.softplus(raw[mask, 4])
    if not terms or not bool(torch.all((risks == 1.) | (risks == 3.))):
        raise ValueError('Declared risk values 1 and 3 required.')
    outcome = torch.stack(terms).mean()
    cost = F.mse_loss(F.softplus(raw[:, 4]), costs)
    weight = advantages.abs() * eligible.to(advantages.dtype)
    decision = F.binary_cross_entropy_with_logits(
        (predicted - direct_values) / TEMPERATURE,
        (advantages > 0).to(raw.dtype), reduction='none')
    regret = (weight * decision).sum() / weight.sum().clamp_min(1e-8)
    loss = .5 * cost
    if arm != 'decision_regret':
        loss = loss + outcome
    if arm != 'outcome':
        loss = loss + regret
    return dict(loss=loss, outcome=outcome, cost=cost, regret=regret,
                predicted_inspection_value=predicted)
