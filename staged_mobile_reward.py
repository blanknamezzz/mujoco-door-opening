"""Measured navigation/position/grasp credit for the continuous mobile task.

These values are offline training targets and post-decision diagnostics only.
No counterfactual outcomes, obstruction labels or planner results enter policy
features. Infeasibility always means the tested finite action/planner budget.
"""
import numpy as np

VERSION = 'mobile_stage_credit_v1'


def navigation_return(episode):
    distance_cost = .05*min(float(episode.get('navigation_path_length_m',episode.get('measured_path_length_m',0.)))/6.,1.)
    return (1. if episode['nav_success'] else -1.)-2.*bool(episode['nav_collision'])-distance_cost


def opening_utility(episode):
    # Navigation errors are attributed to P/navigation; no fictitious opening
    # outcome is invented for episodes where the opening skill was never run.
    if not episode['opening_tested']:
        return None
    return 2.*bool(episode['success'])-1.-bool(episode['collision'])


def credit_targets(episodes, expected_bases=4, expected_grasps=2):
    if len(episodes) != expected_bases*expected_grasps or len({e['scene_id'] for e in episodes}) != 1:
        raise ValueError('Require one complete scene counterfactual table.')
    table = {(e['base_id'],e['grasp_id']):e for e in episodes}
    if set(table) != {(b,g) for b in range(expected_bases) for g in range(expected_grasps)}:
        raise ValueError('Missing or duplicated base/grasp outcomes.')
    for b in range(expected_bases):
        group=[table[b,g] for g in range(expected_grasps)]
        if any(e['side'] not in ('front','back') for e in group):
            raise ValueError('Unknown commanded side.')
        if any(len({e[k] for e in group})!=1 for k in ('side','nav_success','nav_collision')):
            raise ValueError('The same base must have the same measured navigation prefix.')
        if any(e['success'] and not (e['nav_success'] and e['opening_tested']) for e in group):
            raise ValueError('Full success requires physical navigation and opening.')
    sides = np.array([0 if table[b,0]['side']=='front' else 1 for b in range(expected_bases)])
    if set(sides) != {0,1}:
        raise ValueError('Both physical sides required.')
    feasible = np.array([any(table[b,g]['success'] for g in range(expected_grasps)) for b in range(expected_bases)])
    side_possible = np.array([bool(feasible[sides==s].any()) for s in (0,1)])
    side_reward = np.array([2. if ok else -2. if side_possible.any() else -1. for ok in side_possible])
    navigation = np.array([navigation_return(table[b,0]) for b in range(expected_bases)],np.float32)
    base_credit = np.array([0. if not side_possible[sides[b]] else 1. if feasible[b] else -1.
                            for b in range(expected_bases)],np.float32)
    position=[]
    for b in range(expected_bases):
        if not table[b,0]['nav_success']:
            position.append(-2.+.1*navigation[b])
        else:
            position.append(side_reward[sides[b]]+.25*base_credit[b]+.1*navigation[b])
    opening = np.array([[opening_utility(table[b,g]) if table[b,g]['opening_tested'] else np.nan
                         for g in range(expected_grasps)] for b in range(expected_bases)],np.float32)
    # Preparation failures after a visible grasp are G/action failures, not a
    # made-up door outcome. They receive a distinct, explicit negative target.
    grasp = np.array([[opening[b,g] if np.isfinite(opening[b,g]) else -1. for g in range(expected_grasps)]
                      for b in range(expected_bases)],np.float32)
    return dict(version=VERSION, navigation=navigation, side_rewards=side_reward.astype(np.float32),
                base_credit=base_credit, position=np.array([*position,0.],np.float32),
                opening=opening, grasp=grasp, feasible_bases=feasible, feasible_sides=side_possible,
                grasp_training_mask=np.array([[table[b,g]['grasp_observed'] for g in range(expected_grasps)]
                                              for b in range(expected_bases)],bool))


def shared_targets(episodes, grasp_probabilities):
    """Matched control: both heads receive the old combined downstream return.

    P integrates the same frozen starting G distribution in both training arms;
    only the decomposed arm removes this dependence from its position target.
    """
    credit_targets(episodes)
    table={(e['base_id'],e['grasp_id']):e for e in episodes}
    reward=np.array([[table[b,g]['reward']/5. + .1*navigation_return(table[b,g])
                      for g in range(2)] for b in range(4)],np.float32)
    probability=np.asarray(grasp_probabilities,dtype=np.float32)
    if probability.shape != (4,2) or not np.isfinite(probability).all() or (probability<0).any():
        raise ValueError('Expected finite nonnegative frozen G probabilities.')
    if not np.allclose(probability.sum(-1),1.):
        raise ValueError('Frozen G probabilities must sum to one per base.')
    return dict(position=np.r_[(reward*probability).sum(-1),0.].astype(np.float32),grasp=reward)


def reward_contract():
    return dict(version=VERSION,scope='complete measured finite candidate/planner tables, training labels only',
        navigation='R_N = +/-1 for arrived/failed - 2*navigation_collision - 0.05*min(path_m/6,1)',
        position='R_P = R_side + 0.25*R_base + 0.1*R_N; navigation failure overrides to -2+0.1*R_N; refusal=0',
        side='openable +2; wrong side when other opens -2; neither opens -1',
        base='within an openable side: successful base +1, failed base -1; otherwise 0',
        opening='R_G = 2*full_open_and_hold - 1 - opening_collision; no navigation charge',
        preparation='visible-handle preparation failure: G=-1, no invented opening-component target',
        auxiliary='position/alignment/grasp/safe progress/shortfall remain diagnostic labels, not terminal bonuses',
        shared_control='P/G use historical combined physical return/5 plus 0.1 R_N; P expectation uses frozen initial G',
        policy_inputs='current RGB and candidate base/grasp commands only; no feasibility/outcome/planner features',
        limitation='deterministic geometry navigation and parallel-jaw skill; finite-candidate offline endpoint LoRA')
