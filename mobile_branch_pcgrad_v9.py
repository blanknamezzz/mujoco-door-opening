"""Two-branch PCGrad on trainable adapters/heads; original core stays frozen.

For two tasks, each projection uses the other's ORIGINAL gradient, as in
Algorithm 1 of Yu et al., NeurIPS 2020. Gradients are summed, matching the
unprojected objective's scale. AdamW can change the resulting direction.
"""
from __future__ import annotations

ARMS = {
    'paired_sum': dict(sampler='paired', gradient='sum'),
    'paired_pcgrad': dict(sampler='paired', gradient='pcgrad'),
}


def branch_losses(prediction, target, scene):
    """Retain v8 P + .5 G supervision, including its observability masks."""
    import torch
    from torch.nn import functional as F
    losses = []
    for branch in (0, 1):
        terms = []
        logits = prediction[f'p{branch}']
        if branch == 0 or target['post_mask']:
            terms.append(F.cross_entropy(logits.float(), torch.as_tensor(
                target[f'p{branch}'], device=logits.device, dtype=torch.long)))
        grasp = prediction[f'g{branch}']
        mask = torch.as_tensor(scene['branches'][branch]['grasp_mask'],
                               device=grasp.device, dtype=torch.bool)
        if mask.any():
            labels = torch.as_tensor(target[f'g{branch}'], device=grasp.device,
                                     dtype=torch.long)
            terms.append(.5 * F.cross_entropy(grasp[mask].float(), labels[mask]))
        # An unavailable post branch contributes exactly zero, without adding
        # labels for unobserved grasps or an unsafe post-position decision.
        losses.append(sum(terms) if terms else logits.float().sum() * 0.)
    return tuple(losses)


def gradient_stats(first, second):
    import torch
    first, second = first.float(), second.float()
    dot = torch.dot(first, second)
    n0, n1 = torch.dot(first, first), torch.dot(second, second)
    valid = bool((n0 > 0) & (n1 > 0))
    return dict(dot=float(dot), norm_direct=float(n0.sqrt()),
                norm_post=float(n1.sqrt()), cosine=float(dot / (n0*n1).sqrt()) if valid else 0.,
                cosine_defined=valid, conflicting=bool(dot < 0))


def two_task_update(first, second, mode):
    import torch
    if mode not in ('sum', 'pcgrad'):
        raise ValueError('Predeclared sum or pcgrad mode required.')
    if first.ndim != 1 or first.shape != second.shape or not first.numel():
        raise ValueError('Two nonempty matching flat gradients required.')
    first, second = first.float(), second.float()
    if not bool(torch.isfinite(first).all() & torch.isfinite(second).all()):
        raise FloatingPointError('Nonfinite branch gradient.')
    stats = gradient_stats(first, second)
    projected = mode == 'pcgrad' and stats['conflicting']
    if projected:
        dot = torch.dot(first, second)
        # Negative dot implies both norms are strictly positive. No epsilon
        # changes the exact projection for very small but nonzero gradients.
        g0 = first - dot / torch.dot(second, second) * second
        g1 = second - dot / torch.dot(first, first) * first
        update = g0 + g1
        stats['projected_pair_dot'] = float(torch.dot(g0, g1))
    else:
        update = first + second
        stats['projected_pair_dot'] = stats['dot']
    stats.update(projected=projected, direction_dot_direct=float(torch.dot(update, first)),
                 direction_dot_post=float(torch.dot(update, second)))
    return update, stats


def assign_branch_gradients(named_parameters, losses, mode):
    """One graph, two backwards, no mutation of frozen/wholly unused tensors."""
    import torch
    named = list(named_parameters)
    if len(losses) != 2 or not named or any(not p.requires_grad for _, p in named):
        raise ValueError('Exactly two losses and trainable parameters required.')
    parameters = [p for _, p in named]
    gradients = [torch.autograd.grad(losses[0], parameters, retain_graph=True, allow_unused=True),
                 torch.autograd.grad(losses[1], parameters, allow_unused=True)]
    flat = [torch.cat([(torch.zeros_like(p) if g is None else g).detach().float().reshape(-1)
                       for p, g in zip(parameters, branch)]) for branch in gradients]
    update, stats = two_task_update(*flat, mode)
    spans = {'lora': [], 'position': [], 'grasp': []}
    offset = 0
    for index, (name, parameter) in enumerate(named):
        end = offset + parameter.numel()
        parameter.grad = (None if all(branch[index] is None for branch in gradients)
                          else update[offset:end].reshape_as(parameter).to(parameter.dtype))
        group = ('lora' if 'lora_' in name else 'position' if name.startswith('position_policy.')
                 else 'grasp' if name.startswith('grasp_policy.') else None)
        if group:
            spans[group].append((offset, end))
        offset = end
    stats['groups'] = {group: gradient_stats(*[
        torch.cat([vector[a:b] for a, b in ranges]) for vector in flat])
        for group, ranges in spans.items() if ranges}
    return stats
