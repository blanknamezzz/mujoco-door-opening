"""Matched learning-rate / frozen-output retention objectives for LoRA PG."""
from mobile_branch_pcgrad_v9 import branch_losses

ARMS = {
    'standard': dict(lr_scale=1., retention_weight=0.),
    'low_lr': dict(lr_scale=.1, retention_weight=0.),
    'retention': dict(lr_scale=1., retention_weight=1.),
    'low_lr_retention': dict(lr_scale=.1, retention_weight=1.),
}


def pg_rates(scale):
    if scale not in (1., .1):
        raise ValueError('Only predeclared learning-rate scales 1 and .1 are allowed.')
    return dict(head=3e-4*scale, lora=2e-5*scale)


def output_kl(student, teacher):
    """KL(teacher || student), temperature1, averaged over supervised rows."""
    import torch
    from torch.nn import functional as F
    if student.ndim != 2 or student.shape[-1] != 3 or student.shape != teacher.shape:
        raise ValueError('Matching supervised three-class row logits are required.')
    if not student.shape[0]:
        raise ValueError('Empty supervision is masked by the caller.')
    reference = torch.as_tensor(teacher, device=student.device).detach().float()
    return F.kl_div(F.log_softmax(student.float(), dim=-1),
                    F.softmax(reference, dim=-1), reduction='batchmean')


def retention_components(prediction, teacher, target, scene, weight):
    if weight not in (0., 1.):
        raise ValueError('Only predeclared retention weights 0 or 1 are allowed.')
    import torch
    ce = branch_losses(prediction, target, scene)
    kl = []
    for branch in (0, 1):
        terms = []
        if branch == 0 or target['post_mask']:
            terms.append(output_kl(prediction[f'p{branch}'], teacher[f'p{branch}']))
        grasp = prediction[f'g{branch}']
        mask = torch.as_tensor(scene['branches'][branch]['grasp_mask'],
                               device=grasp.device, dtype=torch.bool)
        if mask.any():
            reference = torch.as_tensor(teacher[f'g{branch}'], device=grasp.device)
            terms.append(.5*output_kl(grasp[mask], reference[mask]))
        kl.append(sum(terms) if terms else prediction[f'p{branch}'].float().sum()*0.)
    return dict(loss=sum(ce)+weight*sum(kl), ce=sum(ce), kl=sum(kl),
                ce_direct=ce[0], ce_post=ce[1], kl_direct=kl[0], kl_post=kl[1])
