"""Reversible low-rank updates: original H-JEPA tensors remain frozen.

No external PEFT dependency or merged writes. Linear weights retain their
original state-dict keys, with only lora_A/lora_B keys added.
"""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=8, alpha=16., dropout=.05):
        super().__init__()
        if not 1 <= rank <= min(linear.in_features, linear.out_features):
            raise ValueError('LoRA rank must fit both linear dimensions.')
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight, self.bias = linear.weight, linear.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(linear.weight.new_empty(rank, linear.in_features))
        self.lora_B = nn.Parameter(linear.weight.new_zeros(linear.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = float(alpha)/rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)+self.scale*F.linear(
            F.linear(self.dropout(x), self.lora_A), self.lora_B)


def install_lora(model, rank=8, alpha=16., dropout=.05):
    if any(isinstance(m, LoRALinear) for m in model.model.modules()):
        raise ValueError('LoRA is already installed.')
    model.requires_grad_(False)
    # All eligible core linear layers, including both hierarchical predictors.
    # Scalar task heads are trained normally; no original core weight is trained.
    targets = [(n, m) for n, m in model.model.named_modules()
               if isinstance(m, nn.Linear) and min(m.in_features, m.out_features) >= rank]
    for name, layer in targets:
        parent_name, _, child = name.rpartition('.')
        parent = model.model.get_submodule(parent_name) if parent_name else model.model
        setattr(parent, child, LoRALinear(layer, rank, alpha, dropout))
    for name in ('position_policy', 'grasp_policy', 'value_low', 'value_high', 'progress_low', 'progress_high'):
        if hasattr(model, name):
            getattr(model, name).requires_grad_(True)
    model.config['lora'] = dict(rank=rank, alpha=alpha, dropout=dropout,
                                targets=[n for n, _ in targets], merged=False)
    freeze_core_statistics(model)
    return dict(core_parameters=sum(p.numel() for p in model.model.parameters()),
                adapter_parameters=sum(p.numel() for n, p in model.model.named_parameters() if 'lora_' in n),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                target_layers=len(targets))


def resume_lora(model, rank=8, alpha=16., dropout=.05):
    """Continue existing low-rank weights without stacking a second adapter."""
    config = model.config.get('lora')
    if not config:
        return install_lora(model, rank, alpha, dropout)
    if (config['rank'], config['alpha'], config['dropout']) != (rank, alpha, dropout):
        raise ValueError('Continuation must use the saved LoRA rank, alpha and dropout.')
    model.requires_grad_(False)
    layers = [m for m in model.model.modules() if isinstance(m, LoRALinear)]
    if len(layers) != len(config['targets']):
        raise ValueError('Missing LoRA modules in continuation checkpoint.')
    for layer in layers:
        layer.lora_A.requires_grad_(True)
        layer.lora_B.requires_grad_(True)
    for name in ('position_policy', 'grasp_policy', 'value_low', 'value_high', 'progress_low', 'progress_high'):
        if hasattr(model, name):
            getattr(model, name).requires_grad_(True)
    freeze_core_statistics(model)
    return dict(core_parameters=sum(p.numel() for p in model.model.parameters()),
                adapter_parameters=sum(p.numel() for n, p in model.model.named_parameters() if 'lora_' in n),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                target_layers=len(layers), continued_existing_adapter=True)


def freeze_core_statistics(model):
    # BatchNorm running statistics are part of the frozen base too. Updating
    # them would invalidate exact restoration even with a zero adapter.
    for module in model.model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
        if hasattr(module, '_update_latent_action_queue'):
            # JEPA's training flag gates its mutable planner-prior queue only;
            # child adapter dropout remains in the requested training mode.
            module.training = False


def frozen_core_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.model.state_dict().items()):
        if 'lora_' not in name:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def adapter_state(model):
    return {n: v.detach().cpu().clone() for n, v in model.state_dict().items()
            if 'lora_' in n or n.startswith(('position_policy.', 'grasp_policy.', 'value_low.', 'value_high.', 'progress_low.', 'progress_high.'))}
