"""Actual native ViT forward/backward parity for spatial checkpointing."""
import argparse
from pathlib import Path
import hjepa_runtime
import torch
from experiment_reproducibility import seed_everything
from hjepa_model import load_checkpoint
from hjepa_lora import frozen_core_digest
from run_hjepa_lora_experiment import write_json


def verify(output):
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError('Preserve the prior verification output.')
    reproducibility=seed_everything(42);torch.set_num_threads(4)
    model,_=load_checkpoint(Path('runs/hjepa/lora_continued_model/selected.pt'),'cuda')
    model.requires_grad_(False)
    vit=model.model.get_level(1).encoder.encoder
    for n,p in vit.named_parameters():p.requires_grad_('lora_' in n)
    for m in vit.modules():
        if isinstance(m,torch.nn.Dropout):m.p=0.
    # Disable stochasticity for this numerical equivalence check only.
    # The wrapper captures its chunk size when installed (historical32).
    # A33-image batch exercises at least two chunks without changing weights.
    pixels=torch.rand(33,3,224,224,device='cuda');state=torch.zeros(33,19,device='cuda')
    frozen=frozen_core_digest(model);results={}
    for mode in ('backbone_cls','backbone_patch_mean'):
        model.config['position_encoder']=mode
        vit.train();vit.training=False
        direct=model.encode_position(pixels,state)
        direct.square().mean().backward()
        before={n:p.grad.clone() for n,p in vit.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        vit.train()
        checkpointed=model.encode_position(pixels,state)
        checkpointed.square().mean().backward()
        torch.testing.assert_close(checkpointed,direct,rtol=1e-4,atol=2e-5)
        after={n:p.grad for n,p in vit.named_parameters() if p.grad is not None}
        assert before.keys()==after.keys() and len(before)>0
        for name in before:torch.testing.assert_close(after[name],before[name],rtol=.002,atol=2e-6)
        results[mode]=dict(output_shape=list(direct.shape),output_max_error=float((checkpointed-direct).detach().abs().max()),
            gradient_tensors=len(before),gradient_max_error=max(float((before[n]-after[n]).abs().max()) for n in before),
            nonzero_gradient_tensors=sum(bool(v.abs().sum()>0) for v in after.values()))
        model.zero_grad(set_to_none=True)
        del direct,checkpointed,before,after
    assert frozen_core_digest(model)==frozen
    write_json(output,dict(reproducibility=reproducibility,results=results,original_core_unchanged=True,
        native_image_size=224,native_hidden_dim=192,chunk_size=32,batch_images=33))
    print(results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    verify(p.parse_args().output)
