"""Official H-JEPA model factory and door task heads, with legacy compact loading."""
from pathlib import Path
import subprocess
import sys
import hjepa_runtime  # configure inherited allocator options before importing torch
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
UPSTREAM_URL = 'https://github.com/kevinghst/H-JEPA.git'
UPSTREAM_COMMIT = '97054ecaff83e13839195ec4b92d74c08a0127c5'
UPSTREAM = ROOT/'external'/'H-JEPA'


def resolve_device(device='auto'):
    if device == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable in PyTorch {torch.__version__}; install a CUDA build or use --device cpu.')
    return device


def load_upstream():
    if not (UPSTREAM/'h_jepa'/'models'/'hjepa.py').is_file():
        raise RuntimeError('Run python setup_hjepa.py first to fetch the official release.')
    revision = subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(f'Expected H-JEPA {UPSTREAM_COMMIT}, found {revision}.')
    sys.path.insert(0, str(UPSTREAM/'h_jepa'))


load_upstream()
from hjepa_utils import create_world_model, hjepa_forward  # noqa: E402
from loss import SIGReg  # noqa: E402
from models.hjepa import HJEPA  # noqa: E402
from models.jepa import JEPA, FusionEncoder, ProjectedEncoder  # noqa: E402
from models.module import Embedder, MLP, ResidualLatentMLP  # noqa: E402
from models.predictors.predictors import ARPredictor  # noqa: E402
from models.encoders.seq_encoder import SequenceMLPEncoder  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 16, 5, 2, 2), nn.GELU(),
                                 nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),
                                 nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
                                 nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(), nn.Linear(256, 64))

    def forward(self, x, interpolate_pos_encoding=True):
        return self.net(x)


def compact_config(proprio_dim=19):
    levels = {}
    for level, dim in [(1, 96), (2, 64)]:
        levels[f'level{level}'] = dict(
            wm=dict(history_size=2, rollout_n=1, embed_dim=dim, detach_pred_target=False),
            loss=dict(embed=dict(sigreg=dict(enabled=True, weight=.09))))
    return dict(num_levels=2, stride=4, clip_frames=9, proprio_dim=proprio_dim,
                profile='compact', frame_skip=1, raw_clip_frames=9,
                image_size=64, observation_interval_s=.5, **levels)


def default_config(proprio_dim=19):
    """Resolve the actual pinned pusht_hjepa_l2 YAMLs; change only action_dim.

    These two YAMLs have a single base include. Merging them is equivalent to
    Hydra composition without loading unrelated environment/trainer packages.
    """
    if not OmegaConf.has_resolver('eval'):
        register = getattr(OmegaConf, 'register_resolver', OmegaConf.register_new_resolver)
        register('eval', lambda expr: eval(expr, {'__builtins__': {}}, {}))
    directory = UPSTREAM/'h_jepa'/'config'/'train'
    root = OmegaConf.merge(OmegaConf.load(directory/'base'/'pusht.yaml'),
                           OmegaConf.load(directory/'pusht_hjepa_l2.yaml'))
    # Resolve against the original root so its cross-level references survive.
    levels = {f'level{i}': OmegaConf.to_container(root[f'level{i}'], resolve=True) for i in (1, 2)}
    levels['level1']['wm']['action_dim'] = 6
    data = {f'level{i}': OmegaConf.to_container(root.data.dataset[f'level{i}'], resolve=True) for i in (1, 2)}
    frame_skip = int(data['level1']['frameskip'])
    upper = data['level2']
    frames = max(data['level1']['num_steps'],
                 (upper['num_steps']-1)*upper['frameskip']+upper['window_size'])
    return dict(profile='official', upstream_config='pusht_hjepa_l2', num_levels=2,
                img_size=int(root.img_size), image_size=int(root.img_size), patch_size=int(root.patch_size),
                stride=int(levels['level2']['stride']), frame_skip=frame_skip,
                clip_frames=frames, raw_clip_frames=frames*frame_skip,
                proprio_dim=proprio_dim, observation_interval_s=.5,
                raw_observation_interval_s=.1, data={'dataset': data},
                loader=OmegaConf.to_container(root.loader, resolve=True),
                trainer=OmegaConf.to_container(root.trainer, resolve=True),
                optimizer=OmegaConf.to_container(root.optimizer, resolve=True),
                adaptations={'level1.wm.action_dim': {'upstream': 2, 'door': 6}}, **levels)


class DoorHJEPA(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or default_config()
        self.cfg = OmegaConf.create(self.config)
        self.is_official = self.config.get('profile') == 'official'
        if self.is_official:
            self.model, losses = create_world_model(self.cfg)
            for name, module in losses.items():
                self.add_module(name, module)
            self.register_buffer('action_mean', torch.zeros(6))
            self.register_buffer('action_std', torch.ones(6))
        else:
            self._build_compact()
        low_dim = int(self.cfg.level1.wm.embed_dim)
        high_dim = int(self.cfg.level2.wm.embed_dim)
        self.position_policy = MLP(low_dim+4, 128, 1)
        self.grasp_policy = MLP(low_dim+5, 128, 1)
        self.value_low = MLP(low_dim, 128, 1)
        self.value_high = MLP(high_dim, 128, 1)
        if self.config.get('lora'):
            from hjepa_lora import install_lora
            lora = dict(self.config['lora'])
            install_lora(self, lora['rank'], lora['alpha'], lora['dropout'])
            if self.config['lora']['targets'] != lora['targets']:
                raise ValueError('LoRA target modules differ from the saved architecture.')
        if self.config.get('activation_checkpointing'):
            self.enable_activation_checkpointing(self.config.get('encoder_chunk_size', 32))

    def _build_compact(self):
        low_encoder = FusionEncoder(
            pixel_encoder=ProjectedEncoder(ImageEncoder()),
            proprio_encoder=ProjectedEncoder(MLP(self.config['proprio_dim'], 64, 32)))
        levels = []
        for level, dim in [(1, 96), (2, 64)]:
            upper = level == 2
            encoder = (ProjectedEncoder(ResidualLatentMLP(96, 128, 64, final_ln=False))
                       if upper else low_encoder)
            action_encoder = (SequenceMLPEncoder(16, [128], 96, self.config['stride'])
                              if upper else None)
            levels.append(JEPA(
                encoder=encoder,
                predictor=ARPredictor(num_frames=2, depth=2, heads=4, mlp_dim=128,
                                      input_dim=dim, hidden_dim=dim, dim_head=16),
                action_embed=Embedder(input_dim=16 if upper else 6, emb_dim=dim, patch_embed=False),
                action_encoder=action_encoder, level=level,
                temporal_stride=self.config['stride'] if upper else 1,
                target_length=3))
        self.model = HJEPA(levels)
        self.sigreg_level1 = SIGReg(num_proj=64)
        self.sigreg_level2 = SIGReg(num_proj=64)

    def enable_activation_checkpointing(self, chunk_size=32):
        """Recompute the unchanged official ViT per image chunk to save VRAM.

        Projector BatchNorm and SIGReg still see the entire configured batch.
        No gradient accumulation or reduced model/sequence/image size is used.
        """
        if not self.is_official or hasattr(self, '_checkpointed_encoder'):
            return
        from types import SimpleNamespace
        from torch.utils.checkpoint import checkpoint
        vit = self.model.get_level(1).encoder.encoder
        original_forward = vit.forward

        def forward(pixel_values, **kwargs):
            if not vit.training or not torch.is_grad_enabled():
                return original_forward(pixel_values, **kwargs)
            def encode_chunk(chunk):
                return original_forward(chunk, **kwargs).last_hidden_state[:, :1]
            outputs = [checkpoint(encode_chunk, chunk, use_reentrant=False, preserve_rng_state=True)
                       for chunk in pixel_values.split(chunk_size)]
            return SimpleNamespace(last_hidden_state=torch.cat(outputs))
        vit.forward = forward
        self._checkpointed_encoder = True
        self.config.update(activation_checkpointing=True, encoder_chunk_size=chunk_size)

    def normalize_pixels(self, pixels):
        if pixels.shape[-2:] != (self.config['image_size'], self.config['image_size']):
            raise ValueError(f"Expected native {self.config['image_size']}x{self.config['image_size']} RGB; recollect data.")
        owned = pixels.dtype == torch.uint8
        pixels = pixels.float().div_(255.) if owned else pixels
        if self.is_official:
            shape = [1]*pixels.ndim
            shape[-3] = 3
            mean = pixels.new_tensor([.485, .456, .406]).view(shape)
            std = pixels.new_tensor([.229, .224, .225]).view(shape)
            pixels = pixels.sub_(mean).div_(std) if owned else (pixels-mean)/std
        return pixels

    def log_dict(self, *args, **kwargs):
        pass  # upstream forward's optional Lightning logger

    def predictive_loss(self, batch, stage='train'):
        def normalize(batch):
            for key in list(batch):
                if key.startswith('pixels'):
                    batch[key] = self.normalize_pixels(batch[key])
                elif key == 'action_level1' and self.is_official:
                    batch[key] = self.normalize_actions(batch[key])
        return hjepa_forward(self, dict(batch), stage, self.cfg, normalize_batch=normalize)

    def normalize_actions(self, actions):
        if not self.is_official:
            return actions
        shape = actions.shape
        grouped = actions.reshape(*shape[:-1], -1, 6)
        return torch.nan_to_num((grouped-self.action_mean)/self.action_std,
                                nan=0., posinf=0., neginf=0.).reshape(shape)

    def encode(self, pixels, proprio):
        pixels = self.normalize_pixels(pixels)
        out = self.model.get_level(1).encoder.encode_info(
            {'pixels': pixels[:, None], 'proprio': proprio[:, None]}, key='pixels')
        return out['embed']

    def encode_position(self, pixels, proprio):
        """Optional downstream probe of the same ViT before JEPA projection.

        Dynamics/value/grasp latents remain the official projected embedding.
        This adds no parameters and never changes world-model inference.
        """
        if self.config.get('position_encoder') == 'backbone_cls':
            if not self.is_official:
                raise ValueError('Backbone CLS positioning requires the official ViT.')
            return self.model.get_level(1).encoder.encoder(self.normalize_pixels(pixels)).last_hidden_state[:, 0]
        return self.encode(pixels, proprio)

    def imagine(self, latent, actions, level=1):
        """Finite candidate rollout, reusing official predictors/action encoders.

        This is task-specific candidate ranking, not the upstream continuous
        coarse-to-fine subgoal optimizer. No future IK or physics is queried.
        """
        jepa = self.model.get_level(level)
        actions = self.normalize_actions(actions)
        if level == 1:
            embedded = jepa.action_embed(actions)
        else:
            stride = self.config['stride']
            n = actions.size(1)//stride*stride
            primitive = self.model.get_level(1).action_embed(actions[:, :n])
            chunks = primitive.reshape(actions.size(0), n//stride, stride, int(self.cfg.level1.wm.embed_dim))
            embedded = jepa.action_embed(jepa.action_encoder(chunks))
            latent = jepa.encoder(latent)
        history = latent[:, None]
        history_size = int(self.cfg[f'level{level}'].wm.history_size)
        for i in range(embedded.size(1)):
            start = max(0, i-history_size+1)
            prediction = jepa.predict(history[:, -history_size:], embedded[:, start:i+1])[:, -1:]
            history = torch.cat([history, prediction], dim=1)
        return history[:, -1]


def save_checkpoint(path, model, state_mean, state_std, **metadata):
    torch.save(dict(model=model.state_dict(), config=model.config,
                    state_mean=state_mean, state_std=state_std,
                    upstream_commit=UPSTREAM_COMMIT, **metadata), path)


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if 'adapter' in checkpoint:
        import hashlib
        base = Path(checkpoint['base_checkpoint']['path'])
        with base.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != checkpoint['base_checkpoint']['sha256']:
            raise ValueError('Adapter base checkpoint hash mismatch; refusing to mix weights.')
        model, source = load_checkpoint(base, device)
        from hjepa_lora import install_lora
        config = checkpoint['config']
        lora = config['lora']
        install_lora(model, lora['rank'], lora['alpha'], lora['dropout'])
        if model.config['lora']['targets'] != lora['targets']:
            raise ValueError('Adapter target layer mismatch.')
        expected = {n for n in model.state_dict() if 'lora_' in n or
                    n.startswith(('position_policy.', 'grasp_policy.', 'value_low.', 'value_high.'))}
        if set(checkpoint['adapter']) != expected:
            raise ValueError('Adapter/task-head tensors are missing or unexpected.')
        model.load_state_dict(checkpoint['adapter'], strict=False)
        model.config.update(config)
        source.update(checkpoint.get('metadata', {}), config=config, model=model.state_dict(),
                      state_mean=checkpoint['state_mean'], state_std=checkpoint['state_std'])
        return model.eval(), source
    if checkpoint['upstream_commit'] != UPSTREAM_COMMIT:
        raise ValueError('Checkpoint requires a different upstream revision.')
    model = DoorHJEPA(checkpoint['config']).to(device)
    model.load_state_dict(checkpoint['model'])
    return model.eval(), checkpoint
