"""Explicit reproducibility settings for newly launched seed-42 experiments."""
import os
import random
import numpy as np


def seed_everything(seed=42):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import hjepa_runtime
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    return dict(seed=seed,python_seed=seed,numpy_seed=seed,torch_seed=seed,cuda_seed=seed,
        python_hash_seed_at_process_start=os.environ.get('PYTHONHASHSEED'),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        deterministic_algorithms=True,deterministic_warn_only=True,cudnn_benchmark=False,
        cudnn_deterministic=True,allow_tf32=False,
        limitation='Best effort: nondeterministic operators warn; OpenGL rendering can differ by sparse 1/255 raster roundoff.')


def worker_environment(seed=42):
    return dict(os.environ,PYTHONHASHSEED=str(seed),CUBLAS_WORKSPACE_CONFIG=':4096:8',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
