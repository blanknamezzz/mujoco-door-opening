"""Measure actual official-model training capacity, including optimizer state."""
import argparse
import json
from pathlib import Path
import time
import hjepa_runtime
import torch
from train_hjepa import train


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/official_data'))
    p.add_argument('--output', type=Path, default=Path('runs/hjepa/official_capacity'))
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--steps', type=int, default=2)
    p.add_argument('--activation-checkpointing', action='store_true')
    a = p.parse_args()
    torch.set_num_threads(4)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    name = 'checkpointed' if a.activation_checkpointing else 'vanilla'
    report = dict(profile='official', batch_size=a.batch_size, steps=a.steps,
                  precision='bf16-mixed', activation_checkpointing=a.activation_checkpointing,
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    try:
        train(a.data, a.output/name, steps=a.steps, batch_size=a.batch_size, device='cuda',
              activation_checkpointing=a.activation_checkpointing)
        torch.cuda.synchronize()
        report['status'] = 'ok'
    except torch.OutOfMemoryError as error:
        report.update(status='out_of_memory', error=str(error))
    report.update(seconds=time.perf_counter()-start,
                  peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output/f'{name}_batch{a.batch_size}.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('OFFICIAL_CAPACITY '+json.dumps(report), flush=True)
