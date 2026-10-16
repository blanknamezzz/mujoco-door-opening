"""Measure real end-to-end training time and CUDA memory on this scene data."""
import argparse
import gc
import json
from pathlib import Path
import time
import hjepa_runtime  # process-local CUDA environment fix
import torch
from train_hjepa import train


def benchmark(data, output, device, batches, steps, profile='official'):
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA unavailable in PyTorch {torch.__version__}. Install a CUDA build first.')
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for batch in batches:
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        try:
            train(data, output/f'{device}_batch{batch}', steps=steps, batch_size=batch, device=device, profile=profile)
            if device == 'cuda':
                torch.cuda.synchronize()
            seconds = time.perf_counter()-start
            entry = dict(device=device, torch_version=torch.__version__, steps=steps,
                         batch_size=batch, end_to_end_seconds=seconds,
                         seconds_per_step_including_io_and_validation=seconds/steps,
                         status='ok')
            if device == 'cuda':
                entry.update(gpu=torch.cuda.get_device_name(),
                    total_memory_mib=torch.cuda.get_device_properties(0).total_memory/2**20,
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        except torch.OutOfMemoryError as error:
            entry = dict(device=device, steps=steps, batch_size=batch, status='out_of_memory', error=str(error))
        results.append(entry)
        print('BENCHMARK '+json.dumps(entry), flush=True)
        (output/f'{device}_benchmark.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/official_data'))
    p.add_argument('--output', type=Path, default=Path('runs/hjepa/official_device_benchmark'))
    p.add_argument('--profile', choices=['official', 'compact'], default='official')
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--batches', type=int, nargs='+', default=None)
    p.add_argument('--steps', type=int, default=100)
    a = p.parse_args()
    torch.set_num_threads(4)
    benchmark(a.data, a.output, a.device, a.batches or ([128] if a.profile == 'official' else [16]), a.steps, a.profile)
