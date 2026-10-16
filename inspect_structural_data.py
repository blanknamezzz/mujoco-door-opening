"""Audit trajectory contracts and show actual policy observations, train only."""
import argparse
import json
from pathlib import Path
import numpy as np
from train_hjepa import read_data


def inspect(data, output):
    manifest, records = read_data(data)
    rejected = 0
    frames = 0
    for ep, x in records:
        n = len(x['pixels'])
        if x['pixels'].shape != (n, 224, 224, 3) or x['pixels'].dtype != np.uint8:
            raise ValueError(f'Wrong pixel contract: {ep["file"]}')
        if len(x['action']) != n or len(x['proprio']) != n:
            raise ValueError('Unaligned observation/action/state records.')
        if not np.isfinite(x['action']).all() or not np.isfinite(x['proprio']).all():
            raise ValueError('Non-finite control/state values.')
        if n > 1 and not np.allclose(np.diff(x['time_s']), .1, atol=2e-6, rtol=0):
            raise ValueError('Nonuniform native observation interval.')
        if ep.get('positioning_only'):
            rejected += 1
            if n or ep.get('grasp_observed') or np.any(x['base']):
                raise ValueError('Refusal record contains a fictitious physical observation/action.')
        frames += n
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    specs = [s for s in manifest['scenes'] if s['split'] == 'train']
    families = list(dict.fromkeys(s['family'] for s in specs))
    fig, axes = plt.subplots(len(families), 3, figsize=(9, 2.5*len(families)), layout='constrained')
    for row, family in enumerate(families):
        scene_id = next(s['scene_id'] for s in specs if s['family'] == family)
        selected = {ep['base_id']: x for ep, x in records if ep['scene_id'] == scene_id and ep['grasp_id'] == 0}
        for col, pixels in enumerate([selected[0]['initial_rgb'], selected[0]['arrival_rgb'], selected[2]['arrival_rgb']]):
            axes[row, col].imshow(pixels)
            axes[row, col].axis('off')
            axes[row, col].set_title(f'{family}\n'+['Before position', 'Front arrival', 'Back arrival'][col], fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
    report = dict(passed=True, records=len(records), frames=frames, virtual_refusals=rejected,
                  splits={s: sum(x['split'] == s for x in manifest['scenes']) for s in ['train', 'val', 'test']},
                  observation_montage='Training scenes only; actual RGB policy inputs, no simulator segmentation.')
    (data/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/structural_data'))
    p.add_argument('--output', type=Path, default=Path('docs/hjepa_structural_dataset.png'))
    a = p.parse_args()
    inspect(a.data, a.output)
