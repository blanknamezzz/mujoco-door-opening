"""Separate action-selection failures from physical candidate limitations."""
import argparse
from collections import Counter
import json
from pathlib import Path
import hjepa_runtime
import torch
from evaluate_structural_training import evaluate_records
from hjepa_model import load_checkpoint
from train_hjepa import read_data


def diagnose(data, checkpoint, output, splits=('train', 'val')):
    torch.set_num_threads(2)
    manifest, records = read_data(data, required_splits=splits, include_splits=splits)
    model, source = load_checkpoint(checkpoint, 'cpu')
    report = {}
    for split in splits:
        summary, decisions = evaluate_records(model, source, records, manifest, split=split)
        for decision in decisions:
            episodes = [ep for ep, _ in records if ep['scene_id'] == decision['scene_id']]
            best_by_base = {str(b): max(ep['reward'] for ep in episodes if ep['base_id'] == b)
                            for b in sorted({ep['base_id'] for ep in episodes})}
            decision['actual_best_reward_by_base'] = best_by_base
            decision['return_regret'] = max(best_by_base.values())-decision['reward']
        summary['outcomes'] = dict(Counter(d['result']['outcome'] for d in decisions))
        summary['mean_return_regret'] = sum(d['return_regret'] for d in decisions)/len(decisions)
        report[split] = dict(summary=summary, decisions=decisions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v['summary'] for k, v in report.items()}, indent=2))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=Path('runs/hjepa/structural_data'))
    p.add_argument('--checkpoint', type=Path, default=Path('runs/hjepa/structural_model/selected.pt'))
    p.add_argument('--output', type=Path, default=Path('runs/hjepa/lora_diagnosis/before.json'))
    a = p.parse_args()
    diagnose(a.data, a.checkpoint, a.output)
