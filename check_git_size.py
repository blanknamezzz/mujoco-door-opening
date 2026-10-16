"""Reject oversized Git-visible files, including already tracked artifacts.

Run before staging/committing. Ignore patterns cannot exclude files by size,
and do not exclude files which Git already tracks.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def audit(max_mib=10.):
    if max_mib <= 0:
        raise ValueError('max_mib must be positive')
    prefix = ['git', '-c', f'safe.directory={ROOT.as_posix()}']

    def paths(*args):
        return {os.fsdecode(p) for p in subprocess.check_output(
            [*prefix, *args, '-z'], cwd=ROOT).split(b'\0') if p}

    tracked = paths('ls-files')
    untracked = paths('ls-files', '--others', '--exclude-standard')
    staged = paths('diff', '--cached', '--name-only', '--diff-filter=ACMR')
    files = []
    for name in sorted(tracked | untracked | staged):
        path = ROOT/name
        if path.is_file() or name in staged:
            files.append(dict(path=name, bytes=path.stat().st_size if path.is_file() else 0,
                              tracked=name in tracked, staged=name in staged))
    # Also inspect staged blobs: a large version might already be staged while
    # the working tree version has since been replaced by something smaller.
    for entry in files:
        if entry['staged']:
            entry['staged_bytes'] = int(subprocess.check_output(
                [*prefix, 'cat-file', '-s', f':{entry["path"]}'], cwd=ROOT))
    oversized = [f for f in files if max(f['bytes'], f.get('staged_bytes', 0)) > max_mib*1024**2]
    largest = sorted(files, key=lambda f: max(f['bytes'], f.get('staged_bytes', 0)), reverse=True)[:10]
    return dict(limit_mib=max_mib, tracked_count=len(tracked), untracked_count=len(untracked),
                staged_count=len(staged), oversized=oversized, largest=largest,
                passed=not oversized)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-mib', type=float, default=10.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit(args.max_mib)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)
