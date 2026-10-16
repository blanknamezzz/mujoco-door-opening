"""Fetch the pinned official H-JEPA source and install the scene dependencies."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = 'https://github.com/kevinghst/H-JEPA.git'
COMMIT = '97054ecaff83e13839195ec4b92d74c08a0127c5'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-install', action='store_true')
    args = p.parse_args()
    destination = ROOT/'external'/'H-JEPA'
    if not destination.exists():
        subprocess.run(['git', 'clone', URL, str(destination)], check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', COMMIT], check=True)
    actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise SystemExit(f'Existing checkout is {actual}; expected {COMMIT}. Preserve local work before changing it.')
    if not args.skip_install:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                        '-r', str(ROOT/'requirements-hjepa.txt')], check=True)
    print(f'Official H-JEPA source ready: {destination}\nRevision: {actual}')


if __name__ == '__main__':
    main()
