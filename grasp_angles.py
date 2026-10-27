"""Shared discrete approach-angle grids for supervised learning and REINFORCE."""
import math

LEGACY_ANGLES = tuple(float(a) for a in range(-20, 21, 2))
DIVERSE_ANGLES = tuple(sorted((*LEGACY_ANGLES, -90., -60., -45., -30., 30., 45., 60., 90.)))


def validate_angles(values, require_zero=False):
    angles = [float(a) for a in values]
    if len(angles) < 2 or not all(math.isfinite(a) for a in angles):
        raise ValueError('Provide at least two finite angles.')
    if len(set(angles)) != len(angles):
        raise ValueError('Angles must be unique; duplicate actions distort exploration.')
    if require_zero and 0. not in angles:
        raise ValueError('Include 0 degrees for the fixed-zero comparison baseline.')
    return sorted(angles)


def add_angle_arguments(parser):
    parser.add_argument('--angle-grid', choices=['diverse29', 'legacy21'],
                        help='Default: diverse29 (legacy fine grid plus +/-30,45,60,90).')
    parser.add_argument('--angles-deg', nargs='+', type=float, help='Explicit candidate angles.')
    parser.add_argument('--angle-min', type=float)
    parser.add_argument('--angle-max', type=float)
    parser.add_argument('--angle-step', type=float)


def resolve_angles(args, require_zero=False):
    preset = getattr(args, 'angle_grid', None)
    explicit = getattr(args, 'angles_deg', None)
    bounds = [getattr(args, n, None) for n in ('angle_min', 'angle_max', 'angle_step')]
    ranged = any(v is not None for v in bounds)
    if sum((preset is not None, explicit is not None, ranged)) > 1:
        raise ValueError('Choose one of --angle-grid, --angles-deg, or range arguments.')
    if explicit is not None:
        values = explicit
    elif ranged:
        low, high, step = [v if v is not None else d for v, d in zip(bounds, (-20., 20., 2.))]
        if not all(math.isfinite(v) for v in (low, high, step)) or step <= 0 or high <= low:
            raise ValueError('Angle range must be finite and increasing, with positive step.')
        count = (high-low)/step
        if count > 10000 or not math.isclose(count, round(count), abs_tol=1e-8):
            raise ValueError('Angle range must contain at most 10001 points and be divisible by step.')
        values = [low+i*step for i in range(round(count)+1)]
    else:
        values = LEGACY_ANGLES if preset == 'legacy21' else DIVERSE_ANGLES
    return validate_angles(values, require_zero)


def has_explicit_angles(args):
    return any(getattr(args, n, None) is not None for n in
               ('angle_grid', 'angles_deg', 'angle_min', 'angle_max', 'angle_step'))
