"""Paired, label-independent perturbations; original files are never changed."""
import numpy as np

ERROR_CONFIG = {
    'bias': [1.50, .50, .04, 1.50],
    'linear_span': [3.00, 1.00, .09, 2.50],
    'wander': [.75, .25, .02, .75],
    'jitter_std': [.08, .04, .008, .20],
    'wander_cycles': [.35, 1.25],
    'units': ['microseconds', 'MHz', 'PW ratio', 'dB'],
}


def nested_keep(n, loss, seed):
    if not 0 <= loss < 1:
        raise ValueError('loss must lie in [0,1)')
    keep = np.ones(n, dtype=bool)
    order = np.random.default_rng(seed).permutation(n)
    keep[order[:int(n * loss)]] = False
    return keep

def corrupt(sample, kind, level, seed):
    n = len(sample['pdw_data'])
    out = dict(sample)
    if kind == 'random_loss':
        keep = nested_keep(n, level, seed)
        for key in ['pdw_data', 'labels', 'clean_relative_pdw', 'radar_pdw', 'jammer_pdw', 'pair_index']:
            if key in sample:
                out[key] = np.asarray(sample[key])[keep].copy()
        boundaries = np.cumsum(keep)[np.asarray(sample['change_points'], dtype=int) - 1]
        out['change_points'] = np.unique(boundaries[(boundaries > 0) & (boundaries < sum(keep))])
        out['observed_change_points'] = out['change_points'].copy()
        out['retained_original_indices'] = np.flatnonzero(keep)
        out['loss_ratio'] = np.array([level])
        out['effective_loss_ratio'] = np.array([1 - sum(keep) / n])
        out['total_intended_jam_pulses'] = np.array([n])
        out['total_missing_jam_pulses'] = np.array([n - sum(keep)])
        out['total_observed_jam_pulses'] = np.array([sum(keep)])
    elif kind == 'receiver_error':
        if level < 0:
            raise ValueError('error level must be nonnegative')
        rng = np.random.default_rng(seed)
        x = np.linspace(0., 1., n)[:, None]
        bias = np.asarray(ERROR_CONFIG['bias']) * rng.choice([-1., 1.], 4) * rng.uniform(.75, 1.25, 4)
        span = np.asarray(ERROR_CONFIG['linear_span']) * rng.choice([-1., 1.], 4) * rng.uniform(.75, 1.25, 4)
        phase = rng.uniform(0, 2 * np.pi, 4)
        cycles = rng.uniform(*ERROR_CONFIG['wander_cycles'], 4)
        wander = np.asarray(ERROR_CONFIG['wander']) * rng.choice([-1., 1.], 4)
        field = bias + span * (x - .5) + wander * np.sin(2 * np.pi * cycles * x + phase)
        field += rng.normal(size=(n, 4)) * np.asarray(ERROR_CONFIG['jitter_std'])
        out['injected_error'] = level * field
        observed = np.asarray(sample['pdw_data'], dtype=float) + out['injected_error']
        out['pw_floor_count'] = int(sum(observed[:, 2] < .02))
        observed[:, 2] = np.maximum(observed[:, 2], .02)
        out['pdw_data'] = observed
        if 'radar_pdw' in sample:
            radar = np.asarray(sample['radar_pdw'])
            jammer = radar + observed
            jammer[:, 2] = radar[:, 2] * observed[:, 2]
            out['jammer_pdw'] = jammer
    else:
        raise ValueError(kind)
    return out
