import numpy as np


JAMMING_PDW_LOWER_BOUNDS = np.array([0.0, -10.0, 0.0, -15.0], dtype=float)


JAMMING_PDW_UPPER_BOUNDS = np.array([180.0, 10.0, 4.0, 30.0], dtype=float)


def normalize_jamming_pdw(pdw_data, lower_bounds=None, upper_bounds=None):
    lower_bounds = JAMMING_PDW_LOWER_BOUNDS if lower_bounds is None else lower_bounds
    upper_bounds = JAMMING_PDW_UPPER_BOUNDS if upper_bounds is None else upper_bounds
    pdw_range = upper_bounds - lower_bounds
    pdw_range[pdw_range == 0] = 1e-6
    return 2.0 * (pdw_data - lower_bounds) / pdw_range - 1.0
