from typing import Dict,Sequence,List
import numpy as np
from scipy.signal import find_peaks
from pdw_normalization import normalize_jamming_pdw


def normalized_pdw(sample: Dict) -> np.ndarray:
    pdw = sample["pdw_data"].astype(np.float32)
    return np.clip(normalize_jamming_pdw(pdw).astype(np.float32), -3.0, 3.0)


def change_points_to_labels(length: int, change_points: Sequence[int], radius: int) -> np.ndarray:
    labels = np.zeros(length, dtype=np.float32)
    for cp in change_points:
        start = max(0, int(cp) - radius)
        end = min(length, int(cp) + radius + 1)
        labels[start:end] = 1.0
    return labels


def probabilities_to_change_points(
    probabilities: np.ndarray,
    threshold: float = 0.5,
    min_distance: int = 20,
    prominence: float = 0.05,
) -> List[int]:
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.size == 0:
        return []
    peaks, _ = find_peaks(
        probabilities,
        height=threshold,
        distance=min_distance,
        prominence=prominence,
    )
    return [int(peak) for peak in peaks]
