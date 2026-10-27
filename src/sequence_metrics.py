import numpy as np
from behavior_sequence_metrics import _run_alignment,sequence_metrics


def timeline_runs(labels):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not len(labels):
        return []
    boundaries = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(int(labels[a]), int(b-a)) for a, b in zip(boundaries[:-1], boundaries[1:])]


def expand_timeline(labels, boundaries):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    boundaries = np.asarray(boundaries, dtype=np.int64).reshape(-1)
    if (len(boundaries) != len(labels) + 1 or not len(labels)
            or boundaries[0] != 0 or np.any(np.diff(boundaries) <= 0)):
        raise ValueError("labels must define consecutive nonempty intervals starting at zero")
    return np.repeat(labels, np.diff(boundaries))


def score_timelines(truth, predicted):
    truth, predicted = np.asarray(truth).reshape(-1), np.asarray(predicted).reshape(-1)
    if not len(truth) or len(truth) != len(predicted):
        raise ValueError("truth and prediction must cover the same nonempty pulse timeline")
    reference, hypothesis = timeline_runs(truth), timeline_runs(predicted)
    result = sequence_metrics(reference, hypothesis)
    # Preserve the exact legacy value as diagnostic, but repair its omission of
    # substitutions in the main duration metric. A substitution has two unmatched
    # runs and incurs the sum of their durations, just as delete + insert.
    result["legacy_duration_nmae"] = result.pop("duration_nmae")
    numerator = 0
    for operation, ref_index, pred_index in _run_alignment(reference, hypothesis):
        if operation == "match":
            numerator += abs(reference[ref_index][1] - hypothesis[pred_index][1])
        elif operation == "substitute":
            numerator += reference[ref_index][1] + hypothesis[pred_index][1]
        elif operation == "delete":
            numerator += reference[ref_index][1]
        elif operation == "insert":
            numerator += hypothesis[pred_index][1]
    result["duration_edit_nmae"] = numerator / len(truth)
    result["pulse_error_rate"] = float(np.mean(truth != predicted))
    return result
