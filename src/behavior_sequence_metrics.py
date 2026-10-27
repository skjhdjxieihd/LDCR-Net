from __future__ import annotations
from collections import Counter,defaultdict
from typing import Iterable,Sequence
import numpy as np


Run = tuple[int, int]


def levenshtein_distance(reference: Sequence[int], hypothesis: Sequence[int]) -> int:
    """Return the insertion/deletion/substitution distance between two label strings."""

    reference = list(reference)
    hypothesis = list(hypothesis)
    previous = list(range(len(hypothesis) + 1))
    for i, ref_label in enumerate(reference, start=1):
        current = [i]
        for j, hyp_label in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + int(ref_label != hyp_label)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def _run_alignment(reference: Sequence[Run], hypothesis: Sequence[Run]) -> list[tuple[str, int | None, int | None]]:
    """Align runs by edit distance and return match/delete/insert operations."""

    reference = [(int(label), int(duration)) for label, duration in reference]
    hypothesis = [(int(label), int(duration)) for label, duration in hypothesis]
    n, m = len(reference), len(hypothesis)
    costs = np.zeros((n + 1, m + 1), dtype=np.int64)
    for i in range(1, n + 1):
        costs[i, 0] = i
    for j in range(1, m + 1):
        costs[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            costs[i, j] = min(
                costs[i - 1, j - 1] + int(reference[i - 1][0] != hypothesis[j - 1][0]),
                costs[i - 1, j] + 1,
                costs[i, j - 1] + 1,
            )

    operations: list[tuple[str, int | None, int | None]] = []
    i, j = n, m
    while i or j:
        if i and j:
            diagonal = costs[i - 1, j - 1] + int(reference[i - 1][0] != hypothesis[j - 1][0])
            if costs[i, j] == diagonal:
                operations.append(("match" if reference[i - 1][0] == hypothesis[j - 1][0] else "substitute", i - 1, j - 1))
                i -= 1
                j -= 1
                continue
        if i and costs[i, j] == costs[i - 1, j] + 1:
            operations.append(("delete", i - 1, None))
            i -= 1
        else:
            operations.append(("insert", None, j - 1))
            j -= 1
    return list(reversed(operations))


def _edge_set(runs: Sequence[Run]) -> set[tuple[int, int]]:
    labels = [int(label) for label, _duration in runs]
    return {(source, target) for source, target in zip(labels[:-1], labels[1:]) if source != target}


def sequence_metrics(reference_runs: Sequence[Run], hypothesis_runs: Sequence[Run]) -> dict[str, float]:
    """Compute sequence edit distance, transition-edge F1, and duration NMAE."""

    reference_runs = list(reference_runs)
    hypothesis_runs = list(hypothesis_runs)
    reference_labels = [int(label) for label, _duration in reference_runs]
    hypothesis_labels = [int(label) for label, _duration in hypothesis_runs]
    max_length = max(len(reference_labels), len(hypothesis_labels), 1)
    edit_distance = levenshtein_distance(reference_labels, hypothesis_labels)

    reference_edges = _edge_set(reference_runs)
    hypothesis_edges = _edge_set(hypothesis_runs)
    intersection = reference_edges & hypothesis_edges
    precision = len(intersection) / len(hypothesis_edges) if hypothesis_edges else (1.0 if not reference_edges else 0.0)
    recall = len(intersection) / len(reference_edges) if reference_edges else (1.0 if not hypothesis_edges else 0.0)
    edge_f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    alignment = _run_alignment(reference_runs, hypothesis_runs)
    duration_numerator = 0.0
    duration_denominator = float(sum(duration for _label, duration in reference_runs))
    for operation, reference_index, hypothesis_index in alignment:
        if operation == "match":
            duration_numerator += abs(reference_runs[reference_index][1] - hypothesis_runs[hypothesis_index][1])
        elif operation == "delete":
            duration_numerator += reference_runs[reference_index][1]
        elif operation == "insert":
            duration_numerator += hypothesis_runs[hypothesis_index][1]
    duration_nmae = duration_numerator / max(duration_denominator, 1.0)
    return {
        "normalized_edit_distance": float(edit_distance / max_length),
        "transition_edge_precision": float(precision),
        "transition_edge_recall": float(recall),
        "transition_edge_f1": float(edge_f1),
        "duration_nmae": float(duration_nmae),
        "reference_word_count": float(len(reference_runs)),
        "hypothesis_word_count": float(len(hypothesis_runs)),
    }
