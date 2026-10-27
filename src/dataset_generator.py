"""Exact descriptor-level generator used for the primary paired corpus."""
import os
from typing import Dict,List,Tuple
import numpy as np
from generate_discrete_signal import generate_markov_chain


JAMMING_STATE_CONFIGS: Dict[int, Dict[str, str]] = {
    1: {"name": "Fixed false target", "family": "deception"},
    2: {"name": "RGPO", "family": "deception"},
    3: {"name": "VGPO", "family": "deception"},
    4: {"name": "R-VGPO", "family": "deception"},
    5: {"name": "ISRJ", "family": "deception"},
    6: {"name": "Dense false target", "family": "deception"},
    7: {"name": "Noise suppression", "family": "suppression"},
}


FEATURE_NAMES = ["DTOA", "delta_RF", "PW_ratio", "delta_PA"]


ABSOLUTE_PDW_NAMES = ["TOA", "RF", "PW", "PA"]


DEFAULT_RADAR_CONFIG = {
    "pri_us_range": (80.0, 160.0),
    "rf_mhz_range": (8500.0, 10500.0),
    "pw_us_range": (5.0, 20.0),
    "pa_db_range": (-3.0, 3.0),
    "toa_jitter_std_us": 0.003,
    "rf_jitter_std_mhz": 0.002,
    "pw_jitter_std_us": 0.01,
    "pa_jitter_std_db": 0.2,
    "jammer_range_km": 15.0,
    "jammer_processing_delay_us": 1.0,
}


DEFAULT_MEASUREMENT_NOISE = {
    "dtoa_std_us": 0.003,
    "rf_std_mhz": 0.0008,
    "pw_ratio_std": 0.003,
    "pa_std_db": 0.15,
}


DEFAULT_JAMMING_PDW_PERTURB_CONFIG = {
    "loss_ratio": 0.05,
}


DECEPTION_JSR_DB_RANGE = (14.0, 16.0)


DECEPTION_JSR_STD_DB = 0.12


VGPO_RF_PULL_MHZ_RANGE = (4.0, 8.0)


R_VGPO_RF_PLATFORM_MHZ_RANGE = (4.0, 8.0)


LIGHT_SPEED_M_PER_S = 299_792_458.0


def create_random_transition_matrix(num_states: int) -> np.ndarray:
    if num_states <= 1:
        raise ValueError("Number of states must be greater than 1.")
    transition_matrix = np.random.rand(num_states, num_states)
    np.fill_diagonal(transition_matrix, 0.0)
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    if np.any(zero_rows):
        transition_matrix[zero_rows, :] = 1.0 / (num_states - 1)
        np.fill_diagonal(transition_matrix, 0.0)
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
    return transition_matrix / np.maximum(row_sums, 1e-12)


def sample_radar_parameters(radar_config: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
    jammer_range_km = radar_config["jammer_range_km"]
    one_way_delay_us = jammer_range_km * 1000.0 / LIGHT_SPEED_M_PER_S * 1e6
    processing_delay_us = radar_config["jammer_processing_delay_us"]
    dtoa_reference_us = 2.0 * one_way_delay_us + processing_delay_us
    return {
        "pri_us": np.random.uniform(*radar_config["pri_us_range"]),
        "rf_mhz": np.random.uniform(*radar_config["rf_mhz_range"]),
        "pw_us": np.random.uniform(*radar_config["pw_us_range"]),
        "pa_db": np.random.uniform(*radar_config["pa_db_range"]),
        "jammer_range_km": jammer_range_km,
        "one_way_propagation_delay_us": one_way_delay_us,
        "jammer_processing_delay_us": processing_delay_us,
        "dtoa_reference_us": dtoa_reference_us,
    }


def build_paired_radar_pdw(pair_index: np.ndarray, radar_params: Dict[str, float], radar_config: Dict) -> np.ndarray:
    pair_index = pair_index.astype(float)
    # In this scenario, radar_pdw[:, 0] is the cooperative radar transmit time.
    toa = radar_params["pri_us"] * pair_index
    toa += np.random.normal(0.0, radar_config["toa_jitter_std_us"], len(pair_index))
    rf = np.full(len(pair_index), radar_params["rf_mhz"])
    rf += np.random.normal(0.0, radar_config["rf_jitter_std_mhz"], len(pair_index))
    pw = np.full(len(pair_index), radar_params["pw_us"])
    pw += np.random.normal(0.0, radar_config["pw_jitter_std_us"], len(pair_index))
    pa = np.full(len(pair_index), radar_params["pa_db"])
    pa += np.random.normal(0.0, radar_config["pa_jitter_std_db"], len(pair_index))
    pw = np.maximum(pw, 0.2)
    return np.vstack([toa, rf, pw, pa]).T


def compute_relative_pdw(radar_pdw: np.ndarray, jammer_pdw: np.ndarray) -> np.ndarray:
    dtoa = jammer_pdw[:, 0] - radar_pdw[:, 0]
    delta_rf = jammer_pdw[:, 1] - radar_pdw[:, 1]
    pw_ratio = jammer_pdw[:, 2] / np.maximum(radar_pdw[:, 2], 1e-6)
    delta_pa = jammer_pdw[:, 3] - radar_pdw[:, 3]
    return np.vstack([dtoa, delta_rf, pw_ratio, delta_pa]).T


def add_relative_measurement_noise(relative_pdw: np.ndarray, noise_config: Dict[str, float]) -> np.ndarray:
    noisy = relative_pdw.copy()
    noisy[:, 0] += np.random.normal(0.0, noise_config["dtoa_std_us"], len(noisy))
    noisy[:, 1] += np.random.normal(0.0, noise_config["rf_std_mhz"], len(noisy))
    noisy[:, 2] += np.random.normal(0.0, noise_config["pw_ratio_std"], len(noisy))
    noisy[:, 3] += np.random.normal(0.0, noise_config["pa_std_db"], len(noisy))
    noisy[:, 2] = np.maximum(noisy[:, 2], 0.02)
    return noisy


def make_missing_pulse_mask(num_pulses: int, loss_ratio: float, rng=None) -> np.ndarray:
    if num_pulses <= 0:
        return np.zeros(0, dtype=bool)
    if loss_ratio < 0.0 or loss_ratio >= 1.0:
        raise ValueError("loss_ratio must satisfy 0.0 <= loss_ratio < 1.0")

    survivor_mask = np.ones(num_pulses, dtype=bool)
    num_to_lose = int(num_pulses * loss_ratio)
    if num_to_lose > 0:
        choice = np.random.choice if rng is None else rng.choice
        indices_to_lose = choice(num_pulses, num_to_lose, replace=False)
        survivor_mask[indices_to_lose] = False
    return survivor_mask


def apply_missing_pulses_to_segment(segment: Dict[str, np.ndarray], loss_ratio: float, rng=None) -> Dict[str, np.ndarray]:
    survivor_mask = make_missing_pulse_mask(segment["pdw_data"].shape[0], loss_ratio, rng=rng)
    pulse_level_keys = {"pdw_data", "clean_relative_pdw", "radar_pdw", "jammer_pdw", "pair_index"}
    filtered = {}
    for key, value in segment.items():
        if key in pulse_level_keys:
            filtered[key] = value[survivor_mask]
        else:
            filtered[key] = value
    filtered["survivor_mask"] = survivor_mask
    filtered["missing_indices"] = np.flatnonzero(~survivor_mask).astype(np.int64)
    filtered["num_missing_pulses"] = np.array([np.count_nonzero(~survivor_mask)], dtype=np.int64)
    return filtered


def piecewise_pull(n: int, start: float, stop: float, dwell=0.2, pull=0.55, hold=0.2) -> np.ndarray:
    dwell_len = max(1, int(n * dwell))
    pull_len = max(1, int(n * pull))
    hold_len = max(1, int(n * hold))
    if dwell_len + pull_len + hold_len > n:
        hold_len = max(0, n - dwell_len - pull_len)
    rest_len = max(0, n - dwell_len - pull_len - hold_len)
    return np.concatenate([
        np.full(dwell_len, start),
        np.linspace(start, stop, pull_len),
        np.full(hold_len, stop),
        np.full(rest_len, stop),
    ])[:n]


def piecewise_step(n: int, first_level: float, second_level: float, dwell=0.2, pull=0.55, hold=0.2) -> np.ndarray:
    dwell_len = max(1, int(n * dwell))
    pull_len = max(1, int(n * pull))
    hold_len = max(1, int(n * hold))
    if dwell_len + pull_len + hold_len > n:
        hold_len = max(0, n - dwell_len - pull_len)
    rest_len = max(0, n - dwell_len - pull_len - hold_len)
    return np.concatenate([
        np.full(dwell_len, first_level),
        np.full(pull_len, second_level),
        np.full(hold_len, second_level),
        np.full(rest_len, second_level),
    ])[:n]


def build_jammer_from_relative(radar_pdw: np.ndarray, relative_pdw: np.ndarray) -> np.ndarray:
    jammer_toa = radar_pdw[:, 0] + relative_pdw[:, 0]
    jammer_rf = radar_pdw[:, 1] + relative_pdw[:, 1]
    jammer_pw = radar_pdw[:, 2] * relative_pdw[:, 2]
    jammer_pa = radar_pdw[:, 3] + relative_pdw[:, 3]
    return np.vstack([jammer_toa, jammer_rf, jammer_pw, jammer_pa]).T


def generate_segment_pair_indices(state_code: int, num_jam_pulses: int) -> Tuple[np.ndarray, Dict[str, int]]:
    if state_code == 1:
        multiplicity = np.random.randint(3, 6)
    elif state_code == 5:
        multiplicity = np.random.randint(4, 8)
    elif state_code == 6:
        multiplicity = np.random.randint(16, 25)
    else:
        multiplicity = 1
    num_radar_pulses = int(np.ceil(num_jam_pulses / multiplicity))
    pair_index = np.repeat(np.arange(num_radar_pulses), multiplicity)[:num_jam_pulses]
    return pair_index, {"multiplicity": multiplicity, "num_radar_pulses": num_radar_pulses}


def generate_relative_model(
    state_code: int,
    pair_index: np.ndarray,
    radar_params: Dict[str, float],
    aux: Dict[str, int],
) -> np.ndarray:
    n = len(pair_index)
    dtoa = np.zeros(n)
    delta_rf = np.zeros(n)
    pw_ratio = np.ones(n)
    delta_pa = np.zeros(n)

    base_delay = radar_params["dtoa_reference_us"]
    deception_jsr_db = np.random.uniform(*DECEPTION_JSR_DB_RANGE)
    multiplicity = aux["multiplicity"]
    local_order = np.arange(n) % multiplicity

    if state_code == 1:
        # Fixed false targets: a small number of stable, radar-matched delay replicas.
        spacing = np.random.uniform(3.0, 5.0)
        offset = np.random.uniform(4.0, 8.0)
        dtoa = base_delay + offset + spacing * local_order
        delta_rf = np.random.normal(0.0, 0.0008, n)
        pw_ratio = np.ones(n)
        delta_pa = deception_jsr_db + np.random.normal(0.0, DECEPTION_JSR_STD_DB, n)

    elif state_code == 2:
        # RGPO: large DTOA pull-off only; RF stays almost exactly radar-matched.
        max_pull = np.random.uniform(38.0, 55.0)
        start_delay = base_delay + np.random.uniform(1.0, 3.0)
        dtoa_by_radar = piecewise_pull(aux["num_radar_pulses"], start_delay, start_delay + max_pull, dwell=0.10, pull=0.80, hold=0.10)
        dtoa = dtoa_by_radar[pair_index]
        delta_rf = np.random.normal(0.0, 0.0008, n)
        pw_ratio = np.ones(n)
        delta_pa = deception_jsr_db + np.random.normal(0.0, DECEPTION_JSR_STD_DB, n)

    elif state_code == 3:
        # VGPO: downward delta_RF pull-off only; DTOA is almost flat.
        max_rf_mhz = -np.random.uniform(*VGPO_RF_PULL_MHZ_RANGE)
        dtoa = base_delay + np.random.uniform(1.0, 3.0) + np.random.normal(0.0, 0.015, n)
        delta_rf_by_radar = piecewise_pull(aux["num_radar_pulses"], 0.0, max_rf_mhz, dwell=0.10, pull=0.80, hold=0.10)
        delta_rf = delta_rf_by_radar[pair_index] + np.random.normal(0.0, 0.0015, n)
        pw_ratio = np.ones(n)
        delta_pa = deception_jsr_db + np.random.normal(0.0, DECEPTION_JSR_STD_DB, n)

    elif state_code == 4:
        # R-VGPO: upward DTOA pull-off; delta_RF steps down to a lower platform.
        num_radar = aux["num_radar_pulses"]
        max_pull = np.random.uniform(38.0, 55.0)
        start_delay = base_delay + np.random.uniform(1.0, 3.0)
        dtoa_by_radar = piecewise_pull(num_radar, start_delay, start_delay + max_pull, dwell=0.10, pull=0.80, hold=0.10)
        lower_rf_platform = -np.random.uniform(*R_VGPO_RF_PLATFORM_MHZ_RANGE)
        delta_rf_mhz = piecewise_step(num_radar, 0.0, lower_rf_platform, dwell=0.10, pull=0.80, hold=0.10)
        dtoa = dtoa_by_radar[pair_index]
        delta_rf = delta_rf_mhz[pair_index] + np.random.normal(0.0, 0.0015, n)
        pw_ratio = np.ones(n)
        delta_pa = deception_jsr_db + np.random.normal(0.0, DECEPTION_JSR_STD_DB, n)

    elif state_code == 5:
        # ISRJ: several narrow sub-pulse slices inside one radar pulse.
        slice_span = np.random.uniform(0.35, 0.80)
        dtoa = base_delay + 0.20 + slice_span * local_order / max(multiplicity - 1, 1)
        delta_rf = np.random.normal(0.0, 0.0015, n)
        pw_ratio = np.random.uniform(0.04, 0.12, n)
        delta_pa = deception_jsr_db + np.random.normal(0.0, DECEPTION_JSR_STD_DB, n)

    elif state_code == 6:
        # Dense false target: many replicas over a much wider delay span.
        band_width = np.random.uniform(28.0, 45.0)
        offset = np.random.uniform(8.0, 12.0)
        if multiplicity > 1:
            dtoa = base_delay + offset + band_width * local_order / (multiplicity - 1)
        else:
            dtoa = np.full(n, base_delay + offset)
        dtoa += np.random.normal(0.0, 0.015, n)
        delta_rf = np.random.normal(0.0, 0.0015, n)
        pw_ratio = np.ones(n)
        delta_pa = deception_jsr_db + np.random.normal(0.0, DECEPTION_JSR_STD_DB, n)

    elif state_code == 7:
        # Noise suppression: high-entropy pseudo-pulses with broad RF, PW and PA statistics.
        dtoa = base_delay + np.random.uniform(0.0, radar_params["pri_us"], n)
        delta_rf = np.random.uniform(-25.0, 25.0, n)
        pw_ratio = np.random.uniform(0.25, 4.00, n)
        delta_pa = np.random.uniform(-5.0, 25.0, n) + np.random.normal(0.0, 0.8, n)

    return np.vstack([dtoa, delta_rf, pw_ratio, delta_pa]).T


def generate_jamming_segment(
    state_code: int,
    num_jam_pulses: int,
    radar_params: Dict[str, float],
    radar_config: Dict,
    noise_config: Dict[str, float],
    radar_index_offset: int = 0,
) -> Dict[str, np.ndarray]:
    pair_index_local, aux = generate_segment_pair_indices(state_code, num_jam_pulses)
    pair_index_global = pair_index_local + radar_index_offset
    radar_pdw = build_paired_radar_pdw(pair_index_global, radar_params, radar_config)
    clean_relative_pdw = generate_relative_model(state_code, pair_index_local, radar_params, aux)
    jammer_pdw = build_jammer_from_relative(radar_pdw, clean_relative_pdw)
    observed_relative_pdw = add_relative_measurement_noise(clean_relative_pdw, noise_config)
    return {
        "pdw_data": observed_relative_pdw,
        "clean_relative_pdw": clean_relative_pdw,
        "radar_pdw": radar_pdw,
        "jammer_pdw": jammer_pdw,
        "pair_index": pair_index_global.astype(np.int64),
        "num_radar_pulses": np.array([aux["num_radar_pulses"]], dtype=np.int64),
    }


def generate_jamming_pdw_dataset(
    num_sequences: int,
    output_directory: str = "generated_jamming_pdw_data_1_tx_dtoa_separable",
    num_state_changes_range: Tuple[int, int] = (4, 10),
    num_pulses_per_segment_range: Tuple[int, int] = (80, 180),
    radar_config: Dict = None,
    noise_config: Dict = None,
    perturb_config: Dict = None,
    seed: int = None,
) -> List[Dict[str, np.ndarray]]:
    if seed is not None:
        np.random.seed(seed)
    radar_config = radar_config or DEFAULT_RADAR_CONFIG
    noise_config = noise_config or DEFAULT_MEASUREMENT_NOISE
    perturb_config = perturb_config or DEFAULT_JAMMING_PDW_PERTURB_CONFIG
    loss_ratio = float(perturb_config.get("loss_ratio", 0.0))
    loss_seed = perturb_config.get("loss_seed", seed)
    loss_rng = np.random.default_rng(loss_seed)
    if loss_ratio < 0.0 or loss_ratio >= 1.0:
        raise ValueError("loss_ratio must satisfy 0.0 <= loss_ratio < 1.0")
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory)

    dataset = []
    num_states = len(JAMMING_STATE_CONFIGS)
    if output_directory:
        print(f"Starting generation of {num_sequences} radar-paired jamming PDW sequences...")

    for i in range(num_sequences):
        radar_params = sample_radar_parameters(radar_config)
        num_state_changes = np.random.randint(num_state_changes_range[0], num_state_changes_range[1] + 1)
        transition_matrix = create_random_transition_matrix(num_states)
        initial_state_idx = np.random.randint(0, num_states)
        state_indices = generate_markov_chain(transition_matrix, initial_state_idx, num_state_changes)

        rel_segments, clean_rel_segments, radar_segments, jammer_segments = [], [], [], []
        pair_indices, labels, change_points, state_codes_per_segment = [], [], [], []
        radar_axis_change_points, intended_jam_axis_change_points = [], []
        survivor_masks, missing_indices_per_segment, missing_counts = [], [], []
        current_jam_count = 0
        current_intended_jam_count = 0
        current_radar_offset = 0

        for state_idx in state_indices:
            state_code = int(state_idx) + 1
            segment_len = np.random.randint(num_pulses_per_segment_range[0], num_pulses_per_segment_range[1] + 1)

            segment = generate_jamming_segment(
                state_code=state_code,
                num_jam_pulses=segment_len,
                radar_params=radar_params,
                radar_config=radar_config,
                noise_config=noise_config,
                radar_index_offset=current_radar_offset,
            )
            segment = apply_missing_pulses_to_segment(segment, loss_ratio, rng=loss_rng)
            n = segment["pdw_data"].shape[0]
            if n == 0:
                current_radar_offset += int(segment["num_radar_pulses"][0])
                current_intended_jam_count += segment_len
                continue

            if current_jam_count > 0:
                change_points.append(current_jam_count)
                radar_axis_change_points.append(current_radar_offset)
                intended_jam_axis_change_points.append(current_intended_jam_count)

            rel_segments.append(segment["pdw_data"])
            clean_rel_segments.append(segment["clean_relative_pdw"])
            radar_segments.append(segment["radar_pdw"])
            jammer_segments.append(segment["jammer_pdw"])
            pair_indices.append(segment["pair_index"])
            labels.extend([state_code] * n)
            state_codes_per_segment.append(state_code)
            survivor_masks.append(segment["survivor_mask"])
            missing_indices_per_segment.append(segment["missing_indices"])
            missing_counts.append(segment["num_missing_pulses"][0])
            current_jam_count += n
            current_intended_jam_count += segment_len
            current_radar_offset += int(segment["num_radar_pulses"][0])

        if not rel_segments:
            continue

        sample = {
            "pdw_data": np.concatenate(rel_segments, axis=0),
            "clean_relative_pdw": np.concatenate(clean_rel_segments, axis=0),
            "radar_pdw": np.concatenate(radar_segments, axis=0),
            "jammer_pdw": np.concatenate(jammer_segments, axis=0),
            "pair_index": np.concatenate(pair_indices, axis=0),
            "labels": np.array(labels, dtype=np.int64).reshape(-1, 1),
            "change_points": np.array(change_points, dtype=np.int64),
            "observed_change_points": np.array(change_points, dtype=np.int64),
            "radar_axis_change_points": np.array(radar_axis_change_points, dtype=np.int64),
            "intended_jam_axis_change_points": np.array(intended_jam_axis_change_points, dtype=np.int64),
            "jamming_types_per_segment": np.array(state_codes_per_segment, dtype=np.int64),
            "segment_survivor_masks": np.array(survivor_masks, dtype=object),
            "segment_missing_indices": np.array(missing_indices_per_segment, dtype=object),
            "segment_missing_counts": np.array(missing_counts, dtype=np.int64),
            "total_intended_jam_pulses": np.array([current_intended_jam_count], dtype=np.int64),
            "total_observed_jam_pulses": np.array([current_jam_count], dtype=np.int64),
            "total_missing_jam_pulses": np.array([int(np.sum(missing_counts))], dtype=np.int64),
            "effective_loss_ratio": np.array(
                [float(np.sum(missing_counts)) / max(current_intended_jam_count, 1)],
                dtype=np.float32,
            ),
            "loss_ratio": np.array([loss_ratio], dtype=np.float32),
            "feature_names": np.array(FEATURE_NAMES),
            "absolute_pdw_names": np.array(ABSOLUTE_PDW_NAMES),
            "radar_params": radar_params,
        }
        dataset.append(sample)
        if output_directory and ((i + 1) % 100 == 0 or (i + 1) == num_sequences):
            print(f"  Generated {i + 1}/{num_sequences} sequences...")

    if output_directory:
        output_path = os.path.join(output_directory, "jamming_pdw_dataset.npz")
        np.savez(
            output_path,
            data=dataset,
            state_configs=JAMMING_STATE_CONFIGS,
            feature_names=np.array(FEATURE_NAMES),
            absolute_pdw_names=np.array(ABSOLUTE_PDW_NAMES),
            radar_config=radar_config,
            measurement_noise=noise_config,
            perturb_config=perturb_config,
            loss_ratio=np.array([loss_ratio], dtype=np.float32),
            change_point_axis_note=np.array([
                "sample['change_points'] and sample['observed_change_points'] are compressed observed-PDW indices; "
                "sample['radar_axis_change_points'] are original radar pulse-axis indices."
            ]),
            separable_design=np.array([True]),
            design_note=np.array([
                "Strongly separable diagnostic dataset: class parameter ranges are intentionally non-overlapping."
            ]),
            allow_pickle=True,
        )
        print(f"Dataset successfully saved to: {output_path}")
    return dataset
