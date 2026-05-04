#!/usr/bin/env python3

from copy import deepcopy
from multiprocessing import Pool
import numpy as np
import pandas as pd


def count_max_probes_with_max_overlap(probe_dict: dict, max_overlap: int = 20):
    """
    Count the maximum number of probes that can be selected per transcript
    under a hard pairwise overlap constraint.

    Assumes all probes within a transcript have the same length.
    """
    count_dict = {
        'gene': [],
        'transcript': [],
        'n_probes': [],
        'n_max_probes': [],
    }

    for gk in probe_dict.keys():
        for tk in probe_dict[gk].keys():
            probe_df = probe_dict[gk][tk].sort_values('shift').reset_index(drop=True)

            if probe_df.empty:
                count_dict['gene'].append(gk)
                count_dict['transcript'].append(tk)
                count_dict['n_probes'].append(0)
                count_dict['n_max_probes'].append(0)
                continue

            probe_length = len(probe_df.iloc[0]['target_sequence'])
            min_spacing = probe_length - max_overlap  # for 30 nt and max_overlap=20 -> 10

            last_selected_shift = None
            n_max_probes = 0

            for _, row in probe_df.iterrows():
                shift = row['shift']

                if last_selected_shift is None or shift - last_selected_shift >= min_spacing:
                    n_max_probes += 1
                    last_selected_shift = shift

            count_dict['gene'].append(gk)
            count_dict['transcript'].append(tk)
            count_dict['n_probes'].append(probe_df.shape[0])
            count_dict['n_max_probes'].append(n_max_probes)

    return pd.DataFrame(count_dict)


def select_probes_greedy_stochastic_one_df_trial(
    df,
    N_probes_per_transcript,
    N_on_bits,
    max_overlap=20,
    rng=None
):
    if df.empty:
        return df.iloc[:0].copy(), {}

    if rng is None:
        rng = np.random.default_rng()

    df = df.reset_index(drop=True)

    # All probes have same length
    probe_lengths = df['target_sequence'].str.len().unique()
    if len(probe_lengths) != 1:
        raise ValueError("All probes in a transcript must have the same length.")
    probe_length = probe_lengths[0]

    min_spacing = probe_length - max_overlap
    if min_spacing < 0:
        raise ValueError("max_overlap cannot exceed probe length.")

    # Recover transcript-level ON bits from all probe barcodes
    on_bits = sorted({
        i
        for bc in df['probe_barcode']
        for i, bit in enumerate(bc)
        if bit == '1'
    })

    if len(on_bits) != N_on_bits:
        print(
            f'Warning: Probes for {df.iloc[0]["gene_id"]}:{df.iloc[0]["transcript_id"]} '
            f'have {len(on_bits)} transcript-level on-bits instead of {N_on_bits}. '
            'Returning no probes for this transcript.'
        )
        return df.iloc[:0].copy(), {ob: 0 for ob in on_bits}

    # Track only the transcript-relevant ON bits
    on_bit_coverage = {ob: 0 for ob in on_bits}

    shifts = df['shift'].to_numpy()
    barcodes = df['probe_barcode'].tolist()

    selected_indices = []
    selected_shifts = []
    rest_indices = list(range(df.shape[0]))

    for _ in range(N_probes_per_transcript):
        valid_candidates = []
        trial_scores = []

        for r_id in rest_indices:
            candidate_shift = shifts[r_id]

            # Hard pairwise overlap rule
            too_close = any(abs(candidate_shift - s) < min_spacing for s in selected_shifts)
            if too_close:
                continue

            trial_on_bit_coverage = on_bit_coverage.copy()
            bc = barcodes[r_id]

            for j, bit in enumerate(bc):
                if bit == '1' and j in trial_on_bit_coverage:
                    trial_on_bit_coverage[j] += 1

            score = (
                max(trial_on_bit_coverage.values()) /
                (1 + min(trial_on_bit_coverage.values()))
            )

            valid_candidates.append(r_id)
            trial_scores.append(score)

        if not valid_candidates:
            break

        score_min = min(trial_scores)
        lowest_score_ids = [
            valid_candidates[j]
            for j in range(len(valid_candidates))
            if trial_scores[j] == score_min
        ]

        selected_id = rng.choice(lowest_score_ids)
        selected_indices.append(selected_id)
        selected_shifts.append(shifts[selected_id])
        rest_indices.remove(selected_id)

        bc = barcodes[selected_id]
        for j, bit in enumerate(bc):
            if bit == '1' and j in on_bit_coverage:
                on_bit_coverage[j] += 1

    return df.iloc[selected_indices].copy(), on_bit_coverage


def select_probes_greedy_stochastic_one_df(
    df: pd.DataFrame,
    N_probes_per_transcript: int,
    N_on_bits: int,
    n_trials: int = 1,
    max_overlap: int = 20,
    seed: int = None
):
    """
    Run multiple stochastic trials and keep the best selection.
    Preference:
      1) maximize number of selected probes
      2) minimize on-bit imbalance
    """
    if df.empty:
        return df.copy()

    # if N_probes_per_transcript >= df.shape[0]:
    #     print(f'There are only {df.shape[0]} probes while {N_probes_per_transcript} are required! Just return everything!')
    #     return df.copy()

    best_selected_df = df.iloc[:0].copy()
    best_n_selected = -1
    best_imbalance = np.inf

    base_rng = np.random.default_rng(seed)

    for _ in range(n_trials):
        trial_seed = int(base_rng.integers(0, 2**32 - 1))
        rng = np.random.default_rng(trial_seed)

        selected_df, on_bit_coverage = select_probes_greedy_stochastic_one_df_trial(
            df=df,
            N_probes_per_transcript=N_probes_per_transcript,
            N_on_bits=N_on_bits,
            max_overlap=max_overlap,
            rng=rng
        )

        n_selected = len(selected_df)
        imbalance = max(on_bit_coverage.values()) - min(on_bit_coverage.values())

        if (
            n_selected > best_n_selected or
            (n_selected == best_n_selected and imbalance < best_imbalance)
        ):
            best_selected_df = selected_df
            best_n_selected = n_selected
            best_imbalance = imbalance

        if best_n_selected == N_probes_per_transcript and best_imbalance == 0:
            break

    gene_id = df.iloc[0]["gene_id"] if "gene_id" in df.columns else "NA"
    transcript_id = df.iloc[0]["transcript_id"] if "transcript_id" in df.columns else "NA"
    print(
        f'{gene_id}:{transcript_id}: selected {best_n_selected}/{df.shape[0]} probes '
        f'with max_overlap={max_overlap} (hard pairwise constraint), '
        f'on-bit imbalance={best_imbalance}.'
    )

    return best_selected_df


def select_probes_greedy_stochastic(
    probe_dict: dict,
    N_probes_per_transcript: int,
    N_on_bits: int = 4,
    N_threads: int = 1,
    n_trials: int = 1,
    max_overlap: int = 20,
    seed: int = None
):
    """
    Apply greedy stochastic selection with hard pairwise overlap constraint
    across all transcripts.
    """
    probe_dict_temp = deepcopy(probe_dict)
    keys = []
    args = []

    for gk in probe_dict.keys():
        for tk in probe_dict[gk].keys():
            keys.append((gk, tk))
            args.append((
                probe_dict_temp[gk][tk],
                N_probes_per_transcript,
                N_on_bits,
                n_trials,
                max_overlap,
                seed
            ))

    with Pool(N_threads) as p:
        results = p.starmap(select_probes_greedy_stochastic_one_df, args)

    for i, (gk, tk) in enumerate(keys):
        probe_dict_temp[gk][tk] = results[i]

    return probe_dict_temp