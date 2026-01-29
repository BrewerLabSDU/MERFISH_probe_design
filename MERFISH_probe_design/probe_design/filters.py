#!/usr/bin/env python3

from multiprocessing import Pool

import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Dict
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils import MeltingTemp

# GLOBAL VARIABLES
from . import seq2int

def filter_probe_dict_by_metric(probe_dict:pd.core.frame.DataFrame, column_key:str, 
        lower_bound:float=-np.inf, upper_bound:float=np.inf, verbose:bool=False):
    '''Filter the probe dictionary by a metric.'''
    for gk in probe_dict.keys():
        if verbose:
            print(gk)
        for tk in probe_dict[gk].keys():
            new_df= probe_dict[gk][tk][
                probe_dict[gk][tk][column_key].gt(lower_bound) & 
                probe_dict[gk][tk][column_key].lt(upper_bound)
            ]
            if verbose:
                print(f'\t{tk}: {new_df.shape[0]} / {probe_dict[gk][tk].shape[0]} probes passed the filter {lower_bound} < {column_key} <  {upper_bound}.')
            probe_dict[gk][tk] = new_df

def calc_gc_for_probe_dict(probe_dict:pd.core.frame.DataFrame, 
        column_key_seq:str='target_sequence', column_key_write='target_GC'):
    '''Calculate GC content of sequences under the column_key_seq column in the probe dictionary.
    The GC content is reported in percentile.
    '''
    for gk in probe_dict.keys():
        for tk in probe_dict[gk].keys():
            
            gcs = []
            for seq in probe_dict[gk][tk][column_key_seq]:
                gcs.append(gc_fraction(seq) * 100)

            probe_dict[gk][tk][column_key_write] = pd.Series(gcs, index=probe_dict[gk][tk].index)

def calc_tm_for_probe_dict(probe_dict:pd.core.frame.DataFrame, Na_conc:float, fmd_percentile:float, probe_conc:float=1,
        column_key_seq:str='target_sequence', column_key_write='target_Tm'):
    '''Calculate melting temperatures of the target sequences of the probe dictionary.
    Arguments:
        Na_conc: concentration of the the Na+ ion in mM.
        fmd_percentile: the percentile of formamide.
        probe_conc: concentration of the individual probes in nM.
    '''
    for gk in probe_dict.keys():
        for tk in probe_dict[gk].keys():
            
            tms = []
            for seq in probe_dict[gk][tk][column_key_seq]:
                tm_raw = MeltingTemp.Tm_NN(seq, nn_table=MeltingTemp.DNA_NN4, Na=Na_conc,
                        dnac1=probe_conc, dnac2=0)
                
                tms.append(MeltingTemp.chem_correction(tm_raw, fmd=fmd_percentile))

            probe_dict[gk][tk][column_key_write] = pd.Series(tms, index=probe_dict[gk][tk].index)

def calc_tm_JM(sequence:str, monovalentSalt:float=0.3, probeConc:float=5e-9):
    '''TM calculation used in JM's original MATLAB code.
    Adapted by Rongxin Fang.
    '''
    intSeq = np.array([["A", "C", "G", "T"].index(s) for s in list(sequence)])
    nnID = intSeq[:(len(intSeq)-1)] * 4 + intSeq[1:]
    end = len(intSeq)

    isValid = np.array([
        False not in (a, b, c, d) for (a, b, c, d) in zip(intSeq[:(end-1)] <= 3,
        intSeq[1:end] <= 3,
        intSeq[:(end-1)] >= 0,
        intSeq[1:end] >=0) ])

    H = np.array([
        -7.6, -8.4, -7.8, -7.2, -8.5, -8.0, -10.6, -7.8,
        -8.2, -9.8, -8.0, -8.4, -7.2, -8.2, -8.5, -7.6])   
    S = np.array([
        -21.3, -22.4, -21.0, -20.4, -22.7, -19.9, -27.2, -21.0,
        -22.2, -24.4, -19.9, -22.4, -21.3, -22.2, -22.7, -21.3])

    dG = np.zeros([2, len(intSeq)-1])
    dG[0,:] = H[nnID[isValid]]
    dG[1,:] = S[nnID[isValid]]

    H = np.cumsum(dG[0,:])[-1]
    S = np.cumsum(dG[1,:])[-1]

    # Determine ends
    fivePrimeAT = (intSeq[0] == 0) | (intSeq[0]  == 3);
    threePrimeAT = (intSeq[-1] == 0) | (intSeq[-1] == 3);

    H = H + 0.2 + 2.2*fivePrimeAT + 2.2*threePrimeAT;
    S = S + -5.7 + 6.9*fivePrimeAT + 6.9*threePrimeAT;

    S = S + 0.368*(len(sequence)-1)*np.log(monovalentSalt);

    return H*1000 / (S + 1.9872 * np.log(probeConc)) - 273.15;

def calc_tm_JM_for_transcript(df, monovalentSalt, probe_conc, column_key_seq, column_key_write):
    tms = []
    for seq in df[column_key_seq]:
        tms.append(calc_tm_JM(seq, monovalentSalt, probe_conc))

    df[column_key_write] = pd.Series(tms, index=df.index)
    return df

def calc_tm_JM_for_probe_dict(probe_dict:dict, monovalentSalt:float, probe_conc:float=1,
        column_key_seq:str='target_sequence', column_key_write='target_Tm', n_threads=1):
    '''Calculate melting temperatures of the target sequences of the probe dictionary.
    Use the TM calculation method in JM's original MATLAB code.
    Arguments:
        Na_conc: concentration of the the Na+ ion in mM.
        fmd_percentile: the percentile of formamide.
        probe_conc: concentration of the individual probes in nM.
    '''
    # Iterate through all genes and get the arguments for parallel processing
    ks = []
    args = []
    for gk in probe_dict.keys():
        for tk in probe_dict[gk].keys():
            ks.append((gk, tk))
            args.append([probe_dict[gk][tk], monovalentSalt, probe_conc, column_key_seq, column_key_write])
    # Add readout probes in parallel
    with Pool(n_threads) as p:
        results = p.starmap(calc_tm_JM_for_transcript, args)
    
    # Update the probe dictionary
    for i, kk in enumerate(ks):
        gk, tk = kk
        probe_dict[gk][tk] = results[i]

## Add functions related to Flex probe design:
# calculate GC for left 25 bases:

def calc_GC_for_left_probe_dict(
    probe_dict:pd.core.frame.DataFrame, 
    left_length:int=25,
    column_key_seq:str='target_sequence',
    column_key_write='target_GC_left'):
    """Calculate GC content of the left part of the sequences under the column_key_seq column in the probe dictionary.
    The GC content is reported in percentile.
    """
    for gk in probe_dict.keys():
        for tk in probe_dict[gk].keys():
            gcs = []
            for seq in probe_dict[gk][tk][column_key_seq]:
                gcs.append(gc_fraction(seq[:left_length]) * 100)

            probe_dict[gk][tk][column_key_write+str(left_length)] = pd.Series(gcs, index=probe_dict[gk][tk].index)
    
def calc_GC_for_right_probe_dict(
    probe_dict:pd.core.frame.DataFrame, 
    right_length:int=25,
    column_key_seq:str='target_sequence',
    column_key_write='target_GC_right'):
    """Calculate GC content of the right part of the sequences under the column_key_seq column in the probe dictionary.
    The GC content is reported in percentile.
    """
    for gk in probe_dict.keys():
        for tk in probe_dict[gk].keys():
            gcs = []
            for seq in probe_dict[gk][tk][column_key_seq]:
                gcs.append(gc_fraction(seq[-right_length:]) * 100)

            probe_dict[gk][tk][column_key_write+str(right_length)] = pd.Series(gcs, index=probe_dict[gk][tk].index)  

def calc_base_at_location_probe_dict(
    probe_dict:pd.core.frame.DataFrame, 
    location:int=25,
    column_key_seq:str='target_sequence',
    column_key_write='target_base_at_location'):
    """Calculate the base at a specific location in the sequences under the column_key_seq column in the probe dictionary.
    The base is reported as a string.
    """
    
    for gk in probe_dict.keys():
        for tk in probe_dict[gk].keys():
            bases = []
            for seq in probe_dict[gk][tk][column_key_seq]:
                bases.append(seq2int[seq[location].upper()])
            probe_dict[gk][tk][column_key_write+str(location)] = pd.Series(bases, index=probe_dict[gk][tk].index)

# function to remove probes with homopolymers:
def find_homopolymer(seq, max_length=4):
    """Check if the sequence has a homopolymer longer than max_length."""
    for base in 'ATCG':
        if base * max_length in seq.upper():
            return seq2int[base]  # Return the base if a homopolymer is found
    return 0
def detect_homopolymer_probe_dict(
    probe_dict:pd.core.frame.DataFrame, 
    column_key_seq:str='target_sequence',    
    min_length:int=5,
    column_key_write='has_homopolymer',
    ):
    """Remove probes with homopolymers longer than min_length."""
    
    for gk in probe_dict.keys():
        for tk in probe_dict[gk].keys():
            seqs = probe_dict[gk][tk][column_key_seq]
            probe_dict[gk][tk][column_key_write+str(min_length)] = seqs.apply(
                lambda seq: find_homopolymer(seq, min_length))
            
def adaptive_isospecificity_filter(
    probe_dict: Dict[str, Dict[str, pd.DataFrame]], 
    min_probes: int = 50,
    initial_threshold: float = 0.9,
    min_threshold: float = 0.0,
    threshold_step: float = 0.05,
    column_key: str = 'target_isospecificity',
    verbose: bool = True
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Adaptively filter probes by isoform specificity, relaxing the threshold
    as needed to ensure minimum probe count per transcript.
    
    Parameters:
    -----------
    probe_dict : Dict[str, Dict[str, pd.DataFrame]]
        Nested dictionary of probe dataframes
    min_probes : int
        Minimum number of probes required per transcript
    initial_threshold : float
        Starting threshold for isospecificity (higher is stricter)
    min_threshold : float
        Minimum threshold to try (will not go below this)
    threshold_step : float
        Step size for decreasing threshold
    column_key : str
        Column name containing the isospecificity values
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    filtered_dict : Dict[str, Dict[str, pd.DataFrame]]
        Filtered probe dictionary
    thresholds_used : Dict[str, Dict[str, float]]
        Dictionary tracking the actual threshold used for each transcript
    """
    
    # Create a deep copy to avoid modifying the original
    filtered_dict = deepcopy(probe_dict)
    thresholds_used = {}
    
    # Statistics tracking
    total_transcripts = sum(len(probe_dict[gene]) for gene in probe_dict)
    processed = 0
    
    for gene_id in probe_dict.keys():
        thresholds_used[gene_id] = {}
        
        if verbose:
            print(f"\nProcessing gene: {gene_id}")
        
        for transcript_id in probe_dict[gene_id].keys():
            processed += 1
            original_df = probe_dict[gene_id][transcript_id].copy()
            
            # Check if the column exists
            if column_key not in original_df.columns:
                if verbose:
                    print(f"  Warning: {transcript_id} missing column '{column_key}', skipping")
                thresholds_used[gene_id][transcript_id] = None
                continue
            
            # Start with high threshold and progressively relax
            current_threshold = initial_threshold
            best_df = pd.DataFrame()  # Empty dataframe initially
            best_threshold = None
            
            while current_threshold >= min_threshold:
                # Filter probes at current threshold
                filtered_df = original_df[original_df[column_key] >= current_threshold]
                
                if len(filtered_df) >= min_probes:
                    # Found enough probes at this threshold
                    best_df = filtered_df
                    best_threshold = current_threshold
                    break
                
                # Decrease threshold for next iteration
                current_threshold -= threshold_step
            
            # If we still don't have enough probes, take all above min_threshold
            if len(best_df) < min_probes:
                best_df = original_df[original_df[column_key] >= min_threshold]
                best_threshold = min_threshold
                
                # If still not enough, take the top min_probes by isospecificity
                if len(best_df) < min_probes and len(original_df) >= min_probes:
                    best_df = original_df.nlargest(min_probes, column_key)
                    best_threshold = best_df[column_key].min()
                elif len(best_df) < min_probes:
                    # Take all available probes if we have fewer than min_probes
                    best_df = original_df
                    best_threshold = original_df[column_key].min() if len(original_df) > 0 else 0
            
            # Store the filtered results
            filtered_dict[gene_id][transcript_id] = best_df
            thresholds_used[gene_id][transcript_id] = best_threshold
            
            if verbose:
                print(f"  {transcript_id}: {len(best_df)}/{len(original_df)} probes "
                      f"(threshold: {best_threshold:.3f})")
            
            # Progress indicator for large datasets
            if verbose and processed % 50 == 0:
                print(f"  Progress: {processed}/{total_transcripts} transcripts processed")
    
    if verbose:
        print(f"\nFiltering complete. Processed {processed} transcripts.")
        print_filter_summary(probe_dict, filtered_dict, thresholds_used)
    
    return filtered_dict


def print_filter_summary(
    original_dict: Dict[str, Dict[str, pd.DataFrame]], 
    filtered_dict: Dict[str, Dict[str, pd.DataFrame]], 
    thresholds_used: Dict[str, Dict[str, float]]
) -> None:
    """
    Print a summary of the filtering results.
    """
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    
    total_original = 0
    total_filtered = 0
    threshold_stats = []
    
    for gene_id in original_dict.keys():
        for transcript_id in original_dict[gene_id].keys():
            orig_count = len(original_dict[gene_id][transcript_id])
            filt_count = len(filtered_dict[gene_id][transcript_id])
            threshold = thresholds_used[gene_id].get(transcript_id)
            
            total_original += orig_count
            total_filtered += filt_count
            
            if threshold is not None:
                threshold_stats.append(threshold)
    
    print(f"Total probes: {total_original:,} -> {total_filtered:,} "
          f"({100*total_filtered/total_original:.1f}% retained)")
    
    if threshold_stats:
        print(f"Threshold statistics:")
        print(f"  Mean: {np.mean(threshold_stats):.3f}")
        print(f"  Median: {np.median(threshold_stats):.3f}")
        print(f"  Min: {np.min(threshold_stats):.3f}")
        print(f"  Max: {np.max(threshold_stats):.3f}")