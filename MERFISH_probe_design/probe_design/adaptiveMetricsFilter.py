import pandas as pd
import numpy as np
from copy import deepcopy
from typing import Dict, Optional, List, Tuple


def adaptive_isospecificity_filter(
    probe_dict: Dict[str, Dict[str, pd.DataFrame]], 
    min_probes: int = 48,
    initial_threshold: float = 0.9,
    min_threshold: float = 0.0,
    threshold_step: float = 0.05,
    column_key: str = 'target_isospecificity',
    verbose: bool = True
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], Dict[str, Dict[str, float]]]:
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
    
    return filtered_dict, thresholds_used


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


def get_transcripts_below_threshold(
    thresholds_used: Dict[str, Dict[str, float]], 
    threshold: float = 0.75
) -> List[Tuple[str, str, float]]:
    """
    Get list of transcripts that required relaxing below a given threshold.
    
    Returns:
    --------
    List of tuples (gene_id, transcript_id, actual_threshold_used)
    """
    relaxed_transcripts = []
    
    for gene_id in thresholds_used:
        for transcript_id, used_threshold in thresholds_used[gene_id].items():
            if used_threshold is not None and used_threshold < threshold:
                relaxed_transcripts.append((gene_id, transcript_id, used_threshold))
    
    # Sort by threshold used (lowest first)
    relaxed_transcripts.sort(key=lambda x: x[2])
    
    return relaxed_transcripts


# Example usage:
if __name__ == "__main__":
    # Example of how to use the function
    
    # Assuming you have your probe_dict loaded
    # probe_dict = load_your_probe_dict()
    
    # Apply adaptive filtering
    # filtered_probes, thresholds = adaptive_isospecificity_filter(
    #     probe_dict,
    #     min_probes=48,
    #     initial_threshold=0.9,
    #     min_threshold=0.0,
    #     threshold_step=0.05,
    #     verbose=True
    # )
    
    # Check which transcripts required threshold relaxation
    # relaxed = get_transcripts_below_threshold(thresholds, threshold=0.75)
    # if relaxed:
    #     print(f"\n{len(relaxed)} transcripts required threshold < 0.75:")
    #     for gene, transcript, thresh in relaxed[:10]:  # Show first 10
    #         print(f"  {gene}/{transcript}: threshold = {thresh:.3f}")
    
    pass