#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_values_from_probe_dict(probe_dict:dict, column_key:str):
    '''Get values under the column_key from a probe dictionary.
    Return a list of values.
    '''
    values = []
    for gk in probe_dict.keys():
        for tk in probe_dict[gk].keys():
            values += list(probe_dict[gk][tk][column_key])
    
    return values

def plot_hist(probe_dict:dict, column_key:str, y_max=None, bins=30, log_y:bool=False):
    '''Plot histogram of values under the column_key'''
    plt.hist(get_values_from_probe_dict(probe_dict, column_key), bins=bins)

    if y_max is not None:
        bottom, top = plt.ylim()
        plt.ylim(bottom, y_max)

    if log_y:
        plt.yscale('log')

    plt.xlabel(column_key)
    plt.ylabel('Count')
    plt.show()

def plot_sequence_coverage(df:pd.core.frame.DataFrame, seq_length:int):
    '''Plot the sequence coverage of a sequences.'''
    coverage = np.zeros(seq_length, dtype=int)

    shifts = list(df['shift'])
    target_seqs = list(df['target_sequence'])

    for i in range(len(shifts)):
        shift = shifts[i]
        t_len = len(target_seqs[i])
        coverage[shift : shift + t_len] += 1

    plt.plot(np.arange(seq_length), coverage)
    plt.xlabel('Sequence position')
    plt.ylabel('Coverage')
    plt.show()

def plot_probes_barplot(probe_dict: dict, log_y: bool = False, hlines: list = [20, 50], return_fig: bool = False):
    fig, ax = plt.subplots(figsize=(15,6))
    transcript_names = []
    n_selected_probes = []
    for gene_id in probe_dict:
        for transcript_id in probe_dict[gene_id]:
            transcript_names.append(gene_id)
            n_selected_probes.append(probe_dict[gene_id][transcript_id].shape[0])
    x = np.arange(len(transcript_names))
    ax.bar(x, n_selected_probes)
    if log_y:
        ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(transcript_names, rotation=65)
    # Legend
    if hlines is not None:
        for indicater in hlines:
            ax.axhline(y=indicater, linestyle='--', linewidth=2, color=plt.cm.Accent.colors[hlines.index(indicater)%10])
            ax.text(-6, indicater*1.1, f'{indicater} probes', fontsize=8)
    # Reduce font size on x axis
    for tick in ax.get_xticklabels():
        tick.set_fontsize(6)
    ax.set_ylabel('Number of selected probes')
    plt.show()

    if return_fig:
        return fig

def plot_probes_comparison_barplot(
    probe_dict: dict,
    filtered_probe_dict: dict,
    log_y: bool = False,
    hlines: list = None,
    title_filter: str = "",
):
    """
    Visualize how filtering affects the number of probes per transcript
    by plotting original vs. filtered counts side-by-side.

    Parameters
    ----------
    probe_dict : dict
        Nested dictionary {gene_id: {transcript_id: <array_like or DataFrame>}}.
        It is assumed that probe_dict[gene_id][transcript_id] supports len() or .shape[0].
    filtered_probe_dict : dict
        Same structure as probe_dict but after filtering.
    log_y : bool, default False
        Whether to apply log scale to the y-axis.
    hlines : list or None, default [20, 50]
        Horizontal reference lines to draw; set to None to disable.
    title : str, default "Probes per transcript: original vs. filtered"
        Title of the plot.
    """

    # Collect a unified list of (gene_id, transcript_id) keys present in either dict.
    # This ensures alignment and no missing bars.
    keys = []
    # From original
    for gene_id in probe_dict:
        for transcript_id in probe_dict[gene_id]:
            keys.append((gene_id, transcript_id))
    # From filtered (add if not already present)
    for gene_id in filtered_probe_dict:
        for transcript_id in filtered_probe_dict[gene_id]:
            tup = (gene_id, transcript_id)
            if tup not in keys:
                keys.append(tup)

    # Sort keys to get deterministic order (optional)
    keys = sorted(keys, key=lambda x: (str(x[0]), str(x[1])))

    # Build labels (use gene_id like your original; optionally append transcript_id)
    # If you prefer keeping only gene_id (as in your code), uncomment the first line below
    # labels = [gene_id for gene_id, transcript_id in keys]
    # More informative (gene|transcript) default:
    labels = [f"{gene_id}" for gene_id, transcript_id in keys]

    # Extract counts from both dicts, defaulting to 0 if missing
    def _count(dct, gene_id, transcript_id):
        try:
            obj = dct[gene_id][transcript_id]
            # Support both DataFrame/ndarray with .shape[0] or general sequences via len()
            return getattr(obj, "shape", None)[0] if hasattr(obj, "shape") else len(obj)
        except Exception:
            return 0

    original_counts = [ _count(probe_dict, g, t) for (g, t) in keys ]
    filtered_counts = [ _count(filtered_probe_dict, g, t) for (g, t) in keys ]

    # Create the bar plot
    fig, ax = plt.subplots(figsize=(16, 6))
    x = np.arange(len(labels))
    width = 0.42  # bar width

    bars1 = ax.bar(x - width/2, original_counts, width, label="Original", color="#4C78A8")
    bars2 = ax.bar(x + width/2, filtered_counts, width, label="Filtered", color="#F58518")

    # Optional: log scale
    if log_y:
        ax.set_yscale('log')

    # X ticks/labels
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65)
    for tick in ax.get_xticklabels():
        tick.set_fontsize(6)

    # Horizontal indicator lines
    if hlines is not None:
        # Use Accent colormap for varied colors
        palette = plt.cm.Accent.colors
        for i, yval in enumerate(hlines):
            ax.axhline(y=yval, linestyle='--', linewidth=2, color=palette[i % len(palette)])
            # Place text slightly to the left
            ax.text(-5.5, yval * 1.1, f'{yval} probes', fontsize=8, color=palette[i % len(palette)])

    # Labels, legend, grid and title
    ax.set_ylabel('Number of probes')
    ax.set_title(f"Probes per transcript: original vs. filtered | {title_filter}")
    ax.legend()
    ax.grid(axis='y', alpha=0.25)

    # Tight layout to reduce label clipping
    fig.tight_layout()
    plt.show()

def _moving_average(arr: np.ndarray, window: int | None) -> np.ndarray:
    """Simple moving-average smoothing (centered via 'same')."""
    if not window or window <= 1:
        return arr
    k = np.ones(int(window), dtype=float)
    k /= k.sum()
    return np.convolve(arr, k, mode="same")

def _normalize(arr: np.ndarray, do_norm: bool) -> np.ndarray:
    if not do_norm:
        return arr
    vmax = arr.max() if arr.size else 0.0
    return (arr / vmax) if vmax > 0 else arr

def _compute_coverage_for_transcript(probe_df: pd.DataFrame, seq_length: int) -> np.ndarray:
    """Compute per-base coverage for a single transcript."""
    cov = np.zeros(seq_length, dtype=float)
    # Expect columns: 'shift' (0-based start), 'target_sequence' (nt string)
    for shift, seq in zip(probe_df['shift'].to_numpy(), probe_df['target_sequence'].astype(str).to_numpy()):
        if pd.isna(shift):
            continue
        try:
            start = int(shift)
        except Exception:
            continue
        end = start + len(seq)
        # clip to transcript bounds
        s = max(0, start)
        e = min(seq_length, end)
        if s < e:
            cov[s:e] += 1.0
    return cov

def _transcript_lengths_from_transcriptome(transcriptome: pd.DataFrame,
                                           id_col: str = "transcript_id",
                                           seq_col: str = "sequence") -> dict[str, int]:
    """Return {transcript_id -> length} from a transcriptome table."""
    return {row[id_col]: len(str(row[seq_col])) for _, row in transcriptome[[id_col, seq_col]].iterrows()}

# ---------- plotting: single transcript ----------

def plot_single_transcript_track(
    probe_df: pd.DataFrame,
    seq_length: int,
    transcript_id: str | None = None,
    gene_name: str | None = None,
    smooth: int | None = None,
    normalize: bool = False,
    ax=None,
    color: str = "C0",
):
    """
    Plot coverage track (line + filled area) for a single transcript.
    """
    cov = _compute_coverage_for_transcript(probe_df, seq_length)
    cov = _moving_average(cov, smooth)
    cov = _normalize(cov, normalize)

    if ax is None:
        sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 3), dpi=130) if ax is None else (ax.figure, ax)

    x = np.arange(seq_length)
    ax.plot(x, cov, color=color, lw=1.6, label=transcript_id or "transcript")
    ax.fill_between(x, 0, cov, color=color, alpha=0.25)
    title = "Coverage"
    if gene_name:
        title += f" · {gene_name}"
    if transcript_id:
        title += f" · {transcript_id}"
    if smooth and smooth > 1:
        title += f" (smoothed={smooth})"
    if normalize:
        title += " (norm)"
    ax.set_title(title)
    ax.set_ylabel("Coverage" + (" (norm)" if normalize else ""))
    ax.set_xlabel("Transcript position (bases)")
    # ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    return fig, ax

def plot_ridge_coverage_for_all_genes(
    probe_dict: dict[str, dict[str, pd.DataFrame]],
    transcriptome: pd.DataFrame,
    genes: list[str] | None = None,
    transcripts_by_gene: dict[str, list[str]] | None = None,
    id_col: str = "transcript_id",
    seq_col: str = "sequence",
    smooth: int | None = None,
    normalize: bool = True,
    height: float = 1.0,
    spacing: float = 0.8,
    cmap: str = "crest",
    figsize=(10, 15),
    dpi: int = 100,
    sort_within_gene: str = "length",  # "length" | "maxcov" | "name" | None
    inter_gene_gap: float = 0.5,       # extra space between gene blocks in combined mode
):
    """
    Ridgeplot across ALL genes in probe_dict.

    Modes:
      - per_gene=True: returns {gene -> (fig, ax)}. One figure per gene.
      - per_gene=False: a single figure combining all genes. Each gene is a block of ridges with extra vertical gap.

    Parameters mirror the single-gene function plus:
      - transcripts_by_gene: optional subset {gene -> [tid, ...]}.
      - inter_gene_gap: vertical space between gene blocks in combined mode.
      - save_dir: if provided and per_gene=True, saves '<save_dir>/<gene>_ridgeplot.png'.
    """
    gene_list = list(probe_dict.keys()) if genes is None else list(genes)
    if not gene_list:
        raise ValueError("No genes to plot.")

    # Length map (shared)
    tlen_map = _transcript_lengths_from_transcriptome(transcriptome, id_col=id_col, seq_col=seq_col)

    # ----- Single combined figure for ALL genes (as one big ridgeplot) -----
    sns.set_theme(style="white", rc={"axes.facecolor": "white"})

    import matplotlib.cm as cm
    cmap_fn = cm.get_cmap(cmap)

    # First pass: gather per-gene ordered transcript lists and cov arrays
    blocks = []
    for g in gene_list:
        tids = list(probe_dict[g].keys())
        if transcripts_by_gene and (g in transcripts_by_gene):
            tids = [t for t in tids if t in transcripts_by_gene[g]]
        metrics = []
        covs = {}
        for tid in tids:
            L = tlen_map.get(tid)
            if L is None:
                continue
            cov = _compute_coverage_for_transcript(probe_dict[g][tid], L)
            cov_s = _moving_average(cov, smooth)
            cov_n = _normalize(cov_s, normalize)
            covs[tid] = (cov_n, L)
            metrics.append({"tid": tid, "length": L, "maxcov": float(cov_s.max() if cov_s.size else 0.0), "name": tid})
        if not covs:
            continue
        if sort_within_gene == "length":
            order = [m["tid"] for m in sorted(metrics, key=lambda m: m["length"], reverse=True)]
        elif sort_within_gene == "maxcov":
            order = [m["tid"] for m in sorted(metrics, key=lambda m: m["maxcov"], reverse=True)]
        elif sort_within_gene == "name":
            order = [m["tid"] for m in sorted(metrics, key=lambda m: m["name"])]
        else:
            order = list(covs.keys())
        blocks.append((g, order, covs))

    # Compute total height
    total_ridges = sum(len(order) for _, order, _ in blocks)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Draw all blocks with extra inter-gene spacing
    y_cursor = 0.0
    ridge_idx_global = 0
    max_L = 0
    for g, order, covs in blocks:
        # Optional: gene label to the left
        ax.text(-0.02, y_cursor + (len(order) * spacing) / 2, g,
                ha="right", va="center", transform=ax.get_yaxis_transform(), fontsize=10, weight="bold")
        for i, tid in enumerate(order):
            cov_n, L = covs[tid]
            max_L = max(max_L, L)
            x = np.arange(L)
            baseline = y_cursor + i * spacing
            color = cmap_fn(ridge_idx_global / max(1, total_ridges - 1))
            y = baseline + height * cov_n
            ax.fill_between(x, baseline, y, color=color, alpha=0.85)
            ax.plot(x, y, color=color, lw=1.0)
            # label on the right
            ax.text(L + 5, baseline + 0.5 * height, tid, va="center", fontsize=8, color="dimgray")
            ridge_idx_global += 1
        y_cursor += len(order) * spacing + inter_gene_gap

    ax.set_xlabel("Transcript position (bases)")
    # ax.set_ylabel("Transcripts (grouped by gene)")
    ax.set_xlim(0, max_L + 50)
    ax.set_ylim(-0.1 * height, y_cursor + 0.5 * height)
    ax.set_title("Probe coverage ridgeplot · All genes"
                 + (f" · smoothed={smooth}" if (smooth and smooth > 1) else "")
                 + (" · normalized" if normalize else ""))
    fig.tight_layout()
    return fig, ax


def plot_ridge_coverage_for_all_genes_split_by_length(
    probe_dict: dict[str, dict[str, pd.DataFrame]],
    transcriptome: pd.DataFrame,
    # Length thresholds
    short_max_len: int = 3000,
    long_min_len: int = 3001,
    # Optional selection
    genes: list[str] | None = None,
    transcripts_by_gene: dict[str, list[str]] | None = None,
    # Transcriptome columns
    id_col: str = "transcript_id",
    seq_col: str = "sequence",
    # Plot options forwarded to plot_ridge_coverage_for_all_genes(...)
    smooth: int | None = None,
    normalize: bool = True,
    height: float = 1.0,
    spacing: float = 0.8,
    cmap: str = "crest",
    figsize=(10, 15),
    dpi: int = 100,
    sort_within_gene: str = "length",  # "length" | "maxcov" | "name" | None
    inter_gene_gap: float = 0.5
):
    """
    Create separate ridgeplots for short and long transcripts (and optionally mid-length).

    Returns a dict with keys:
      - 'short': per-gene dict or (fig, ax), depending on per_gene
      - 'long' : per-gene dict or (fig, ax)
      - 'mid'  : per-gene dict or (fig, ax) if include_mid_group=True, else None
    """
    # Build transcript length map once
    tlen_map = {row[id_col]: len(str(row[seq_col])) for _, row in transcriptome[[id_col, seq_col]].iterrows()}

    # Which genes to consider
    gene_list = list(probe_dict.keys()) if genes is None else list(genes)

    # Optional: pre-restrict by transcripts_by_gene
    def _allowed_tids(g):
        tids = list(probe_dict[g].keys())
        if transcripts_by_gene and (g in transcripts_by_gene):
            tids = [t for t in tids if t in transcripts_by_gene[g]]
        return tids

    # Build per-group transcript filters
    short_map: dict[str, list[str]] = {}
    long_map: dict[str, list[str]]  = {}

    for g in gene_list:
        tids = _allowed_tids(g)
        for tid in tids:
            L = tlen_map.get(tid)
            if L is None:
                continue
            if (short_max_len is not None) and (L <= short_max_len):
                short_map.setdefault(g, []).append(tid)
            elif (long_min_len is not None) and (L >= long_min_len):
                long_map.setdefault(g, []).append(tid)

    # Helper to drop empty genes from a transcripts_by_gene mapping
    def _prune_empty(mapping: dict[str, list[str]]) -> tuple[list[str], dict[str, list[str]]]:
        genes_present = [g for g, lst in mapping.items() if lst]
        mapping_pruned = {g: mapping[g] for g in genes_present}
        return genes_present, mapping_pruned

    out = {"short": None, "long": None}

    # SHORT group
    if short_map:
        short_genes, short_map = _prune_empty(short_map)
        out["short"] = plot_ridge_coverage_for_all_genes(
            probe_dict=probe_dict,
            transcriptome=transcriptome,
            genes=short_genes,
            transcripts_by_gene=short_map,
            id_col=id_col,
            seq_col=seq_col,
            smooth=smooth,
            normalize=normalize,
            height=height,
            spacing=spacing,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            sort_within_gene=sort_within_gene,
            inter_gene_gap=inter_gene_gap
        )

    # LONG group
    if long_map:
        long_genes, long_map = _prune_empty(long_map)
        out["long"] = plot_ridge_coverage_for_all_genes(
            probe_dict=probe_dict,
            transcriptome=transcriptome,
            genes=long_genes,
            transcripts_by_gene=long_map,
            id_col=id_col,
            seq_col=seq_col,
            smooth=smooth,
            normalize=normalize,
            height=height,
            spacing=spacing,
            cmap=cmap,
            figsize=figsize,
            dpi=dpi,
            sort_within_gene=sort_within_gene,
            inter_gene_gap=inter_gene_gap,
        )

    return out
