import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def single_heatmap(data_dict: dict):
    """Create a single heatmap using only the first (key, value) from data_dict.

    Chooses the first y_label configuration (VLPSoftmax) and applies the same
    filtering/pivot rules as in plot_perplexity.
    """

    y_labels = ['VLPSoftmax', 'VLPActivation', 'PWLSoftmax', 'PWLActivation', 'TaylorSoftmax']
    silu_labels = ['llama']
    gelu_labels = ['whisper', 'swin', 'vivit']

    # Use only the first provided model/key
    try:
        key, value = next(iter(data_dict.items()))
    except StopIteration:
        raise ValueError("data_dict is empty; expected at least one (key, value)")

    # We'll generate a single heatmap. Use the first y_label configuration.
    y_label = y_labels[0]

    data = pd.read_csv(value)
    data = data.drop(columns=['model', 'modality', 'function_name', 'patch_attention', 'patch_ffn'])

    plot_data = data.copy()

    if y_label == 'VLPSoftmax':
        plot_data = plot_data.drop(columns=['attn_lut_build', 'ffn_exp_dim','ffn_max_pos_exp','ffn_window_size','attn_segments','attn_segment_0','ffn_segments','ffn_segment_0','attn_degree_center','attn_degrees'])
    elif y_label == 'VLPActivation':
        plot_data = plot_data.drop(columns=['attn_exp_dim','attn_max_exp','attn_min_exp','attn_window_size','attn_lut_build','attn_segments','attn_segment_0','ffn_segments','ffn_segment_0','attn_degree_center','attn_degrees'])
    elif y_label == 'PWLSoftmax':
        plot_data = plot_data.drop(columns=['attn_exp_dim','attn_max_exp','attn_min_exp','attn_window_size','attn_lut_build','ffn_exp_dim','ffn_max_pos_exp','ffn_window_size','ffn_segments','ffn_segment_0','attn_degree_center','attn_degrees'])
    elif y_label == 'PWLActivation':
        plot_data = plot_data.drop(columns=['attn_exp_dim','attn_max_exp','attn_min_exp','attn_window_size','attn_lut_build','ffn_exp_dim','ffn_max_pos_exp','ffn_window_size','attn_segments','attn_segment_0','attn_degree_center','attn_degrees'])
    elif y_label == 'TaylorSoftmax':
        plot_data = plot_data.drop(columns=['attn_exp_dim','attn_max_exp','attn_min_exp','attn_window_size','attn_lut_build','ffn_exp_dim','ffn_max_pos_exp','ffn_window_size','attn_segments','attn_segment_0','ffn_segments','ffn_segment_0'])

    if key.split('_')[0] in silu_labels and 'Activation' in y_label:
        y_label = y_label.replace('Activation', 'Silu')
        plot_data = plot_data[plot_data['ffn_fn'] == y_label]
    elif key.split('_')[0] in gelu_labels and 'Activation' in y_label:
        y_label = y_label.replace('Activation', 'Gelu')
        plot_data = plot_data[plot_data['ffn_fn'] == y_label]
    elif 'Softmax' in y_label:
        plot_data = plot_data[plot_data['attn_fn'] == y_label]

    plot_data = plot_data.drop(columns=['attn_fn', 'ffn_fn'])

    # Ensure numeric and drop missing 'value's to avoid duplicate/NaN issues
    plot_data['value'] = pd.to_numeric(plot_data['value'], errors='coerce')
    plot_data = plot_data.dropna(subset=['value'])

    if y_label == 'VLPSoftmax':
        heatmap_data = plot_data.pivot_table(index='attn_exp_dim', columns='attn_max_exp', values='value', aggfunc='mean')
    elif y_label == 'VLPActivation':
        heatmap_data = plot_data.pivot_table(index='ffn_exp_dim', columns='ffn_max_pos_exp', values='value', aggfunc='mean')
    elif y_label == 'PWLSoftmax':
        heatmap_data = plot_data.pivot_table(index='attn_segments', columns='attn_segment_0', values='value', aggfunc='mean')
    elif y_label == 'PWLActivation':
        heatmap_data = plot_data.pivot_table(index='ffn_segments', columns='ffn_segment_0', values='value', aggfunc='mean')
    elif y_label == 'TaylorSoftmax':
        heatmap_data = plot_data.pivot_table(index='attn_degree_center', columns='attn_degrees', values='value', aggfunc='mean')
    else:
        raise ValueError(f"Unsupported y_label: {y_label}")

    # If empty after filtering, create a tiny placeholder to avoid errors
    if heatmap_data.size == 0:
        heatmap_data = pd.DataFrame([[np.nan]])

    # Sort index/columns for consistent layout
    try:
        heatmap_data = heatmap_data.sort_index().sort_index(axis=1)
    except Exception:
        pass

    fig_width = 506 / 72.27
    fig_height = fig_width * 0.5
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    # Build mask and custom annotations to hide NaN cells
    mask = heatmap_data.isna()
    ann = heatmap_data.copy()
    ann = ann.applymap(lambda v: f"{v:.2f}" if pd.notna(v) else "")
    sns.heatmap(
        heatmap_data,
        mask=mask,
        ax=ax,
        annot=ann.values,
        fmt="",
        cmap='viridis',
        cbar=False,
        xticklabels=False,
        yticklabels=False
    )

    save_dir = os.path.expanduser('figures')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'perplexity_plot.png'), dpi=1200)

def plot_perplexity(data_dict: dict):
    """Create a 5 x N grid of heatmaps (rows = methods, cols = models).

    Fixes:
    - Keep y_label immutable for pivot logic; use a separate filter label for ffn/attn.
    - Ensure consistent axes shape with squeeze=False.
    - Scale figure size with grid so each subplot matches the single heatmap cell size.
    - Expand save path correctly.
    """

    y_labels = ['VLPSoftmax', 'VLPActivation', 'PWLSoftmax', 'PWLActivation', 'TaylorSoftmax']

    silu_labels = ['llama']
    gelu_labels = ['whisper', 'swin', 'vivit']

    n_models = len(data_dict)

    # Use the same per-axes size as single_heatmap, then scale by grid dimensions
    per_fig_width = 506 / 72.27
    per_fig_height = per_fig_width * 0.5

    fig_width = per_fig_width * max(1, n_models)
    fig_height = per_fig_height * 5

    fig, axes = plt.subplots(5, n_models, figsize=(fig_width, fig_height), squeeze=False)

    for i, (key, value) in enumerate(data_dict.items()):
        data = pd.read_csv(value)
        data = data.drop(columns=['model', 'modality', 'function_name', 'patch_attention', 'patch_ffn'])

        for j, y_label in enumerate(y_labels):
            plot_data = data.copy()

            # Drop unrelated columns based on the original y_label (immutable for pivot logic)
            if y_label == 'VLPSoftmax':
                plot_data = plot_data.drop(columns=['attn_lut_build', 'ffn_exp_dim','ffn_max_pos_exp','ffn_window_size','attn_segments','attn_segment_0','ffn_segments','ffn_segment_0','attn_degree_center','attn_degrees'])
            elif y_label == 'VLPActivation':
                plot_data = plot_data.drop(columns=['attn_exp_dim','attn_max_exp','attn_min_exp','attn_window_size','attn_lut_build','attn_segments','attn_segment_0','ffn_segments','ffn_segment_0','attn_degree_center','attn_degrees'])
            elif y_label == 'PWLSoftmax':
                plot_data = plot_data.drop(columns=['attn_exp_dim','attn_max_exp','attn_min_exp','attn_window_size','attn_lut_build','ffn_exp_dim','ffn_max_pos_exp','ffn_window_size','ffn_segments','ffn_segment_0','attn_degree_center','attn_degrees'])
            elif y_label == 'PWLActivation':
                plot_data = plot_data.drop(columns=['attn_exp_dim','attn_max_exp','attn_min_exp','attn_window_size','attn_lut_build','ffn_exp_dim','ffn_max_pos_exp','ffn_window_size','attn_segments','attn_segment_0','attn_degree_center','attn_degrees'])
            elif y_label == 'TaylorSoftmax':
                plot_data = plot_data.drop(columns=['attn_exp_dim','attn_max_exp','attn_min_exp','attn_window_size','attn_lut_build','ffn_exp_dim','ffn_max_pos_exp','ffn_window_size','attn_segments','attn_segment_0','ffn_segments','ffn_segment_0'])

            # Build the function filter without mutating y_label
            filter_label = y_label
            if key.split('_')[0] in silu_labels and 'Activation' in y_label:
                filter_label = filter_label.replace('Activation', 'Silu')
                plot_data = plot_data[plot_data['ffn_fn'] == filter_label]
            elif key.split('_')[0] in gelu_labels and 'Activation' in y_label:
                filter_label = filter_label.replace('Activation', 'Gelu')
                plot_data = plot_data[plot_data['ffn_fn'] == filter_label]
            elif 'Softmax' in filter_label:
                plot_data = plot_data[plot_data['attn_fn'] == filter_label]

            # Remove fn columns now that filtering is applied
            plot_data = plot_data.drop(columns=['attn_fn', 'ffn_fn'])

            # Ensure numeric and drop missing 'value's to avoid duplicate/NaN issues
            plot_data['value'] = pd.to_numeric(plot_data['value'], errors='coerce')
            plot_data = plot_data.dropna(subset=['value'])

            # Pivot based on the ORIGINAL y_label
            if y_label == 'VLPSoftmax':
                heatmap_data = plot_data.pivot_table(index='attn_exp_dim', columns='attn_max_exp', values='value', aggfunc='mean')
            elif y_label == 'VLPActivation':
                heatmap_data = plot_data.pivot_table(index='ffn_exp_dim', columns='ffn_max_pos_exp', values='value', aggfunc='mean')
            elif y_label == 'PWLSoftmax':
                plot_data = plot_data[(plot_data['attn_segment_0'] >= -24) & (plot_data['attn_segment_0'] <= -19)]
                heatmap_data = plot_data.pivot_table(index='attn_segments', columns='attn_segment_0', values='value', aggfunc='mean')
            elif y_label == 'PWLActivation':
                plot_data = plot_data[(plot_data['ffn_segment_0'] >= 3) & (plot_data['ffn_segment_0'] <= 8)]
                heatmap_data = plot_data.pivot_table(index='ffn_segments', columns='ffn_segment_0', values='value', aggfunc='mean')
            elif y_label == 'TaylorSoftmax':
                plot_data = plot_data[(plot_data['attn_degree_center'] >= -7) & (plot_data['attn_degree_center'] <= -2)]
                heatmap_data = plot_data.pivot_table(index='attn_degree_center', columns='attn_degrees', values='value', aggfunc='mean')
            else:
                raise ValueError(f"Unsupported y_label: {y_label}")

            # Guard: if empty after filtering, place a placeholder to avoid errors
            if heatmap_data.size == 0:
                heatmap_data = pd.DataFrame([[np.nan]])

            # Sort index/columns for consistent layout
            try:
                heatmap_data = heatmap_data.sort_index().sort_index(axis=1)
            except Exception:
                pass

            ax = axes[j, i]
            show_y = (i == 0)
            # Build mask and custom annotations to hide NaN cells
            mask = heatmap_data.isna()
            ann = heatmap_data.copy().applymap(lambda v: f"{v:.2f}" if pd.notna(v) else "")
            sns.heatmap(
                heatmap_data,
                mask=mask,
                ax=ax,
                annot=ann.values,
                fmt="",
                cmap='viridis',
                cbar=False,
                xticklabels=False,
                yticklabels=show_y
            )
            # Add model key as title on the top row only
            if j == 0:
                ax.set_title(key)
            if show_y:
                try:
                    ax.set_yticklabels([str(v) for v in heatmap_data.index], rotation=0)
                except Exception:
                    pass

    # Save under local figures/ like single_heatmap and ensure path exists
    save_dir = os.path.expanduser('figures')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'perplexity_plot.png'), dpi=300)

if __name__ == "__main__":
    data_dict = {
        'llama_2_7b': 'csv/meta-llama/Llama-2-7b-hf/metric.csv',
        'llama_2_13b': 'csv/meta-llama/Llama-2-13b-hf/metric.csv',
        'llama_2_7b': 'csv/meta-llama/Llama-2-7b-hf/metric.csv',
        'llama_2_13b': 'csv/meta-llama/Llama-2-13b-hf/metric.csv',
        'whisper_tiny': 'csv/openai/whisper-tiny/metric.csv',
        'whisper_large': 'csv/openai/whisper-large/metric.csv',
        'swin_tiny': 'csv/microsoft/swinv2-tiny-patch4-window8-256/metric.csv',
        'swin_large': 'csv/microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft/metric.csv',
        'vivit': 'csv/google/vivit-b-16x2/metric.csv'
    }

    #single_heatmap(data_dict)
    plot_perplexity(data_dict)