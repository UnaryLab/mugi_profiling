import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import numpy as np
import os

base_acc = {
    'vivit': 1.77392578125,
    'llama_2_7b': 5.754334063454066,
    'llama_2_13b': 5.192732655174028,
    'swin_tiny': 0.90771484375,
    'swin_large': 0.71240234375,
    'whisper_tiny': 22.232662691545976,
    'whisper_large': 5.059855323136026
}

# Set global font size
plt.rcParams.update({'font.size': 7})

def plot_perplexity(data_dict: dict, highlight_color: str = 'red'):
    """Create a 5 x N grid of heatmaps (rows = methods, cols = models).

    Args:
        data_dict: Dictionary mapping model names to CSV file paths
        highlight_color: Color to highlight the minimum value cell (default: 'red')

    Fixes:
    - Keep y_label immutable for pivot logic; use a separate filter label for ffn/attn.
    - Ensure consistent axes shape with squeeze=False.
    - Scale figure size with grid so each subplot matches the single heatmap cell size.
    - Expand save path correctly.
    - Highlight minimum value cell with customizable color.
    """

    y_labels = ['VLPSoftmax', 'VLPActivation', 'PWLSoftmax', 'PWLActivation', 'TaylorSoftmax']

    silu_labels = ['llama']
    gelu_labels = ['whisper', 'swin', 'vivit']

    n_models = len(data_dict)

    ppl_dict = {
        'Llama 2 7B': '6.21/5.68',
        'Llama 2 13B': '6.00/5.21',
        'Whisper Tiny': '11.1/22.2',
        'Whisper Large': '3.52/5.05',
        'SwinV2 Tiny': '0.83/0.91',
        'SwinV2 Large': '0.67/0.71',
        'ViViT': '1.75/1.77'
    }

    torch_dict = {
        'Llama 2 7B': '5.75',
        'Llama 2 13B': '5.19',
        'Whisper Tiny': '22.2',
        'Whisper Large': '5.06',
        'SwinV2 Tiny': '0.91',
        'SwinV2 Large': '0.71',
        'ViViT': '1.77'
    }

    # Use the same per-axes size as single_heatmap, then scale by grid dimensions
    per_fig_width = 506 / 72.27
    per_fig_height = per_fig_width * 0.575

    fig, axes = plt.subplots(6, n_models, figsize=(per_fig_width, per_fig_height))
    # Ensure axes is always 2D for consistent indexing
    if n_models == 1:
        axes = axes.reshape(6, 1)
    
    # Increase hspace to make room for x-tick labels at the bottom of each heatmap
    plt.subplots_adjust(wspace=0.03, hspace=0.2)

    for i, (key, value) in enumerate(data_dict.items()):
        data = pd.read_csv(value)
        data = data.drop(columns=['model', 'modality', 'function_name', 'patch_attention', 'patch_ffn'])

        for j, y_label in enumerate(y_labels):
            plot_data = data.copy()

            # Drop unrelated columns based on the original y_label (immutable for pivot logic)
            if y_label == 'VLPSoftmax':
                plot_data = plot_data.drop(columns=['ffn_exp_dim','ffn_max_pos_exp','ffn_window_size','attn_segments','attn_segment_0','ffn_segments','ffn_segment_0','attn_degree_center','attn_degrees'])
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
            # plot_data = plot_data.drop(columns=['attn_fn', 'ffn_fn'])

            # Ensure numeric and drop missing 'value's to avoid duplicate/NaN issues
            plot_data['value'] = pd.to_numeric(plot_data['value'], errors='coerce')
            plot_data = plot_data.dropna(subset=['value'])

            # Pivot based on the ORIGINAL y_label

            window = 2

            if y_label == 'VLPSoftmax':
                lut_max_data = plot_data[(plot_data['attn_lut_build'] == 'max')]
                lut_min_data = plot_data[plot_data['attn_lut_build'] == 'min']

                lut_min_min_row = lut_min_data.loc[lut_min_data['value'].idxmin()]
                lut_max_min_row = lut_max_data.loc[lut_max_data['value'].idxmin()]

                min_heatmap_value = lut_min_min_row['value']
                max_heatmap_value = lut_max_min_row['value']

                lut_min_exp = lut_min_min_row['attn_min_exp']
                lut_max_exp = lut_max_min_row['attn_max_exp']

                lut_min_min_exp = lut_min_data['attn_min_exp'].min()
                lut_min_max_exp = lut_min_data['attn_min_exp'].max()
                lut_max_max_exp = lut_max_data['attn_max_exp'].max()
                lut_max_min_exp = lut_max_data['attn_max_exp'].min()

                lut_min_lower_bound = max(lut_min_exp - window, lut_min_min_exp)
                lut_min_upper_bound = min(lut_min_exp + window, lut_min_max_exp)
                lut_max_lower_bound = max(lut_max_exp - window, lut_max_min_exp)
                lut_max_upper_bound = min(lut_max_exp + window, lut_max_max_exp)
                
                if lut_min_lower_bound == lut_min_min_exp:
                    lut_min_upper_bound += (abs(lut_min_exp - window) - abs(lut_min_min_exp))
                if lut_max_lower_bound == lut_max_min_exp:
                    lut_max_upper_bound += (abs(lut_max_exp - window) - abs(lut_max_min_exp))
                if lut_min_upper_bound == lut_min_max_exp:
                    lut_min_lower_bound -= (abs(lut_min_exp + window) - abs(lut_min_max_exp))
                if lut_max_upper_bound == lut_max_max_exp:
                    lut_max_lower_bound -= (abs(lut_max_exp + window) - abs(lut_max_max_exp))

                lut_min_slice = lut_min_data[lut_min_data['attn_min_exp'].between(lut_min_lower_bound, lut_min_upper_bound)]
                lut_max_slice = lut_max_data[lut_max_data['attn_max_exp'].between(lut_max_lower_bound, lut_max_upper_bound)]

                if min_heatmap_value < max_heatmap_value:
                    heatmap_data = lut_min_slice.pivot_table(index='attn_exp_dim', columns='attn_min_exp', values='value')
                else:
                    heatmap_data = lut_max_slice.pivot_table(index='attn_exp_dim', columns='attn_max_exp', values='value')

            elif y_label == 'VLPActivation':

                if '13' in key:
                    lut_max_data = plot_data
                    lut_max_min_row = lut_max_data.loc[lut_max_data['value'].idxmin()]
                    max_heatmap_value = lut_max_min_row['value']
                    lut_max_exp = lut_max_min_row['ffn_max_pos_exp']
                    lut_min_align = lut_max_min_row['ffn_max_pos_exp']
                    lut_max_max_exp = lut_max_data['ffn_max_pos_exp'].max()
                    lut_max_min_exp = lut_max_data['ffn_max_pos_exp'].min()
                    lut_max_lower_bound = max(lut_max_exp - window, lut_max_min_exp)
                    lut_max_upper_bound = min(lut_max_exp + window, lut_max_max_exp)
                    if lut_min_upper_bound == lut_min_max_exp:
                        lut_min_lower_bound -= (abs(lut_min_exp + window) - abs(lut_min_max_exp))
                    if lut_max_upper_bound == lut_max_max_exp:
                        lut_max_lower_bound -= (abs(lut_max_exp + window) - abs(lut_max_max_exp))
                    lut_max_slice = lut_max_data[(lut_max_data['ffn_max_pos_exp'].between(lut_max_lower_bound, lut_max_upper_bound))]
                    heatmap_data = lut_max_slice.pivot_table(index='ffn_exp_dim', columns='ffn_max_pos_exp', values='value')
                else:
                    if 'whisper_tiny' in key.lower():
                        lut_max_data = plot_data[((plot_data['ffn_lut_build'] == 'max') & (plot_data['ffn_max_pos_exp'] == 1)) | ((plot_data['ffn_lut_build'] != 'min') & ((plot_data['ffn_fn'] == 'VLPSilu') | (plot_data['ffn_fn'] == 'VLPGelu')))]
                        lut_min_data = plot_data[(plot_data['ffn_lut_build'] == 'min') & (plot_data['ffn_min_pos_exp'] == -7)]
                    else:
                        lut_max_data = plot_data[((plot_data['ffn_lut_build'] == 'max')) | ((plot_data['ffn_lut_build'] != 'min') & ((plot_data['ffn_fn'] == 'VLPSilu') | (plot_data['ffn_fn'] == 'VLPGelu')))]
                        lut_min_data = plot_data[(plot_data['ffn_lut_build'] == 'min')]

                    lut_min_min_row = lut_min_data.loc[lut_min_data['value'].idxmin()]
                    lut_max_min_row = lut_max_data.loc[lut_max_data['value'].idxmin()]

                    min_heatmap_value = lut_min_min_row['value']
                    max_heatmap_value = lut_max_min_row['value']

                    lut_min_exp = lut_min_min_row['ffn_min_pos_exp']
                    lut_max_exp = lut_max_min_row['ffn_max_pos_exp']

                    lut_min_align = lut_max_min_row['ffn_max_pos_exp']
                    lut_max_align = lut_min_min_row['ffn_min_pos_exp']

                    lut_min_min_exp = lut_min_data['ffn_min_pos_exp'].min()
                    lut_min_max_exp = lut_min_data['ffn_min_pos_exp'].max()
                    lut_max_max_exp = lut_max_data['ffn_max_pos_exp'].max()
                    lut_max_min_exp = lut_max_data['ffn_max_pos_exp'].min()

                    lut_min_lower_bound = max(lut_min_exp - window, lut_min_min_exp)
                    lut_min_upper_bound = min(lut_min_exp + window, lut_min_max_exp)
                    lut_max_lower_bound = max(lut_max_exp - window, lut_max_min_exp)
                    lut_max_upper_bound = min(lut_max_exp + window, lut_max_max_exp)

                    if lut_min_lower_bound == lut_min_min_exp:
                        lut_min_upper_bound += (abs(lut_min_exp - window) - abs(lut_min_min_exp))
                    if lut_max_lower_bound == lut_max_min_exp:
                        lut_max_upper_bound += (abs(lut_max_exp - window) - abs(lut_max_min_exp))
                    if lut_min_upper_bound == lut_min_max_exp:
                        lut_min_lower_bound -= (abs(lut_min_exp + window) - abs(lut_min_max_exp))
                    if lut_max_upper_bound == lut_max_max_exp:
                        lut_max_lower_bound -= (abs(lut_max_exp + window) - abs(lut_max_max_exp))

                    lut_min_slice = lut_min_data[(lut_min_data['ffn_min_pos_exp'].between(lut_min_lower_bound, lut_min_upper_bound))]
                    lut_max_slice = lut_max_data[(lut_max_data['ffn_max_pos_exp'].between(lut_max_lower_bound, lut_max_upper_bound))]

                    if min_heatmap_value < max_heatmap_value:
                        heatmap_data = lut_min_slice.pivot_table(index='ffn_exp_dim', columns='ffn_min_pos_exp', values='value')
                    else:
                        heatmap_data = lut_max_slice.pivot_table(index='ffn_exp_dim', columns='ffn_max_pos_exp', values='value')

            elif y_label == 'PWLSoftmax':


                min_segment_row = plot_data.loc[plot_data['value'].idxmin()]

                min_value = min_segment_row['value']
                min_segment = min_segment_row['attn_segment_0']
                smallest_segment = plot_data['attn_segment_0'].min()
                largest_segment = plot_data['attn_segment_0'].max()

                lut_lower_bound = max(min_segment - window, smallest_segment)
                lut_upper_bound = min(min_segment + window, largest_segment)

                if 'swin' not in key.lower() and 'vivit' not in key.lower() and 'whisper_large' not in key.lower():
                    if lut_lower_bound == smallest_segment:
                        lut_upper_bound += ((abs(min_segment) - abs(window)) - abs(smallest_segment))
                    if lut_upper_bound == largest_segment:
                        lut_lower_bound -= ((abs(min_segment) + abs(window) - abs(largest_segment)))
                else:
                    if lut_lower_bound == smallest_segment:
                        lut_upper_bound += ((abs(min_segment - window)) - abs(smallest_segment))
                    if lut_upper_bound == largest_segment:
                        lut_lower_bound -= ((abs(min_segment + window) - abs(largest_segment)))

                lut_slice = plot_data[plot_data['attn_segment_0'].between(lut_lower_bound, lut_upper_bound)]

                heatmap_data = lut_slice.pivot_table(index='attn_segments', columns='attn_segment_0', values='value')
                max_row_idx = heatmap_data.stack().idxmax()[0]
                heatmap_data = heatmap_data.drop(index=max_row_idx)

            elif y_label == 'PWLActivation':
                min_segment_row = plot_data.loc[plot_data['value'].idxmin()]

                min_value = min_segment_row['value']
                min_segment = min_segment_row['ffn_segment_0']
                smallest_segment = plot_data['ffn_segment_0'].min()
                largest_segment = plot_data['ffn_segment_0'].max()

                lut_lower_bound = max(min_segment - window, smallest_segment)
                lut_upper_bound = min(min_segment + window, largest_segment)

                if 'whisper_tiny' not in key.lower():
                    if lut_lower_bound == smallest_segment:
                        lut_upper_bound += abs((abs(min_segment) - abs(window)) - abs(smallest_segment))
                    if lut_upper_bound == largest_segment:
                        lut_lower_bound -= ((abs(min_segment) + abs(window) - abs(largest_segment)))
                else:
                    if lut_lower_bound == smallest_segment:
                        lut_upper_bound += abs((abs(min_segment - window)) - abs(smallest_segment))
                    if lut_upper_bound == largest_segment:
                        lut_lower_bound -= ((abs(min_segment + window) - abs(largest_segment)))

                lut_slice = plot_data[plot_data['ffn_segment_0'].between(lut_lower_bound, lut_upper_bound)]


                heatmap_data = lut_slice.pivot_table(index='ffn_segments', columns='ffn_segment_0', values='value')
                max_row_idx = heatmap_data.stack().idxmax()[0]
                heatmap_data = heatmap_data.drop(index=max_row_idx)
               
            elif y_label == 'TaylorSoftmax':
                min_segment_row = plot_data.loc[plot_data['value'].idxmin()]

                min_value = min_segment_row['value']
                min_segment = min_segment_row['attn_degree_center']
                smallest_segment = plot_data['attn_degree_center'].min()
                largest_segment = plot_data['attn_degree_center'].max()

                lut_lower_bound = max(min_segment - window, smallest_segment)
                lut_upper_bound = min(min_segment + window, largest_segment)

                if lut_lower_bound == smallest_segment:
                    lut_upper_bound += (abs(min_segment - window) - abs(smallest_segment))
                if lut_upper_bound == largest_segment:
                    lut_lower_bound -= (abs(min_segment + window) - abs(largest_segment))

                lut_slice = plot_data[plot_data['attn_degree_center'].between(lut_lower_bound, lut_upper_bound)]
                
                
                

                heatmap_data = lut_slice.pivot_table(index='attn_degrees', columns='attn_degree_center', values='value')
                max_row_idx = heatmap_data.stack().idxmax()[0]
                heatmap_data = heatmap_data.drop(index=max_row_idx)
                

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

            # Convert index and column labels to integers for cleaner display
            try:
                heatmap_data.index = heatmap_data.index.astype(int)
                heatmap_data.columns = heatmap_data.columns.astype(int)
            except Exception:
                pass  # If conversion fails, keep original labels

            ax = axes[j, i]

            # Find the minimum value position for highlighting
            min_value = heatmap_data.min().min()
            min_pos = None
            if pd.notna(min_value):
                # Find the position of the minimum value
                min_idx = heatmap_data.stack().idxmin()
                if pd.notna(min_idx):
                    min_pos = (heatmap_data.index.get_loc(min_idx[0]), 
                              heatmap_data.columns.get_loc(min_idx[1]))
            
            if 'whisper_tiny' in key.lower():
            # Build mask to hide only NaN cells, keep colored cells for values > 10
                mask = heatmap_data.isna()
                ann = heatmap_data.copy().applymap(lambda v: f"{v:.1f}" if pd.notna(v) and v <= 25 else "")
            else:
                mask = heatmap_data.isna()
                ann = heatmap_data.copy().applymap(lambda v: f"{v:.2f}" if pd.notna(v) and v <= 10 else "")
            
            if i == 0:
                yticklabels = True
            else:
                yticklabels = False

            sns.heatmap(
                heatmap_data,
                mask=mask,
                ax=ax,
                annot=ann.values,
                fmt="",
                cmap='viridis',
                cbar=False,
                xticklabels=False,
                yticklabels=yticklabels,
                annot_kws={"fontsize": 4.5}
            )
            
            # FORCE x-tick labels to show on EVERY heatmap at the bottom
            ax.set_xticks([x + 0.5 for x in range(len(heatmap_data.columns))])  # Offset by 0.5 to center on cells
            ax.set_xticklabels(heatmap_data.columns, visible=True, fontsize=1)
            ax.tick_params(axis='x', labelbottom=True, labeltop=False, labelsize=4.5, pad=0)  # pad=0 moves labels closer
            ax.xaxis.set_ticks_position('bottom')
            ax.xaxis.set_label_position('bottom')
            
            # Move x-tick labels closer to the heatmap by adjusting their position
            for label in ax.get_xticklabels():
                label.set_verticalalignment('top')
                label.set_y(0.08)  # Move labels up closer to the heatmap
            
            # Remove axis labels that seaborn adds automatically
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            # Add right-side y-axis labels only for the last column, positioned between method pairs
            if i == n_models - 1:  # Last column
                right_label = ''
                show_label = False
                
                # Show labels positioned between the method pairs
                if y_label == 'VLPActivation':  # Second row of VLP pair
                    right_label = 'Min/Max Exp'
                    show_label = True
                elif y_label == 'PWLActivation':  # Second row of PWL pair
                    right_label = 'Segment Range'
                    show_label = True
                elif y_label == 'TaylorSoftmax':  # Only row for Taylor
                    right_label = 'Degree Center'
                    show_label = True
                
                if show_label:
                    # Position the label between the current and previous row for pairs
                    if y_label in ['VLPActivation', 'PWLActivation']:
                        # Create text positioned between this row and the previous row
                        fig.text(0.905, (axes[j, i].get_position().y0 + axes[j-1, i].get_position().y1) / 2, 
                                right_label, fontsize=5, rotation=270, ha='left', va='center')
                    else:  # TaylorSoftmax - centered on its own row
                        fig.text(0.905, (axes[j, i].get_position().y0 + axes[j, i].get_position().y1) / 2,
                                right_label, fontsize=5, rotation=270, ha='left', va='center')
            
            # Remove x-axis labels
            ax.set_xlabel('')
            
            # Highlight the minimum value cell
            if min_pos is not None:
                # Add a colored rectangle around the minimum value cell
                rect = Rectangle((min_pos[1], min_pos[0]), 1, 1, 
                               linewidth=0.75, edgecolor=highlight_color, 
                               facecolor='none', clip_on=False)
                ax.add_patch(rect)
            # Add model key as title on the top row only

            key_title = key.replace('_', ' ').title()
            if 'swin' in key_title.lower():
                key_title = key_title.replace('Swin', 'SwinV2')
            if 'vivit' in key_title.lower():
                key_title = 'ViViT'
            
            if j == 0:  
                # Set title positioned in the center
                ax.text(0.5, 1.02, key_title, transform=ax.transAxes, 
                    fontsize=6.75, ha='center', va='bottom')

                # ax.text(0.08, 1.25, torch_dict[key_title], transform=ax.transAxes, 
                #         fontsize=3.5, ha='center', va='center', color='black',)
                # Split ppl values and display vertically with line separator
                # ppl_values = str(ppl_dict[key_title]).split('/')
                # if len(ppl_values) == 2:
                #     # First value (top)
                #     ax.text(.91, 1.32, ppl_values[0], transform=ax.transAxes, 
                #            fontsize=5.5, ha='center', va='center')
                #     # Separator line
                #     ax.text(.91, 1.29, '_____', transform=ax.transAxes, 
                #            fontsize=5.0, ha='center', va='center')
                #     # Second value (bottom)
                #     ax.text(.91, 1.1, ppl_values[1], transform=ax.transAxes, 
                #            fontsize=5.5, ha='center', va='center')
                # else:
                #     # Fallback for single values
                #     ax.text(0.5, 1.08, str(ppl_dict[key_title]), transform=ax.transAxes, 
                #            fontsize=5.5, ha='center', va='bottom')
            
            # Add method labels on the y-axis for the first column only
            y_label_title = 'VLP SM' if y_label == 'VLPSoftmax' else \
                      'VLP S/G' if y_label == 'VLPActivation' else \
                      'PWL SM' if y_label == 'PWLSoftmax' else \
                      'PWL S/G' if y_label == 'PWLActivation' else \
                      'Taylor SM'

            if i == 0:
                ax.set_ylabel(y_label_title, fontsize=7, rotation=90, labelpad=11, 
                             bbox=dict(boxstyle='round,pad=0.1', facecolor="#C5C7C9", edgecolor="#4A92DA", linewidth=0.5))
            
            # Add left-side y-axis labels for y-data only for the first column, positioned between method pairs
            if i == 0:  # First column
                left_label = ''
                show_left_label = False
                
                # Show labels positioned between the method pairs for y-axis data
                if y_label == 'VLPActivation':  # Second row of VLP pair
                    left_label = 'LUT Size'
                    show_left_label = True
                elif y_label == 'PWLActivation':  # Second row of PWL pair
                    left_label = 'Segments'
                    show_left_label = True
                elif y_label == 'TaylorSoftmax':  # Only row for Taylor
                    left_label = 'Degrees'
                    show_left_label = True
                
                if show_left_label:
                    # Position the label between the current and previous row for pairs
                    if y_label in ['VLPActivation', 'PWLActivation']:
                        # Create text positioned between this row and the previous row
                        fig.text(0.109, (axes[j, i].get_position().y0 + axes[j-1, i].get_position().y1) / 2, 
                                left_label, fontsize=6, rotation=90, ha='right', va='center')
                    else:  # TaylorSoftmax - centered on its own row
                        fig.text(0.109, (axes[j, i].get_position().y0 + axes[j, i].get_position().y1) / 2,
                                left_label, fontsize=6, rotation=90, ha='right', va='center')
            
            # Set tick parameters for consistent styling
            ax.tick_params(axis='x', which='major', pad=0)
            ax.tick_params(axis='y', which='major', pad=-3)
            ax.tick_params(axis='y', which='major', left=False)  # Turn off y tick marks but keep labels
            ax.tick_params(axis='x', which='major', bottom=False)  # Turn off x tick marks but keep labels
            
            # Ensure y-tick labels are handled properly
            if yticklabels:
                if y_label == 'TaylorSoftmax':
                    plt.setp(ax.get_yticklabels(), visible=True, fontsize=4.5, rotation=0)
                else:
                    plt.setp(ax.get_yticklabels(), visible=True, fontsize=4.5)

        # Create table for best perplexity values in the bottom row
        table_ax = axes[5, i]  # Bottom row for table
        table_ax.axis('off')  # Turn off axis
        
        # Get perplexity values from ppl_dict (same as shown at top of figure)
        ppl_values = str(ppl_dict[key_title]).split('/')
        best_vlp = ppl_values[0]  # First value (VLP)
        best_pwl = ppl_values[1]  # Second value (PWL) 
        
        # Get best Taylor values from CSV
        data = pd.read_csv(value)
        data = data.drop(columns=['model', 'modality', 'function_name', 'patch_attention', 'patch_ffn'])
        data['value'] = pd.to_numeric(data['value'], errors='coerce')
        data = data.dropna(subset=['value'])
        taylor_data = data[data['attn_fn'] == 'TaylorSoftmax']
        best_taylor = f'{taylor_data["value"].min():.2f}'
        
        # Get base accuracy for this model
        base_value = base_acc.get(key, 'N/A')
        if isinstance(base_value, (int, float)):
            base_str = f'{base_value:.2f}'
        else:
            base_str = str(base_value)
        
        # Create table data
        table_data = [['Base', 'VLP', 'PWL', 'T'], 
                     [base_str, best_vlp, best_pwl, best_taylor]]
        
        # Create table
        table = table_ax.table(cellText=table_data,
                              cellLoc='center',
                              loc='center',
                              colWidths=[0.25, 0.25, 0.25, 0.25])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(4.5)
        
        # Style header row and make border lines thinner
        for j in range(4):
            table[(0, j)].set_facecolor('#E8E8E8')
            table[(0, j)].set_text_props(weight='bold')
            table[(0, j)].set_linewidth(0.5)
            table[(1, j)].set_facecolor('#F5F5F5')
            table[(1, j)].set_linewidth(0.5)
        
    # Add "Full PPL" label for the first column only (outside the loop)
    if len(data_dict) > 0:
        table_ax_first = axes[5, 0]  # First column table
        fig.text(0.0845, (table_ax_first.get_position().y0 + table_ax_first.get_position().y1) / 2,
                'Full PPL', fontsize=7, rotation=90, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.1', facecolor="#C5C7C9", edgecolor="#4A92DA", linewidth=0.5))

    # Save under local figures/ like single_heatmap and ensure path exists
    save_dir = os.path.expanduser('figures')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.savefig(os.path.join(save_dir, 'perplexity_plot.png'), dpi=1200, bbox_inches='tight', pad_inches=0.05)
    plt.savefig(os.path.join(save_dir, 'perplexity_plot.pdf'), dpi=1200, bbox_inches='tight', pad_inches=0.05)

if __name__ == "__main__":
    data_dict = {
        'llama_2_7b': 'csv/meta-llama/Llama-2-7b-hf/metric.csv',
        'llama_2_13b': 'csv/meta-llama/Llama-2-13b-hf/metric.csv',
        #'llama_3_8b': 'csv/meta-llama/Llama-3.1-8B/metric.csv',
        #'llama_2_13b': 'csv/meta-llama/Llama-2-7b-hf/metric.csv',
        'whisper_tiny': 'csv/openai/whisper-tiny/nonlinear_config/metric.csv',
        'whisper_large': 'csv/openai/whisper-large/metric.csv',
        'swin_tiny': 'csv/microsoft/swinv2-tiny-patch4-window8-256/metric.csv',
        'swin_large': 'csv/microsoft/swinv2-large-patch4-window12to16-192to256-22kto1k-ft/metric.csv',
        'vivit': 'csv/google/vivit-b-16x2/metric.csv'
    }

    # You can customize the highlight color here (e.g., 'red', 'orange', 'yellow', '#FF5733', etc.)
    highlight_color = "#FF5E00"  # Example: a shade of orange
    
    #single_heatmap(data_dict)
    plot_perplexity(data_dict, highlight_color=highlight_color)