import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_perplexity(data_dict: dict):
    
    y_labels = ['VLPSoftmax', 'VLPActivation', 'PWLSoftmax', 'PWLActivation', 'TaylorSoftmax']

    silu_labels = ['llama']
    gelu_labels = ['whisper', 'swin', 'vivit']


    n_models = len(data_dict)

    fig_width = 506 / 72.27
    fig_height = fig_width * 0.5

    fig, axes = plt.subplots(3, 10, figsize=(fig_width, fig_height))

    for i, key, value in enumerate(data_dict.items()):

        data = pd.read_csv(value)
        data = data.drop(columns=['model', 'modality', 'function_name', 'patch_attention', 'patch_ffn', 'attn_fn', 'ffn_fn'])

        for j, y_label in enumerate(y_labels):

            plot_data = data.copy()

            if y_label == 'VLPSoftmax':
                plot_data = plot_data.drop(columns=ffn_exp_dim,ffn_max_pos_exp,ffn_window_size,attn_segments,attn_segment_0,ffn_segments,ffn_segment_0,attn_degree_center,attn_degrees)
            elif y_label == 'VLPActivation':
                plot_data = plot_data.drop(columns=attn_exp_dim,attn_max_exp,attn_min_exp,attn_window_size,attn_lut_build,attn_segments,attn_segment_0,ffn_segments,ffn_segment_0,attn_degree_center,attn_degrees)
            elif y_label == 'PWLSoftmax':
                plot_data = plot_data.drop(columns=attn_exp_dim,attn_max_exp,attn_min_exp,attn_window_size,attn_lut_build,ffn_exp_dim,ffn_max_pos_exp,ffn_window_size,ffn_segments,ffn_segment_0,attn_degree_center,attn_degrees)
            elif y_label == 'PWLActivation':
                plot_data = plot_data.drop(columns=attn_exp_dim,attn_max_exp,attn_min_exp,attn_window_size,attn_lut_build,ffn_exp_dim,ffn_max_pos_exp,ffn_window_size,attn_segments,attn_segment_0,attn_degree_center,attn_degrees)
            elif y_label == 'TaylorSoftmax':
                plot_data = plot_data.drop(columns=attn_exp_dim,attn_max_exp,attn_min_exp,attn_window_size,attn_lut_build,ffn_exp_dim,ffn_max_pos_exp,ffn_window_size,attn_segments,attn_segment_0,ffn_segments,ffn_segment_0)

            if key in silu_labels and 'Activation' in y_label:
                y_label = y_label.replace('Activation', 'Silu')
                plot_data = data[data['ffn_fn'] == y_label]
            elif key in gelu_labels and 'Activation' in y_label:
                y_label = y_label.replace('Activation', 'Gelu')
                plot_data = data[data['ffn_fn'] == y_label]
            elif key in silu_labels and 'Softmax' in y_label:
                plot_data = data[data['attn_fn'] == y_label]

            np_data = plot_data.to_numpy()
            sns.heatmap(np_data, ax=axes[j, i], annot=True, fmt=".2f", cmap='viridis', cbar=False, xticklabels=False, yticklabels=False)

    save_dir = '~/mugi_profiling/figures'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.tight_layout()
    plt.save(save_dir + '/perplexity_plot.png', dpi=1200)

data_dict = {
    'llama_2_7b': 'csv/meta-llama/Llama-2-7b-hf/metric.csv',
    'llama_2_70b': 'csv/meta-llama/Llama-2-70b-hf/metric.csv',
    'llama_3_8b': 'csv/meta-llama/Llama-3-8b-hf/metric.csv',
    'llama_3_70b': 'csv/meta-llama/Llama-3-70b-hf/metric.csv',
    'whisper_tiny': 'csv/openai/whisper-tiny/metric.csv',
    'whisper_large': 'csv/openai/whisper-large/metric.csv',
    'swin_tiny': 'csv/microsoft/swinv2-tiny-patch4-window8-256',
    'swin_base': 'csv/microsoft/swinv2-base-patch4-window8-256',
    'vivit': 'csv/google/vivit-b-16x2/metric.csv'
}