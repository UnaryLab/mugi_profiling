import torch
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator
from matplotlib.patches import Rectangle
from matplotlib import gridspec
from scipy.interpolate import make_interp_spline
import numpy as np

def get_min_list(arr, length=3):

    idx = np.searchsorted(arr, 0)
    
    # Two-pointer expansion around 0
    left, right = idx - 1, idx
    result = []
    
    while len(result) < length and (left >= 0 or right < len(arr)):
        if left < 0:
            result.append(arr[right])
            right += 1
        elif right >= len(arr):
            result.append(arr[left])
            left -= 1
        else:
            if abs(arr[left]) <= abs(arr[right]):
                result.append(arr[left])
                left -= 1
            else:
                result.append(arr[right])
                right += 1
    
    return np.array(sorted(result))

def convert_to_percentage(tensor):
    tensor = tensor.to(torch.float32)
    return tensor / tensor.sum() * 100

def set_ax_title(ax1, ax2, title):
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()

    left = pos1.x0
    right = pos2.x1
    top = pos1.y1
    height = pos1.height

    new_ax = fig.add_axes([left, top, right-left, 0.001])
    new_ax.set_title(title, fontsize=6, ha='center', va='center', pad=9)
    new_ax.axis('off')

def trim_distribution(x, y, threshold=0.05, pad=5):
    # Find indices where y is above the threshold
    indices = np.where(y > threshold)[0]
    
    if len(indices) == 0:
        # nothing above threshold
        return np.array([]), np.array([])
    
    # Determine start and end indices with padding
    start = max(indices[0] - pad, 0)
    end = min(indices[-1] + pad + 1, len(y))
    
    # Slice the arrays
    x_trimmed = x[start:end]
    y_trimmed = y[start:end]
    
    return x_trimmed, y_trimmed

def sort_int_string_list(l):
    """
    Sorts a list of strings that represent integers.
    """
    return sorted(l, key=lambda x: int(x.split('_')[-1]) if '_' in x else int(x))

def sort_dict(d, sorted_list):
    sorted_dict = {}
    for key in sorted_list:
        if key in d:
            sorted_dict[key] = d[key]
    return sorted_dict

def smooth_curve(x, y):
    x_new = np.linspace(x.min(), x.max(), 50)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_new)

    return x_new, y_smooth

def compute_outliers(y, threshold=3):
    y = np.array(y)
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    return np.abs(y - median) > threshold * mad

def moving_average(y, window_size=5, threshold=30, detect_outliers=True):
    y = np.array(y)

    if detect_outliers:
        outliers = compute_outliers(y, threshold)
    
    y_smooth = np.convolve(y, np.ones(window_size)/window_size, mode='same')
    
    if detect_outliers:
        y_smooth[outliers] = y[outliers]
    
    return y_smooth

def exp_distribution(model, figure_dict, ax, nonlinear_op, s, linewidths, threshold, pad, window_size, color_map):
    key_1 = list(figure_dict.keys())[0]

    x = np.arange(-16, 16)
    x = x[6:-6]

    layer_labels = []
    layer_lines = []
    lines = {}
    seq_0 = None

    if 'layer' in key_1:
        for i, (layer, layer_value) in enumerate(figure_dict.items()):
            layer = int(layer.split('_')[-1]) + 1
            label = f'Layer {layer}'
            if label not in layer_labels:
                layer_lines.append(plt.Line2D([0], [0], color='white', label=label))
                layer_labels.append(label)

            for j, (seq, seq_value) in enumerate(layer_value.items()):
                tensor = torch.load(seq_value, map_location='cpu')
                tensor = convert_to_percentage(tensor)
                y = tensor.numpy()

                y[-1] = 0
                
                y = y[6:-6]

                x_trimmed, y_trimmed = trim_distribution(x, y, threshold=threshold, pad=pad)

                seq = int(seq.split('_')[-1]) + 1 if isinstance(seq, str) else seq

                color = color_map[i][j]

                l, = ax.plot(x_trimmed, y_trimmed, linewidth=linewidths, label=f'{seq}', color=color, alpha=s)

                if seq not in lines:
                    lines[seq] = [plt.Line2D([0], [0], color='white')]

                lines[seq].append(l)
    
                
    elif 'encoder' in key_1 or 'decoder' in key_1:
        for i, (layer_type, layer_type_value) in enumerate(figure_dict.items()):
            for j, (layer, layer_value) in enumerate(layer_type_value.items()):
                layer = int(layer.split('_')[-1]) + 1
                label = f'Layer {layer}'
                if label not in layer_labels:
                    layer_lines.append(plt.Line2D([0], [0], color='white', label=label))
                    layer_labels.append(label)
                for k, (seq, seq_value) in enumerate(layer_value.items()):
                    tensor = torch.load(seq_value, map_location='cpu')

                    

                    if seq == 'seq_len_0':
                        seq_0 = tensor
                        continue
                    elif seq == 'seq_len_111' and seq_0 is not None:
                        tensor += seq_0
                    elif seq == 'seq_len_1':
                        seq = 'seq_len_223'

                    tensor = convert_to_percentage(tensor)
                    y = tensor.numpy()
                    y[-1] = 0
                    y = y[6:-6]

                    

                    color = color_map[(i*(len(figure_dict))) + j][k]

                    seq = int(seq.split('_')[-1]) + 1 if isinstance(seq, str) else seq

                    x_trimmed, y_trimmed = trim_distribution(x, y, threshold=threshold, pad=pad)
                    #y_smooth = moving_average(y_trimmed, window_size)
                    l, = ax.plot(x_trimmed, y_trimmed, linewidth=linewidths, label=f'{seq}', color=color, alpha=s)

                    if seq not in lines:
                        if seq == 375:
                            lines[seq] = [plt.Line2D([0], [0], color='white', label='Enc')]
                        elif seq == 112:
                            lines[seq] = [plt.Line2D([0], [0], color='white', label='Dec')]
                        else:
                            lines[seq] = [plt.Line2D([0], [0], color='white')]

                    lines[seq].append(l)
    else:
        for i, (block, block_value) in enumerate(figure_dict.items()):
            block = int(block.split('_')[-1])
            label = f'Stage {block}'
            if label not in layer_labels:
                layer_lines.append(plt.Line2D([0], [0], color='white', label=label))
                layer_labels.append(label)
            for j, (layer, layer_value) in enumerate(block_value.items()):
                for k, (seq, seq_value) in enumerate(layer_value.items()):
                    tensor = torch.load(seq_value, map_location='cpu')
                    tensor = convert_to_percentage(tensor)
                    y = tensor.numpy()
                    
                    y[-1] = 0
                    y = y[6:-6]


                    seq = int(seq.split('_')[-1]) + 1 if isinstance(seq, str) else seq

                    color = color_map[(i*(len(figure_dict))) + j][k]

                    x_trimmed, y_trimmed = trim_distribution(x, y, threshold=threshold, pad=pad)
                    #y_smooth = moving_average(y_trimmed, window_size)
                    l, = ax.plot(x_trimmed, y_trimmed, linewidth=linewidths, label=f'{seq}', color=color, alpha=s)

                    if seq not in lines:
                        if seq == 1024:
                            lines[seq] = [plt.Line2D([0], [0], color='white'), plt.Line2D([0], [0], color='white', label='Layer 1')]
                        elif seq == 16:
                            lines[seq] = [plt.Line2D([0], [0], color='white'), plt.Line2D([0], [0], color='white', label='Layer 2')]
                        else:
                            lines[seq] = [plt.Line2D([0], [0], color='white'), plt.Line2D([0], [0], color='white')]

                    lines[seq].append(l)

    for seq, seq_list in lines.items():
        layer_lines.extend(seq_list)
    #exit()
    return layer_lines

def value_distribution(model, figure_dict, ax, nonlinear_op, s, linewidths, threshold, pad, window_size, color_map, window=False):
    key_1 = list(figure_dict.keys())[0]

    if nonlinear_op == 'softmax':
        x = np.arange(-20, 0, 0.05)
    else:
        x = np.arange(-10, 10, 0.05)

    seq_0 = None

    if 'layer' in key_1:
        for i, (layer, layer_value) in enumerate(figure_dict.items()):
            for j, (seq, seq_value) in enumerate(layer_value.items()):
                tensor = torch.load(seq_value, map_location='cpu')
                tensor = convert_to_percentage(tensor)
                y = tensor.numpy()
                if len(y) > 400:
                    y = y[:400]

                x_trimmed, y_trimmed = trim_distribution(x, y, threshold=threshold, pad=pad)
                if window:
                    y_trimmed = moving_average(y_trimmed, window_size=20, threshold=80, detect_outliers=False)
                    
                color = color_map[i][j]

                ax.plot(x_trimmed, y_trimmed, linewidth=linewidths, color=color, alpha=s)

    elif 'encoder' in key_1 or 'decoder' in key_1:
        for i, (layer_type, layer_type_value) in enumerate(figure_dict.items()):
            for j, (layer, layer_value) in enumerate(layer_type_value.items()):
                for k, (seq, seq_value) in enumerate(layer_value.items()):
                    
                    tensor = torch.load(seq_value, map_location='cpu')

                    if torch.all(tensor == 0):
                        continue

                    if seq == 'seq_len_0':
                        seq_0 = tensor
                        continue
                    elif seq == 'seq_len_111' and seq_0 is not None:
                        tensor += seq_0
                    elif seq == 'seq_len_1':
                        seq = 'seq_len_223'
                    
                    tensor = convert_to_percentage(tensor)
                    y = tensor.numpy()

                    color = color_map[(i*(len(figure_dict))) + j][k]

                    x_trimmed, y_trimmed = trim_distribution(x, y, threshold=threshold, pad=pad)
                    if window:
                        y_trimmed = moving_average(y_trimmed, window_size=20, threshold=10000, detect_outliers=False)
                    ax.plot(x_trimmed, y_trimmed, linewidth=linewidths, color=color, alpha=s)
                    
    else:
        for i, (block, block_value) in enumerate(figure_dict.items()):
            for j, (layer, layer_value) in enumerate(block_value.items()):
                for k, (seq, seq_value) in enumerate(layer_value.items()):
                    tensor = torch.load(seq_value, map_location='cpu')
                    tensor = convert_to_percentage(tensor)
                    y = tensor.numpy()

                    color = color_map[(i*(len(figure_dict))) + j][k]

                    x_trimmed, y_trimmed = trim_distribution(x, y, threshold=threshold, pad=pad)
                    if window:
                        y_trimmed = moving_average(y_trimmed, window_size=5, threshold=0, detect_outliers=False)
                    ax.plot(x_trimmed, y_trimmed, linewidth=linewidths, color=color, alpha=s)

    return lines

def merge_dicts(d1, d2):

    if not d1:
        return d2
    if not d2:
        return d1

    d1_keys = list(d1.keys())
    d2_keys = list(d2.keys())


    for k in d2_keys:
        if k in d1_keys:
            new = merge_dicts(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1

def get_subdirs(base_dir, model_dict):
    print(base_dir)
    if os.path.isdir(base_dir):
        output_dict = {}
        for subdir in os.listdir(base_dir):
            subdir = os.path.join(base_dir, subdir)
            path_dict = get_subdirs(subdir, model_dict)
            
            if isinstance(path_dict, dict) and path_dict:
                output_dict = merge_dicts(output_dict, path_dict)

        return output_dict

    else:
        if os.path.isfile(base_dir):
            path_dir = base_dir.split('.pt')[0]
            split_path = path_dir.split('/')

            model = split_path[2]
            nonlinear_op = split_path[4]
            pre_post = split_path[5]
            distribution = split_path[6]
            layer = split_path[7]

            if len(split_path) >= 10:
                block = split_path[8]
                block_split = block.split('_')
                seq = split_path[9]
            else:
                block = None
                seq = split_path[8]

            layer_split = layer.split('_')
            seq_split = seq.split('_')

            path_dict = {}
            if model in model_dict and int(layer_split[-1]) in model_dict[model]['layers']:
                if 'seq' in model_dict[model]:
                    if int(seq_split[-1]) in model_dict[model]['seq']:
                        path_dict[model] = {nonlinear_op: {pre_post: {distribution: {layer: {seq: base_dir}}}}}
                    
                elif 'enc_seq' in model_dict[model]:
                    if int(seq_split[-1]) in model_dict[model]['enc_seq']:
                        path_dict[model] = {nonlinear_op: {pre_post: {distribution: {'encoder': {layer: {seq: base_dir}}}}}}

                    if int(seq_split[-1]) in model_dict[model]['dec_seq']:
                        path_dict[model] = {nonlinear_op: {pre_post: {distribution: {'decoder': {layer: {seq: base_dir}}}}}}
                    
                else:
                    if int(block_split[-1]) in model_dict[model]['blocks']:
                        if int(seq_split[-1]) in model_dict[model][f'layer_{int(block_split[-1]) + 1}_seq']:
                            path_dict[model] = {nonlinear_op: {pre_post: {distribution: {f'block_{int(block_split[-1]) + 1}': {layer: {seq: base_dir}}}}}}
            return path_dict

base_dir = 'profile/'

model_config = {
    'Llama-2-7b-hf': {
        'layers': [0, 15, 31],
        'seq': [1023, 2047, 4095],
    },
    'Llama-2-70b-hf': {
        'layers': [0, 39, 79],
        'seq': [1023, 2047, 4095],
    },
    'Llama-3.1-8B': {
        'layers': [0, 15, 31],
        'seq': [1023, 2047, 4095],
    },
    'Llama-3.1-405B': {
        'layers': [0, 62, 125],
        'seq': [1023, 2047, 4095],
    },
    'whisper-tiny': {
        'layers': [0, 1, 3],
        'enc_seq': [374, 749, 1499],
        'dec_seq': [1, 111],
    },
    'whisper-large': {
        'layers': [0, 15, 31],
        'enc_seq': [374, 749, 1499],
        'dec_seq': [0, 1, 111],
    },
    'swinv2-tiny-patch4-window8-256': {
        'layers': [0, 1],
        'blocks': [0, 3],
        'layer_1_seq': [1023, 2047, 4095],
        # 'layer_2_seq': [225, 511, 1023],
        # 'layer_3_seq': [63, 127, 255],
        'layer_4_seq': [15, 31, 63],
    },
    'swinv2-large-patch4-window12to16-192to256-22kto1k-ft': {
        'layers': [0, 1],
        'blocks': [0, 3],
        'layer_1_seq': [1023, 2047, 4095],
        # 'layer_2_seq': [225, 511, 1023],
        # 'layer_3_seq': [63, 127, 255],
        'layer_4_seq': [15, 31, 63],
    },
    'vivit-b-16x2': {
        'layers': [0, 5, 11],
        'seq': [784, 1568, 3136],
    },
}

s = 0.6
linewidths = 0.35
threshold = 0.175
pad = 15
window_size = 3

model_dict = get_subdirs(base_dir, model_config)

print(model_dict)

per_fig_width = 506 / 72.27
per_fig_height = per_fig_width * 0.3

# Create figure with gridspec for custom spacing
fig = plt.figure(figsize=(per_fig_width, per_fig_height))

# Create gridspec with proper column arrangement for seamless pairs
# Columns: Llama2_L, Llama2_R, gap, Llama3_L, Llama3_R, gap, Whisper_L, Whisper_R, gap, SwinV2_L, SwinV2_R, gap, ViViT
gs = gridspec.GridSpec(4, 13, figure=fig, hspace=0.285, wspace=0,
                       width_ratios=[1, 1, 0.25, 1, 1, 0.25, 1, 1, 0.25, 1, 1, 0.25, 1])

# Create axes manually - pairs are adjacent (no gap), with spacing between model families
axes = {}

# Llama 2 (columns 0,1 - adjacent)
for row in range(4):
    axes[(row, 0)] = fig.add_subplot(gs[row, 0])
    axes[(row, 1)] = fig.add_subplot(gs[row, 1])

# Llama 3 (columns 3,4 - adjacent) 
for row in range(4):
    axes[(row, 2)] = fig.add_subplot(gs[row, 3])
    axes[(row, 3)] = fig.add_subplot(gs[row, 4])

# Whisper (columns 6,7 - adjacent)
for row in range(4):
    axes[(row, 4)] = fig.add_subplot(gs[row, 6])
    axes[(row, 5)] = fig.add_subplot(gs[row, 7])

# SwinV2 (columns 9,10 - adjacent)
for row in range(4):
    axes[(row, 6)] = fig.add_subplot(gs[row, 9])
    axes[(row, 7)] = fig.add_subplot(gs[row, 10])

# ViViT (column 12 - solo)
for row in range(4):
    axes[(row, 8)] = fig.add_subplot(gs[row, 12])

model_dict_sorted = {}
model_dict_sorted['Llama 2 7B'] = model_dict['Llama-2-7b-hf']
model_dict_sorted['Llama 2 70B'] = model_dict['Llama-2-70b-hf']
model_dict_sorted['Llama 3.1 8B'] = model_dict['Llama-3.1-8B']
model_dict_sorted['Llama 3.1 405B'] = model_dict['Llama-3.1-405B']
model_dict_sorted['Whisper Tiny'] = model_dict['whisper-tiny']
model_dict_sorted['Whisper Large'] = model_dict['whisper-large']
model_dict_sorted['SwinV2 Tiny'] = model_dict['swinv2-tiny-patch4-window8-256']
model_dict_sorted['SwinV2 Large'] = model_dict['swinv2-large-patch4-window12to16-192to256-22kto1k-ft']
model_dict_sorted['ViViT Base'] = model_dict['vivit-b-16x2']

model_list = ['Llama 2 7B', 'Llama 2 70B', 'Llama 3.1 8B', 'Llama 3.1 405B', 'Whisper Tiny', 'Whisper Large', 'SwinV2 Tiny', 'SwinV2 Large', 'ViViT Base']
nonlinear_ops = ['softmax', 'silu', 'gelu']
distribution_list = ['value_dist', 'exp_dist']
layer_type_list = ['encoder', 'decoder']

model_dict_sorted = sort_dict(model_dict_sorted, model_list)
for model in model_dict_sorted.keys():
    model_dict_sorted[model] = sort_dict(model_dict_sorted[model], nonlinear_ops)
    for nonlinear_op in model_dict_sorted[model].keys():
        for pre_post in model_dict_sorted[model][nonlinear_op].keys():
            model_dict_sorted[model][nonlinear_op][pre_post] = sort_dict(model_dict_sorted[model][nonlinear_op][pre_post], distribution_list)
            for distribution in model_dict_sorted[model][nonlinear_op][pre_post].keys():
                if 'Llama' in model or 'ViViT' in model:
                    layer_list = sort_int_string_list(list(model_dict_sorted[model][nonlinear_op][pre_post][distribution].keys()))
                    model_dict_sorted[model][nonlinear_op][pre_post][distribution] = sort_dict(model_dict_sorted[model][nonlinear_op][pre_post][distribution], layer_list)
                    for layer in model_dict_sorted[model][nonlinear_op][pre_post][distribution].keys():
                        seq_list = sort_int_string_list(list(model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer].keys()))
                        model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer] = sort_dict(model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer], seq_list)
                elif 'Whisper' in model:
                    model_dict_sorted[model][nonlinear_op][pre_post][distribution] = sort_dict(model_dict_sorted[model][nonlinear_op][pre_post][distribution], layer_type_list)
                    for layer_type in model_dict_sorted[model][nonlinear_op][pre_post][distribution].keys():
                        layer_list = sort_int_string_list(list(model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer_type].keys()))
                        model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer_type] = sort_dict(model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer_type], layer_list)
                        for layer in model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer_type].keys():
                            if 'seq_len_1' in model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer_type][layer]:
                                model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer_type][layer]['seq_len_223'] = model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer_type][layer].pop('seq_len_1')
                            seq_list = sort_int_string_list(list(model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer_type][layer].keys()))
                            model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer_type][layer] = sort_dict(model_dict_sorted[model][nonlinear_op][pre_post][distribution][layer_type][layer], seq_list)
                else:
                    block_list = sort_int_string_list(list(model_dict_sorted[model][nonlinear_op][pre_post][distribution].keys()))
                    model_dict_sorted[model][nonlinear_op][pre_post][distribution] = sort_dict(model_dict_sorted[model][nonlinear_op][pre_post][distribution], block_list)
                    for block in model_dict_sorted[model][nonlinear_op][pre_post][distribution].keys():
                        layer_list = sort_int_string_list(list(model_dict_sorted[model][nonlinear_op][pre_post][distribution][block].keys()))
                        model_dict_sorted[model][nonlinear_op][pre_post][distribution][block] = sort_dict(model_dict_sorted[model][nonlinear_op][pre_post][distribution][block], layer_list)
                        for layer in model_dict_sorted[model][nonlinear_op][pre_post][distribution][block].keys():
                            seq_list = sort_int_string_list(list(model_dict_sorted[model][nonlinear_op][pre_post][distribution][block][layer].keys()))
                            model_dict_sorted[model][nonlinear_op][pre_post][distribution][block][layer] = sort_dict(model_dict_sorted[model][nonlinear_op][pre_post][distribution][block][layer], seq_list)

lines = []
labels = []

color_map = {
    0: {0:'#85BEFF', 1:'#5492E3', 2:'#3A6DAB', 3:'#22436E'},
    1: {0:'#84F593', 1:'#3FBF50', 2:'#1B8028', 3:'#085212'},
    2: {0:'#FFC687', 1:'#EB9F49', 2:'#DE8623', 3:'#E07400'},
    3: {0:'#FFA6A6', 1:'#E06060', 2:'#C93434', 3:'#960000'},
    4: {0:'#FFB366', 1:'#E8A34B', 2:'#B37F32', 3:'#6E5526'},
    5: {0:'#66FFB3', 1:'#4BE8A3', 2:'#32B37F', 3:'#1E6E55'}
}

# Share y-axes for model pairs to make them look like one continuous graph
axes[(0, 1)].sharey(axes[(0, 0)])  # Llama 2s
axes[(1, 1)].sharey(axes[(1, 0)])
axes[(2, 1)].sharey(axes[(2, 0)])
axes[(3, 1)].sharey(axes[(3, 0)])

axes[(0, 3)].sharey(axes[(0, 2)])  # Llama 3s
axes[(1, 3)].sharey(axes[(1, 2)])
axes[(2, 3)].sharey(axes[(2, 2)])
axes[(3, 3)].sharey(axes[(3, 2)])

axes[(0, 5)].sharey(axes[(0, 4)])  # Whispers
axes[(1, 5)].sharey(axes[(1, 4)])
axes[(2, 5)].sharey(axes[(2, 4)])
axes[(3, 5)].sharey(axes[(3, 4)])

axes[(0, 7)].sharey(axes[(0, 6)])  # SwinV2s
axes[(1, 7)].sharey(axes[(1, 6)])
axes[(2, 7)].sharey(axes[(2, 6)])
axes[(3, 7)].sharey(axes[(3, 6)])

# Hide y-axis labels and ticks on right subplots of each pair to create seamless appearance
for row in range(4):
    axes[(row, 1)].tick_params(left=False, labelleft=False)  # Llama 2 right
    axes[(row, 3)].tick_params(left=False, labelleft=False)  # Llama 3 right
    axes[(row, 5)].tick_params(left=False, labelleft=False)  # Whisper right
    axes[(row, 7)].tick_params(left=False, labelleft=False)  # SwinV2 right

for i, (model, model_value) in enumerate(model_dict_sorted.items()):

    # if 'Llama 2 7B' in model:
    #     fig.text(0.1, 0.95, 'Llama 2', fontsize=6, ha='center', va='center')
    # elif 'Llama 3.1 8B' in model:
    #     fig.text(0.3, 0.95, 'Llama 3.1', fontsize=6, ha='center', va='center')
    # elif 'Whisper Tiny' in model:
    #     fig.text(0.5, 0.95, 'Whisper', fontsize=6, ha='center', va='center')
    # elif 'SwinV2 Tiny' in model:
    #     fig.text(0.7, 0.95, 'SwinV2', fontsize=6, ha='center', va='center')
    # elif 'ViViT B-16x2' in model:
    #     fig.text(0.9, 0.95, 'ViViT', fontsize=6, ha='center', va='center')

    if 'SwinV2 Large' in model or 'ViViT' in model:
        lines.append(plt.Line2D([0], [0], color='white'))
        labels.append(f'')

    lines.append(plt.Line2D([0], [0], color='white'))
    labels.append(f'{model}')

    for nonlinear_op, nonlinear_value in model_value.items():
        for pre_post, pre_post_value in nonlinear_value.items():
            if 'post' in pre_post:
                continue
            for distribution, distribution_value in pre_post_value.items():
                if 'exp' in distribution:
                    ax = axes[(2, i)] if nonlinear_op == 'softmax' else axes[(3, i)]
                    ax.tick_params(axis='both', labelsize=5, length=1.25, width=0.375, pad=0.2)

                    new_lines = exp_distribution(model, distribution_value, ax, nonlinear_op, s, linewidths, threshold, pad, window_size, color_map)

                    if nonlinear_op != 'softmax':
                        lines.extend(new_lines)
                        labels.extend([line.get_label() for line in new_lines])

                elif 'value' in distribution:
                    ax = axes[(0, i)] if nonlinear_op == 'softmax' else axes[(1, i)]
                    ax.tick_params(axis='both', labelsize=5, length=1.25, width=0.375, pad=0.2)

                    if nonlinear_op == 'softmax' or 'vivit' in model.lower() or 'swin' in model.lower() or 'whisper' in model.lower():
                        window = True
                    else:
                        window = False

                    value_distribution(model, distribution_value, ax, nonlinear_op, s, linewidths, threshold, pad, window_size, color_map, window)

                    if nonlinear_op == 'softmax':
                        ax.set_title(f'{model.split(' ')[-1]}', fontsize=6, pad=2)

                ax_text = 'SM' if nonlinear_op == 'softmax' else 'S' if 'llama' in model.lower() else 'G'

                if i % 2 != 1:
                    ax.text(
                        0.075, 0.925,           
                        ax_text,       
                        transform=ax.transAxes, 
                        fontsize=5,
                        verticalalignment='top',   
                        horizontalalignment='left' 
                    )

                ax.grid(True)
                ax.grid(which='major', axis='both', linestyle='--', linewidth=0.25, color='gray', alpha=0.7)

# fig.legend(lines, labels, loc='upper center', ncol=46, fontsize=4, frameon=True, handlelength=0.25, handleheight=0.5)

range_value = 10


for row in range(4):
    for col in range(9):  # Still 9 logical columns for our axes dict
        ax = axes[(row, col)]

        

        for k, line in enumerate(reversed(lines)):
            line.set_zorder(k+1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

        if row > 1:
            x_ticks = ax.get_xticks()
            x_ticks = get_min_list(x_ticks, 3)

            for i, tick in enumerate(x_ticks):
                if tick >= range_value:
                    x_ticks[i] = range_value
                elif tick <= -range_value:
                    x_ticks[i] = -range_value

            ax.set_xticks(x_ticks)
            ax.set_xticklabels([f'{int(tick)}' for tick in x_ticks], fontsize=5, rotation=0, ha='center', va='top')

        # Limit axis ranges to -12 to 12
        for spine in ax.spines.values():
            spine.set_linewidth(0.375)

set_ax_title(axes[(0, 0)], axes[(0, 1)], 'Llama 2')
set_ax_title(axes[(0, 2)], axes[(0, 3)], 'Llama 3.1')
set_ax_title(axes[(0, 4)], axes[(0, 5)], 'Whisper')
set_ax_title(axes[(0, 6)], axes[(0, 7)], 'SwinV2')
set_ax_title(axes[(0, 8)], axes[(0, 8)], 'ViViT')

value_pos_test = axes[(0, 0)].get_position().y1
value_pos1 = axes[(0, 0)].get_position().y0
value_pos2 = axes[(1, 0)].get_position().y1
value_pos = (value_pos1 + value_pos2) / 2

exp_pos_test = axes[(2, 0)].get_position().y1
exp_pos1 = axes[(2, 0)].get_position().y0
exp_pos2 = axes[(3, 0)].get_position().y1
exp_pos = (exp_pos1 + exp_pos2) / 2

fig.text(0.09, value_pos + 0.01, "Value Distribution", va='center', rotation='vertical', fontsize=6)
fig.text(0.09, exp_pos - 0.01, "Exp Distribution", va='center', rotation='vertical', fontsize=6)
# fig.text(0.05, value_pos_test + 0.01, "Value", va='center', fontsize=6)
# fig.text(0.05, exp_pos_test + 0.01, "Exp", va='center', fontsize=6)


# Set exactly 3 y-axis ticks and format to 1 decimal place

#plt.subplots_adjust(left=0.075, right=0.975, top=0.95, bottom=0.05)

if not os.path.exists('figures'):
    os.makedirs('figures')

plt.savefig('figures/distribution_figure.png', dpi=300, bbox_inches='tight')
plt.savefig('figures/distribution_figure.pdf', dpi=1200, bbox_inches='tight')
