import matplotlib.pyplot as plt
import matplotlib.colors as mc
import pandas as pd  # Import pandas library
import numpy as np
import sys
import os
from matplotlib.lines import Line2D
import math
import matplotlib.ticker as mticker  # Import ticker for formatting

def read_csv_file(file_path, x_min, x_max):
    try:
        df = pd.read_csv(file_path)
        df.dropna(subset=['x', 'y'], inplace=True)
        df = df[(df['x'] >= x_min) & (df['x'] <= x_max)]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)
    return df

def plot_curve(ax, file_path, x_min, x_max, color, label):

    df = read_csv_file(file_path, x_min, x_max).sort_values(by='x')
    ax.plot(df['x'], df['y'], color=color, linestyle='--', linewidth=0.5, label=label)
    ax.set_xticks([x_min, (x_min + x_max) / 2, x_max])
    # ax.set_xlabel('x', fontsize=6)
    # ax.set_ylabel('y', fontsize=6)
    ax.grid(True, linestyle='-', linewidth=0.5, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.tick_params(axis='both', which='major', labelsize=8, direction='in', length=2, width=0.5, pad=1)

def format_yaxis_as_percent(ax):
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0f}%'))

def main():

    fig_width_pt = 240             # Single-column width in points
    fig_width = fig_width_pt / 72    # Convert points to inches (~3.56 inches)
    # Height per subplot (adjust as needed)
    single_subplot_height = fig_width * 1.05
    # For a 3x3 grid, overall height is three times the single subplot height.
    overall_width = fig_width         # Overall width remains as single-column width
    overall_height = single_subplot_height


    exp_files = {
        'pwl': 'csv/nonlinear_error/exp_pwl.csv',
        'taylor': 'csv/nonlinear_error/exp_taylor.csv',
        'mugi': 'csv/nonlinear_error/exp_mugi.csv'
    }

    silu_files = {
        'pwl': 'csv/nonlinear_error/silu_pwl.csv',
        'mobilenet_pwl': 'csv/nonlinear_error/silu_mobilenet_pwl.csv',
        'mugi': 'csv/nonlinear_error/silu_mugi.csv'
    }

    gelu_files = {
        'pwl': 'csv/nonlinear_error/gelu_pwl.csv',
        'mugi': 'csv/nonlinear_error/gelu_mugi.csv'
    }

    # Model order for columns
    exp_models = ['pwl', 'taylor', 'mugi']
    silu_models = ['pwl', 'mobilenet_pwl', 'mugi']
    gelu_models = ['pwl', 'mugi']

    # Define colors for each model
    exp_colors = {'pwl': 'blue', 'taylor': 'blue', 'mugi': 'blue'}
    silu_colors = {'pwl': 'green', 'mobilenet_pwl': 'green', 'mugi': 'green'}
    gelu_colors = {'pwl': 'red', 'mugi': 'red'}

    # Create a 3x3 subplot grid
    fig, axs = plt.subplots(3, 3, figsize=(overall_width, overall_height))
    
    # Spacing control - adjust this value to make figures closer/farther apart
    wspace = 0.65  # Width spacing between subplots (lower = closer together)
    hspace = 0.5  # Height spacing between rows (lower = closer together)
    
    # GELU positioning parameter - adjust these values to shift GELU plots left/right
    # 0 = leftmost position, 1 = middle position, 2 = rightmost position
    # For centering two plots, use positions like 0.5 and 1.5 (halfway between columns)
    gelu_col_offset = 0.8  # Adjust this value to shift GELU plots horizontally
    

    # Plot Exp-group curves in the top row (x-range: [-20, 0])
    for i, model in enumerate(exp_models):
        ax = axs[0, i]
        x_min = -8 if model == 'taylor' else -16
        plot_curve(ax, exp_files[model], x_min=x_min, x_max=0,
                   color=exp_colors[model], label=f'Exp {model.upper()}')
        format_yaxis_as_percent(ax)  # Format y-axis as percent
        if model == 'taylor':
            ax.set_title("Exp Taylor", fontsize=9, pad = 0.5)
        elif model == 'mugi':
            ax.set_title("Exp Mugi", fontsize=9, pad = 0.5)
        else:
            ax.set_title(f"Exp {model.upper()}", fontsize=9, pad = 0.5)
        # Add inset axes for zooming into x-range [-1, 0]
        if model == 'pwl':
            axins = ax.inset_axes([0.45, 0.25, 0.4, 0.4])
        elif model == 'taylor':
            axins = ax.inset_axes([0.5, 0.2, 0.4, 0.4])
        else:
            axins = ax.inset_axes([0.75, 0.4, 0.4, 0.4])

        df_zoom = read_csv_file(exp_files[model], x_min=-0.5, x_max=0).sort_values(by='x')

        y_min = round(df_zoom['y'].min())
        y_max = round(df_zoom['y'].max())
        
        # Special handling for Taylor model to show 1% as top tick but use 1.4% as actual max
        if model == 'taylor':
            y_max_display = 1.4  # Actual max value for positioning
            y_max_tick = 1       # Value to display as top tick
        else:
            y_max_display = y_max
            y_max_tick = y_max

        axins.plot(df_zoom['x'], df_zoom['y'], color=exp_colors[model],
                   linestyle='--', linewidth=0.5)
        axins.set_xlim(-0.5, 0)
        # Optionally, remove tick labels on the inset
        axins.set_xticklabels([-0.5, 0])
        if model in ['mugi']:
          axins.xaxis.tick_top()

        axins.set_ylim(y_min, y_max_display)
        if model == 'taylor':
            # Put the "1%" tick at the 1.5% position
            axins.set_yticks([y_min, 0, y_max_display])  # Use actual positions
            axins.set_yticklabels([f'{y_min:.0f}%', '0%', '1%'])  # But label the top as 1%
        else:
            axins.set_yticks([y_min, 0, y_max_tick])

        axins.tick_params(axis='x', pad=0.5, labelsize=7, length=2, width=0.5)

        #axins.set_yticklabels([])
        axins.tick_params(axis='y', pad=2, labelsize=7, length = 2, width=0.5)
        if model in ['pwl', 'taylor']:
            axins.yaxis.tick_left()
            axins.yaxis.set_label_position('left')
        else:
            axins.yaxis.tick_right()
            axins.yaxis.set_label_position('right')
        
        axins.spines['top'].set_linewidth(0.5)
        axins.spines['right'].set_linewidth(0.5)
        axins.spines['left'].set_linewidth(0.5)
        axins.spines['bottom'].set_linewidth(0.5)

        format_yaxis_as_percent(axins)  # Format inset y-axis as percent
        ax.indicate_inset_zoom(axins, edgecolor="grey")

    # Plot Silu-group curves in the bottom row (x-range: [-5, 5])
    for i, model in enumerate(silu_models):
        ax = axs[1, i]
        plot_curve(ax, silu_files[model], x_min=-5, x_max=5,
                   color=silu_colors[model], label=f'Silu {model.upper()}')
        format_yaxis_as_percent(ax)  # Format y-axis as percent
        if model == 'mobilenet_pwl':
            ax.set_title("SiLU PA", fontsize=9, pad = 0.5)
        elif model == 'pwl':
            ax.set_title("SiLU PWL", fontsize=9, pad = 0.5)
        elif model == 'mugi':
            ax.set_title("SiLU Mugi", fontsize=9, pad = 0.5)
        else:
            ax.set_title(f"SiLU {model}", fontsize=9, pad = 0.5)

        if model == 'pwl':
            axins = ax.inset_axes([0.6, 0.225, 0.4, 0.4])
        elif model == 'mobilenet_pwl':
            axins = ax.inset_axes([0.72, 0.18, 0.4, 0.4])
        else:
            axins = ax.inset_axes([0.5, 0.3, 0.4, 0.4])
        #ax.set_ylim(-100, 100)
        # Cap x-axis for main axes (again, if needed)
        ax.set_xlim(-5, 5)

        # Add inset axes for zooming into x-range [-1, 0]
        x_min = -0.5
        x_max = 0.5

        df_zoom = read_csv_file(silu_files[model], x_min=x_min, x_max=x_max).sort_values(by='x')
        y_min = round(df_zoom['y'].min())
        y_max = round(df_zoom['y'].max())

        axins.plot(df_zoom['x'], df_zoom['y'], color=silu_colors[model],
                   linestyle='--', linewidth=0.5)
        axins.set_xlim(x_min, x_max)
        # Explicitly set ticks and labels to avoid misalignment
        axins.set_xticks([x_min, x_max])
        axins.set_xticklabels([x_min, x_max])
        if model in ['mugi']:
            axins.xaxis.tick_top()
        axins.tick_params(axis='x', pad=0.5, labelsize=7, length=2, width=0.5)

        #axins.set_yticklabels([])
        axins.set_ylim(y_min, y_max)
        axins.set_yticks([y_min, 0,  y_max])
        axins.tick_params(axis='y', labelsize=7, length = 2, width=0.5)
        if model in ['pwl', 'mobilenet_pwl']:
            axins.yaxis.tick_left()
            axins.yaxis.set_label_position('left')
        else:
            axins.yaxis.tick_right()
            axins.yaxis.set_label_position('right')

        axins.spines['top'].set_linewidth(0.5)
        axins.spines['right'].set_linewidth(0.5)
        axins.spines['left'].set_linewidth(0.5)
        axins.spines['bottom'].set_linewidth(0.5)

        format_yaxis_as_percent(axins)  # Format inset y-axis as percent
        ax.indicate_inset_zoom(axins, edgecolor="grey")

    # Plot GELU-group curves in the third row (x-range: [-5, 5])
    # Create manually positioned subplots for GELU models, completely independent of the grid
    
    # Get actual dimensions from existing subplots to ensure exact matching
    # Use the first subplot from the top row to get actual width and height
    bbox = axs[0, 0].get_position()
    actual_subplot_width = bbox.width
    actual_subplot_height = bbox.height
    
    # Calculate y position for third row to match the grid spacing exactly
    # Use the same vertical spacing as between first and second rows
    first_row_bottom = axs[0, 0].get_position().y0
    second_row_bottom = axs[1, 0].get_position().y0
    row_spacing = first_row_bottom - second_row_bottom
    
    # Position GELU figures with the same spacing as the grid and shift down
    y_pos = second_row_bottom - row_spacing - 0.055  # Shift down by 0.055 (moved up by 0.01)
    
    # Calculate x positions for centered symmetrical placement with more space between figures
    # Get the total figure width used by the 3 columns
    total_subplot_area_width = 3 * actual_subplot_width + 2 * (axs[0, 1].get_position().x0 - (axs[0, 0].get_position().x0 + actual_subplot_width))
    
    # Calculate spacing to center 2 subplots within the area of 3 subplots with adjustable gap
    space_for_two_plots = 2 * actual_subplot_width
    remaining_space = total_subplot_area_width - space_for_two_plots
    
    # Use gelu_col_offset to control the gap between GELU plots
    # gelu_col_offset of 0.5 = normal spacing, lower = closer together, higher = farther apart
    center_gap = remaining_space * gelu_col_offset
    side_margin = (remaining_space - center_gap) / 2
    
    x_left = axs[0, 0].get_position().x0 + side_margin
    x_right = x_left + actual_subplot_width + center_gap
    
    # Create the two GELU subplots with manual positioning and increased height
    # Make GELU subplot height larger to match visual appearance of grid subplots
    gelu_height = actual_subplot_height * 0.85  # Increase height by 20%
    ax_gelu_left = fig.add_axes([x_left, y_pos, actual_subplot_width, gelu_height])
    ax_gelu_right = fig.add_axes([x_right, y_pos, actual_subplot_width, gelu_height])
    
    gelu_axes = [ax_gelu_left, ax_gelu_right]
    
    for i, model in enumerate(gelu_models):
        ax = gelu_axes[i]
        plot_curve(ax, gelu_files[model], x_min=-5, x_max=5,
                   color=gelu_colors[model], label=f'GELU {model.upper()}')
        format_yaxis_as_percent(ax)  # Format y-axis as percent
        if model == 'pwl':
            ax.set_title("GELU PWL", fontsize=9, pad = 0.5)
        elif model == 'mugi':
            ax.set_title("GELU Mugi", fontsize=9, pad = 0.5)
        else:
            ax.set_title(f"GELU {model}", fontsize=9, pad = 0.5)

        if model == 'pwl':
            axins = ax.inset_axes([0.6, 0.175, 0.4, 0.4])
        else:
            axins = ax.inset_axes([0.5, 0.45, 0.4, 0.4])
        
        # Cap x-axis for main axes (again, if needed)
        ax.set_xlim(-5, 5)

        # Add inset axes for zooming into x-range [-0.5, 0.5]
        x_min = -0.5
        x_max = 0.5

        df_zoom = read_csv_file(gelu_files[model], x_min=x_min, x_max=x_max).sort_values(by='x')
        y_min = round(df_zoom['y'].min())
        y_max = round(df_zoom['y'].max())

        axins.plot(df_zoom['x'], df_zoom['y'], color=gelu_colors[model],
                   linestyle='--', linewidth=0.5)
        axins.set_xlim(x_min, x_max)
        # Explicitly set ticks and labels to avoid misalignment
        axins.set_xticks([x_min, x_max])
        axins.set_xticklabels([x_min, x_max])
        if model in ['mugi']:
            axins.xaxis.tick_top()
        axins.tick_params(axis='x', pad=0.5, labelsize=7, length=2, width=0.5)

        axins.set_ylim(y_min, y_max)
        axins.set_yticks([y_min, 0, y_max])
        axins.tick_params(axis='y', labelsize=7, length=2, width=0.5)
        if model in ['pwl']:
            axins.yaxis.tick_right()
            axins.yaxis.set_label_position('right')
        else:
            axins.yaxis.tick_right()
            axins.yaxis.set_label_position('right')

        axins.spines['top'].set_linewidth(0.5)
        axins.spines['right'].set_linewidth(0.5)
        axins.spines['left'].set_linewidth(0.5)
        axins.spines['bottom'].set_linewidth(0.5)

        format_yaxis_as_percent(axins)  # Format inset y-axis as percent
        ax.indicate_inset_zoom(axins, edgecolor="grey")

    # Hide the original third row subplots since we're using manually positioned ones
    for j in range(3):
        axs[2, j].axis('off')

    # Add a legend to the top of the figure
    handles = [Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='Exp Models'),
               Line2D([0], [0], color='green', linestyle='--', linewidth=1, label='SiLU Models'),
               Line2D([0], [0], color='red', linestyle='--', linewidth=1, label='GELU Models')]


    # Reduce whitespace: adjust padding between subplots.
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=.3)
    plt.subplots_adjust(top=0.84, bottom=0.05, left=0.05, right=0.95, wspace=wspace, hspace=hspace)

    plt.savefig("figures/output/nonlinear_error.pdf", dpi=1200, bbox_inches="tight")
    plt.savefig("figures/output/nonlinear_error.png", dpi=1200, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    main()