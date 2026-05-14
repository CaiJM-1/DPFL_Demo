"""


"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from config import Config
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_accuracy_points(x_values, y_values, title='Accuracy Over Time', xlabel='Rounds', ylabel='Accuracy (%)', label=None, color='blue', marker='o', markersize=4, linewidth=1.5, linestyle='-', figsize=(8, 6), dpi=600, grid=False, save_path=None, show_values=False, xticks=None, equal_spacing=False):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if equal_spacing:
        x_positions = range(len(x_values))
        ax.plot(x_positions, y_values, color=color, marker=marker, markersize=markersize, linewidth=linewidth, linestyle=linestyle, label=label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_values)
        if show_values:
            for x_pos, y in zip(x_positions, y_values):
                ax.text(x_pos, y, f'{y:.2f}', fontsize=14, ha='center', va='bottom')
    else:
        ax.plot(x_values, y_values, color=color, marker=marker, markersize=markersize, linewidth=linewidth, linestyle=linestyle, label=label)
        if show_values:
            for x, y in zip(x_values, y_values):
                ax.text(x, y, f'{y:.2f}', fontsize=14, ha='center', va='bottom')
        if xticks is not None:
            ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    if label:
        ax.legend(loc='best', fontsize=16, framealpha=0.9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return (fig, ax)

def plot_multiple_accuracy_curves(histories, title=None, xlabel='Rounds', ylabel='Accuracy', figsize=(8, 6), dpi=600, grid=False, save_path=None, legend_loc='lower right', show_marker=False, x_values=None, linestyle=None, accuracy_scale='percent', ylim=None, mark_interval=None, smoothing=None, smoothing_param=0.4):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if linestyle is None:
        line_styles = ['-', '--', '-.', ':']
    elif isinstance(linestyle, str):
        line_styles = [linestyle] * 10
    else:
        line_styles = linestyle
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '*']
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = default_colors * 2
    auto_detected_scale = accuracy_scale
    if accuracy_scale == 'percent':
        first_history = next(iter(histories.values()))
        if isinstance(first_history, dict) and 'global_accuracy' in first_history:
            max_val = max(first_history['global_accuracy'])
            if max_val <= 1.0:
                auto_detected_scale = 'decimal'
    for i, (strategy, history) in enumerate(histories.items()):
        if isinstance(history, dict) and 'global_accuracy' in history:
            if 'rounds' in history and history['rounds'] is not None:
                rounds = history['rounds']
            elif x_values is not None:
                rounds = x_values
            else:
                rounds = range(1, len(history['global_accuracy']) + 1)
            accuracy_data = history['global_accuracy']
            if auto_detected_scale == 'decimal':
                accuracy_data = [acc * 100 for acc in accuracy_data]
            style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)] if style == '-' and show_marker else None
            if smoothing is not None:
                ax.plot(rounds, accuracy_data, linestyle=style, color=color, alpha=0.2, linewidth=1.0, zorder=1)
                if smoothing.lower() == 'ema':
                    alpha = smoothing_param
                    smoothed_data = []
                    ema = accuracy_data[0]
                    for val in accuracy_data:
                        ema = alpha * val + (1 - alpha) * ema
                        smoothed_data.append(ema)
                elif smoothing.lower() == 'ma':
                    window = int(smoothing_param) if smoothing_param > 1 else 5
                    smoothed_data = []
                    for j in range(len(accuracy_data)):
                        start = max(0, j - window // 2)
                        end = min(len(accuracy_data), j + window // 2 + 1)
                        smoothed_data.append(sum(accuracy_data[start:end]) / (end - start))
                else:
                    smoothed_data = accuracy_data
                ax.plot(rounds, smoothed_data, label=strategy, linestyle=style, marker=marker, color=color, markersize=0 if not show_marker else 4, linewidth=1.5, zorder=2)
                accuracy_data = smoothed_data
            else:
                ax.plot(rounds, accuracy_data, label=strategy, linestyle=style, marker=marker, color=color, markersize=0 if not show_marker else 4, linewidth=1.5)
            if mark_interval is not None:
                marked_rounds = []
                marked_accuracy = []
                for round_num, acc_val in zip(rounds, accuracy_data):
                    if round_num % mark_interval == 0:
                        marked_rounds.append(round_num)
                        marked_accuracy.append(acc_val)
                if marked_rounds:
                    ax.scatter(marked_rounds, marked_accuracy, color=color, s=50, marker='s', alpha=0.8, zorder=4, edgecolors='white', linewidth=0.5)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    if ylim is not None:
        ax.set_ylim(ylim)
    elif auto_detected_scale == 'decimal':
        ax.set_ylim(0, 100)
    ax.set_xlim(left=0)
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16, loc=legend_loc, framealpha=0.9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return (fig, ax)

def plot_dual_axis(x_values, y1_values, y2_values, y1_label='Left Axis', y2_label='Right Axis', xlabel='X Axis', title=None, y1_color='#1f77b4', y2_color='#ff7f0e', y2_max_color='#7f7f7f', y2_min_color='#c7c7c7', fill_alpha=0.1, range_line_alpha=1.0, y1_marker='o', y2_marker='s', y1_linestyle='-', y2_linestyle='-', markersize=6, linewidth=1.2, figsize=(8, 6), dpi=600, grid=False, legend_loc='lower right', y1_lim=None, y2_lim=None, equal_spacing=False, save_path=None, y2_max_values=None, y2_min_values=None, show_range=True):
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    from matplotlib.colors import to_rgba, to_rgb
    if y2_max_color is None:
        y2_max_color = '#9DB6F2'
    if y2_min_color is None:
        y2_min_color = '#F4C7A1'
    fill_rgb = tuple(((a + b) / 2 for a, b in zip(to_rgb(y2_max_color), to_rgb(y2_min_color))))
    fill_color = (*fill_rgb, fill_alpha)
    max_line_color = to_rgba(y2_max_color, range_line_alpha)
    min_line_color = to_rgba(y2_min_color, range_line_alpha)
    if equal_spacing:
        x_positions = range(len(x_values))
        line1 = ax1.plot(x_positions, y1_values, color=y1_color, marker=y1_marker, markersize=markersize, linewidth=linewidth, linestyle=y1_linestyle, label='Test Accuracy')
        ax2 = ax1.twinx()
        range_lines = []
        if show_range and y2_max_values is not None and (y2_min_values is not None):
            ax2.fill_between(x_positions, y2_min_values, y2_max_values, color=fill_color, zorder=1)
            max_line = ax2.plot(x_positions, y2_max_values, color=max_line_color, linewidth=1.0, linestyle='--', label=f'Max reuse rate', zorder=2)
            min_line = ax2.plot(x_positions, y2_min_values, color=min_line_color, linewidth=1.0, linestyle='--', label=f'Min reuse rate', zorder=2)
            range_lines += max_line + min_line
        line2 = ax2.plot(x_positions, y2_values, color=y2_color, marker=y2_marker, markersize=markersize, linewidth=linewidth, linestyle=y2_linestyle, label='Average reuse rate', zorder=3)
        ax1.set_xticks(x_positions)
        ax1.set_xticklabels(x_values)
    else:
        range_lines = []
        line1 = ax1.plot(x_values, y1_values, color=y1_color, marker=y1_marker, markersize=markersize, linewidth=linewidth, linestyle=y1_linestyle, label=y1_label)
        ax2 = ax1.twinx()
        if show_range and y2_max_values is not None and (y2_min_values is not None):
            ax2.fill_between(x_values, y2_min_values, y2_max_values, color=fill_color, zorder=1)
            max_line = ax2.plot(x_values, y2_max_values, color=y2_max_color, linewidth=linewidth, linestyle='--', label=f'{y2_label} (Max)', zorder=2)
            min_line = ax2.plot(x_values, y2_min_values, color=y2_min_color, linewidth=linewidth, linestyle=':', label=f'{y2_label} (Min)', zorder=2)
            range_lines += max_line + min_line
        line2 = ax2.plot(x_values, y2_values, color=y2_color, marker=y2_marker, markersize=markersize, linewidth=linewidth, linestyle=y2_linestyle, label=y2_label, zorder=3)
    ax1.set_xlabel(xlabel, fontsize=18, color='black')
    ax1.set_ylabel(y1_label, fontsize=18, color='black')
    ax1.tick_params(axis='both', labelcolor='black', labelsize=16)
    ax2.set_ylabel(y2_label, fontsize=18, color='black')
    ax2.tick_params(axis='y', labelcolor='black', labelsize=16)
    if y1_lim is not None:
        ax1.set_ylim(y1_lim)
    if y2_lim is not None:
        ax2.set_ylim(y2_lim)
    if grid:
        ax1.grid(True, alpha=0.3, linestyle='--')
    lines = line1 + range_lines + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc=legend_loc, fontsize=16, framealpha=0.9)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return (fig, ax1, ax2)

def plot_reuse_accuracy_scatter(reuse_rates, accuracies, k_values, tau_values, figsize=(8, 6), dpi=600, xlabel='Average Reuse Rate (%)', ylabel='Final Test Accuracy (%)', save_path=None):
    assert len(reuse_rates) == len(accuracies) == len(k_values) == len(tau_values), 'Input lengths must match'
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    unique_k = sorted(list(set(k_values)))
    unique_tau = sorted(list(set(tau_values)))
    palette = plt.get_cmap('tab10')
    color_map = {k: palette(i % 10) for i, k in enumerate(unique_k)}
    marker_cycle = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    marker_map = {tau: marker_cycle[i % len(marker_cycle)] for i, tau in enumerate(unique_tau)}
    for x, y, k, tau in zip(reuse_rates, accuracies, k_values, tau_values):
        ax.scatter(x, y, color=color_map[k], marker=marker_map[tau], s=100, edgecolor='black', linewidths=0.8, alpha=0.85, zorder=3)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    handles_k = [plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=color_map[k], markeredgecolor='black', markersize=10, linestyle='') for k in unique_k]
    labels_k = [f'K={k}' for k in unique_k]
    handles_tau = [plt.Line2D([0], [0], marker=marker_map[tau], color='black', linestyle='', markerfacecolor='white', markersize=10) for tau in unique_tau]
    labels_tau = [f'tau={tau}' for tau in unique_tau]
    legend1 = ax.legend(handles_k, labels_k, title='K (Max Modes)', loc='upper left', fontsize=14, title_fontsize=14, frameon=True, framealpha=0.9)
    legend2 = ax.legend(handles_tau, labels_tau, title='Tau (Threshold)', loc='lower right', fontsize=14, title_fontsize=14, frameon=True, framealpha=0.9)
    ax.add_artist(legend1)
    fig.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return (fig, ax)
    return (fig, ax)

def plot_method_comparison_dual_axis(our_method_data, comparison_methods, method_labels, figsize=(8, 6), dpi=600, y1_label='Accuracy (%)', y2_label='Average reuse rate (%)', y1_lim=None, y2_lim=None, save_path=None):
    n_methods = len(comparison_methods)
    assert len(method_labels) == n_methods, 'method_labels length must match comparison_methods'
    fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
    n_points = n_methods + 1
    x_positions = list(range(n_points))
    x_labels = ['Ours'] + method_labels
    accuracy_values = [our_method_data['accuracy']] + [m['accuracy'] for m in comparison_methods]
    reuse_avg_values = [our_method_data['reuse']] + [m['reuse'] for m in comparison_methods]
    reuse_max_values = [our_method_data.get('reuse_max', our_method_data['reuse'])] + [m.get('reuse_max', m['reuse']) for m in comparison_methods]
    reuse_min_values = [our_method_data.get('reuse_min', our_method_data['reuse'])] + [m.get('reuse_min', m['reuse']) for m in comparison_methods]
    from matplotlib.colors import to_rgba, to_rgb
    acc_color = '#1f77b4'
    reuse_avg_color = '#ff7f0e'
    reuse_max_color = '#7f7f7f'
    reuse_min_color = '#c7c7c7'
    fill_rgb = tuple(((a + b) / 2 for a, b in zip(to_rgb(reuse_max_color), to_rgb(reuse_min_color))))
    fill_color = (*fill_rgb, 0.1)
    max_line_color = to_rgba(reuse_max_color, 1.0)
    min_line_color = to_rgba(reuse_min_color, 1.0)
    ax1.plot(x_positions, accuracy_values, color=acc_color, marker='o', markersize=6, linewidth=1.2, linestyle='-', label='Test Accuracy', zorder=3)
    ax2 = ax1.twinx()
    ax2.fill_between(x_positions, reuse_min_values, reuse_max_values, color=fill_color, zorder=1)
    ax2.plot(x_positions, reuse_max_values, color=max_line_color, linewidth=1.0, linestyle='--', label='Max reuse rate', zorder=2)
    ax2.plot(x_positions, reuse_min_values, color=min_line_color, linewidth=1.0, linestyle='--', label='Min reuse rate', zorder=2)
    ax2.plot(x_positions, reuse_avg_values, color=reuse_avg_color, marker='s', markersize=6, linewidth=1.2, linestyle='-', label='Average reuse rate', zorder=3)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(x_labels, fontsize=16)
    ax1.set_ylabel(y1_label, fontsize=18, color='black')
    ax2.set_ylabel(y2_label, fontsize=18, color='black')
    ax1.tick_params(axis='both', labelsize=16, labelcolor='black')
    ax2.tick_params(axis='y', labelsize=16, labelcolor='black')
    if y1_lim:
        ax1.set_ylim(y1_lim)
    if y2_lim:
        ax2.set_ylim(y2_lim)
    ax1.grid(False)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=14, framealpha=0.9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return (fig, ax1, ax2)

def plot_multi_seed_datasets_grid(log_files_by_dataset_alpha, smoothing='ema', smoothing_param=0.6, figsize=(18, 9), dpi=600, xlabel='Communication Rounds', ylabel='Accuracy (%)', max_rounds=None, colors=None, y_lim=None, save_path=None):
    import re
    import numpy as np
    if not log_files_by_dataset_alpha:
        return None
    dataset_names = list(log_files_by_dataset_alpha.keys())
    first_dataset = dataset_names[0]
    first_value = log_files_by_dataset_alpha[first_dataset]
    if isinstance(first_value, list):
        log_files_by_dataset_alpha = {'Dataset': log_files_by_dataset_alpha}
        dataset_names = ['Dataset']
        first_dataset = 'Dataset'
    alpha_labels = list(log_files_by_dataset_alpha[first_dataset].keys())
    n_datasets = len(dataset_names)
    n_alphas = len(alpha_labels)
    fig, axes = plt.subplots(n_datasets, n_alphas, figsize=figsize, dpi=dpi)
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    elif n_alphas == 1:
        axes = axes.reshape(-1, 1)
    all_strategies_set = set()
    for dataset_dict in log_files_by_dataset_alpha.values():
        for log_files in dataset_dict.values():
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            match = re.match('([^:]+):\\s*Round', line)
                            if match:
                                all_strategies_set.add(match.group(1).strip())
                except FileNotFoundError:
                    continue
    legend_order = ['Fed-CDP', 'Fed-dynS', 'Fed-AC-DP', 'Addp-DP-FL', 'MoG-AdDP', 'TUNE-DPFL']
    all_strategies = [s for s in legend_order if s in all_strategies_set]
    for s in sorted(all_strategies_set):
        if s not in all_strategies:
            all_strategies.append(s)
    if colors is None:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {}
        for i, strategy in enumerate(all_strategies):
            colors[strategy] = default_colors[i % len(default_colors)]

    def apply_smoothing(data, method, param):
        if method is None or method.lower() == 'none':
            return data
        if method.lower() == 'ema':
            smoothed = [data[0]]
            for val in data[1:]:
                smoothed.append(param * smoothed[-1] + (1 - param) * val)
            return np.array(smoothed)
        if method.lower() == 'moving_avg':
            window = int(param)
            smoothed = []
            for i in range(len(data)):
                start = max(0, i - window // 2)
                end = min(len(data), i + window // 2 + 1)
                smoothed.append(np.mean(data[start:end]))
            return np.array(smoothed)
        return data

    def get_seed_id(log_file):
        basename = os.path.basename(log_file).replace('.log', '')
        seed_match = re.match('^(\\d+)', basename)
        return seed_match.group(1) if seed_match else basename
    for i, dataset_name in enumerate(dataset_names):
        dataset_dict = log_files_by_dataset_alpha.get(dataset_name, {})
        for j, alpha_label in enumerate(alpha_labels):
            ax = axes[i, j]
            log_files = dataset_dict.get(alpha_label, [])
            if not log_files:
                ax.set_axis_off()
                continue
            seeds_data = {}
            for log_file in log_files:
                seed_id = get_seed_id(log_file)
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            match = re.match('([^:]+):\\s*Round\\s+(\\d+):\\s*.*roundAccuracy\\s*=\\s*([\\d.]+)\\s*%round', line)
                            if match:
                                strategy = match.group(1).strip()
                                round_num = int(match.group(2))
                                accuracy = float(match.group(3))
                                if strategy not in seeds_data:
                                    seeds_data[strategy] = {}
                                if seed_id not in seeds_data[strategy]:
                                    seeds_data[strategy][seed_id] = {}
                                if max_rounds is None or round_num <= max_rounds:
                                    seeds_data[strategy][seed_id][round_num] = accuracy
                except FileNotFoundError:
                    continue
            subplot_max_round = 0
            for strategy in all_strategies:
                if strategy not in seeds_data:
                    continue
                all_rounds = set()
                for seed_dict in seeds_data[strategy].values():
                    for round_num in seed_dict.keys():
                        if max_rounds is None or round_num <= max_rounds:
                            all_rounds.add(round_num)
                if not all_rounds:
                    continue
                all_rounds = sorted(all_rounds)
                subplot_max_round = max(subplot_max_round, all_rounds[-1])
                seeds_accuracy = []
                for seed_dict in seeds_data[strategy].values():
                    if not seed_dict:
                        continue
                    seed_rounds = sorted(seed_dict.keys())
                    first_val = seed_dict[seed_rounds[0]]
                    last_val = None
                    acc_list = []
                    for r in all_rounds:
                        if r in seed_dict:
                            last_val = seed_dict[r]
                            acc_list.append(last_val)
                        elif last_val is not None:
                            acc_list.append(last_val)
                        else:
                            acc_list.append(first_val)
                    seeds_accuracy.append(acc_list)
                if not seeds_accuracy:
                    continue
                seeds_accuracy = np.array(seeds_accuracy)
                mean_acc = np.mean(seeds_accuracy, axis=0)
                std_acc = np.std(seeds_accuracy, axis=0)
                mean_acc_smoothed = apply_smoothing(mean_acc, smoothing, smoothing_param)
                std_acc_smoothed = apply_smoothing(std_acc, smoothing, smoothing_param)
                ax.fill_between(all_rounds, mean_acc_smoothed - std_acc_smoothed, mean_acc_smoothed + std_acc_smoothed, color=colors.get(strategy, 'C0'), alpha=0.2, zorder=1)
                ax.plot(all_rounds, mean_acc_smoothed, color=colors.get(strategy, 'C0'), linewidth=2.0, alpha=0.9, zorder=3)
            ax.set_title(f'{dataset_name} {alpha_label}', fontsize=13, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.tick_params(axis='both', labelsize=10)
            ax.grid(False)
            if subplot_max_round > 0:
                if subplot_max_round <= 100:
                    xtick_interval = 20
                elif subplot_max_round <= 250:
                    xtick_interval = 50
                elif subplot_max_round <= 500:
                    xtick_interval = 100
                elif subplot_max_round <= 1000:
                    xtick_interval = 200
                else:
                    xtick_interval = 500
                ax.set_xticks(list(range(xtick_interval, subplot_max_round + 1, xtick_interval)))
                ax.set_xlim(0, subplot_max_round)
            if y_lim is not None:
                ax.set_ylim(y_lim)
    if all_strategies:
        handles = []
        labels = []
        for strategy in all_strategies:
            handles.append(plt.Line2D([0], [0], color=colors.get(strategy, 'C0'), linewidth=2.0))
            labels.append(strategy)
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels), fontsize=14, frameon=False, borderpad=1.0, labelspacing=0.6, handlelength=2.4, handletextpad=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig

def plot_label_distribution_stacked(data_dict, figsize=(18, 6), dpi=600, xlabel='Client ID', ylabel='Number of Samples', save_path=None):
    import matplotlib.patches as mpatches
    n_settings = len(data_dict)
    setting_names = list(data_dict.keys())
    fig, axes = plt.subplots(1, n_settings, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes[0]
    colors_classes = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, setting_name in enumerate(setting_names):
        ax = axes[idx]
        client_data = data_dict[setting_name]
        clients = sorted(client_data.keys())
        n_clients = len(clients)
        n_classes = 10
        bottom_values = np.zeros(n_clients)
        for class_id in range(n_classes):
            class_counts = []
            for client_id in clients:
                if client_id < len(client_data) and class_id < len(client_data[client_id]):
                    class_counts.append(client_data[client_id][class_id])
                else:
                    class_counts.append(0)
            ax.bar(np.arange(n_clients), class_counts, bottom=bottom_values, label=f'Class {class_id}', color=colors_classes[class_id], edgecolor='white', linewidth=0.5)
            bottom_values += np.array(class_counts)
        if '=' in setting_name:
            title_text = setting_name.replace('=', ' = ')
        else:
            title_text = setting_name
        ax.set_title(title_text, fontsize=18, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.tick_params(axis='both', labelsize=16)
        ax.grid(False)
        ax.set_xticks(np.arange(n_clients))
        ax.set_xticklabels([str(i + 1) for i in range(n_clients)])
        ax.set_xlim(-0.5, n_clients - 0.5)
    handles = []
    labels = []
    for class_id in range(10):
        handles.append(mpatches.Patch(color=colors_classes[class_id], label=f'Class {class_id}'))
        labels.append(f'Class {class_id}')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=10, fontsize=14, frameon=False, borderpad=1.2, labelspacing=0.8, handlelength=1.8, handletextpad=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig

def plot_label_distribution_heatmap(data_dict, figsize=(20, 4.5), dpi=600, cmap='YlGnBu', save_path=None):
    n_settings = len(data_dict)
    setting_names = list(data_dict.keys())
    fig, axes = plt.subplots(1, n_settings, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes[0]
    all_values = []
    for setting_data in data_dict.values():
        for client_data in setting_data.values():
            all_values.extend(client_data)
    global_max = max(all_values)
    vmax_adjusted = global_max
    for idx, setting_name in enumerate(setting_names):
        ax = axes[idx]
        client_data = data_dict[setting_name]
        clients = sorted(client_data.keys())
        n_classes = 10
        matrix = np.zeros((n_classes, len(clients)))
        for col_idx, client_id in enumerate(clients):
            for class_id in range(n_classes):
                if class_id < len(client_data[client_id]):
                    matrix[class_id, col_idx] = client_data[client_id][class_id]
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=vmax_adjusted)
        if '=' in setting_name:
            title_text = setting_name.replace('=', ' = ')
        else:
            title_text = setting_name
        ax.set_title(title_text, fontsize=18, fontweight='bold')
        ax.set_xlabel('Client ID', fontsize=18)
        ax.set_ylabel('Class ID', fontsize=18)
        ax.set_xticks(np.arange(len(clients)))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels([str(i + 1) for i in clients], fontsize=16)
        ax.set_yticklabels([str(i) for i in range(n_classes)], fontsize=16)
        ax.set_xticks(np.arange(len(clients)) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_classes) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
        ax.tick_params(which='minor', size=0)
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.08, aspect=40, shrink=0.8, location='top')
    cbar.set_label('Number of Samples', fontsize=18, labelpad=8)
    cbar.ax.tick_params(labelsize=14, top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.tight_layout(rect=[0, 0, 1, 0.7])
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig

def plot_class_cooccurrence_heatmap(data_dict, figsize=(16, 4.8), dpi=600, cmap='YlOrRd', save_path=None):
    n_settings = len(data_dict)
    setting_names = list(data_dict.keys())
    fig, axes = plt.subplots(1, n_settings, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes[0]
    for idx, setting_name in enumerate(setting_names):
        ax = axes[idx]
        client_assignment = data_dict[setting_name]
        n_classes = 10
        cooccurrence_matrix = np.zeros((n_classes, n_classes))
        for client_id, class_list in client_assignment.items():
            for i in range(len(class_list)):
                for j in range(len(class_list)):
                    class_i = class_list[i]
                    class_j = class_list[j]
                    cooccurrence_matrix[class_i, class_j] += 1
        vmax = np.max(cooccurrence_matrix) if np.max(cooccurrence_matrix) > 0 else 1
        im = ax.imshow(cooccurrence_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=vmax)
        ax.set_title(setting_name, fontsize=18, fontweight='bold')
        ax.set_xlabel('Class ID', fontsize=18)
        ax.set_ylabel('Class ID', fontsize=18)
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels([str(i) for i in range(n_classes)], fontsize=16)
        ax.set_yticklabels([str(i) for i in range(n_classes)], fontsize=16)
        ax.set_xticks(np.arange(n_classes) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_classes) - 0.5, minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=1.5)
        ax.tick_params(which='minor', size=0)
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.08, aspect=40, shrink=0.8, location='top')
    cbar.set_label('Co-occurrence Frequency', fontsize=18, labelpad=8)
    cbar.ax.tick_params(labelsize=14, top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.tight_layout(rect=[0, 0, 1, 0.7])
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig

def plot_class_assignment_heatmap(data_dict, figsize=(16, 4.8), dpi=600, cmap='Blues', save_path=None):
    n_settings = len(data_dict)
    setting_names = list(data_dict.keys())
    fig, axes = plt.subplots(1, n_settings, figsize=figsize, dpi=dpi, squeeze=False)
    axes = axes[0]
    for idx, setting_name in enumerate(setting_names):
        ax = axes[idx]
        client_assignment = data_dict[setting_name]
        clients = sorted(client_assignment.keys())
        n_classes = 10
        matrix = np.zeros((n_classes, len(clients)))
        for col_idx, client_id in enumerate(clients):
            for class_id in client_assignment[client_id]:
                matrix[class_id, col_idx] = 1
        im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1, interpolation='nearest')
        ax.set_title(setting_name, fontsize=18, fontweight='bold')
        ax.set_xlabel('Client ID', fontsize=18)
        ax.set_ylabel('Class ID', fontsize=18)
        ax.set_xticks(np.arange(len(clients)))
        ax.set_yticks(np.arange(n_classes))
        ax.set_xticklabels([str(i + 1) for i in clients], fontsize=16)
        ax.set_yticklabels([str(i) for i in range(n_classes)], fontsize=16)
        ax.set_xticks(np.arange(len(clients)) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_classes) - 0.5, minor=True)
        ax.grid(which='minor', color='yellow', linestyle='-', linewidth=1, alpha=0.5)
        ax.tick_params(which='minor', size=0)
    cbar = fig.colorbar(im, ax=axes, orientation='horizontal', pad=0.08, aspect=40, shrink=0.8, location='top')
    cbar.set_label('Class Assignment', fontsize=18, labelpad=8)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['No', 'Yes'], fontsize=14)
    cbar.ax.tick_params(labelsize=14, top=True, labeltop=True, bottom=False, labelbottom=False)
    plt.tight_layout(rect=[0, 0, 1, 0.7])
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig

def plot_dirichlet_accuracy_bars(alpha_labels, strategy_to_values, ylabel='Accuracy (%)', ylim=None, colors=None, hatches=None, bar_width=0.11, bar_spacing=0, figsize=(8, 4), dpi=600, grid=False, legend_loc='lower right', save_path=None):
    strategies = list(strategy_to_values.keys())
    m = len(alpha_labels)
    n = len(strategies)
    x = np.arange(m)
    if colors is None:
        morandi_palette = ['#9B59B6', '#1EC664', '#EB3926', '#1f77b4', '#ff7f0e', '#1ABC9C']
        colors = (morandi_palette * (n // len(morandi_palette) + 1))[:n]
    if hatches is None:
        hatches = ['/', 'o', '\\', '.', 'x', '-', '+', '*']
    hatch_list = (hatches * (n // len(hatches) + 1))[:n]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for idx, strat in enumerate(strategies):
        vals = strategy_to_values[strat]
        if len(vals) != m:
            raise ValueError(f'Strategy {strat} has length {len(vals)} but alpha_labels has length {m}')
        offset = (idx - (n - 1) / 2) * (bar_width + bar_spacing)
        ax.bar(x + offset, vals, width=bar_width, color=colors[idx], hatch=hatch_list[idx], label=strat, edgecolor='black', linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(alpha_labels)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    if ylim:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.legend(fontsize=16, loc=legend_loc, framealpha=0.9, ncol=3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return (fig, ax)

def plot_dirichlet_datasets_row(alpha_labels, datasets_strategies, dataset_titles, ylims=None, colors=None, hatches=None, bar_width=0.14, bar_spacing=0.01, figsize=(18, 6), dpi=600, save_path=None):
    assert len(datasets_strategies) == 3, 'datasets_strategies '
    assert len(dataset_titles) == 3, 'dataset_titles '
    strategies = list(datasets_strategies[0].keys())
    n = len(strategies)
    m = len(alpha_labels)
    x = np.arange(m)
    if colors is None:
        morandi_palette = ['#9B59B6', '#1EC664', '#EB3926', '#1f77b4', '#ff7f0e', '#1ABC9C']
        colors = (morandi_palette * (n // len(morandi_palette) + 1))[:n]
    if hatches is None:
        hatches = ['/', 'o', '\\', '.', 'x', '-', '+', '*']
    hatch_list = (hatches * (n // len(hatches) + 1))[:n]
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)
    handles_map = {}
    for idx_ds in range(3):
        ax = axes[idx_ds]
        strat_to_vals = datasets_strategies[idx_ds]
        for i, strat in enumerate(strategies):
            vals = strat_to_vals.get(strat, [0] * m)
            offset = (i - (n - 1) / 2) * (bar_width + bar_spacing)
            bars = ax.bar(x + offset, vals, width=bar_width, color=colors[i % len(colors)], hatch=hatch_list[i], edgecolor='white', linewidth=0.7)
            if idx_ds == 0 and strat not in handles_map:
                handles_map[strat] = bars
        ax.set_xticks(x)
        ax.set_xticklabels(alpha_labels, fontsize=12)
        ax.set_xlabel('alpha', fontsize=14)
        ax.set_ylabel('Accuracy (%)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        if ylims and ylims[idx_ds]:
            ax.set_ylim(ylims[idx_ds])
        ax.set_title(dataset_titles[idx_ds], fontsize=14, fontweight='bold')
        try:
            ax.set_box_aspect(0.65)
        except Exception:
            pass
    legend_order = ['Fed-CDP', 'Fed-dynS', 'Fed-AC-DP', 'Addp-DP-FL', 'TUNE-DPFL']
    handles = []
    labels = []
    for s in legend_order:
        if s in handles_map:
            handles.append(handles_map[s])
            labels.append(s)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels), fontsize=16, frameon=False, borderpad=1, labelspacing=0.6, handlelength=2.4, handletextpad=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return (fig, axes)

def plot_lambda_bar_chart(lambda_labels, accuracies, xlabel='Data distribution', ylabel='Accuracy (%)', y_lim=None, figsize=(8, 6), dpi=600, colors=None, annotate=True, save_path=None):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x = np.arange(len(lambda_labels))
    if colors is None:
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % cmap.N) for i in range(len(lambda_labels))]
    bars = ax.bar(x, accuracies, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(lambda_labels)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    if annotate:
        for bar in bars:
            h = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            ax.text(x, h, f'{h:.2f}', ha='center', va='bottom', fontsize=11, color='black', fontweight='bold')
    if y_lim is not None:
        ax.set_ylim(y_lim)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return (fig, ax)

def plot_relative_drop_vs_alpha(log_files_dict, figsize=(15, 9), dpi=600, xlabel='alpha (Dirichlet Parameter)', ylabel='Relative Drop = Acc(IID) - Acc(alpha) (%)', colors=None, save_path=None):
    import re
    datasets_alpha_dict = {}
    for dataset_name, log_files in log_files_dict.items():
        datasets_alpha_dict[dataset_name] = {}
        for log_file in log_files:
            if 'IID' in log_file or 'iid' in log_file:
                alpha = 'IID'
            else:
                match = re.search('(\\d+\\.\\d+)', log_file)
                if match:
                    alpha = float(match.group(1))
                else:
                    alpha = log_file
            datasets_alpha_dict[dataset_name][alpha] = log_file
    dataset_names = list(datasets_alpha_dict.keys())
    alpha_values_set = set()
    for dataset_dict in datasets_alpha_dict.values():
        for alpha in dataset_dict.keys():
            if alpha != 'IID':
                alpha_values_set.add(alpha)
    alpha_values = sorted(list(alpha_values_set))
    n_datasets = len(dataset_names)
    n_alphas = len(alpha_values)
    fig, axes = plt.subplots(n_datasets, n_alphas, figsize=figsize, dpi=dpi)
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    elif n_alphas == 1:
        axes = axes.reshape(-1, 1)
    all_strategies = set()
    first_dataset = dataset_names[0]
    first_log_file = list(datasets_alpha_dict[first_dataset].values())[0]
    try:
        with open(first_log_file, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match('([^:]+):\\s*Round', line)
                if match:
                    strategy = match.group(1).strip()
                    all_strategies.add(strategy)
    except FileNotFoundError:
        pass
    legend_order = ['Fed-CDP', 'Fed-dynS', 'Fed-AC-DP', 'Addp-DP-FL', 'TUNE-DPFL']
    all_strategies = [s for s in legend_order if s in all_strategies]
    if colors is None:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {}
        for i, strategy in enumerate(all_strategies):
            colors[strategy] = default_colors[i % len(default_colors)]
    iid_accs = {}
    for dataset_name in dataset_names:
        iid_accs[dataset_name] = {}
        if 'IID' in datasets_alpha_dict[dataset_name]:
            iid_log_file = datasets_alpha_dict[dataset_name]['IID']
            try:
                with open(iid_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for strategy in all_strategies:
                        for line in reversed(lines):
                            match = re.match(f'{re.escape(strategy)}:\\s*Round\\s+(\\d+):\\s*.*roundAccuracy\\s*=\\s*([\\d.]+)', line)
                            if match:
                                iid_accs[dataset_name][strategy] = float(match.group(2))
                                break
            except FileNotFoundError:
                pass
                pass
                pass
                pass
    for i, dataset_name in enumerate(dataset_names):
        for j, alpha_val in enumerate(alpha_values):
            ax = axes[i, j]
            drops_data = []
            log_file = datasets_alpha_dict[dataset_name][alpha_val]
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for strategy in all_strategies:
                        alpha_acc = None
                        for line in reversed(lines):
                            match = re.match(f'{re.escape(strategy)}:\\s*Round\\s+(\\d+):\\s*.*roundAccuracy\\s*=\\s*([\\d.]+)', line)
                            if match:
                                alpha_acc = float(match.group(2))
                                break
                        if alpha_acc is not None and strategy in iid_accs[dataset_name]:
                            iid_acc = iid_accs[dataset_name][strategy]
                            drop = iid_acc - alpha_acc
                            drops_data.append(drop)
            except FileNotFoundError:
                pass
            if drops_data:
                x_pos = np.arange(len(all_strategies))
                bars = ax.bar(x_pos, drops_data, color=[colors.get(s, 'C0') for s in all_strategies], alpha=0.8, edgecolor='black', linewidth=0.5)
                ax.set_xticks([])
            ax.set_title(f'{dataset_name} ={alpha_val}', fontsize=13, fontweight='bold')
            ax.set_ylabel(ylabel, fontsize=14)
            ax.tick_params(axis='y', labelsize=12)
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6, axis='y')
            ax.set_axisbelow(True)
            ax.set_ylim(bottom=0)
    handles = []
    labels = []
    for strategy in legend_order:
        if strategy in all_strategies:
            color = colors.get(strategy, 'C0')
            handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='black', linewidth=0.5, alpha=0.8))
            labels.append(strategy)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels), fontsize=16, frameon=False, borderpad=1, labelspacing=0.6, handlelength=2.4, handletextpad=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig

def plot_multiple_datasets_subplots(log_files_dict, figsize=(18, 9), dpi=600, xlabel='Communication Rounds', ylabel='Final Accuracy (%)', sampling_interval=10, colors=None, save_path=None):
    import re
    datasets_alpha_dict = {}
    if log_files_dict:
        first_val = next(iter(log_files_dict.values()))
        if isinstance(first_val, list):
            for dataset_name, log_files in log_files_dict.items():
                datasets_alpha_dict[dataset_name] = {}
                for log_file in log_files:
                    match = re.search('(\\d+\\.\\d+)', log_file)
                    if match:
                        alpha = f'={match.group(1)}'
                    else:
                        alpha = log_file
                    datasets_alpha_dict[dataset_name][alpha] = log_file
        else:
            datasets_alpha_dict = log_files_dict
    dataset_names = list(datasets_alpha_dict.keys())
    alpha_values = list(datasets_alpha_dict[dataset_names[0]].keys()) if dataset_names else []
    n_datasets = len(dataset_names)
    n_alphas = len(alpha_values)
    fig, axes = plt.subplots(n_datasets, n_alphas, figsize=figsize, dpi=dpi)
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    elif n_alphas == 1:
        axes = axes.reshape(-1, 1)
    all_strategies = set()
    for dataset in datasets_alpha_dict.values():
        for alpha_val, log_file in dataset.items():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        match = re.match('([^:]+):\\s*Round', line)
                        if match:
                            strategy = match.group(1).strip()
                            all_strategies.add(strategy)
            except FileNotFoundError:
                continue
    legend_order = ['Fed-CDP', 'Fed-dynS', 'Fed-AC-DP', 'Addp-DP-FL', 'MoG-AdDP', 'TUNE-DPFL']
    all_strategies = [s for s in legend_order if s in all_strategies]
    for s in sorted(all_strategies):
        if s not in all_strategies:
            all_strategies.append(s)
    if colors is None:
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = {}
        for i, strategy in enumerate(all_strategies):
            colors[strategy] = default_colors[i % len(default_colors)]
    for i, dataset_name in enumerate(dataset_names):
        for j, alpha_val in enumerate(alpha_values):
            ax = axes[i, j]
            log_file = datasets_alpha_dict[dataset_name][alpha_val]
            strategy_data = {}
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        match = re.match('([^:]+):\\s*Round\\s+(\\d+):\\s*.*roundAccuracy\\s*=\\s*([\\d.]+)', line)
                        if match:
                            strategy = match.group(1).strip()
                            round_num = int(match.group(2))
                            accuracy = float(match.group(3))
                            if strategy not in strategy_data:
                                strategy_data[strategy] = []
                            strategy_data[strategy].append((round_num, accuracy))
            except FileNotFoundError:
                continue
            subplot_max_round = 0
            for strategy in all_strategies:
                if strategy in strategy_data:
                    all_rounds_accs = strategy_data[strategy]
                    all_rounds_accs.sort(key=lambda x: x[0])
                    sampled_rounds = [0]
                    sampled_accuracy = [0]
                    for round_num, accuracy in all_rounds_accs:
                        if round_num % sampling_interval == 0:
                            sampled_rounds.append(round_num)
                            sampled_accuracy.append(accuracy)
                            if round_num > subplot_max_round:
                                subplot_max_round = round_num
                    if sampled_rounds:
                        ax.plot(sampled_rounds, sampled_accuracy, label=strategy, color=colors.get(strategy, 'C0'), linewidth=1.5, marker='o', markersize=4, alpha=0.8)
            ax.set_title(f'{dataset_name} {alpha_val}', fontsize=13, fontweight='bold')
            ax.set_xlabel(xlabel, fontsize=14)
            ax.set_ylabel(ylabel, fontsize=14)
            ax.tick_params(axis='both', labelsize=12)
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
            try:
                ax.set_xlim(left=0)
                if subplot_max_round > 0:
                    ax.set_xticks(list(range(20, subplot_max_round + 1, 20)))
                    ax.set_xlim(0, subplot_max_round)
                ax.set_ylim(bottom=0)
            except Exception:
                pass
            ax.set_aspect('auto')
    legend_order = ['Fed-CDP', 'Fed-dynS', 'Fed-AC-DP', 'Addp-DP-FL', 'MoG-AdDP', 'TUNE-DPFL']
    handles = []
    labels = []
    for strategy in legend_order:
        if strategy in all_strategies:
            color = colors.get(strategy, 'C0')
            handles.append(plt.Line2D([0], [0], color=color, linewidth=2, marker='o', markersize=6))
            labels.append(strategy)
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=len(labels), fontsize=16, frameon=False, borderpad=1, labelspacing=0.6, handlelength=2.4, handletextpad=0.9)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return fig

def plot_communication_rounds_accuracy(log_file_path, sampling_interval=5, colors=None, figsize=(8, 6), dpi=600, xlabel='Communication Rounds', ylabel='Final Accuracy (%)', ylim=(0, 105), grid=False, legend_loc='lower right', save_path=None):
    if colors is None:
        colors = ['#9B59B6', '#1EC664', '#EB3926', '#1f77b4', '#ff7f0e', '#1ABC9C']
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']
    linestyles = ['-', '--', '-.', ':']
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    parsed_data = {}
    strategy_order = []
    try:
        strategy_data = {}
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = re.match('([^:]+):\\s*Round\\s+(\\d+):\\s*.*roundAccuracy\\s*=\\s*([\\d.]+)\\s*%', line)
                if match:
                    strategy = match.group(1).strip()
                    round_num = int(match.group(2))
                    accuracy = float(match.group(3))
                    if strategy not in strategy_data:
                        strategy_data[strategy] = []
                        strategy_order.append(strategy)
                    strategy_data[strategy].append((round_num, accuracy))
        if not strategy_data:
            raise ValueError(f'No data parsed from {log_file_path}')
        for idx, strategy in enumerate(strategy_order):
            all_rounds_accs = strategy_data[strategy]
            all_rounds_accs.sort(key=lambda x: x[0])
            sampled_rounds = [0]
            sampled_accuracy = [0]
            for round_num, accuracy in all_rounds_accs:
                if round_num % sampling_interval == 0:
                    sampled_rounds.append(round_num)
                    sampled_accuracy.append(accuracy)
            if not sampled_rounds:
                continue
            parsed_data[strategy] = {'rounds': sampled_rounds, 'accuracy': sampled_accuracy}
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            linestyle = linestyles[idx % len(linestyles)]
            ax.plot(sampled_rounds, sampled_accuracy, color=color, marker=marker, markersize=5, linewidth=1.5, linestyle=linestyle, label=strategy)
    except FileNotFoundError:
        raise FileNotFoundError(f'Log file not found: {log_file_path}')
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis='both', labelsize=16)
    if ylim:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=16, loc=legend_loc, framealpha=0.9)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return (fig, ax, parsed_data)

def plot_sampling_rate_comparison(data_by_sampling_rate, config_name='66', methods=None, title=None, xlabel='Sampling Rate', ylabel='Accuracy (%)', figsize=(10, 7), dpi=300, colors=None, markers=None, linewidth=2, markersize=8, capsize=5, grid=True, legend_loc='best', ylim=None, save_path=None):
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    if markers is None:
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    sampling_rates = sorted(data_by_sampling_rate.keys())
    if methods is None:
        methods = list(data_by_sampling_rate[sampling_rates[0]].keys())
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for idx, method in enumerate(methods):
        y_means = []
        y_stds = []
        for sr in sampling_rates:
            values = data_by_sampling_rate[sr][method]
            if isinstance(values, list):
                y_means.append(np.mean(values))
                y_stds.append(np.std(values))
            else:
                y_means.append(values)
                y_stds.append(0)
        ax.errorbar(sampling_rates, y_means, yerr=y_stds, color=colors[idx % len(colors)], marker=markers[idx % len(markers)], markersize=markersize, linewidth=linewidth, capsize=capsize, capthick=linewidth, elinewidth=linewidth, linestyle='-', label=method)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc=legend_loc, fontsize=11, framealpha=0.5, edgecolor='gray', fancybox=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    return (fig, ax)
if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    accuracy = [61, 61.2, 62.8, 60.3, 59.95]
    loss_avg = [60.87, 61.66, 64.22, 71.16, 62.54]
    loss_max = [65.2, 75.14, 68.5, 81.22, 66.8]
    loss_min = [56.5, 51.57, 59.9, 58.1, 58.3]
    fig, ax1, ax2 = plot_dual_axis(x_values=x, y1_values=accuracy, y2_values=loss_avg, y1_label='Test Accuracy (%)', y2_label='Average reuse rate (%)', xlabel='Maximum Number of Modes', equal_spacing=True, y1_lim=(55, 66), y2_lim=(45, 95), y2_max_values=loss_max, y2_min_values=loss_min, show_range=True, save_path=os.path.join(Config.output_root, 'imgs', '.png'))
    plt.close(fig)
    log_files_dict = {'CIFAR-10': ['cifar0.1.log', 'cifar0.3.log', 'cifar0.5.log', 'cifar0.7.log', 'cifar0.9.log'], 'FMNIST': ['fmnist0.1.log', 'fmnist0.3.log', 'fmnist0.5.log', 'fmnist0.7.log', 'fmnist0.9.log'], 'MNIST': ['mnist0.1.log', 'mnist0.3.log', 'mnist0.5.log', 'mnist0.7.log', 'mnist0.9.log']}
    log_files_dict_with_iid = {'CIFAR-10': ['cifar0.1.log', 'cifar0.3.log', 'cifar0.5.log', 'cifar0.7.log', 'cifar0.9.log', 'cifarIID.log'], 'FMNIST': ['fmnist0.1.log', 'fmnist0.3.log', 'fmnist0.5.log', 'fmnist0.7.log', 'fmnist0.9.log', 'fmnistIID.log'], 'MNIST': ['mnist0.1.log', 'mnist0.3.log', 'mnist0.5.log', 'mnist0.7.log', 'mnist0.9.log', 'mnistIID.log']}
    log_files_by_setting = {'Sampling rate = 0.2': ['66cifar0.2.log', '108cifar0.2.log', '492cifar0.2.log'], 'Sampling rate = 0.3': ['66cifar0.3.log', '108cifar0.3.log', '492cifar0.3.log'], 'Sampling rate = 0.4': ['66cifar0.4.log', '108cifar0.4.log', '492cifar0.4.log'], 'Sampling rate = 0.5': ['66cifar0.5.log', '108cifar0.5.log', '492cifar0.5.log'], 'Sampling rate = 0.6': ['66cifar0.6.log', '108cifar0.6.log', '492cifar0.6.log']}
    log_files_by_dataset_alpha = {'CIFAR-10': {'alpha=0.1': ['66cifar0.1.log', '108cifar0.1.log', '492cifar0.1.log'], 'alpha=0.3': ['66cifar0.3.log', '108cifar0.3.log', '492cifar0.3.log'], 'alpha=0.5': ['66cifar0.5.log', '108cifar0.5.log', '492cifar0.5.log'], 'alpha=0.7': ['66cifar0.7.log', '108cifar0.7.log', '492cifar0.7.log'], 'alpha=0.9': ['66cifar0.9.log', '108cifar0.9.log', '492cifar0.9.log']}, 'FMNIST': {'alpha=0.1': ['66fmnist0.1.log', '108fmnist0.1.log', '492fmnist0.1.log'], 'alpha=0.3': ['66fmnist0.3.log', '108fmnist0.3.log', '492fmnist0.3.log'], 'alpha=0.5': ['66fmnist0.5.log', '108fmnist0.5.log', '492fmnist0.5.log'], 'alpha=0.7': ['66fmnist0.7.log', '108fmnist0.7.log', '492fmnist0.7.log'], 'alpha=0.9': ['66fmnist0.9.log', '108fmnist0.9.log', '492fmnist0.9.log']}, 'MNIST': {'alpha=0.1': ['66mnist0.1.log', '108mnist0.1.log', '492mnist0.1.log'], 'alpha=0.3': ['66mnist0.3.log', '108mnist0.3.log', '492mnist0.3.log'], 'alpha=0.5': ['66mnist0.5.log', '108mnist0.5.log', '492mnist0.5.log'], 'alpha=0.7': ['66mnist0.7.log', '108mnist0.7.log', '492mnist0.7.log'], 'alpha=0.9': ['66mnist0.9.log', '108mnist0.9.log', '492mnist0.9.log']}}
    label_distribution_data = {'=0.1': {0: [5, 9, 205, 1773, 4, 0, 0, 0, 0, 0], 1: [0, 0, 95, 145, 1, 191, 0, 0, 0, 1566], 2: [0, 0, 0, 0, 124, 795, 1080, 0, 0, 0], 3: [124, 41, 0, 0, 57, 0, 3, 0, 0, 1771], 4: [143, 18, 11, 0, 48, 0, 4, 0, 41, 1663], 5: [22, 0, 674, 0, 0, 293, 2, 993, 5, 0], 6: [27, 76, 1, 17, 0, 1023, 557, 223, 72, 0], 7: [0, 301, 1564, 0, 0, 0, 0, 0, 0, 0], 8: [0, 0, 153, 392, 0, 8, 0, 15, 1418, 0], 9: [164, 18, 0, 0, 698, 1078, 0, 38, 0, 0]}, '=0.3': {0: [99, 143, 782, 869, 53, 10, 0, 27, 13, 0], 1: [2, 2, 480, 417, 102, 106, 5, 4, 8, 870], 2: [3, 18, 1, 27, 252, 509, 1162, 0, 17, 7], 3: [442, 155, 4, 9, 273, 107, 55, 41, 0, 909], 4: [499, 72, 80, 10, 153, 32, 63, 61, 144, 881], 5: [210, 22, 368, 0, 5, 413, 78, 729, 108, 62], 6: [42, 434, 23, 159, 53, 493, 333, 265, 192, 2], 7: [14, 571, 1049, 54, 62, 19, 3, 0, 108, 115], 8: [0, 0, 501, 230, 27, 60, 28, 78, 893, 179], 9: [297, 101, 31, 77, 590, 674, 2, 209, 0, 14]}, '=0.5': {0: [146, 200, 788, 612, 78, 26, 3, 108, 35, 0], 1: [15, 12, 530, 421, 196, 92, 38, 24, 30, 636], 2: [14, 68, 12, 97, 270, 431, 1024, 1, 45, 33], 3: [448, 170, 24, 23, 299, 214, 83, 95, 4, 635], 4: [518, 86, 106, 40, 168, 72, 93, 556, 0, 356], 5: [81, 0, 39, 120, 91, 14, 528, 177, 544, 400], 6: [141, 4, 15, 17, 562, 306, 247, 34, 221, 447], 7: [8, 92, 880, 16, 182, 3, 85, 5, 713, 12], 8: [53, 167, 152, 477, 107, 78, 159, 78, 525, 200], 9: [292, 129, 88, 164, 488, 525, 12, 252, 7, 37]}, '=0.7': {0: [166, 220, 732, 501, 91, 39, 13, 179, 54, 0], 1: [30, 25, 512, 396, 242, 88, 81, 50, 50, 521], 2: [27, 110, 32, 155, 266, 381, 892, 5, 65, 62], 3: [424, 172, 48, 34, 296, 269, 98, 130, 10, 516], 4: [455, 86, 107, 62, 157, 90, 99, 632, 0, 307], 5: [86, 3, 60, 145, 142, 28, 468, 185, 488, 391], 6: [180, 9, 34, 29, 447, 260, 309, 73, 250, 403], 7: [26, 96, 800, 44, 195, 10, 105, 11, 681, 26], 8: [73, 163, 145, 442, 123, 85, 185, 90, 458, 229], 9: [277, 140, 130, 211, 427, 448, 26, 260, 19, 55]}, '=0.9': {0: [176, 228, 679, 439, 100, 49, 26, 230, 68, 1], 1: [44, 37, 483, 371, 262, 87, 118, 73, 66, 453], 2: [38, 140, 53, 193, 257, 347, 792, 11, 79, 85], 3: [401, 172, 69, 43, 288, 296, 107, 152, 17, 451], 4: [418, 87, 109, 79, 152, 103, 104, 660, 0, 282], 5: [89, 8, 75, 159, 178, 41, 428, 188, 450, 378], 6: [200, 13, 52, 39, 384, 235, 335, 108, 260, 369], 7: [47, 100, 734, 75, 201, 19, 118, 19, 642, 41], 8: [87, 162, 143, 416, 133, 91, 200, 99, 419, 244], 9: [265, 146, 158, 237, 388, 402, 38, 260, 33, 68]}}
    class_assignment_data = {'2 Classes': {0: [0, 1], 1: [1, 2], 2: [2, 3], 3: [3, 4], 4: [4, 5], 5: [5, 6], 6: [6, 7], 7: [7, 8], 8: [8, 9], 9: [9, 0]}, '3 Classes': {0: [0, 1, 2], 1: [1, 2, 3], 2: [2, 3, 4], 3: [3, 4, 5], 4: [4, 5, 6], 5: [5, 6, 7], 6: [6, 7, 8], 7: [7, 8, 9], 8: [8, 9, 0], 9: [9, 0, 1]}, '4 Classes': {0: [0, 1, 2, 3], 1: [1, 2, 3, 4], 2: [2, 3, 4, 5], 3: [3, 4, 5, 6], 4: [4, 5, 6, 7], 5: [5, 6, 7, 8], 6: [6, 7, 8, 9], 7: [7, 8, 9, 0], 8: [8, 9, 0, 1], 9: [9, 0, 1, 2]}, '5 Classes': {0: [0, 1, 2, 3, 4], 1: [1, 2, 3, 4, 5], 2: [2, 3, 4, 5, 6], 3: [3, 4, 5, 6, 7], 4: [4, 5, 6, 7, 8], 5: [5, 6, 7, 8, 9], 6: [6, 7, 8, 9, 0], 7: [7, 8, 9, 0, 1], 8: [8, 9, 0, 1, 2], 9: [9, 0, 1, 2, 3]}, '6 Classes': {0: [0, 1, 2, 3, 4, 5], 1: [1, 2, 3, 4, 5, 6], 2: [2, 3, 4, 5, 6, 7], 3: [3, 4, 5, 6, 7, 8], 4: [4, 5, 6, 7, 8, 9], 5: [5, 6, 7, 8, 9, 0], 6: [6, 7, 8, 9, 0, 1], 7: [7, 8, 9, 0, 1, 2], 8: [8, 9, 0, 1, 2, 3], 9: [9, 0, 1, 2, 3, 4]}}
    dropout_rate = {0: {'TUNE-DPFL': [61.4, 61.1, 61.4], 'Addp_DP_FL': [52.35, 51.9, 51.25], 'Fed_AC_DP': [54.7, 54.45, 53.25]}, 0.1: {'TUNE-DPFL': [61.4, 60.45, 60.45], 'Addp_DP_FL': [51.2, 51.65, 51.75], 'Fed_AC_DP': [53.8, 54.9, 54.6]}, 0.3: {'TUNE-DPFL': [60.3, 59.4, 57.5], 'Addp_DP_FL': [51.15, 50.1, 49.1], 'Fed_AC_DP': [53.15, 56.85, 50.75]}, 0.5: {'TUNE-DPFL': [56.4, 56.5, 56.4], 'Addp_DP_FL': [48.5, 46.4, 51.25], 'Fed_AC_DP': [50.85, 49.65, 48.95]}}
    sampling_rate_data = {1: {'TUNE-DPFL': [62.8, 61.7, 61.2], 'Addp_DP_FL': [51.55, 51.25, 54], 'Fed_AC_DP': [54.95, 55.8, 54.45]}, 0.7: {'TUNE-DPFL': [61.4, 61.1, 61.4], 'Addp_DP_FL': [52.35, 51.9, 51.25], 'Fed_AC_DP': [54.7, 54.45, 53.25]}, 0.5: {'TUNE-DPFL': [60.95, 60.25, 61.3], 'Addp_DP_FL': [52.8, 51.9, 52.1], 'Fed_AC_DP': [56.9, 55.85, 54.1]}, 0.3: {'TUNE-DPFL': [59.9, 58, 59.4], 'Addp_DP_FL': [52.6, 51.4, 52.35], 'Fed_AC_DP': [54.4, 54.15, 50.05]}}
    fig, ax = plot_sampling_rate_comparison(data_by_sampling_rate=dropout_rate, config_name='Multi-seed', xlabel='Dropout Rate', ylabel='Test Accuracy (%)', figsize=(8, 6), dpi=600, linewidth=2, markersize=8, capsize=5, grid=False, legend_loc='lower right', ylim=(43, 63), save_path=os.path.join(Config.output_root, 'imgs', 'dropout_rate_errorbar_comparison.png'))
    plt.close(fig)
