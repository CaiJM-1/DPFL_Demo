import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager as fm
import matplotlib.pyplot as plt
from config import Config

def _choose_chinese_font():
    available = {f.name for f in fm.fontManager.ttflist}
    candidates = ['SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK SC', 'Noto Sans CJK', 'DejaVu Sans']
    for name in candidates:
        if name in available:
            return name
    return 'DejaVu Sans'
chosen_font = _choose_chinese_font()
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman', chosen_font]
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.unicode_minus'] = False

def plot_results(histories, save_dir=os.path.join(Config.output_root, 'imgs'), dpi=600):
    os.makedirs(save_dir, exist_ok=True)
    time_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
    line_styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D', '*', 'x']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
    plt.figure(figsize=(4, 3))
    for i, (strategy, history) in enumerate(histories.items()):
        if isinstance(history, dict) and 'global_loss' in history:
            rounds = range(1, len(history['global_loss']) + 1)
            style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)] if style == '-' else None
            plt.plot(rounds, history['global_loss'], label=strategy, linestyle=style, marker=marker, color=color, markersize=0, linewidth=0.8)
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.legend(fontsize=8)
    plt.tight_layout()
    loss_path = os.path.join(save_dir, f'loss_{time_suffix}.png')
    plt.savefig(loss_path, dpi=dpi)
    plt.close()
    plt.figure(figsize=(4, 3))
    for i, (strategy, history) in enumerate(histories.items()):
        if isinstance(history, dict) and 'global_accuracy' in history:
            rounds = range(1, len(history['global_accuracy']) + 1)
            style = line_styles[i % len(line_styles)]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)] if style == '-' else None
            plt.plot(rounds, history['global_accuracy'], label=strategy, linestyle=style, marker=marker, color=color, markersize=0, linewidth=0.8)
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    acc_path = os.path.join(save_dir, f'accuracy_{time_suffix}.png')
    plt.savefig(acc_path, dpi=dpi)
    plt.close()
    plt.figure(figsize=(4, 3))
    strategies = list(histories.keys())
    total_times = [histories[s]['total_time'] for s in strategies]
    plt.bar(strategies, total_times, color=colors[:len(strategies)])
    plt.xlabel('Training Strategy')
    plt.ylabel('Total Training Time (s)')
    for i, v in enumerate(total_times):
        plt.text(i, v, f'{v:.2f}s', ha='center', va='bottom')
    plt.tight_layout()
    time_path = os.path.join(save_dir, f'training_time_comparison_{time_suffix}.png')
    plt.savefig(time_path, dpi=dpi)
    plt.close()
