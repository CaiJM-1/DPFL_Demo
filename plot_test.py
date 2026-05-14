import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import re
from scipy.stats import norm
from config import *
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

def get_all_layers_from_saved_bic(data_dir=None):
    layer_names = set()
    pattern = re.compile('gmm_bic_round\\d+_(.+)\\.pkl')
    if data_dir is None:
        data_dir = os.path.join(Config.output_root, 'saved')
    for fname in os.listdir(data_dir):
        match = pattern.match(fname)
        if match:
            layer_names.add(match.group(1))
    return sorted(layer_names)

def plot_all_rounds_in_one_fig(layer_name, data_dir=None, img_dir=None, normalize=True, client_id=None):
    if data_dir is None:
        data_dir = os.path.join(Config.output_root, 'saved')
    if img_dir is None:
        img_dir = os.path.join(Config.output_root, 'gmm_fit')
    os.makedirs(img_dir, exist_ok=True)
    pattern = re.compile('gmm_bic_round(\\d+)_' + re.escape(layer_name) + '\\.pkl')
    round_files = sorted([(int(match.group(1)), os.path.join(data_dir, f)) for f in os.listdir(data_dir) if (match := pattern.match(f))])
    if not round_files:
        return
    plt.figure(figsize=(12, 6))
    all_means, all_stds = ([], [])
    for _, filepath in round_files:
        with open(filepath, 'rb') as f:
            p = pickle.load(f)
            all_means.extend(p['means'])
            all_stds.extend(p['stds'])
    global_xmin = np.min(all_means - 3 * np.array(all_stds))
    global_xmax = np.max(all_means + 3 * np.array(all_stds))
    x = np.linspace(global_xmin, global_xmax, 1000)
    for round_num, filepath in round_files:
        with open(filepath, 'rb') as f:
            params = pickle.load(f)
        means, stds, weights = (params['means'], params['stds'], params['weights'])
        density = np.zeros_like(x)
        for mu, sigma, w in zip(means, stds, weights):
            density += w * norm.pdf(x, loc=mu, scale=sigma)
        if normalize:
            density /= np.max(density)
        plt.plot(x, density, label=f'Round {round_num + 1}', linewidth=2)
    layer_str = layer_name.replace('_', '.')
    if client_id is not None:
        plt.title(f'Client {client_id} - Layer {layer_str} Noise Model Comparison')
    else:
        plt.title(f'{layer_str} - BIC Fits Across Rounds')
    plt.xlabel('Gradient Value')
    plt.ylabel('Density')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout(rect=[0, 0, 0.65, 1])
    save_path = os.path.join(img_dir, f'all_rounds_overlay_bic_{layer_name}.png')
    plt.savefig(save_path, dpi=600)
    plt.close()
if __name__ == '__main__':
    data_dir = os.path.join(Config.output_root, 'saved')
    img_dir = os.path.join(Config.output_root, 'gmm_fit')
    os.makedirs(img_dir, exist_ok=True)
    all_layers = get_all_layers_from_saved_bic(data_dir=data_dir)
    for layer in all_layers:
        plot_all_rounds_in_one_fig(layer_name=layer, data_dir=data_dir, img_dir=img_dir, normalize=True)
