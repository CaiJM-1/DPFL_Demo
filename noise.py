import torch
import numpy as np
from config import *
from sklearn.mixture import GaussianMixture
import math
import pickle
import os
from collections import defaultdict
CLIENT_DECAY_STEP = defaultdict(int)

def layer_signature(vec):
    v = np.asarray(vec, dtype=np.float64).ravel()
    if v.size == 0:
        return np.zeros(4, dtype=np.float64)
    a = np.abs(v)
    return np.array([v.mean(), a.mean(), a.std(), np.quantile(a, 0.95)], dtype=np.float64)

def rel_change(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12)

def _enforce_rdp_per_step(fused_sens, layer_variances, epsilon_per_step, alpha):
    if epsilon_per_step is None or epsilon_per_step <= 0:
        return 1.0
    total_rdp = 0.0
    for name, sigma2 in layer_variances.items():
        if sigma2 <= 0:
            continue
        sens = fused_sens.get(name, 0.0)
        total_rdp += alpha * sens ** 2 / (2 * sigma2)
    if total_rdp <= epsilon_per_step or total_rdp == 0.0:
        return 1.0
    return max(total_rdp / epsilon_per_step, 1.0)

def generate_mog_noise(size, means, stds, weights):
    num_elements = int(np.prod(size))
    component_ids = np.random.choice(len(weights), size=num_elements, p=weights)
    noise_values = np.random.normal(loc=np.take(means, component_ids), scale=np.take(stds, component_ids))
    noise_tensor = torch.tensor(noise_values.reshape(size), dtype=torch.float32)
    return noise_tensor
from sklearn.mixture import GaussianMixture

def adjust_gmm_variance(gmm_dict, fused_sens, client_id, client_sigma2_init, layer_width_dict, grad_norm_history_dict, grad_history_dict, use_history=True, gamma=1.0, beta=0.5, client_step=None, epsilon_per_step=None, **kwargs):
    if not hasattr(adjust_gmm_variance, 'last_step_cache'):
        adjust_gmm_variance.last_step_cache = {}
    if not hasattr(adjust_gmm_variance, 'rebuild_ref_cache'):
        adjust_gmm_variance.rebuild_ref_cache = {}
    if not hasattr(adjust_gmm_variance, 'prev_fused_sens_cache'):
        adjust_gmm_variance.prev_fused_sens_cache = {}
    if not hasattr(adjust_gmm_variance, 'accumulated_change_cache'):
        adjust_gmm_variance.accumulated_change_cache = {}
    if not hasattr(adjust_gmm_variance, 'reuse_stats_cache'):
        adjust_gmm_variance.reuse_stats_cache = {}
    if not hasattr(adjust_gmm_variance, 'mode_printed'):
        adjust_gmm_variance.mode_printed = {}
    threshold_mode_global = kwargs.get('threshold_mode', 'both')
    if client_id not in adjust_gmm_variance.mode_printed:
        mode_desc = {'step_only': 'step_only: delta_step > tau', 'cumsum_only': 'cumsum_only: delta_ref > tau_c', 'both': 'both: delta_step > tau OR delta_ref > tau_c'}
        adjust_gmm_variance.mode_printed[client_id] = True
    adjusted_gmm_dict = {}
    layer_noise_energy_dict = {}
    layer_variances = {}
    layer_std_arrays = {}
    os.makedirs(os.path.join(Config.output_root, 'logs'), exist_ok=True)
    log_path = os.path.join(Config.output_root, f'logs/client_{client_id}_log.txt')
    log_file = open(log_path, 'a', encoding='utf-8')
    if not hasattr(adjust_gmm_variance, 'prev_fused_sens_cache'):
        adjust_gmm_variance.prev_fused_sens_cache = {}
    prev_fused_sens = adjust_gmm_variance.prev_fused_sens_cache.get(client_id, {})
    if client_step is None:
        t = CLIENT_DECAY_STEP[client_id]
        CLIENT_DECAY_STEP[client_id] += 1
    else:
        t = int(client_step)
        CLIENT_DECAY_STEP[client_id] = t + 1
    for name in gmm_dict:
        if name not in fused_sens:
            continue
        means = np.array(gmm_dict[name]['means'])
        stds = np.array(gmm_dict[name]['stds'])
        weights = np.array(gmm_dict[name]['weights'])
        d = means.copy()
        n_components = len(means)
        S_mix = fused_sens[name]
        S_mix_prev = prev_fused_sens.get(name, 1.0)
        lambda_soft = Config.lambda_soft
        grad_history = grad_history_dict.get(name, [])
        if use_history and len(grad_history) > 0:
            if isinstance(grad_history[0], torch.Tensor):
                grad_samples = torch.stack(grad_history).cpu().numpy().flatten()
            else:
                grad_samples = np.array(grad_history).flatten()
            if grad_samples is not None and grad_samples.size > 0:
                sig_curr = layer_signature(grad_samples)
                key = (client_id, name)
                sig_prev = adjust_gmm_variance.last_step_cache.get(key, None)
                sig_ref = adjust_gmm_variance.rebuild_ref_cache.get(key, None)
                tau, tau_c = (0.4, 4.0)
                threshold_mode = kwargs.get('threshold_mode', 'both')
                need_rebuild = sig_prev is None or sig_ref is None
                if not need_rebuild:
                    delta_step = rel_change(sig_curr, sig_prev)
                    key = (client_id, name)
                    accumulated_change = adjust_gmm_variance.accumulated_change_cache.get(key, 0.0)
                    accumulated_change += delta_step
                    adjust_gmm_variance.accumulated_change_cache[key] = accumulated_change
                    delta_ref = accumulated_change
                    if threshold_mode == 'step_only':
                        need_rebuild = delta_step > tau
                        if t % 10 == 0:
                            pass
                    elif threshold_mode == 'cumsum_only':
                        need_rebuild = delta_ref > tau_c
                        if t % 10 == 0:
                            pass
                    else:
                        need_rebuild = delta_step > tau or delta_ref > tau_c
                        if t % 10 == 0:
                            pass
                    if need_rebuild:
                        adjust_gmm_variance.accumulated_change_cache[key] = 0.0
                if need_rebuild:
                    max_peaks = kwargs.get('max_peaks', 3)
                    use_bic = kwargs.get('use_bic', False)
                    use_fixed_means_bic = kwargs.get('use_fixed_means_bic', False)
                    if use_fixed_means_bic:
                        d, weights = select_gmm_and_get_means_bic_fixed_3peak(grad_samples, name, t, save_gmm=True)
                        means = d
                    elif max_peaks == 1:
                        if use_bic:
                            d, weights = select_gmm_and_get_means_bic_1peak(grad_samples, name, t, save_gmm=True)
                        else:
                            d, weights = select_gmm_and_get_means_1peak(grad_samples, name, t, save_gmm=True)
                        means = build_adaptive_mean_template_1peak()
                    elif max_peaks == 2:
                        if use_bic:
                            d, weights = select_gmm_and_get_means_bic_2peak(grad_samples, name, t, save_gmm=True)
                        else:
                            d, weights = select_gmm_and_get_means_2peak(grad_samples, name, t, save_gmm=True)
                        means = build_adaptive_mean_template_2peak(d)
                    elif max_peaks == 4:
                        if use_bic:
                            d, weights = select_gmm_and_get_means_bic_4peak(grad_samples, name, t, save_gmm=True)
                        else:
                            d, weights = select_gmm_and_get_means_4peak(grad_samples, name, t, save_gmm=True)
                        means = build_adaptive_mean_template_4peak(d)
                    elif max_peaks == 5:
                        if use_bic:
                            d, weights = select_gmm_and_get_means_bic_5peak(grad_samples, name, t, save_gmm=True)
                        else:
                            d, weights = select_gmm_and_get_means_5peak(grad_samples, name, t, save_gmm=True)
                        means = build_adaptive_mean_template_5peak(d)
                    else:
                        if use_bic:
                            d, weights = select_gmm_and_get_means_bic_3peak(grad_samples, name, t, save_gmm=True)
                        else:
                            d, weights = select_gmm_and_get_means(grad_samples, name, t, save_gmm=True)
                        means = build_adaptive_mean_template(d)
                    n_components = len(means)
                    adjust_gmm_variance.rebuild_ref_cache[key] = sig_curr.copy()
                    if key not in adjust_gmm_variance.reuse_stats_cache:
                        adjust_gmm_variance.reuse_stats_cache[key] = {'reused': 0, 'rebuilt': 0, 'tau': tau, 'tau_c': tau_c, 'threshold_mode': threshold_mode}
                    adjust_gmm_variance.reuse_stats_cache[key]['rebuilt'] += 1
                else:
                    prev_gmm = gmm_dict[name]
                    means = np.array(prev_gmm['means'])
                    stds = np.array(prev_gmm['stds'])
                    weights = np.array(prev_gmm['weights'])
                    d = np.array(gmm_dict[name]['means'])
                    n_components = len(means)
                    if key not in adjust_gmm_variance.reuse_stats_cache:
                        adjust_gmm_variance.reuse_stats_cache[key] = {'reused': 0, 'rebuilt': 0, 'tau': 0.4, 'tau_c': 4.0, 'threshold_mode': threshold_mode}
                    adjust_gmm_variance.reuse_stats_cache[key]['reused'] += 1
                adjust_gmm_variance.last_step_cache[key] = sig_curr.copy()
        lambda_soft_adjusted = lambda_soft
        weights = soft_weight_mask_symmetric_safe(n_components, d=d, mu_list=means, fused_sens=S_mix, lambda_soft=lambda_soft_adjusted, rbic_weights=weights)
        sigma2_cfg = client_sigma2_init.get(client_id, {})
        sigma2_history = sigma2_cfg.get('sigma2_pre', {})
        sigma2_init_value = sigma2_history.get(name, sigma2_cfg.get('sigma2_init', 1.0))
        eps_layer = max(Config.epsilon, 1e-06)
        alpha = Config.rdp_alpha
        sigma2_target = alpha * S_mix ** 2 / (2 * eps_layer)

        weights_array = np.array(weights) + 1e-12
        weights_array = weights_array / np.sum(weights_array)
        stds = np.sqrt(weights_array * sigma2_target)
        sigma2_current = np.mean(stds ** 2)
        sensitivity_scaling = np.sqrt(sigma2_target / (sigma2_current + 1e-10))
        stds = stds * sensitivity_scaling
        noise_variance = np.sum(weights * stds ** 2)
        VARIANCE_MIN = 1e-12
        VARIANCE_MAX = 1000000.0
        clipped_variance = float(np.clip(noise_variance, VARIANCE_MIN, VARIANCE_MAX))
        layer_variances[name] = clipped_variance
        layer_std_arrays[name] = stds
        sigma2_current = np.mean(stds ** 2)
        log_file.write(f'[Round {t}] Layer {name} means: {np.array(means).tolist()}\n')
        adjusted_gmm_dict[name] = {'means': means.tolist(), 'stds': stds.tolist(), 'weights': weights}
    for name, variance in layer_variances.items():
        layer_noise_energy_dict[name] = variance
        client_sigma2_init[client_id]['sigma2_pre'][name] = variance
        if name in adjusted_gmm_dict:
            adjusted_gmm_dict[name]['stds'] = layer_std_arrays[name].tolist()
    adjust_gmm_variance.prev_fused_sens_cache[client_id] = fused_sens.copy()
    if t % 10 == 0 and t > 0:
        reuse_log_path = os.path.join(Config.output_root, f'logs/reuse_stats_client_{client_id}.txt')
        os.makedirs(os.path.join(Config.output_root, 'logs'), exist_ok=True)
        with open(reuse_log_path, 'a', encoding='utf-8') as reuse_log:
            reuse_log.write(f'\n========== Round {t} Reuse Stats ==========\n')
            tau_val = None
            tau_c_val = None
            threshold_mode_val = None
            for key, stats in adjust_gmm_variance.reuse_stats_cache.items():
                cid, layer = key
                if cid == client_id and tau_val is None:
                    tau_val = stats.get('tau', 0.4)
                    tau_c_val = stats.get('tau_c', 4.0)
                    threshold_mode_val = stats.get('threshold_mode', 'both')
                    break
            reuse_log.write(f'Thresholds: tau={tau_val}, tau_c={tau_c_val}\n')
            reuse_log.write(f'Threshold mode: {threshold_mode_val}\n')
            mode_desc = {'step_only': 'step_only: delta_step > tau', 'cumsum_only': 'cumsum_only: delta_ref > tau_c', 'both': 'both: delta_step > tau OR delta_ref > tau_c'}
            reuse_log.write(f"Mode details: {mode_desc.get(threshold_mode_val, 'unknown')}\n\n")
            for key, stats in adjust_gmm_variance.reuse_stats_cache.items():
                cid, layer = key
                if cid != client_id:
                    continue
                total = stats['reused'] + stats['rebuilt']
                if total > 0:
                    reuse_rate = stats['reused'] / total * 100
                    reuse_log.write(
                        f"Layer {layer}:\n"
                        f"  reused: {stats['reused']}\n"
                        f"  rebuilt: {stats['rebuilt']}\n"
                        f"  total: {total}\n"
                        f"  reuse_rate: {reuse_rate:.2f}%\n\n"
                    )
    return (adjusted_gmm_dict, layer_noise_energy_dict)

def soft_weight_mask_symmetric_safe(n_components, d, mu_list, fused_sens=1.0, lambda_soft=1.0, rbic_weights=None, sigmags=0.1, min_weight=0.05):
    mu = np.array(mu_list)
    d = np.array(d)
    if mu is None or rbic_weights is None:
        raise ValueError('mu_list and rbic_weights must not be None')
    if len(d) != len(rbic_weights):
        raise ValueError(f'RBIC means length ({len(d)}) does not match rbic_weights length ({len(rbic_weights)})')
    final_weights = np.zeros(len(mu_list))
    for i, rbic_value in enumerate(d):
        distances = np.array([np.linalg.norm(np.array(rbic_value) - np.array(mu)) for mu in mu_list])
        closest_mu_index = np.argmin(distances)
        final_weights[closest_mu_index] += rbic_weights[i]
    if len(final_weights) == 3:
        avg_side = (final_weights[0] + final_weights[2]) / 2
        final_weights[0] = avg_side
        final_weights[2] = avg_side
    elif len(final_weights) == 2:
        avg_pair = np.sum(final_weights) / 2.0
        final_weights[0] = avg_pair
        final_weights[1] = avg_pair
    logits = lambda_soft * final_weights
    soft_weights = np.exp(logits - np.max(logits))
    soft_weights = soft_weights / np.sum(soft_weights)
    clipped_weights = np.clip(soft_weights, min_weight, None)
    final_weights = clipped_weights / np.sum(clipped_weights)
    return final_weights.tolist()

def select_gmm_and_get_means(data, name, t, save_gmm, max_components=3):
    data = np.array(data).reshape(-1, 1)
    best_bic = np.inf
    best_gmm = None
    best_n = 1
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(data)
        bic = gmm.bic(data)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n
    means = best_gmm.means_.flatten()
    stds = np.sqrt(best_gmm.covariances_.flatten())
    weights = best_gmm.weights_.flatten()
    if save_gmm:
        name = name.replace('.', '_')
        os.makedirs('saved', exist_ok=True)
        save_path = f'saved/gmm_bic_round{t}_{name}.pkl'
        gmm_dict2 = {'means': means, 'stds': stds, 'weights': weights}
        with open(save_path, 'wb') as f:
            pickle.dump(gmm_dict2, f)
    return (means, weights)

def select_gmm_and_get_means_bic_1peak(data, name, t, save_gmm, max_components=1):
    data = np.array(data).reshape(-1, 1)
    best_bic = np.inf
    best_gmm = None
    best_n = 1
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(data)
        bic = gmm.bic(data)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n
    means = best_gmm.means_.flatten()
    stds = np.sqrt(best_gmm.covariances_.flatten())
    weights = best_gmm.weights_.flatten()
    if save_gmm:
        name = name.replace('.', '_')
        os.makedirs(os.path.join(Config.output_root, 'saved'), exist_ok=True)
        save_path = os.path.join(Config.output_root, 'saved', f'gmm_bic_1peak_round{t}_{name}.pkl')
        gmm_dict2 = {'means': means, 'stds': stds, 'weights': weights}
        with open(save_path, 'wb') as f:
            pickle.dump(gmm_dict2, f)
    return (means, weights)

def select_gmm_and_get_means_bic_2peak(data, name, t, save_gmm, max_components=2):
    data = np.array(data).reshape(-1, 1)
    best_bic = np.inf
    best_gmm = None
    best_n = 1
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(data)
        bic = gmm.bic(data)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n
    means = best_gmm.means_.flatten()
    stds = np.sqrt(best_gmm.covariances_.flatten())
    weights = best_gmm.weights_.flatten()
    if save_gmm:
        name = name.replace('.', '_')
        os.makedirs(os.path.join(Config.output_root, 'saved'), exist_ok=True)
        save_path = os.path.join(Config.output_root, 'saved', f'gmm_bic_2peak_round{t}_{name}.pkl')
        gmm_dict2 = {'means': means, 'stds': stds, 'weights': weights}
        with open(save_path, 'wb') as f:
            pickle.dump(gmm_dict2, f)
    return (means, weights)

def select_gmm_and_get_means_bic_3peak(data, name, t, save_gmm, max_components=3):
    data = np.array(data).reshape(-1, 1)
    best_bic = np.inf
    best_gmm = None
    best_n = 1
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(data)
        bic = gmm.bic(data)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n
    means = best_gmm.means_.flatten()
    stds = np.sqrt(best_gmm.covariances_.flatten())
    weights = best_gmm.weights_.flatten()
    if save_gmm:
        name = name.replace('.', '_')
        os.makedirs(os.path.join(Config.output_root, 'saved'), exist_ok=True)
        save_path = os.path.join(Config.output_root, 'saved', f'gmm_bic_3peak_round{t}_{name}.pkl')
        gmm_dict2 = {'means': means, 'stds': stds, 'weights': weights}
        with open(save_path, 'wb') as f:
            pickle.dump(gmm_dict2, f)
    return (means, weights)

def select_gmm_and_get_means_bic_4peak(data, name, t, save_gmm, max_components=4):
    data = np.array(data).reshape(-1, 1)
    best_bic = np.inf
    best_gmm = None
    best_n = 1
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(data)
        bic = gmm.bic(data)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n
    means = best_gmm.means_.flatten()
    stds = np.sqrt(best_gmm.covariances_.flatten())
    weights = best_gmm.weights_.flatten()
    if save_gmm:
        name = name.replace('.', '_')
        os.makedirs(os.path.join(Config.output_root, 'saved'), exist_ok=True)
        save_path = os.path.join(Config.output_root, 'saved', f'gmm_bic_4peak_round{t}_{name}.pkl')
        gmm_dict2 = {'means': means, 'stds': stds, 'weights': weights}
        with open(save_path, 'wb') as f:
            pickle.dump(gmm_dict2, f)
    return (means, weights)

def select_gmm_and_get_means_bic_5peak(data, name, t, save_gmm, max_components=5):
    data = np.array(data).reshape(-1, 1)
    best_bic = np.inf
    best_gmm = None
    best_n = 1
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(data)
        bic = gmm.bic(data)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n
    means = best_gmm.means_.flatten()
    stds = np.sqrt(best_gmm.covariances_.flatten())
    weights = best_gmm.weights_.flatten()
    if save_gmm:
        name = name.replace('.', '_')
        os.makedirs(os.path.join(Config.output_root, 'saved'), exist_ok=True)
        save_path = os.path.join(Config.output_root, 'saved', f'gmm_bic_5peak_round{t}_{name}.pkl')
        gmm_dict2 = {'means': means, 'stds': stds, 'weights': weights}
        with open(save_path, 'wb') as f:
            pickle.dump(gmm_dict2, f)
    return (means, weights)

def select_gmm_and_get_means_bic_fixed_3peak(data, name, t, save_gmm):
    data = np.array(data).reshape(-1, 1)
    best_bic = np.inf
    best_gmm = None
    best_n = 1
    for n in range(1, 4):
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(data)
        bic = gmm.bic(data)
        if bic < best_bic:
            best_bic = bic
            best_gmm = gmm
            best_n = n
    means_raw = best_gmm.means_.flatten()
    weights_raw = best_gmm.weights_.flatten()
    order = np.argsort(means_raw)
    means_sorted = means_raw[order]
    weights_sorted = weights_raw[order]
    if best_n <= 1:
        means = np.array([0.0])
        weights = np.array([weights_sorted[0]])
    elif best_n == 2:
        means = np.array([-0.003, 0.003])
        weights = weights_sorted[:2]
    else:
        means = np.array([-0.003, 0.0, 0.003])
        weights = weights_sorted[:3]
    if save_gmm:
        name = name.replace('.', '_')
        os.makedirs(os.path.join(Config.output_root, 'saved'), exist_ok=True)
        save_path = os.path.join(Config.output_root, 'saved', f'gmm_bic_fixed_3peak_round{t}_{name}.pkl')
        covs_sorted = best_gmm.covariances_.reshape(best_n, -1)[order]
        stds = np.sqrt(covs_sorted.flatten()[:len(means)])
        gmm_dict = {'means': means, 'stds': stds, 'weights': weights}
        with open(save_path, 'wb') as f:
            pickle.dump(gmm_dict, f)
    return (means, weights)

def select_gmm_and_get_means_2peak(data, name, t, save_gmm, max_components=2, gamma=1.0, beta=0.5, w_min=0.05, overlap_tol=0.05):
    data = np.array(data).reshape(-1, 1)
    best_rbic = np.inf
    best_gmm = None
    best_n = 1
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(data)
        bic = gmm.bic(data)
        means = np.sort(gmm.means_.flatten())
        weights = gmm.weights_.flatten()
        overlap_penalty = 0.0
        if len(means) > 1:
            diffs = np.diff(means)
            overlap_penalty = np.sum(np.exp(-diffs ** 2 / (2 * overlap_tol ** 2)))
        small_weight_penalty = np.sum(weights < w_min)
        rbic = bic + gamma * overlap_penalty + beta * small_weight_penalty
        if rbic < best_rbic:
            best_rbic = rbic
            best_gmm = gmm
            best_n = n
    means = best_gmm.means_.flatten()
    stds = np.sqrt(best_gmm.covariances_.flatten())
    weights = best_gmm.weights_.flatten()
    if save_gmm:
        name = name.replace('.', '_')
        os.makedirs(os.path.join(Config.output_root, 'saved'), exist_ok=True)
        save_path = os.path.join(Config.output_root, 'saved', f'gmm_2peak_round{t}_{name}.pkl')
        gmm_dict2 = {'means': means, 'stds': stds, 'weights': weights}
        with open(save_path, 'wb') as f:
            pickle.dump(gmm_dict2, f)
    return (means, weights)

def select_gmm_and_get_means_4peak(data, name, t, save_gmm, max_components=4, gamma=0.7, beta=0.4, w_min=0.04, overlap_tol=0.05):
    data = np.array(data).reshape(-1, 1)
    best_rbic = np.inf
    best_gmm = None
    best_n = 1
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(data)
        bic = gmm.bic(data)
        means = np.sort(gmm.means_.flatten())
        weights = gmm.weights_.flatten()
        overlap_penalty = 0.0
        if len(means) > 1:
            diffs = np.diff(means)
            overlap_penalty = np.sum(np.exp(-diffs ** 2 / (2 * overlap_tol ** 2)))
        small_weight_penalty = np.sum(weights < w_min)
        rbic = bic + gamma * overlap_penalty + beta * small_weight_penalty
        if rbic < best_rbic:
            best_rbic = rbic
            best_gmm = gmm
            best_n = n
    means = best_gmm.means_.flatten()
    stds = np.sqrt(best_gmm.covariances_.flatten())
    weights = best_gmm.weights_.flatten()
    if save_gmm:
        name = name.replace('.', '_')
        os.makedirs(os.path.join(Config.output_root, 'saved'), exist_ok=True)
        save_path = os.path.join(Config.output_root, 'saved', f'gmm_4peak_round{t}_{name}.pkl')
        gmm_dict2 = {'means': means, 'stds': stds, 'weights': weights}
        with open(save_path, 'wb') as f:
            pickle.dump(gmm_dict2, f)
    return (means, weights)

def select_gmm_and_get_means_5peak(data, name, t, save_gmm, max_components=5, gamma=0.5, beta=0.3, w_min=0.03, overlap_tol=0.05):
    data = np.array(data).reshape(-1, 1)
    best_rbic = np.inf
    best_gmm = None
    best_n = 1
    for n in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(data)
        bic = gmm.bic(data)
        means = np.sort(gmm.means_.flatten())
        weights = gmm.weights_.flatten()
        overlap_penalty = 0.0
        if len(means) > 1:
            diffs = np.diff(means)
            overlap_penalty = np.sum(np.exp(-diffs ** 2 / (2 * overlap_tol ** 2)))
        small_weight_penalty = np.sum(weights < w_min)
        rbic = bic + gamma * overlap_penalty + beta * small_weight_penalty
        if rbic < best_rbic:
            best_rbic = rbic
            best_gmm = gmm
            best_n = n
    means = best_gmm.means_.flatten()
    stds = np.sqrt(best_gmm.covariances_.flatten())
    weights = best_gmm.weights_.flatten()
    if save_gmm:
        name = name.replace('.', '_')
        os.makedirs(os.path.join(Config.output_root, 'saved'), exist_ok=True)
        save_path = os.path.join(Config.output_root, 'saved', f'gmm_5peak_round{t}_{name}.pkl')
        gmm_dict2 = {'means': means, 'stds': stds, 'weights': weights}
        with open(save_path, 'wb') as f:
            pickle.dump(gmm_dict2, f)
    return (means, weights)

def select_gmm_and_get_means_1peak(data, name, t, save_gmm):
    means = np.array([0.0], dtype=float)
    weights = np.array([1.0], dtype=float)
    if save_gmm:
        name = name.replace('.', '_')
        os.makedirs(os.path.join(Config.output_root, 'saved'), exist_ok=True)
        save_path = os.path.join(Config.output_root, 'saved', f'gmm_1peak_round{t}_{name}.pkl')
        gmm_dict2 = {'means': means, 'stds': np.array([1.0], dtype=float), 'weights': weights}
        with open(save_path, 'wb') as f:
            pickle.dump(gmm_dict2, f)
    return (means, weights)

def build_adaptive_mean_template_2peak(raw_means, min_magnitude=0.001):
    raw = np.array(raw_means, dtype=float).reshape(-1)
    if raw.size == 0:
        return np.array([0.0], dtype=float)
    finite_raw = raw[np.isfinite(raw)]
    if finite_raw.size == 0:
        return np.array([0.0], dtype=float)
    max_abs = float(np.max(np.abs(finite_raw)))
    if max_abs < min_magnitude:
        return np.array([0.0], dtype=float)
    pos_peaks = finite_raw[finite_raw > min_magnitude]
    neg_peaks = finite_raw[finite_raw < -min_magnitude]
    has_pos = len(pos_peaks) > 0
    has_neg = len(neg_peaks) > 0
    if has_pos and has_neg:
        b = max_abs
        return np.array([-b, b], dtype=float)
    if has_pos or has_neg:
        peaks = pos_peaks if has_pos else -neg_peaks
        if len(peaks) >= 2:
            a = max_abs
            return np.array([-a, a], dtype=float)
        else:
            b = max_abs
            return np.array([-b, b], dtype=float)
    return np.array([0.0], dtype=float)

def build_adaptive_mean_template_4peak(raw_means, min_magnitude=0.001):
    raw = np.array(raw_means, dtype=float).reshape(-1)
    if raw.size == 0:
        return np.array([0.0], dtype=float)
    finite_raw = raw[np.isfinite(raw)]
    if finite_raw.size == 0:
        return np.array([0.0], dtype=float)
    max_abs = float(np.max(np.abs(finite_raw)))
    if max_abs < min_magnitude:
        return np.array([0.0], dtype=float)
    pos_peaks = finite_raw[finite_raw > min_magnitude]
    neg_peaks = finite_raw[finite_raw < -min_magnitude]
    merge_threshold = 0.005
    if len(pos_peaks) >= 2:
        pos_sorted = np.sort(pos_peaks)[::-1]
        filtered_pos = [pos_sorted[0]]
        for peak in pos_sorted[1:]:
            if abs(peak - filtered_pos[-1]) >= merge_threshold:
                filtered_pos.append(peak)
        pos_peaks = np.array(filtered_pos)
    if len(neg_peaks) >= 2:
        neg_sorted = np.sort(neg_peaks)
        filtered_neg = [neg_sorted[0]]
        for peak in neg_sorted[1:]:
            if abs(peak - filtered_neg[-1]) >= merge_threshold:
                filtered_neg.append(peak)
        neg_peaks = np.array(filtered_neg)
    has_pos = len(pos_peaks) > 0
    has_neg = len(neg_peaks) > 0
    if has_pos and has_neg and (len(pos_peaks) >= 2) and (len(neg_peaks) >= 2):
        pos_sorted = np.sort(pos_peaks)[::-1]
        neg_sorted = np.sort(neg_peaks)
        a_pos = abs(pos_sorted[0])
        a_neg = abs(neg_sorted[0])
        if abs(a_pos - a_neg) < merge_threshold:
            a = max(a_pos, a_neg)
        else:
            a = float(max(a_pos, a_neg))
        b_pos = abs(pos_sorted[1])
        b_neg = abs(neg_sorted[1])
        if abs(b_pos - b_neg) < merge_threshold:
            b = max(b_pos, b_neg)
        else:
            b = float(max(b_pos, b_neg))
        if abs(a) < merge_threshold:
            a = 0.0
        if abs(b) < merge_threshold:
            b = 0.0
        if a != 0.0 and b != 0.0 and (abs(a - b) < merge_threshold):
            return np.array([-a, a], dtype=float)
        if a == 0.0 and b == 0.0:
            return np.array([0.0], dtype=float)
        elif a == 0.0:
            return np.array([-b, b], dtype=float)
        elif b == 0.0:
            return np.array([-a, a], dtype=float)
        else:
            return np.array([-a, -b, b, a], dtype=float)
    if has_pos or has_neg:
        peaks = pos_peaks if has_pos else -neg_peaks
        peaks_sorted = np.sort(peaks)[::-1]
        if len(peaks_sorted) >= 2:
            a = float(peaks_sorted[0])
            b = float(peaks_sorted[1])
            if abs(a) < merge_threshold:
                a = 0.0
            if abs(b) < merge_threshold:
                b = 0.0
            if a != 0.0 and b != 0.0 and (abs(a - b) < merge_threshold):
                return np.array([-a, a], dtype=float)
            if a == 0.0 and b == 0.0:
                return np.array([0.0], dtype=float)
            elif a == 0.0:
                return np.array([-b, b], dtype=float)
            elif b == 0.0:
                return np.array([-a, a], dtype=float)
            else:
                return np.array([-a, -b, b, a], dtype=float)
        if len(peaks_sorted) == 1:
            b = max_abs
            return np.array([-b, b], dtype=float)
    if has_pos and has_neg:
        b = max_abs
        return np.array([-b, 0.0, b], dtype=float)
    return np.array([0.0], dtype=float)

def build_adaptive_mean_template(raw_means, min_magnitude=0.001):
    raw = np.array(raw_means, dtype=float).reshape(-1)
    if raw.size == 0:
        return np.array([0.0], dtype=float)
    finite_raw = raw[np.isfinite(raw)]
    if finite_raw.size == 0:
        return np.array([0.0], dtype=float)
    max_abs = float(np.max(np.abs(finite_raw)))
    if max_abs < min_magnitude:
        return np.array([0.0], dtype=float)
    has_pos = np.any(finite_raw > 0)
    has_neg = np.any(finite_raw < 0)
    if has_pos and has_neg:
        return np.array([-max_abs, 0.0, max_abs], dtype=float)
    else:
        return np.array([-max_abs, max_abs], dtype=float)

def build_adaptive_mean_template_5peak(raw_means, min_magnitude=0.001):
    raw = np.array(raw_means, dtype=float).reshape(-1)
    if raw.size == 0:
        return np.array([0.0], dtype=float)
    finite_raw = raw[np.isfinite(raw)]
    if finite_raw.size == 0:
        return np.array([0.0], dtype=float)
    max_abs = float(np.max(np.abs(finite_raw)))
    if max_abs < min_magnitude:
        return np.array([0.0], dtype=float)
    pos_peaks = finite_raw[finite_raw > min_magnitude]
    neg_peaks = finite_raw[finite_raw < -min_magnitude]
    merge_threshold = 0.005
    if len(pos_peaks) >= 2:
        pos_sorted = np.sort(pos_peaks)[::-1]
        filtered_pos = [pos_sorted[0]]
        for peak in pos_sorted[1:]:
            if abs(peak - filtered_pos[-1]) >= merge_threshold:
                filtered_pos.append(peak)
        pos_peaks = np.array(filtered_pos)
    if len(neg_peaks) >= 2:
        neg_sorted = np.sort(neg_peaks)
        filtered_neg = [neg_sorted[0]]
        for peak in neg_sorted[1:]:
            if abs(peak - filtered_neg[-1]) >= merge_threshold:
                filtered_neg.append(peak)
        neg_peaks = np.array(filtered_neg)
    has_pos = len(pos_peaks) > 0
    has_neg = len(neg_peaks) > 0
    if has_pos and has_neg and (len(pos_peaks) >= 2) and (len(neg_peaks) >= 2):
        pos_sorted = np.sort(pos_peaks)[::-1]
        neg_sorted = np.sort(neg_peaks)
        a_pos = abs(pos_sorted[0])
        a_neg = abs(neg_sorted[0])
        if abs(a_pos - a_neg) < merge_threshold:
            a = max(a_pos, a_neg)
        else:
            a = float(max(a_pos, a_neg))
        b_pos = abs(pos_sorted[1])
        b_neg = abs(neg_sorted[1])
        if abs(b_pos - b_neg) < merge_threshold:
            b = max(b_pos, b_neg)
        else:
            b = float(max(b_pos, b_neg))
        if abs(a) < merge_threshold:
            a = 0.0
        if abs(b) < merge_threshold:
            b = 0.0
        if a != 0.0 and b != 0.0 and (abs(a - b) < merge_threshold):
            return np.array([-a, 0.0, a], dtype=float)
        if a == 0.0 and b == 0.0:
            return np.array([0.0], dtype=float)
        elif a == 0.0:
            return np.array([-b, 0.0, b], dtype=float)
        elif b == 0.0:
            return np.array([-a, 0.0, a], dtype=float)
        else:
            return np.array([-a, -b, 0.0, b, a], dtype=float)
    if has_pos and has_neg:
        b = max_abs
        return np.array([-b, 0.0, b], dtype=float)
    if has_pos or has_neg:
        peaks = pos_peaks if has_pos else -neg_peaks
        peaks_sorted = np.sort(peaks)[::-1]
        if len(peaks_sorted) >= 2:
            a = float(peaks_sorted[0])
            b = float(peaks_sorted[1])
            if abs(a) < merge_threshold:
                a = 0.0
            if abs(b) < merge_threshold:
                b = 0.0
            if a != 0.0 and b != 0.0 and (abs(a - b) < merge_threshold):
                return np.array([-a, a], dtype=float)
            if a == 0.0 and b == 0.0:
                return np.array([0.0], dtype=float)
            elif a == 0.0:
                return np.array([-b, b], dtype=float)
            elif b == 0.0:
                return np.array([-a, a], dtype=float)
            else:
                return np.array([-a, -b, b, a], dtype=float)
        else:
            b = max_abs
            return np.array([-b, b], dtype=float)
    return np.array([0.0], dtype=float)

def build_adaptive_mean_template_1peak(raw_means=None, min_magnitude=None):
    return np.array([0.0], dtype=float)

def softmin(x, beta=5.0, temp=1.0, epsilon=1e-06):
    x = np.array(x)
    x = x / temp
    weights = np.exp(-beta * x)
    return weights / (np.sum(weights) + epsilon)
import numpy as np
from scipy.special import softmax

def build_symmetric_gmm_template(sigma2_target, mu_gap=0.1, beta=50, n_components=3):
    assert n_components % 2 == 1, 'n_components must be odd for symmetric templates'
    half = n_components // 2
    means = np.array([mu_gap * (i - half) for i in range(n_components)])
    distances = np.abs(means)
    logits = -beta * distances
    weights = softmax(logits)
    initial_stds = np.ones(n_components)
    current_variance = np.sum(weights * (initial_stds ** 2 + means ** 2))
    scale = np.sqrt(sigma2_target / current_variance)
    scaled_stds = initial_stds * scale
    return (means, scaled_stds, weights)

def should_rebuild(layer_name, theta_prev, theta_curr, tau=0.02, tau_c=0.1):
    delta = np.linalg.norm(theta_curr - theta_prev) / (np.linalg.norm(theta_prev) + 1e-08)
    if not hasattr(should_rebuild, 'cumulative_drift_cache'):
        should_rebuild.cumulative_drift_cache = {}
    C = should_rebuild.cumulative_drift_cache.get(layer_name, 0.0)
    C += delta
    if delta > tau or C > tau_c:
        should_rebuild.cumulative_drift_cache[layer_name] = 0.0
        return (True, delta)
    else:
        should_rebuild.cumulative_drift_cache[layer_name] = C
        return (False, delta)
