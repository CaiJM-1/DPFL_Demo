import torch
import numpy as np
from config import *
from collections import defaultdict
MAX_GRAD_SNAPSHOT_DIM = 4096

def update_grad_history(model, grad_history_dict, grad_norm_history_dict, client_loader, client_id, history_window_epoch=3):
    width_dict = defaultdict(float)
    grad_history_snapshot = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            full_grad = param.grad.detach().cpu().numpy().reshape(-1)
            if full_grad.size > MAX_GRAD_SNAPSHOT_DIM:
                idx = np.linspace(0, full_grad.size - 1, MAX_GRAD_SNAPSHOT_DIM, dtype=int)
                grad_snapshot = full_grad[idx]
            else:
                grad_snapshot = full_grad.copy()
            grad_history_dict.setdefault(name, []).append(grad_snapshot)
            grad_history_snapshot[name] = grad_snapshot.copy()
            grad_norm_history_dict.setdefault(name, []).append(np.linalg.norm(full_grad))
            max_snapshots = max(1, int(len(client_loader) * history_window_epoch))
            if len(grad_history_dict[name]) > max_snapshots:
                grad_history_dict[name] = grad_history_dict[name][-max_snapshots:]
    return (width_dict, grad_history_snapshot)

def estimate_batch_sensitivity_from_snapshot(grad_snapshot, client_id=None, quantile=0.95, ema_decay=0.8, clipping_threshold=None):
    if not grad_snapshot:
        return {}
    if not hasattr(estimate_batch_sensitivity_from_snapshot, 'ema_state'):
        estimate_batch_sensitivity_from_snapshot.ema_state = {}
    sens_dict = {}
    for name, grad_array in grad_snapshot.items():
        if grad_array is None:
            continue
        if isinstance(grad_array, torch.Tensor):
            vals = grad_array.detach().cpu().numpy().reshape(-1)
        else:
            vals = np.asarray(grad_array).reshape(-1)
        if vals.size == 0:
            continue
        abs_vals = np.abs(vals)
        q_val = np.quantile(abs_vals, quantile)
        sens_value = float(q_val)
        if clipping_threshold is not None and clipping_threshold > 0:
            sens_value = min(sens_value, clipping_threshold)
        key = (client_id, name) if client_id is not None else name
        prev = estimate_batch_sensitivity_from_snapshot.ema_state.get(key, sens_value)
        smoothed = ema_decay * prev + (1 - ema_decay) * sens_value
        estimate_batch_sensitivity_from_snapshot.ema_state[key] = smoothed
        sens_dict[name] = smoothed
    return sens_dict

def robust_normalize(x_list, lower_q=5, upper_q=95, eps=1e-08):
    if len(x_list) == 0:
        return 0.0
    low = np.percentile(x_list, lower_q)
    high = np.percentile(x_list, upper_q)
    scale = max(high - low, eps)
    return float(np.clip((x_list[-1] - low) / scale, 0.0, 1.0))

def estimate_local_sensitivity(model, grad_norm_history_dict, grad_history_dict=None, quantile=0.98, alpha=0.5, beta=0.5, ema_decay=0.9):
    if not hasattr(estimate_local_sensitivity, 'ema_state'):
        estimate_local_sensitivity.ema_state = {}
    if not hasattr(estimate_local_sensitivity, 'fisher_hist'):
        estimate_local_sensitivity.fisher_hist = {}
    if not hasattr(estimate_local_sensitivity, 'var_hist'):
        estimate_local_sensitivity.var_hist = {}
    sensitivity_dict = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        norms = np.array(grad_norm_history_dict.get(name, []), dtype=np.float64)
        if len(norms) > 0:
            S_local = float(np.quantile(norms, quantile))
        else:
            S_local = 1.0
        grads = grad_history_dict.get(name, []) if grad_history_dict is not None else []
        if len(grads) > 0:
            grads_np = np.stack([g.cpu().numpy().flatten() if isinstance(g, torch.Tensor) else np.asarray(g).flatten() for g in grads])
            grad_sq_mean = np.mean(grads_np ** 2)
            fisher_val = float(np.mean(np.mean(grads_np ** 2, axis=0)))
            var_val = float(np.mean(np.var(grads_np, axis=0)))
        else:
            grad_sq_mean, fisher_val, var_val = (1.0, 0.0, 0.0)
        ema_prev = estimate_local_sensitivity.ema_state.get(name, fisher_val)
        fisher_ema = ema_decay * ema_prev + (1 - ema_decay) * fisher_val
        estimate_local_sensitivity.ema_state[name] = fisher_ema
        estimate_local_sensitivity.fisher_hist.setdefault(name, []).append(fisher_ema)
        estimate_local_sensitivity.var_hist.setdefault(name, []).append(var_val)
        fisher_norm = robust_normalize(estimate_local_sensitivity.fisher_hist[name])
        var_norm = robust_normalize(estimate_local_sensitivity.var_hist[name])
        S_tilde_sq = S_local * (1 + alpha * fisher_norm) * (1 + beta * var_norm)
        sensitivity_dict[name] = float(np.sqrt(max(S_tilde_sq, 0.0)))
    return sensitivity_dict
from collections import defaultdict
import numpy as np

def aggregate_global_sensitivity(local_sensitivities, method='mean'):
    layer_values = defaultdict(list)
    for client_entry in local_sensitivities:
        sensitivity = client_entry['sensitivity']
        if sensitivity is None:
            continue
        for layer, value in sensitivity.items():
            layer_values[layer].append(value)
    global_sensitivity = {}
    for layer, values in layer_values.items():
        if method == 'mean':
            global_sensitivity[layer] = float(np.mean(values))
        elif method == 'median':
            global_sensitivity[layer] = float(np.median(values))
        else:
            raise ValueError(f'Unsupported aggregation method: {method}')
    return global_sensitivity

def fuse_local_global_sensitivity(local_sens, global_sens):
    lambda_t = Config.lambda_t
    fused_sens = {}
    for name in local_sens:
        s_local = local_sens[name]
        s_global = global_sens.get(name, 0.0)
        fused_sens[name] = lambda_t * s_local + (1 - lambda_t) * s_global
    return fused_sens
