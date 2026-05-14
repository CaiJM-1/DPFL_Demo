import torch
from torch.nn import CrossEntropyLoss
import time
import math
from config import *
from resnet18 import resnet18_model
from Sensitivity import estimate_local_sensitivity, fuse_local_global_sensitivity, update_grad_history, estimate_batch_sensitivity_from_snapshot
from noise import generate_mog_noise, adjust_gmm_variance, build_symmetric_gmm_template
import os
import pickle
import numpy as np
from rdp_accountant import RDPAccountant
from collections import defaultdict
GMM_CACHE = {}
client_privacy_managers = {}
client_round_counter = {}
client_adjusted_gmm_history = {}
client_rdp_accountants = {}
client_grad_history = {}
client_grad_norm_history = {}
client_prev_fused_sens = {}
client_prev_sigma_sq = {}
gmm_dict = {}
client_sigma2_init = {}
total_rdp_per_client = {}
frozen_clients = set()
client_pretrained_states = {}
client_cumulative_sigma_sq = defaultdict(float)
client_cumulative_delta_sq = defaultdict(float)

def tune(client, client_loaders, round_num, global_model_state_dict, rho_target, global_sensitivity, lr_this_round=None, optim_state=None, test_loader=None):
    global frozen_clients
    if client in frozen_clients:
        return None
    model = resnet18_model(num_classes=10).to(Config.device)
    pretrained_state = client_pretrained_states.get(client)
    prefer_pretrained = getattr(Config, 'prefer_pretrained_init', False) and pretrained_state is not None and (round_num == 1)
    if prefer_pretrained:
        model.load_state_dict(pretrained_state)
    elif global_model_state_dict is not None:
        model.load_state_dict(global_model_state_dict)
    grad_history_dict = client_grad_history.get(client, {})
    grad_norm_history_dict = client_grad_norm_history.get(client, {})
    train_loader = client_loaders[client]['train']
    batches_per_epoch = _estimate_batches_per_epoch(train_loader)
    steps_per_round = max(1, batches_per_epoch * Config.local_epochs)
    epsilon_per_step = None
    if round_num == 1:
        client_cumulative_sigma_sq[client] = 0.0
        client_cumulative_delta_sq[client] = 0.0
    layer_names = [name for name, param in model.named_parameters()]
    if client not in client_rdp_accountants:
        client_rdp_accountants[client] = RDPAccountant(layer_names)
    rdp_accountant = client_rdp_accountants[client]
    if client not in client_grad_history:
        client_grad_history[client] = {}
    grad_history_dict = client_grad_history[client]
    if client not in client_grad_norm_history:
        client_grad_norm_history[client] = {}
    grad_norm_history_dict = client_grad_norm_history[client]
    privacy_history = []
    if client in GMM_CACHE:
        del GMM_CACHE[client]
    if client not in GMM_CACHE:
        gmm_model_path = os.path.join(Config.output_root, 'mog_models', f'client_{client}_mog.pkl')
        if os.path.exists(gmm_model_path):
            try:
                with open(gmm_model_path, 'rb') as f:
                    GMM_CACHE[client] = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f'[Client {client}] Failed to load GMM model: {e}')
        else:
            raise FileNotFoundError(f'[Error] GMM model for client {client} not found. Run initialization first.')
    gmm_dict = GMM_CACHE.get(client, {})
    layer_width_dict = {}
    loss = CrossEntropyLoss().cuda()
    use_lr = float(lr_this_round) if lr_this_round is not None else float(Config.lr)
    if Config.optimizer.upper() == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=use_lr, betas=(Config.beta1, Config.beta2), weight_decay=Config.weight_decay)
    start_time = time.time()
    fused_sens, adjusted_gmm_dict, layer_noise_energy_dict = prepare_noise_for_round(client=client, model=model, global_sensitivity=global_sensitivity, grad_norm_history_dict=grad_norm_history_dict, gmm_dict=gmm_dict, grad_history_dict=grad_history_dict, layer_width_dict=layer_width_dict, epsilon_per_step=epsilon_per_step)
    actual_variance_accumulator = {name: 0.0 for name in adjusted_gmm_dict.keys()}
    actual_variance_counts = {name: 0 for name in adjusted_gmm_dict.keys()}
    use_batch_sensitivity = getattr(Config, 'use_batch_sensitivity', True)
    rdp_alpha = getattr(Config, 'rdp_alpha', 10)
    target_epsilon_rdp = max(getattr(Config, 'epsilon', 1.0), 1e-06)
    round_variance_accumulator = {name: 0.0 for name in adjusted_gmm_dict.keys()}
    round_sensitivity_sq = {name: 0.0 for name in adjusted_gmm_dict.keys()}
    round_variance_counts = {name: 0 for name in adjusted_gmm_dict.keys()}
    for i in range(Config.local_epochs):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        epoch_variance_accumulator = {name: 0.0 for name in adjusted_gmm_dict.keys()}
        epoch_sensitivity_sq = {name: 0.0 for name in adjusted_gmm_dict.keys()}
        epoch_variance_counts = {name: 0 for name in adjusted_gmm_dict.keys()}
        for batch_idx, data in enumerate(train_loader, 1):
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            optimizer.zero_grad()
            outputs = model(imgs)
            loss_value = loss(outputs, targets)
            loss_value.backward()
            if Config.clipping_threshold is not None and Config.clipping_threshold > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.clipping_threshold)
            layer_width_dict, grad_history_snapshot = update_grad_history(model=model, grad_history_dict=grad_history_dict, grad_norm_history_dict=grad_norm_history_dict, client_loader=train_loader, client_id=client, history_window_epoch=3)
            batch_layer_sensitivities = {}
            if use_batch_sensitivity:
                batch_layer_sensitivities = estimate_batch_sensitivity_from_snapshot(grad_snapshot=grad_history_snapshot, client_id=client, quantile=0.95, ema_decay=0.8, clipping_threshold=Config.clipping_threshold)
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None and (name in gmm_dict):
                    template_entry = adjusted_gmm_dict.get(name, gmm_dict[name])
                    if use_batch_sensitivity:
                        batch_sensitivity = batch_layer_sensitivities.get(name, float(fused_sens.get(name, 1.0)))
                    else:
                        batch_sensitivity = float(fused_sens.get(name, 1.0))
                    if Config.clipping_threshold is not None and Config.clipping_threshold > 0:
                        batch_sensitivity = min(batch_sensitivity, Config.clipping_threshold)
                    actual_variance = _apply_mog_noise_from_template(param=param, template_entry=template_entry, sensitivity=batch_sensitivity, target_epsilon_rdp=target_epsilon_rdp, rdp_alpha=rdp_alpha, scale_with_budget=use_batch_sensitivity)
                    epoch_variance_accumulator[name] = epoch_variance_accumulator.get(name, 0.0) + actual_variance
                    epoch_sensitivity_sq[name] = epoch_sensitivity_sq.get(name, 0.0) + batch_sensitivity ** 2
                    epoch_variance_counts[name] = epoch_variance_counts.get(name, 0) + 1
                    actual_variance_accumulator[name] = actual_variance_accumulator.get(name, 0.0) + actual_variance
                    actual_variance_counts[name] = actual_variance_counts.get(name, 0) + 1
            optimizer.step()
            total_loss += loss_value.item() * imgs.size(0)
            total_correct += (targets == outputs.argmax(1)).sum().item()
            total_samples += targets.size(0)
        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        for name in epoch_variance_accumulator:
            round_variance_accumulator[name] = round_variance_accumulator.get(name, 0.0) + epoch_variance_accumulator.get(name, 0.0)
            round_sensitivity_sq[name] = round_sensitivity_sq.get(name, 0.0) + epoch_sensitivity_sq.get(name, 0.0)
            round_variance_counts[name] = round_variance_counts.get(name, 0) + epoch_variance_counts.get(name, 0)
    increment_epsilon, total_epsilon = _log_round_noise_and_privacy(client=client, round_num=round_num, epoch_idx='all', layer_variances=round_variance_accumulator, layer_sens_sq=round_sensitivity_sq, layer_counts=round_variance_counts, fused_sens=fused_sens, epoch_noise_file=None, rdp_log_path=None, rdp_accountant=rdp_accountant)
    if increment_epsilon is not None:
        privacy_history.append({'round': round_num, 'epoch': 'all', 'increment_epsilon': increment_epsilon, 'total_epsilon': total_epsilon})
    try:
        from federated_training import frozen_clients as global_frozen_clients
        threshold = getattr(Config, 'dp_epsilon_max', None)
        if threshold is None:
            threshold = getattr(Config, 'rdp_max', None)
        if total_epsilon is not None and threshold is not None and (total_epsilon > threshold):
            global_frozen_clients.add(client)
            return None
    except Exception:
        pass
    avg_noise_energy = {}
    for name, total_var in actual_variance_accumulator.items():
        count = actual_variance_counts.get(name, 0)
        if count > 0:
            avg_noise_energy[name] = total_var / count
    local_sensitivity = estimate_local_sensitivity(model, grad_norm_history_dict, quantile=0.95)
    return {'client_id': client, 'model_state_dict': model.state_dict(), 'local_sensitivity': local_sensitivity, 'adjusted_gmm': adjusted_gmm_dict, 'privacy_history': privacy_history}

def prepare_noise_for_round(client, model, global_sensitivity, grad_norm_history_dict, gmm_dict, grad_history_dict, layer_width_dict, epsilon_per_step=None):
    local_sensitivity = estimate_local_sensitivity(model, grad_norm_history_dict, grad_history_dict, quantile=0.99)
    fused_sens = fuse_local_global_sensitivity(local_sensitivity, global_sensitivity)
    if client not in client_prev_fused_sens:
        client_prev_fused_sens[client] = {}
    for name in fused_sens:
        client_prev_fused_sens[client][name] = fused_sens[name]
    round_num = client_round_counter.get(client, 0)
    THRESHOLD_MODE = 'both'
    adjusted_gmm_dict, layer_noise_energy_dict = adjust_gmm_variance(gmm_dict=gmm_dict, fused_sens=fused_sens, client_id=client, client_sigma2_init=client_sigma2_init, layer_width_dict=layer_width_dict, grad_norm_history_dict=grad_norm_history_dict, grad_history_dict=grad_history_dict, client_step=round_num, epsilon_per_step=epsilon_per_step, max_peaks=3, threshold_mode=THRESHOLD_MODE)
    GMM_CACHE[client] = adjusted_gmm_dict
    save_dir = os.path.join(Config.output_root, 'mog_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'client_{client}_mog.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(adjusted_gmm_dict, f)
    display_round = round_num + 1
    log_sensitivity_and_variance(client_id=client, round_label=display_round, fused_sens=fused_sens, layer_variance_dict=layer_noise_energy_dict, phase='TRAIN')
    client_round_counter[client] = round_num + 1
    return (fused_sens, adjusted_gmm_dict, layer_noise_energy_dict)

def initialize_gmm_model(client, model, rho_target):
    GMM_CACHE.pop(client, None)
    T = Config.global_rounds
    alpha = Config.rdp_alpha
    L = len([p for p in model.parameters() if p.requires_grad])
    E = Config.local_epochs
    rho_total = rho_target if rho_target is not None and rho_target > 0 else Config.epsilon
    sigma2_init = alpha / (2 * rho_total)
    client_sigma2_init[client] = {'sigma2_init': sigma2_init, 'sigma2_pre': {}}
    local_sensitivity = {name: 1.0 for name, param in model.named_parameters() if param.requires_grad}
    global_sensitivity = {name: 1.0 for name in local_sensitivity.keys()}
    fused_sens = fuse_local_global_sensitivity(local_sensitivity, global_sensitivity)
    gmm_results = {}
    for layer_name in fused_sens.keys():
        sigma2_target = sigma2_init * fused_sens[layer_name] ** 2
        means, stds, weights = build_symmetric_gmm_template(sigma2_target=sigma2_target, mu_gap=0.001, beta=50, n_components=1)
        gmm_results[layer_name] = {'means': means, 'stds': stds, 'weights': weights}
    init_noise_variance = {}
    for layer_name, params in gmm_results.items():
        weights = np.array(params['weights'])
        stds = np.array(params['stds'])
        init_noise_variance[layer_name] = float(np.sum(weights * stds ** 2))
    log_sensitivity_and_variance(client_id=client, round_label='INIT', fused_sens=fused_sens, layer_variance_dict=init_noise_variance, phase='INIT')
    GMM_CACHE[client] = gmm_results
    save_dir = os.path.join(Config.output_root, 'mog_models')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'client_{client}_mog.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(gmm_results, f)

def tune_init_train(client, client_loaders, global_model_state_dict, rho_target):
    model = resnet18_model(num_classes=10).to(Config.device)
    train_loader = client_loaders[client]['train']
    batches_per_epoch = _estimate_batches_per_epoch(train_loader)
    steps_per_round = max(1, batches_per_epoch * Config.local_epochs)
    initialize_gmm_model(client=client, model=model, rho_target=rho_target)

    if global_model_state_dict is not None:
        model.load_state_dict(global_model_state_dict)



    if client not in client_grad_history:
        client_grad_history[client] = {}
    grad_history_dict = client_grad_history[client]

    if client not in client_grad_norm_history:
        client_grad_norm_history[client] = {}
    grad_norm_history_dict = client_grad_norm_history[client]

        
    loss=CrossEntropyLoss().cuda()

    optim = torch.optim.Adam(model.parameters(), lr=Config.lr)
    start_time=time.time()

    layer_names = [name for name, param in model.named_parameters()]
    

    if client not in client_rdp_accountants:
        client_rdp_accountants[client] = RDPAccountant(layer_names)
    rdp_accountant = client_rdp_accountants[client]
    

    if client not in GMM_CACHE:
        gmm_model_path = os.path.join(Config.output_root, "mog_models", f"client_{client}_mog.pkl")
        if os.path.exists(gmm_model_path):
            try:
                with open(gmm_model_path, "rb") as f:
                    GMM_CACHE[client] = pickle.load(f)
            except Exception as e:
                raise RuntimeError(f'[Client {client}] Failed to load GMM model: {e}')
        else:
            raise FileNotFoundError(f'[Error] GMM model for client {client} not found. Run initialization first.')

    gmm_dict = GMM_CACHE.get(client, {})

    layer_width_dict = {}
    use_batch_sensitivity = getattr(Config, "use_batch_sensitivity", True)
    rdp_alpha = getattr(Config, "rdp_alpha", 10)
    target_epsilon_rdp = max(getattr(Config, "epsilon", 1.0), 1e-6)
    
   

    for i in range(Config.ini_epochs):
        model.train() 
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
                     
        for batch_idx, data in enumerate(train_loader, 1):
            imgs,targets=data
            imgs=imgs.cuda()
            targets=targets.cuda() 
            optim.zero_grad()
            outputs=model(imgs)
            result=loss(outputs,targets)
            acc=(targets==outputs.argmax(1)).float().mean()
            result.backward()
            
          
            total_loss += result.item() * imgs.size(0)
            total_correct += (targets == outputs.argmax(1)).sum().item()
            total_samples += targets.size(0)

    
        
            if Config.clipping_threshold is not None and Config.clipping_threshold > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.clipping_threshold)

          
            layer_width_dict, grad_history_snapshot = update_grad_history(
                model=model,
                grad_history_dict=grad_history_dict,
                grad_norm_history_dict=grad_norm_history_dict,
                client_loader=train_loader,
                client_id=client,
                history_window_epoch=3
            )

            batch_layer_sensitivities = {}
            if use_batch_sensitivity:
                batch_layer_sensitivities = estimate_batch_sensitivity_from_snapshot(
                    grad_snapshot=grad_history_snapshot,
                    client_id=client,
                    quantile=0.95,
                    ema_decay=0.8,
                    clipping_threshold=Config.clipping_threshold
                )

            optim.step()

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
       
    end_time=time.time()   
   
    local_sensitivity = estimate_local_sensitivity(model, grad_norm_history_dict, grad_history_dict, quantile=0.99)

    
    client_pretrained_states[client] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    return {
    'client_id': client,
    'model_state_dict': model.state_dict(),
    'local_sensitivity': local_sensitivity
}

def _apply_mog_noise_from_template(param, template_entry, sensitivity, target_epsilon_rdp, rdp_alpha, scale_with_budget=True):
    means = np.array(template_entry['means'])
    stds_template = np.array(template_entry['stds'])
    weights = np.array(template_entry['weights'])
    final_stds = stds_template
    if scale_with_budget and target_epsilon_rdp is not None and (rdp_alpha is not None):
        template_sigma = math.sqrt(max(np.sum(weights * stds_template ** 2), 1e-12))
        if template_sigma > 0:
            sigma_scale = math.sqrt(max(rdp_alpha, 1e-12) / (2 * max(target_epsilon_rdp, 1e-12)))
            target_sigma = sensitivity * sigma_scale
            scale = max(target_sigma / template_sigma, 0.0)
            final_stds = stds_template * scale
    noise = generate_mog_noise(param.grad.shape, means=means.tolist(), stds=final_stds.tolist(), weights=weights.tolist()).to(param.grad.device)
    param.grad += noise
    actual_variance = float(np.sum(weights * final_stds ** 2))
    return actual_variance

def _log_round_noise_and_privacy(client, round_num, epoch_idx, layer_variances, layer_sens_sq, layer_counts, fused_sens, epoch_noise_file, rdp_log_path, rdp_accountant):
    if not layer_variances:
        return (None, None)
    if epoch_noise_file:
        with open(epoch_noise_file, 'a', encoding='utf-8') as f:
            f.write(f'[TRAIN] round {round_num}, local_epoch {epoch_idx}\n')
            for layer_name, variance in layer_variances.items():
                sens_sq = layer_sens_sq.get(layer_name)
                if sens_sq is None:
                    sens = fused_sens.get(layer_name, 1.0)
                    count = layer_counts.get(layer_name, 1)
                    sens_sq = sens ** 2 * max(count, 1)
                batch_count = layer_counts.get(layer_name, 0)
                f.write(f'  {layer_name}: ={variance:.6f}, ={sens_sq:.6f}, ={batch_count}\n')
    default_sens = Config.clipping_threshold if Config.clipping_threshold is not None else 1.0
    global_sigma_sq = 0.0
    global_delta_sq = 0.0
    for name, variance in layer_variances.items():
        global_sigma_sq += max(variance, 1e-12)
        sens_sq = layer_sens_sq.get(name)
        if sens_sq is None or sens_sq <= 0.0:
            sens = fused_sens.get(name, default_sens)
            count = layer_counts.get(name, 1)
            sens_sq = sens ** 2 * max(count, 1)
        global_delta_sq += max(sens_sq, 1e-12)
    client_cumulative_sigma_sq[client] += global_sigma_sq
    client_cumulative_delta_sq[client] += global_delta_sq
    global_sigma = math.sqrt(max(global_sigma_sq, 1e-12))
    global_delta = math.sqrt(max(global_delta_sq, 1e-12))
    alpha = getattr(Config, 'rdp_alpha', 10)
    cumulative_var_based_epsilon = alpha * client_cumulative_delta_sq[client] / (2 * max(client_cumulative_sigma_sq[client], 1e-12))
    increment_rdp, increment_epsilon, total_rdp, total_epsilon = rdp_accountant.accumulate_and_get_global_rdp_increment_and_total(sigma=global_sigma, sensitivity=global_delta, steps=1)
    if rdp_log_path:
        with open(rdp_log_path, 'a', encoding='utf-8') as rdp_log:
            rdp_log.write(f'Round {round_num}, epoch {epoch_idx}: delta_epsilon={increment_epsilon:.6f}, total_epsilon={total_epsilon:.6f}, cum_variance_epsilon={cumulative_var_based_epsilon:.6f}\n')
    global total_rdp_per_client
    total_rdp_per_client[client] = total_epsilon
    return (increment_epsilon, total_epsilon)

def log_sensitivity_and_variance(client_id, round_label, fused_sens, layer_variance_dict, phase='TRAIN'):
    return

def _estimate_batches_per_epoch(train_loader):
    try:
        batches = len(train_loader)
        if isinstance(batches, int) and batches > 0:
            return batches
    except TypeError:
        batches = None
    dataset = getattr(train_loader, 'dataset', None)
    batch_size = getattr(train_loader, 'batch_size', None)
    if dataset is not None and batch_size:
        try:
            data_len = len(dataset)
            if data_len > 0 and batch_size > 0:
                return max(1, math.ceil(data_len / batch_size))
        except TypeError:
            pass
    return 1
