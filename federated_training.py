import os
import random
import time
import torch
from config import *
from resnet18 import resnet18_model
from tune import tune
from initialize import initialize_training
from global_aggregation import global_aggregation
from evaluate_global_model import evaluate_global_model, log_global_evaluation
client_adjusted_gmm_history = {}
frozen_clients = set()

def federated_training(num_rounds, num_clients, client_loaders, test_loader, strategy, client_class_counts):
    if strategy != 'TUNE':
        raise ValueError(f'Unsupported strategy: {strategy}')
    local_history = {'global_loss': [], 'global_accuracy': [], 'total_time': 0.0}
    start_round = 1
    dropped_clients = set()
    dropout_schedule = {}
    if getattr(Config, 'enable_client_dropout', False):
        total_rate = float(getattr(Config, 'client_dropout_total_rate', 0.0))
        if total_rate > 0:
            total_drop = int(round(num_clients * total_rate))
            total_drop = min(total_drop, num_clients)
            if total_drop > 0:
                rng = random.Random(getattr(Config, 'client_dropout_seed', 42))
                candidates = list(range(num_clients))
                rng.shuffle(candidates)
                selected = candidates[:total_drop]
                for c in selected:
                    drop_round = rng.randint(start_round, num_rounds)
                    dropout_schedule.setdefault(drop_round, []).append(c)
                print(f'[Dropout] scheduled {total_drop}/{num_clients} clients')
    if dropout_schedule and start_round > 1:
        for r, clients in dropout_schedule.items():
            if int(r) < start_round:
                for c in clients:
                    dropped_clients.add(c)
    global_model = resnet18_model(num_classes=10).to(Config.device)
    global_model_state_dict = global_model.state_dict()
    global client_adjusted_gmm_history
    client_adjusted_gmm_history = {}
    client_rho_targets = {i: Config.epsilon for i in range(num_clients)}
    client_privacy_history = {i: [] for i in range(num_clients)}
    global_sensitivity = initialize_training(client_loaders, num_clients, global_model_state_dict, rho_targets_dict=client_rho_targets)
    total_start_time = time.time()
    for round_num in range(start_round, num_rounds + 1):
        try:
            if dropout_schedule and round_num in dropout_schedule:
                newly_dropped = []
                for c in dropout_schedule[round_num]:
                    if c not in dropped_clients:
                        dropped_clients.add(c)
                        newly_dropped.append(c)
                if newly_dropped:
                    print(f'[Dropout] round {round_num} dropped clients: {sorted(newly_dropped)}')
            active_clients = [c for c in range(num_clients) if c not in frozen_clients and c not in dropped_clients]
            if not active_clients:
                break
            local_sensitivities = []
            local_models_state_dicts = []
            _attempted = 0
            selected_clients = active_clients
            if getattr(Config, 'enable_client_sampling', False) and active_clients:
                rate = float(getattr(Config, 'client_sampling_rate', 1.0))
                if rate < 1.0:
                    if rate <= 0:
                        selected_clients = []
                    else:
                        k = max(1, int(round(len(active_clients) * rate)))
                        k = min(k, len(active_clients))
                        rng = random.Random(getattr(Config, 'client_sampling_seed', 42) + round_num)
                        selected_clients = sorted(rng.sample(active_clients, k))
                    print(f'[Sampling] round {round_num} selected: {selected_clients} (active={len(active_clients)}, rate={rate})')
            for client in selected_clients:
                if client in frozen_clients:
                    continue
                _attempted += 1
                try:
                    round_gamma = getattr(Config, 'round_gamma', 0.999)
                    lr_this_round = float(Config.lr) * round_gamma ** (round_num - 1)
                    local_result = tune(client, client_loaders, round_num, global_model_state_dict, client_rho_targets[client], global_sensitivity, lr_this_round=lr_this_round)
                    if local_result is not None:
                        if 'adjusted_gmm' in local_result:
                            client_id = local_result['client_id']
                            if client_id not in client_adjusted_gmm_history:
                                client_adjusted_gmm_history[client_id] = []
                            client_adjusted_gmm_history[client_id].append(local_result['adjusted_gmm'])
                        if local_result['model_state_dict'] is not None:
                            local_models_state_dicts.append(local_result['model_state_dict'])
                        local_sensitivities.append({'client_id': local_result['client_id'], 'sensitivity': local_result['local_sensitivity']})
                        if 'privacy_history' in local_result:
                            client_privacy_history[client].extend(local_result['privacy_history'])
                except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as e:
                    error_msg = str(e).lower()
                    is_resource_error = 'memory' in error_msg or 'cuda' in error_msg or 'allocate' in error_msg or ('out of' in error_msg)
                    if is_resource_error:
                        continue
                    raise
            if not local_models_state_dicts:
                break
            global_model_state_dict, global_sensitivity = global_aggregation(local_models_state_dicts, local_sensitivities)
            global_loss, global_accuracy = evaluate_global_model(global_model_state_dict, test_loader)
            local_history['global_loss'].append(global_loss)
            local_history['global_accuracy'].append(global_accuracy)
            log_global_evaluation(round_num, global_loss, global_accuracy, strategy)
        except (RuntimeError, torch.cuda.OutOfMemoryError, MemoryError) as e:
            error_msg = str(e).lower()
            is_resource_error = 'memory' in error_msg or 'cuda' in error_msg or 'allocate' in error_msg or ('out of' in error_msg)
            if is_resource_error:
                continue
            raise
    local_history['total_time'] = time.time() - total_start_time
    return local_history
