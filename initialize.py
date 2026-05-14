from tune import tune_init_train
from Sensitivity import aggregate_global_sensitivity

def initialize_training(client_loaders, num_clients, global_model_state_dict, rho_targets_dict):
    local_sensitivities = []
    local_models_state_dicts = []
    for client in range(num_clients):
        rho_target = rho_targets_dict[client]
        local_result = tune_init_train(client, client_loaders, global_model_state_dict, rho_target)
        local_models_state_dicts.append(local_result['model_state_dict'])
        if local_result['local_sensitivity'] is not None:
            local_sensitivities.append({'client_id': local_result['client_id'], 'sensitivity': local_result['local_sensitivity']})
    global_sensitivity = aggregate_global_sensitivity(local_sensitivities, method='mean')
    return global_sensitivity
