import torch
from Sensitivity import aggregate_global_sensitivity

def global_aggregation(local_models_state_dicts, local_sensitivities):
    global_model_state_dict = local_models_state_dicts[0]
    for key in global_model_state_dict:
        global_model_state_dict[key] = torch.mean(torch.stack([local_model[key].float() for local_model in local_models_state_dicts]), dim=0)
    global_sensitivity = aggregate_global_sensitivity(local_sensitivities, method='mean')
    return (global_model_state_dict, global_sensitivity)
