import torch
import os
class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print(f'Training on CUDA: {torch.cuda.get_device_name(0)}')
    else:
        print('Training on CPU')
    train_samples = 2000
    test_samples = 1000
    num_clients = 10
    batch_size = 256
    clipping_threshold = 4.0
    data_distribution_mode = 'dirichlet'
    few_classes_per_client = 2
    few_classes_random_assign = False
    few_classes_custom_assignment = {0: [0, 1], 1: [2, 3], 2: [4, 5]}
    unequal_sample_ratios = [(0.3, 500), (0.4, 2000), (0.3, 3500)]
    unequal_sample_balanced_classes = False
    apply_batch_sample_rate = False
    batch_sample_rate = 0.3
    enable_client_sampling = False
    client_sampling_rate = 0.5
    client_sampling_seed = 42
    enable_client_dropout = False
    client_dropout_total_rate = 0.5
    client_dropout_seed = 42
    global_rounds = 100
    local_epochs = 5
    ini_epochs = 5
    lr = 0.001
    dirichlet_alpha = 0.5
    epsilon = 3.0
    dp_epsilon_max = 1000000000000
    dp_delta = 1e-05
    rdp_alpha = 10
    round_gamma = 0.9999
    lambda_t = 0.6
    lambda_soft = 10
    optimizer = 'Adam'
    momentum = 0.9
    weight_decay = 0.0
    beta1 = 0.9
    beta2 = 0.999
    prefer_pretrained_init = True
    use_batch_sensitivity = True
    output_root = os.path.abspath(os.path.dirname(__file__))
    data_root = './data'
    dataset_download = False
