from config import *
from data_split import get_data_loaders
from federated_training import federated_training, frozen_clients
from plot_results import plot_results
if __name__ == '__main__':
    client_loaders, test_loader, client_class_counts = get_data_loaders(seed=108, dataset='cifar10')
    strategies = ['TUNE']
    histories = {}
    for strategy in strategies:
        frozen_clients.clear()
        history = federated_training(Config.global_rounds, Config.num_clients, client_loaders, test_loader, strategy, client_class_counts)
        histories[strategy] = history
    plot_results(histories, dpi=600)
