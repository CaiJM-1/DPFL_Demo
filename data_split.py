import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from config import *
from resnet18 import *
import random

def get_data_loaders(seed, dataset):
    mode = Config.data_distribution_mode
    print('=' * 60)
    print(f'Data distribution mode: {mode}')
    print('=' * 60)
    if mode == 'iid':
        client_loaders, test_loader, client_class_counts = data_split(seed, dataset)
    elif mode == 'dirichlet':
        client_loaders, test_loader, client_class_counts = data_split_unequal(seed, dataset)
    elif mode == 'few_classes':
        client_loaders, test_loader, client_class_counts = data_split_few_classes(seed, dataset)
    elif mode == 'unequal_samples':
        client_loaders, test_loader, client_class_counts = data_split_unequal_samples(seed, dataset)
    if hasattr(Config, 'apply_batch_sample_rate') and Config.apply_batch_sample_rate:
        client_loaders = apply_batch_sample_rate(client_loaders, mode)
    return (client_loaders, test_loader, client_class_counts)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'[INFO] Seed fixed to {seed}')

def data_split(seed, dataset):
    set_seed(seed)
    dataset = dataset.lower()
    data_root = Config.data_root
    allow_download = Config.dataset_download
    if dataset == 'mnist':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.MNIST(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.MNIST(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    elif dataset == 'fmnist':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.FashionMNIST(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    elif dataset == 'cifar10':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = torchvision.datasets.CIFAR10(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.CIFAR10(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    train_targets = np.array(train_data.targets)
    test_targets = np.array(test_data.targets)
    num_clients = Config.num_clients
    samples_per_client = Config.train_samples
    samples_per_class_per_client = samples_per_client // num_classes
    assert samples_per_class_per_client * num_classes == samples_per_client, "Each client's sample count must be divisible by the number of classes"
    class_indices = {i: np.where(train_targets == i)[0] for i in range(num_classes)}
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])
    client_loaders = {}
    client_class_counts = {}
    for client_id in range(num_clients):
        client_indices = []
        for cls in range(num_classes):
            selected = class_indices[cls][:samples_per_class_per_client]
            client_indices.extend(selected)
            class_indices[cls] = class_indices[cls][samples_per_class_per_client:]
        np.random.shuffle(client_indices)
        client_subset = Subset(train_data, client_indices)
        train_loader = DataLoader(client_subset, batch_size=Config.batch_size, shuffle=True)
        client_loaders[client_id] = {'train': train_loader}
        client_labels = train_targets[client_indices]
        label_counts = np.bincount(client_labels, minlength=num_classes)
        client_class_counts[client_id] = label_counts.tolist()
        print(f'Client {client_id} train samples: {len(client_indices)}')
        print(f'Label distribution: {label_counts.tolist()}')
    samples_per_class = Config.test_samples // num_classes
    class_indices = {i: np.where(test_targets == i)[0] for i in range(num_classes)}
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])
    selected_test_indices = []
    for cls in range(num_classes):
        selected_test_indices.extend(class_indices[cls][:samples_per_class])
    np.random.shuffle(selected_test_indices)
    test_subset = Subset(test_data, selected_test_indices)
    test_loader = DataLoader(test_subset, batch_size=Config.batch_size, shuffle=False)
    sampled_labels = test_targets[selected_test_indices]
    label_counts = np.bincount(sampled_labels, minlength=num_classes)
    print(f'Test samples: {len(selected_test_indices)}')
    print(f'Test label distribution: {label_counts.tolist()}')
    return (client_loaders, test_loader, client_class_counts)
from torch.distributions.dirichlet import Dirichlet

def data_split_unequal(seed, dataset):
    set_seed(seed)
    dataset = dataset.lower()
    data_root = Config.data_root
    allow_download = Config.dataset_download
    if dataset == 'mnist':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.MNIST(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.MNIST(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    elif dataset == 'fmnist':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.FashionMNIST(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    elif dataset == 'cifar10':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = torchvision.datasets.CIFAR10(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.CIFAR10(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    train_targets = np.array(train_data.targets)
    test_targets = np.array(test_data.targets)
    num_clients = Config.num_clients
    samples_per_client = Config.train_samples
    class_indices = {i: np.where(train_targets == i)[0] for i in range(num_classes)}
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])
    client_loaders = {}
    client_class_counts = {}
    dirichlet = Dirichlet(torch.tensor([Config.dirichlet_alpha] * num_classes))
    for client_id in range(num_clients):
        proportions = dirichlet.sample().numpy()
        proportions = proportions / proportions.sum()
        total_samples = samples_per_client
        client_class_counts[client_id] = {}
        selected_indices = []
        for cls in range(num_classes):
            count = int(proportions[cls] * total_samples)
            cls_samples = class_indices[cls]
            selected_indices.extend(cls_samples[:count])
            class_indices[cls] = cls_samples[count:]
            client_class_counts[client_id][cls] = count
        np.random.shuffle(selected_indices)
        selected_indices = selected_indices[:samples_per_client]
        client_subset = Subset(train_data, selected_indices)
        train_loader = DataLoader(client_subset, batch_size=Config.batch_size, shuffle=True)
        client_loaders[client_id] = {'train': train_loader}
        labels = train_targets[selected_indices]
        print(f'Client {client_id} train samples: {len(selected_indices)}')
        print(f'Label distribution: {np.bincount(labels, minlength=num_classes).tolist()}')
    samples_per_class = Config.test_samples // num_classes
    test_class_indices = {i: np.where(test_targets == i)[0] for i in range(num_classes)}
    for cls in test_class_indices:
        np.random.shuffle(test_class_indices[cls])
    selected_test_indices = []
    for cls in range(num_classes):
        selected_test_indices.extend(test_class_indices[cls][:samples_per_class])
    np.random.shuffle(selected_test_indices)
    global_test_subset = Subset(test_data, selected_test_indices)
    global_test_loader = DataLoader(global_test_subset, batch_size=Config.batch_size, shuffle=False)
    print(f'Test samples: {len(selected_test_indices)}')
    print(f'Test label distribution: {np.bincount(test_targets[selected_test_indices], minlength=num_classes).tolist()}')
    return (client_loaders, global_test_loader, client_class_counts)

def data_split_few_classes(seed, dataset):
    set_seed(seed)
    dataset = dataset.lower()
    data_root = Config.data_root
    allow_download = Config.dataset_download
    if dataset == 'mnist':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.MNIST(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.MNIST(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    elif dataset == 'fmnist':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.FashionMNIST(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    elif dataset == 'cifar10':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = torchvision.datasets.CIFAR10(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.CIFAR10(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    train_targets = np.array(train_data.targets)
    test_targets = np.array(test_data.targets)
    num_clients = Config.num_clients
    samples_per_client = Config.train_samples
    classes_per_client = Config.few_classes_per_client
    random_assign = Config.few_classes_random_assign
    assert classes_per_client <= num_classes, f'Classes per client ({classes_per_client}) cannot exceed total classes ({num_classes})'
    assert samples_per_client % classes_per_client == 0, f'Samples per client ({samples_per_client}) must be divisible by classes per client ({classes_per_client})'
    samples_per_class_per_client = samples_per_client // classes_per_client
    class_indices = {i: np.where(train_targets == i)[0].tolist() for i in range(num_classes)}
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])
    client_loaders = {}
    client_class_counts = {}
    custom_assignment = Config.few_classes_custom_assignment
    if custom_assignment is not None and len(custom_assignment) > 0:
        client_classes = []
        for client_id in range(num_clients):
            if client_id in custom_assignment:
                selected_classes = custom_assignment[client_id]
                if not isinstance(selected_classes, (list, tuple)):
                    raise ValueError(f'Client {client_id} classes must be a list or tuple')
                if len(selected_classes) != classes_per_client:
                    samples_per_class_per_client_dynamic = samples_per_client // len(selected_classes)
                else:
                    samples_per_class_per_client_dynamic = samples_per_class_per_client
                client_classes.append((list(selected_classes), samples_per_class_per_client_dynamic))
            else:
                selected_classes = np.random.choice(num_classes, classes_per_client, replace=False).tolist()
                client_classes.append((selected_classes, samples_per_class_per_client))
    elif random_assign:
        client_classes = []
        for client_id in range(num_clients):
            selected_classes = np.random.choice(num_classes, classes_per_client, replace=False).tolist()
            client_classes.append((selected_classes, samples_per_class_per_client))
    else:
        client_classes = []
        for client_id in range(num_clients):
            start_class = client_id * classes_per_client % num_classes
            selected_classes = [(start_class + i) % num_classes for i in range(classes_per_client)]
            client_classes.append((selected_classes, samples_per_class_per_client))
    for client_id in range(num_clients):
        selected_classes, samples_per_class_current = client_classes[client_id]
        client_indices = []
        for cls in selected_classes:
            available = class_indices[cls]
            if len(available) < samples_per_class_current:
                selected = available[:samples_per_class_current]
            else:
                selected = available[:samples_per_class_current]
            client_indices.extend(selected)
            class_indices[cls] = available[samples_per_class_current:]
        np.random.shuffle(client_indices)
        client_subset = Subset(train_data, client_indices)
        train_loader = DataLoader(client_subset, batch_size=Config.batch_size, shuffle=True)
        client_loaders[client_id] = {'train': train_loader}
        client_labels = train_targets[client_indices]
        label_counts = np.bincount(client_labels, minlength=num_classes)
        client_class_counts[client_id] = label_counts.tolist()
        print(f'Client {client_id} classes: {selected_classes} | samples: {len(client_indices)}')
        print(f'Label distribution: {label_counts.tolist()}')
    samples_per_class = Config.test_samples // num_classes
    test_class_indices = {i: np.where(test_targets == i)[0] for i in range(num_classes)}
    for cls in test_class_indices:
        np.random.shuffle(test_class_indices[cls])
    selected_test_indices = []
    for cls in range(num_classes):
        selected_test_indices.extend(test_class_indices[cls][:samples_per_class])
    np.random.shuffle(selected_test_indices)
    test_subset = Subset(test_data, selected_test_indices)
    test_loader = DataLoader(test_subset, batch_size=Config.batch_size, shuffle=False)
    sampled_labels = test_targets[selected_test_indices]
    label_counts = np.bincount(sampled_labels, minlength=num_classes)
    sampled_labels = test_targets[selected_test_indices]
    label_counts = np.bincount(sampled_labels, minlength=num_classes)
    print(f'Test samples: {len(selected_test_indices)}')
    print(f'Test label distribution: {label_counts.tolist()}')
    return (client_loaders, test_loader, client_class_counts)

class SampledDataLoader:

    def __init__(self, dataloader, sample_rate=1.0):
        self.dataloader = dataloader
        self.sample_rate = max(min(sample_rate, 1.0), 0.0)
        self.total_batches = len(dataloader)
        self.sampled_batches = max(int(self.total_batches * self.sample_rate), 1)

    def __iter__(self):
        batch_count = 0
        for batch in self.dataloader:
            if batch_count >= self.sampled_batches:
                break
            yield batch
            batch_count += 1

    def __len__(self):
        return self.sampled_batches

def apply_batch_sample_rate(client_loaders, distribution_mode):
    num_clients = Config.num_clients
    batch_sample_rate = Config.batch_sample_rate
    if isinstance(batch_sample_rate, (int, float)):
        client_rates = [float(batch_sample_rate)] * num_clients
    elif isinstance(batch_sample_rate, (list, tuple)):
        if len(batch_sample_rate) != num_clients:
            if len(batch_sample_rate) < num_clients:
                client_rates = list(batch_sample_rate) + [0.5] * (num_clients - len(batch_sample_rate))
            else:
                client_rates = list(batch_sample_rate[:num_clients])
        else:
            client_rates = list(batch_sample_rate)
    else:
        raise ValueError(f'batch_sample_rate must be float or list, got {type(batch_sample_rate)}')
    sampled_client_loaders = {}
    for client_id in range(num_clients):
        original_train_loader = client_loaders[client_id]['train']
        sampled_rate = client_rates[client_id]
        sampled_train_loader = SampledDataLoader(original_train_loader, sample_rate=sampled_rate)
        sampled_client_loaders[client_id] = {'train': sampled_train_loader}
    print('========== Batch sampling applied ==========')
    print(f'Original distribution: {distribution_mode}')
    print(f'Client sampling rates: {client_rates}')
    for client_id in range(num_clients):
        original_batches = len(client_loaders[client_id]['train'])
        sampled_batches = len(sampled_client_loaders[client_id]['train'])
        rate = client_rates[client_id]
        print(f'Client {client_id} | rate {rate:.2%} | batches {original_batches} -> {sampled_batches}')
    return sampled_client_loaders

def data_split_unequal_samples(seed, dataset):
    set_seed(seed)
    dataset = dataset.lower()
    data_root = Config.data_root
    allow_download = Config.dataset_download
    if dataset == 'mnist':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.MNIST(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.MNIST(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    elif dataset == 'fmnist':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = torchvision.datasets.FashionMNIST(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.FashionMNIST(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    elif dataset == 'cifar10':
        trans_compose = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = torchvision.datasets.CIFAR10(root=data_root, train=True, transform=trans_compose, download=allow_download)
        test_data = torchvision.datasets.CIFAR10(root=data_root, train=False, transform=trans_compose, download=allow_download)
        num_classes = 10
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    train_targets = np.array(train_data.targets)
    test_targets = np.array(test_data.targets)
    num_clients = Config.num_clients
    sample_ratios = Config.unequal_sample_ratios
    balanced_classes = Config.unequal_sample_balanced_classes
    total_ratio = sum([ratio for ratio, _ in sample_ratios])
    if abs(total_ratio - 1.0) > 0.01:
        print(f'[WARN] sample ratios sum to {total_ratio:.2f}, normalizing.')
        sample_ratios = [(ratio / total_ratio, samples) for ratio, samples in sample_ratios]
    client_sample_assignments = []
    for ratio, sample_count in sample_ratios:
        num_clients_in_group = int(np.round(ratio * num_clients))
        client_sample_assignments.extend([sample_count] * num_clients_in_group)
    while len(client_sample_assignments) < num_clients:
        client_sample_assignments.append(sample_ratios[0][1])
    while len(client_sample_assignments) > num_clients:
        client_sample_assignments.pop()
    np.random.shuffle(client_sample_assignments)
    class_indices = {i: np.where(train_targets == i)[0].tolist() for i in range(num_classes)}
    for cls in class_indices:
        np.random.shuffle(class_indices[cls])
    client_loaders = {}
    client_class_counts = {}
    print('========== Unequal sample distribution ==========')
    print(f'Sample ratios: {sample_ratios}')
    print(f'Balanced classes: {balanced_classes}')
    print(f'Client sample assignments: {client_sample_assignments}')
    for client_id in range(num_clients):
        total_samples = client_sample_assignments[client_id]
        client_indices = []
        if balanced_classes:
            samples_per_class = total_samples // num_classes
            remainder = total_samples % num_classes
            for cls in range(num_classes):
                samples_for_this_class = samples_per_class + (1 if cls < remainder else 0)
                available = class_indices[cls]
                if len(available) < samples_for_this_class:
                    selected = available[:samples_for_this_class]
                else:
                    selected = available[:samples_for_this_class]
                client_indices.extend(selected)
                class_indices[cls] = available[samples_for_this_class:]
        else:
            from torch.distributions.dirichlet import Dirichlet
            dirichlet = Dirichlet(torch.tensor([0.5] * num_classes))
            proportions = dirichlet.sample().numpy()
            proportions = proportions / proportions.sum()
            for cls in range(num_classes):
                samples_for_this_class = int(proportions[cls] * total_samples)
                available = class_indices[cls]
                if len(available) < samples_for_this_class:
                    selected = available[:samples_for_this_class]
                else:
                    selected = available[:samples_for_this_class]
                client_indices.extend(selected)
                class_indices[cls] = available[samples_for_this_class:]
        client_indices = client_indices[:total_samples]
        np.random.shuffle(client_indices)
        client_subset = Subset(train_data, client_indices)
        train_loader = DataLoader(client_subset, batch_size=Config.batch_size, shuffle=True)
        client_loaders[client_id] = {'train': train_loader}
        client_labels = train_targets[client_indices]
        label_counts = np.bincount(client_labels, minlength=num_classes)
        client_class_counts[client_id] = label_counts.tolist()
        print(f'Client {client_id} | target {total_samples} | actual {len(client_indices)}')
        print(f'Label distribution: {label_counts.tolist()}')
    samples_per_class = Config.test_samples // num_classes
    test_class_indices = {i: np.where(test_targets == i)[0] for i in range(num_classes)}
    for cls in test_class_indices:
        np.random.shuffle(test_class_indices[cls])
    selected_test_indices = []
    for cls in range(num_classes):
        selected_test_indices.extend(test_class_indices[cls][:samples_per_class])
    np.random.shuffle(selected_test_indices)
    test_subset = Subset(test_data, selected_test_indices)
    test_loader = DataLoader(test_subset, batch_size=Config.batch_size, shuffle=False)
    sampled_labels = test_targets[selected_test_indices]
    label_counts = np.bincount(sampled_labels, minlength=num_classes)
    print(f'Test samples: {len(selected_test_indices)}')
    print(f'Test label distribution: {label_counts.tolist()}')
    return (client_loaders, test_loader, client_class_counts)
