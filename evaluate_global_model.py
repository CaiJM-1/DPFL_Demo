import torch
import os
from config import *
from resnet18 import resnet18_model
from torch.nn import CrossEntropyLoss

def evaluate_global_model(global_model_state_dict, test_loader):
    model = resnet18_model(num_classes=10).to(Config.device)
    model.load_state_dict(global_model_state_dict)
    loss_fn = CrossEntropyLoss().to(Config.device)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = (imgs.to(Config.device), targets.to(Config.device))
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            predictions = outputs.argmax(dim=1)
            total_correct += (predictions == targets).sum().item()
            total_samples += targets.size(0)
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_samples
    print(f'Accuracy: {accuracy:.4%}')
    return (avg_loss, accuracy)

def log_global_evaluation(round_num, loss, accuracy, strategy, epsilon=None, cumulative_epsilon=None, log_file='test.log'):
    return
