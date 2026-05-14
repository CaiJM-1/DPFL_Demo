# TUNE - Federated Learning with Differential Privacy Framework

## Project Overview

This project implements the TUNE algorithm, a federated learning framework with adaptive differential privacy protection. The framework is built on PyTorch, uses ResNet-18 as the base model, and supports federated learning training under various data distribution scenarios.

## Key Features

-  **Differential Privacy Protection**: Privacy budget tracking based on RDP (Rényi Differential Privacy)
-  **Adaptive Noise Mechanism**: Generates adaptive noise using Gaussian Mixture Models (GMM)
-  **Sensitivity Estimation**: Supports local sensitivity and batch sensitivity estimation
-  **Multiple Data Distributions**: Supports IID, Dirichlet, Few Classes, Unequal Samples and other distribution modes
-  **Client Sampling & Dropout**: Simulates client participation patterns in real federated learning scenarios
-  **Visualization Tools**: Automatically generates training curves and comparison plots

## Project Structure

```
TUNE/
├── main.py                     # Main entry point
├── config.py                   # Global configuration file
├── federated_training.py       # Federated training main process
├── tune.py                     # TUNE algorithm core implementation
├── initialize.py               # Training initialization (pretraining, sensitivity initialization)
├── global_aggregation.py       # Global model aggregation
├── evaluate_global_model.py    # Global model evaluation
├── data_split.py              # Data splitting and loading
├── resnet18.py                # ResNet-18 model definition
├── noise.py                   # Noise generation (GMM)
├── Sensitivity.py             # Sensitivity estimation
├── rdp_accountant.py          # RDP privacy budget tracking
├── gradient_monitor.py        # Gradient monitoring tool
├── global_state.py            # Global state management
├── plot_results.py            # Results visualization
├── plot_utils.py              # Plotting utility functions
├── plot_test.py               # Plotting tests
├── data/                      # Dataset directory
│   ├── cifar-10-batches-py/  # CIFAR-10 dataset
│   └── FashionMNIST/         # Fashion-MNIST dataset
└── mog_models/               # GMM model cache directory
```

## Dependencies

### Python Version
- Python 3.8+

### Main Dependencies
```
torch >= 1.12.0
torchvision >= 0.13.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.5.0
```

### Install Dependencies
```bash
pip install torch torchvision numpy scikit-learn matplotlib
```

## Quick Start

### 1. Basic Usage

Run federated learning training with default configuration:

```bash
python main.py
```

### 2. Configuration

Modify the following key parameters in [config.py](config.py):

#### Dataset Configuration
```python
train_samples = 2000          # Number of training samples per client
test_samples = 1000           # Number of test samples
num_clients = 10              # Number of clients
batch_size = 256              # Batch size
```

#### Data Distribution Mode
```python
data_distribution_mode = 'dirichlet'  # Options: 'iid', 'dirichlet', 'few_classes', 'unequal_samples'
dirichlet_alpha = 0.5         # Dirichlet distribution parameter (smaller = more imbalanced)
```

#### Training Parameters
```python
global_rounds = 100           # Number of global training rounds
local_epochs = 5              # Number of local training epochs
lr = 0.001                    # Learning rate
optimizer = 'Adam'            # Optimizer ('Adam' or 'SGD')
```

#### Differential Privacy Parameters
```python
epsilon = 3.0                 # Privacy budget target
dp_delta = 1e-05             # Differential privacy delta parameter
rdp_alpha = 10               # RDP order
clipping_threshold = 4.0      # Gradient clipping threshold
```

#### Client Behavior Configuration
```python
enable_client_sampling = False    # Enable client sampling
client_sampling_rate = 0.5        # Client sampling rate
enable_client_dropout = False     # Enable client dropout
client_dropout_total_rate = 0.5   # Client dropout rate
```

### 3. Data Distribution Modes Explained

#### IID (Independent and Identically Distributed)
```python
data_distribution_mode = 'iid'
```
All clients have the same data distribution.

#### Dirichlet Distribution (Non-IID)
```python
data_distribution_mode = 'dirichlet'
dirichlet_alpha = 0.5  # Smaller alpha = more imbalanced distribution
```
Uses Dirichlet distribution to simulate data heterogeneity in real scenarios.

#### Few Classes
```python
data_distribution_mode = 'few_classes'
few_classes_per_client = 2  # Number of classes per client
```
Each client only has data from a few classes.

#### Unequal Samples
```python
data_distribution_mode = 'unequal_samples'
unequal_sample_ratios = [(0.3, 500), (0.4, 2000), (0.3, 3500)]
```
Different clients have different amounts of samples.

## Core Algorithm

### TUNE Algorithm Workflow

1. **Initialization Phase**
   - Pretraining: Each client trains independently to obtain initial model
   - Sensitivity Estimation: Calculate global sensitivity baseline

2. **Federated Training Phase** (per round)
   - Client local training
   - Local sensitivity estimation (based on gradient history)
   - Fuse local and global sensitivity
   - Adaptive GMM noise generation
   - Upload updates after adding noise
   - Global aggregation
   - RDP privacy budget tracking

3. **Evaluation Phase**
   - Evaluate global model on test set
   - Record accuracy and loss

### Differential Privacy Mechanism

- **RDP Accountant**: Precisely tracks privacy budget consumption
- **Adaptive Noise**: Dynamically adjusts noise parameters based on gradient distribution
- **Sensitivity Fusion**: Combines historical gradient information with global sensitivity

## Results Visualization

After training completes, the following plots are automatically generated:

- `comparison_plot.png`: Strategy comparison plot
- `loss_curve.png`: Loss curve
- `accuracy_curve.png`: Accuracy curve

You can adjust the image resolution by modifying the `dpi` parameter in [plot_results.py](plot_results.py):

```python
plot_results(histories, dpi=600)  # High resolution output
```

## Advanced Features

### 1. Gradient Clipping

```python
clipping_threshold = 4.0  # Set gradient clipping threshold
```

### 2. Batch Sampling Rate

```python
apply_batch_sample_rate = True
batch_sample_rate = 0.3  # Randomly sample 30% of batches per round
```

### 3. Pretrained Initialization

```python
prefer_pretrained_init = True  # Initialize with pretrained model
ini_epochs = 5  # Number of pretraining epochs
```

### 4. Batch Sensitivity Estimation

```python
use_batch_sensitivity = True  # Enable batch-level sensitivity estimation
```

### 5. Privacy Budget Decay

```python
round_gamma = 0.9999  # Privacy budget decay factor
```

## Experimental Examples

### Example 1: Non-IID Training on CIFAR-10

```python
# config.py
data_distribution_mode = 'dirichlet'
dirichlet_alpha = 0.5
num_clients = 10
global_rounds = 100
epsilon = 3.0
```

```python
# main.py
client_loaders, test_loader, client_class_counts = get_data_loaders(
    seed=108, 
    dataset='cifar10'
)
```

### Example 2: Few Classes Scenario on Fashion-MNIST

```python
# config.py
data_distribution_mode = 'few_classes'
few_classes_per_client = 2
```

```python
# main.py
client_loaders, test_loader, client_class_counts = get_data_loaders(
    seed=108, 
    dataset='fmnist'
)
```

## Output Description

### Training Log Example
```
==================== Round 1/100 ====================
Client 0 | Eps: 0.15 | Loss: 2.3456 | Acc: 10.5%
Client 1 | Eps: 0.16 | Loss: 2.3123 | Acc: 11.2%
...
Global Model | Loss: 2.2890 | Acc: 12.3%
====================================================
```

### Privacy Budget Tracking
The privacy budget of each client is tracked independently. When the target epsilon is reached, the client will be frozen.

## Important Notes

1. **GPU Usage**: If GPU is available, CUDA acceleration will be used automatically
2. **Memory Usage**: GMM models are cached in the `mog_models/` directory, first run may be slow
3. **Random Seed**: Experimental reproducibility can be controlled through the `seed` parameter
4. **Privacy Budget**: Too small epsilon will lead to decreased model performance

## FAQ

### Q1: What to do if training is slow?
- Reduce `global_rounds` or `local_epochs`
- Increase `batch_size`
- Reduce `num_clients`

### Q2: What to do if accuracy is low?
- Increase `epsilon` (relax privacy constraints)
- Adjust learning rate `lr`
- Increase training rounds `global_rounds`
- Try different data distribution modes

### Q3: How to add a new dataset?
Add corresponding data loading logic in [data_split.py](data_split.py), refer to the implementation of MNIST/CIFAR-10.

### Q4: How to modify the model architecture?
Modify the `resnet18_model` function in [resnet18.py](resnet18.py), or replace with another model.


## License

This project is for learning and research purposes only.

