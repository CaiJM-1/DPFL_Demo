# TUNE - 联邦学习差分隐私框架

## 项目简介

本项目实现了 TUNE（**T**raining with **U**nified **N**oise and **E**nsemble）算法，这是一个带有自适应差分隐私保护的联邦学习框架。该框架基于 PyTorch 实现，使用 ResNet-18 作为基础模型，支持在多种数据分布场景下进行联邦学习训练。

## 核心特性

- ✅ **差分隐私保护**：基于 RDP（Rényi Differential Privacy）的隐私预算追踪
- ✅ **自适应噪声机制**：使用高斯混合模型（GMM）生成自适应噪声
- ✅ **敏感度估计**：支持局部敏感度和批次敏感度估计
- ✅ **多种数据分布**：支持 IID、Dirichlet、Few Classes、Unequal Samples 等分布模式
- ✅ **客户端采样与丢弃**：模拟真实联邦学习场景中的客户端参与模式
- ✅ **可视化工具**：自动生成训练曲线和结果对比图

## 项目结构

```
TUNE/
├── main.py                     # 主程序入口
├── config.py                   # 全局配置文件
├── federated_training.py       # 联邦训练主流程
├── tune.py                     # TUNE 算法核心实现
├── initialize.py               # 训练初始化（预训练、敏感度初始化）
├── global_aggregation.py       # 全局模型聚合
├── evaluate_global_model.py    # 全局模型评估
├── data_split.py              # 数据分割与加载
├── resnet18.py                # ResNet-18 模型定义
├── noise.py                   # 噪声生成（GMM）
├── Sensitivity.py             # 敏感度估计
├── rdp_accountant.py          # RDP 隐私预算追踪
├── gradient_monitor.py        # 梯度监控工具
├── global_state.py            # 全局状态管理
├── plot_results.py            # 结果可视化
├── plot_utils.py              # 绘图工具函数
├── plot_test.py               # 绘图测试
├── data/                      # 数据集目录
│   ├── cifar-10-batches-py/  # CIFAR-10 数据集
│   └── FashionMNIST/         # Fashion-MNIST 数据集
└── mog_models/               # GMM 模型缓存目录
```

## 环境依赖

### Python 版本
- Python 3.8+

### 主要依赖库
```
torch >= 1.12.0
torchvision >= 0.13.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.5.0
```

### 安装依赖
```bash
pip install torch torchvision numpy scikit-learn matplotlib
```

## 快速开始

### 1. 基本使用

运行默认配置的联邦学习训练：

```bash
python main.py
```

### 2. 配置说明

在 [config.py](config.py) 中修改以下关键参数：

#### 数据集配置
```python
train_samples = 2000          # 每个客户端的训练样本数
test_samples = 1000           # 测试样本数
num_clients = 10              # 客户端数量
batch_size = 256              # 批次大小
```

#### 数据分布模式
```python
data_distribution_mode = 'dirichlet'  # 可选：'iid', 'dirichlet', 'few_classes', 'unequal_samples'
dirichlet_alpha = 0.5         # Dirichlet 分布参数（越小越不均衡）
```

#### 训练参数
```python
global_rounds = 100           # 全局训练轮数
local_epochs = 5              # 本地训练轮数
lr = 0.001                    # 学习率
optimizer = 'Adam'            # 优化器（'Adam' 或 'SGD'）
```

#### 差分隐私参数
```python
epsilon = 3.0                 # 隐私预算目标
dp_delta = 1e-05             # 差分隐私 delta 参数
rdp_alpha = 10               # RDP 阶数
clipping_threshold = 4.0      # 梯度裁剪阈值
```

#### 客户端行为配置
```python
enable_client_sampling = False    # 是否启用客户端采样
client_sampling_rate = 0.5        # 客户端采样率
enable_client_dropout = False     # 是否启用客户端丢弃
client_dropout_total_rate = 0.5   # 客户端丢弃率
```

### 3. 数据分布模式详解

#### IID（独立同分布）
```python
data_distribution_mode = 'iid'
```
所有客户端数据分布相同。

#### Dirichlet 分布（非 IID）
```python
data_distribution_mode = 'dirichlet'
dirichlet_alpha = 0.5  # alpha 越小，数据分布越不均衡
```
使用 Dirichlet 分布模拟真实场景中的数据异构性。

#### Few Classes（少类别分布）
```python
data_distribution_mode = 'few_classes'
few_classes_per_client = 2  # 每个客户端拥有的类别数
```
每个客户端只拥有少数几个类别的数据。

#### Unequal Samples（不均衡样本）
```python
data_distribution_mode = 'unequal_samples'
unequal_sample_ratios = [(0.3, 500), (0.4, 2000), (0.3, 3500)]
```
不同客户端拥有不同数量的样本。

## 核心算法

### TUNE 算法流程

1. **初始化阶段**
   - 预训练：各客户端独立训练获得初始模型
   - 敏感度估计：计算全局敏感度基准

2. **联邦训练阶段**（每轮）
   - 客户端本地训练
   - 局部敏感度估计（基于梯度历史）
   - 融合局部与全局敏感度
   - 自适应 GMM 噪声生成
   - 添加噪声后上传更新
   - 全局聚合
   - RDP 隐私预算追踪

3. **评估阶段**
   - 全局模型在测试集上评估
   - 记录准确率和损失

### 差分隐私机制

- **RDP Accountant**：精确追踪隐私预算消耗
- **自适应噪声**：根据梯度分布动态调整噪声参数
- **敏感度融合**：结合历史梯度信息和全局敏感度

## 结果可视化

训练完成后，会自动生成以下图表：

- `comparison_plot.png`：不同策略对比图
- `loss_curve.png`：损失曲线
- `accuracy_curve.png`：准确率曲线

可通过修改 [plot_results.py](plot_results.py) 中的 `dpi` 参数调整图片分辨率：

```python
plot_results(histories, dpi=600)  # 高分辨率输出
```

## 高级功能

### 1. 梯度裁剪

```python
clipping_threshold = 4.0  # 设置梯度裁剪阈值
```

### 2. 批次采样率

```python
apply_batch_sample_rate = True
batch_sample_rate = 0.3  # 每轮随机采样 30% 的批次
```

### 3. 预训练初始化

```python
prefer_pretrained_init = True  # 使用预训练模型初始化
ini_epochs = 5  # 预训练轮数
```

### 4. 批次敏感度估计

```python
use_batch_sensitivity = True  # 启用批次级敏感度估计
```

### 5. 隐私预算衰减

```python
round_gamma = 0.9999  # 隐私预算衰减因子
```

## 实验示例

### 示例 1：CIFAR-10 上的非 IID 训练

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

### 示例 2：Fashion-MNIST 上的 Few Classes 场景

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

## 输出说明

### 训练日志示例
```
==================== Round 1/100 ====================
Client 0 | Eps: 0.15 | Loss: 2.3456 | Acc: 10.5%
Client 1 | Eps: 0.16 | Loss: 2.3123 | Acc: 11.2%
...
Global Model | Loss: 2.2890 | Acc: 12.3%
====================================================
```

### 隐私预算追踪
每个客户端的隐私预算会被独立追踪，当达到目标 epsilon 时，该客户端会被冻结（frozen）。

## 注意事项

1. **GPU 使用**：如果有 GPU，会自动使用 CUDA 加速
2. **内存占用**：GMM 模型会缓存在 `mog_models/` 目录，首次运行可能较慢
3. **随机种子**：可通过 `seed` 参数控制实验可重复性
4. **隐私预算**：过小的 epsilon 会导致模型性能下降

## 常见问题

### Q1: 训练速度很慢怎么办？
- 减少 `global_rounds` 或 `local_epochs`
- 增大 `batch_size`
- 减少 `num_clients`

### Q2: 准确率不高怎么办？
- 增大 `epsilon`（放宽隐私约束）
- 调整学习率 `lr`
- 增加训练轮数 `global_rounds`
- 尝试不同的数据分布模式

### Q3: 如何添加新的数据集？
在 [data_split.py](data_split.py) 中添加对应的数据加载逻辑，参考 MNIST/CIFAR-10 的实现。

### Q4: 如何修改模型架构？
在 [resnet18.py](resnet18.py) 中修改 `resnet18_model` 函数，或替换为其他模型。


## 许可证

本项目仅供学习和研究使用。

