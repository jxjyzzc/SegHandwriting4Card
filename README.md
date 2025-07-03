# 手写笔迹擦除模型训练指南

## 项目简介

本项目基于 **SegFormer** 语义分割模型，实现试卷中手写笔迹的自动识别与擦除功能。通过深度学习技术，模型能够准确识别试卷图像中的手写内容，并生成相应的掩码，从而实现笔迹的精确擦除。

### 核心特性

- 🎯 **高精度识别**: 基于 SegFormer 的语义分割技术，实现像素级的手写笔迹识别
- 🚀 **高效训练**: 优化的训练流程，支持 GPU 加速训练
- 📊 **实时监控**: 详细的训练进度显示和损失统计
- 💾 **自动保存**: 自动保存最佳模型权重
- 🔧 **灵活配置**: 丰富的命令行参数，支持多种训练配置

## 环境要求

### 系统要求
- Linux 操作系统
- Python 3.7.4
- CUDA 支持的 GPU（推荐）

### 依赖库
- PaddlePaddle = 2.2.2
- OpenCV
- NumPy
- tqdm
- 其他依赖见 `requirements.txt`

## 项目结构

```
SegFormer4Card/
├── train_demo.py          # 主训练脚本
├── README_DEMO.md         # 本说明文档
├── main.ipynb            # 原始 Jupyter Notebook
├── work/                 # 核心代码目录
│   ├── main.py          # 原始训练脚本
│   ├── dataloader.py    # 数据加载器
│   ├── model.py         # 模型定义
│   ├── loss.py          # 损失函数
│   ├── utils.py         # 工具函数
│   └── data/            # 数据目录
│       └── dehw_train_dataset/  # 训练数据集
└── models/              # 模型保存目录（自动创建）
```

## 数据准备

### 数据集获取与解压

数据集文件通常以压缩包形式提供，例如 `dehw_train_dataset.zip` 和 `dehw_testA_dataset.zip`。

请将这些压缩包解压到 `SegFormer4Card/work/data/` 目录下，解压后应形成以下结构：

```
work/data/
├── dehw_train_dataset/  # 训练数据集根目录
│   ├── train.csv            # 训练文件列表
│   ├── val.csv              # 验证文件列表（可选）
│   ├── images/              # 原始图像
│   │   ├── img001.jpg
│   │   └── ...
│   ├── gts/                 # 干净的目标图像
│   │   ├── img001.jpg
│   │   └── ...
│   └── masks/               # 手写笔迹掩码（可选，可自动生成）
│       ├── img001.png
│       └── ...
└── dehw_testA_dataset/  # 测试数据集根目录（如果存在）
    ├── test.csv
    ├── images/
    └── ...
```

**解压示例命令** (在 `SegFormer4Card/` 目录下执行):

```bash
mkdir -p work/data
unzip /path/to/dehw_train_dataset.zip -d work/data/
unzip /path/to/dehw_testA_dataset.zip -d work/data/
```

### 数据集格式

每个数据集（如 `dehw_train_dataset`）应包含以下子目录和文件：

- `images/`: 存放原始图像文件。
- `gts/`: 存放对应的干净目标图像（即擦除笔迹后的图像）。
- `masks/`: 存放手写笔迹的二值掩码图像（如果未提供，`dataloader.py` 会根据 `images` 和 `gts` 自动生成）。
- `train.csv` / `val.csv` / `test.csv`: 包含图像文件名的列表，每行一个文件名（例如 `images/img001.jpg`）。

**CSV 文件格式示例**:

`train.csv` 和 `val.csv` 文件应包含图像文件相对于其数据集根目录的路径，每行一个路径：

```
images/img001.jpg
images/img002.jpg
images/img003.jpg
...
```

## 快速开始

### 1. 环境检查

首先确保在 `ocr-autoscore` 容器中运行：

```bash
# 检查 PaddlePaddle 安装
python3 -c "import paddle; print(f'PaddlePaddle版本: {paddle.__version__}')"

# 检查 GPU 可用性
python3 -c "import paddle; print(f'GPU数量: {paddle.device.cuda.device_count()}')"
```

### 2. 数据准备

确保训练数据已正确放置在 `/data/dehw_train_dataset/` 目录下。
demo 数据预处理脚本 data_processor.py 可以解包并放置数据集，你可以阅读并按需调整参数。

### 3. 开始训练

#### 基础训练命令

```bash
# 使用默认参数开始训练
python3 train_demo.py
```

#### 自定义参数训练

```bash
# 使用自定义参数训练
python3 train_demo.py \
    --model_arch segformer_b2 \
    --batch_size 16 \
    --epochs 100 \
    --learning_rate 1e-3 \
    --data_root work/data/dehw_train_dataset
```

## 详细参数说明

### 模型相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_arch` | str | `segformer_b2` | 模型架构，可选 `segformer_b1` 或 `segformer_b2` |
| `--pretrained` | bool | `True` | 是否使用预训练权重 |

### 数据相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | str | `work/data/dehw_train_dataset` | 训练数据根目录 |
| `--load_size` | int | `384` | 图像加载尺寸 |
| `--batch_size` | int | `8` | 批次大小 |
| `--num_workers` | int | `4` | 数据加载器工作进程数 |

### 训练相关参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--epochs` | int | `50` | 训练轮数 |
| `--learning_rate` | float | `1e-3` | 初始学习率 |
| `--weight_decay` | float | `1e-3` | 权重衰减系数 |

### 其他参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save_path` | str | `models` | 模型保存路径 |
| `--gpu_id` | int | `0` | GPU 设备 ID |

## 训练过程监控

### 训练输出示例

```
============================================================
手写笔迹擦除模型训练
============================================================
训练配置:
  model_arch: segformer_b2
  pretrained: True
  data_root: work/data/dehw_train_dataset
  load_size: 384
  batch_size: 8
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.001
  save_path: models
  gpu_id: 0
============================================================
使用设备: gpu:0
模型将保存到: models/segformer_b2
正在加载数据集...
训练样本数: 1000
验证样本数: 200
正在构建模型: segformer_b2
模型参数量: 25.37 MB
开始训练...

Epoch 1/50 [Train]: 100%|██████████| 125/125 [02:15<00:00, loss=0.1234, lr=1.00e-03]
Epoch 1/50 [Val]: 100%|██████████| 25/25 [00:30<00:00, loss=0.1156]

Epoch 1/50:
  Train Loss: 0.1234
  Val Loss: 0.1156
  Learning Rate: 1.00e-03
  新的最佳验证损失: 0.1156
最佳模型已保存: models/segformer_b2/best_model_12151430.pdparams
--------------------------------------------------
```

### 关键指标说明

- **Train Loss**: 训练损失，反映模型在训练数据上的拟合程度
- **Val Loss**: 验证损失，反映模型的泛化能力
- **Learning Rate**: 当前学习率，会随训练进度自动衰减

## 模型保存与加载

### 自动保存

训练过程中，当验证损失达到新的最低值时，模型会自动保存到：

```
models/{model_arch}/best_model_{timestamp}.pdparams
```

### 手动加载模型

```python
import paddle
from work.model import segformer_b2

# 加载模型
model = segformer_b2(num_classes=1, pretrained=False)
model_state = paddle.load('models/segformer_b2/best_model_12151430.pdparams')
model.set_state_dict(model_state)
model.eval()
```

## 性能优化建议

### 1. 批次大小调整

- **GPU 内存充足**: 可以增加 `batch_size` 到 16 或 32
- **GPU 内存不足**: 减少 `batch_size` 到 4 或更小

```bash
# 大批次训练（需要更多 GPU 内存）
python3 train_demo.py --batch_size 16

# 小批次训练（适合内存有限的情况）
python3 train_demo.py --batch_size 4
```

### 2. 学习率调整

- **快速收敛**: 使用较大的初始学习率 `1e-2`
- **精细调优**: 使用较小的初始学习率 `1e-4`

```bash
# 快速训练
python3 train_demo.py --learning_rate 1e-2 --epochs 30

# 精细训练
python3 train_demo.py --learning_rate 1e-4 --epochs 100
```

### 3. 模型选择

- **SegFormer-B1**: 更轻量，训练速度快，适合快速实验
- **SegFormer-B2**: 更强大，精度更高，适合最终部署

```bash
# 轻量模型
python3 train_demo.py --model_arch segformer_b1

# 强力模型
python3 train_demo.py --model_arch segformer_b2
```

## 常见问题解决

### 1. 内存不足错误

**错误信息**: `CUDA out of memory`

**解决方案**:
```bash
# 减少批次大小
python3 train_demo.py --batch_size 4

# 减少图像尺寸
python3 train_demo.py --load_size 256

# 减少工作进程数
python3 train_demo.py --num_workers 2
```

### 2. 数据加载错误

**错误信息**: `FileNotFoundError: 数据路径不存在`

**解决方案**:
```bash
# 检查数据路径
ls -la work/data/dehw_train_dataset/

# 指定正确的数据路径
python3 train_demo.py --data_root /path/to/your/dataset
```

### 3. 模型收敛慢

**可能原因**: 学习率过小或数据质量问题

**解决方案**:
```bash
# 增加学习率
python3 train_demo.py --learning_rate 5e-3

# 检查数据质量
python3 -c "from work.dataloader import ErasingData; data = ErasingData('work/data/dehw_train_dataset', (384, 384)); print(f'数据集大小: {len(data)}')"
```

## 进阶使用

### 1. 自定义损失函数

如需修改损失函数，可以编辑 `work/loss.py` 文件：

```python
# 在 work/loss.py 中添加新的损失函数
class CustomMaskLoss(nn.Layer):
    def __init__(self):
        super().__init__()
        # 自定义损失函数实现
        pass
```

### 2. 数据增强

可以在 `work/dataloader.py` 中添加更多数据增强方法：

```python
# 添加新的数据增强函数
def random_rotation(data, max_angle=15):
    # 随机旋转实现
    pass
```

### 3. 模型微调

对于特定领域的数据，可以基于预训练模型进行微调：

```bash
# 使用较小的学习率进行微调
python3 train_demo.py \
    --pretrained True \
    --learning_rate 1e-4 \
    --epochs 20
```

## 结果评估

### 训练完成后的评估

1. **查看训练日志**: 观察损失曲线的变化趋势
2. **验证集性能**: 关注最终的验证损失值
3. **视觉检查**: 使用训练好的模型对测试图像进行预测

### 模型推理示例

```python
import cv2
import numpy as np
import paddle
from work.model import segformer_b2

# 加载训练好的模型
model = segformer_b2(num_classes=1, pretrained=False)
model.set_state_dict(paddle.load('models/segformer_b2/best_model_xxx.pdparams'))
model.eval()

# 加载测试图像
img = cv2.imread('test_image.jpg')
img = cv2.resize(img, (384, 384))
img = img.astype(np.float32) / 255.0
img_tensor = paddle.to_tensor(img.transpose(2, 0, 1)[None, ...])

# 预测
with paddle.no_grad():
    pred_mask = model(1 - img_tensor)
    pred_mask = paddle.nn.functional.sigmoid(pred_mask)

# 后处理
mask = pred_mask.numpy()[0, 0]
mask = (mask > 0.5).astype(np.uint8) * 255

# 保存结果
cv2.imwrite('predicted_mask.png', mask)
```

## 技术支持

如果在使用过程中遇到问题，请：

1. 检查错误信息和日志输出
2. 确认环境配置和依赖安装
3. 验证数据格式和路径设置
4. 参考本文档的常见问题解决方案

## 更新日志

- **v1.0.0**: 初始版本，支持基础的 SegFormer 训练流程
- 后续版本将添加更多功能和优化

---

**注意**: 本脚本基于原始的 `main.ipynb` 和 `work/main.py` 开发，保持了核心算法的一致性，同时提供了更好的用户体验和错误处理机制。