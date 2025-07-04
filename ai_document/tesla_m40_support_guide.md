# Tesla M40 显卡支持指南

## 概述

本文档提供了在 Tesla M40 显卡上运行 SegHandwriting4Card 项目的完整解决方案。Tesla M40 使用 Maxwell 架构 (Compute Capability 5.2)，需要特殊的配置来确保兼容性。

## 问题分析

### 1. PaddlePaddle 兼容性问题

**问题描述**:
- Tesla M40 GPU 架构为 Maxwell (Compute Capability 5.2)
- 当前 PaddlePaddle 编译时支持的架构为 60 61 70 75 80 86，不包含 5.2
- 错误信息: "no kernel image is available for execution on the device"

**根本原因**:
PaddlePaddle 的预编译版本不包含对 Maxwell 架构的支持，需要使用兼容的 CUDA 版本或切换到 PyTorch。

## 解决方案

### 方案一：使用 Tesla M40 优化版 Docker 镜像（推荐）

项目已提供专门针对 Tesla M40 优化的 Docker 配置：

#### 1. 使用优化版构建脚本

```bash
# 构建 Tesla M40 优化版镜像
./build_tesla_m40.sh

# 或者使用详细输出
./build_tesla_m40.sh --verbose

# 构建后验证 GPU 支持
./build_tesla_m40.sh --verify-gpu
```

#### 2. 优化特性

- **CUDA 11.8 兼容性**: 确保与 Tesla M40 完全兼容
- **Compute Capability 5.2**: 专门编译参数支持
- **网络流量优化**: 相比原版本节省约 60% 下载量
- **浅克隆源码**: 减少 PaddlePaddle 源码下载时间

#### 3. 验证安装

```bash
# 运行验证脚本
./verify_tesla_m40.sh

# 或在容器内验证
docker exec seghandwriting-app /app/verify_tesla_m40.sh
```

### 方案二：PyTorch 版本迁移（长期解决方案）

如果 PaddlePaddle 方案仍有问题，可以创建 PyTorch 版本：

#### 1. 创建 PyTorch 分支

```bash
# 创建新分支
git checkout -b pytorch-m40-support

# 创建 PyTorch 版本目录
mkdir -p pytorch_version
```

#### 2. PyTorch 版本优势

- **更好的 Maxwell 支持**: PyTorch 对老显卡支持更完善
- **丰富的预训练模型**: Transformers 库提供 SegFormer 实现
- **活跃的社区支持**: 更容易找到解决方案
- **TensorBoard 集成**: 替代 VisualDL 的可视化方案

#### 3. 迁移计划

**阶段一：环境搭建**
```dockerfile
# PyTorch Tesla M40 Dockerfile
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# 安装 transformers 和相关依赖
RUN pip install transformers datasets tensorboard opencv-python
```

**阶段二：模型迁移**
```python
# 使用 Transformers 库的 SegFormer
from transformers import SegformerForSemanticSegmentation

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512",
    num_labels=1,  # 手写笔迹分割
    ignore_mismatched_sizes=True
)
```

**阶段三：训练循环迁移**
```python
# PyTorch 训练循环
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class PyTorchTrainer:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.writer = SummaryWriter('logs')
    
    def train_epoch(self, dataloader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(self.device), masks.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs.logits, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 记录到 TensorBoard
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Loss/Train_Step', loss.item(), 
                                     batch_idx + epoch * len(dataloader))
        
        return total_loss / len(dataloader)
```

## 当前状态修复

### 已修复的问题

1. **JSON 序列化错误**: 修复了 `training_stats` 中 `defaultdict` 无法序列化的问题
2. **训练监控**: VisualDL 可视化功能正常工作
3. **CPU 回退机制**: 当 GPU 不兼容时自动切换到 CPU 模式

### 修复代码

```python
# 修复 JSON 序列化问题
def log_training_stats(self, epoch, train_loss, val_loss=None, lr=None, epoch_time=None):
    # ... 其他代码 ...
    
    # 保存统计数据到文件
    stats_file = os.path.join(self.log_dir, 'training_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        # 转换defaultdict为普通dict以支持JSON序列化
        serializable_stats = dict(self.training_stats)
        serializable_stats['model_stats'] = dict(serializable_stats['model_stats'])
        json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
```

## 使用建议

### 1. 短期解决方案

- 使用当前修复版本在 CPU 模式下训练
- 虽然速度较慢，但功能完整
- 适合小规模数据集和原型验证

### 2. 中期解决方案

- 尝试 Tesla M40 优化版 Docker 镜像
- 使用 CUDA 11.8 和专门的编译参数
- 如果成功，可以获得 GPU 加速

### 3. 长期解决方案

- 迁移到 PyTorch 版本
- 更好的硬件兼容性
- 更丰富的生态系统支持

## 性能对比

| 方案 | GPU 支持 | 训练速度 | 兼容性 | 维护成本 |
|------|----------|----------|--------|----------|
| 当前 CPU 模式 | ❌ | 慢 | ✅ | 低 |
| Tesla M40 优化版 | ⚠️ | 快 | ⚠️ | 中 |
| PyTorch 版本 | ✅ | 快 | ✅ | 中 |

## 下一步行动

1. **立即**: 当前 CPU 版本已可正常训练
2. **本周**: 测试 Tesla M40 优化版 Docker 镜像
3. **下周**: 如需要，开始 PyTorch 版本迁移

## 相关文件

- `Dockerfile.tesla-m40`: Tesla M40 优化的 Docker 配置
- `build_tesla_m40.sh`: Tesla M40 构建脚本
- `verify_tesla_m40.sh`: Tesla M40 验证脚本
- `train_demo.py`: 已修复 JSON 序列化问题的训练脚本

## 技术支持

如遇到问题，请按以下顺序排查：

1. 检查 NVIDIA 驱动版本
2. 验证 CUDA 版本兼容性
3. 运行 GPU 兼容性测试
4. 查看容器日志获取详细错误信息
5. 考虑使用 PyTorch 替代方案

---

**更新记录**:
- 2024-07-04: 创建文档，修复 JSON 序列化问题，提供 Tesla M40 支持方案