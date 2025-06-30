# SegFormer4Card 手写笔迹擦除模型实现计划

## 项目概述
基于 SegFormer 语义分割模型的试卷手写笔迹识别与擦除系统，实现 main.ipynb 的功能。

## 当前进展

### ✅ 已完成的工作

1. **数据处理模块统一化**
   - 创建了 `data_processor.py` 统一处理数据解压和CSV生成
   - 移除了硬编码路径，支持配置化参数
   - 合并了原来的 `extract_dataset.py` 和 `generate_csv.py`
   - 支持自动检测数据集目录结构

2. **训练脚本优化**
   - 更新了 `train_demo.py` 使用新的数据处理器
   - 修复了模型导入和参数调用问题
   - 添加了配置化的数据路径支持

3. **容器环境测试**
   - 在 `ocr-autoscore` 容器中成功运行数据处理脚本
   - 成功解压训练和测试数据集
   - 生成了训练集CSV (864个样本) 和验证集CSV (217个样本)
   - 生成了测试集CSV (200个样本)

### 🔄 当前问题

1. **依赖兼容性问题**
   - paddleseg 版本与 PaddlePaddle 存在兼容性问题
   - 需要解决 `AttributeError: module 'paddleseg' has no attribute 'optimizers'` 错误

2. **模型架构问题**
   - work/model.py 中的 SegFormer 实现依赖 paddleseg
   - 需要创建独立的 SegFormer 实现或解决依赖问题

3. **路径灵活性问题** (✅ 已解决)
   - `data_processor.py` 和 `train_demo.py` 已更新，通过动态计算项目根目录来管理数据路径，移除了硬编码。

## 文件结构

```
SegFormer4Card/
├── data_processor.py          # 统一数据处理脚本 ✅
├── train_demo.py              # 训练主脚本 ✅
├── work/
│   ├── dataloader.py         # 数据加载器 ✅
│   ├── model.py              # 模型定义 ⚠️ (依赖问题)
│   ├── loss.py               # 损失函数 ✅
│   └── utils.py              # 工具函数 ✅
├── data/
│   ├── data127971/           # 原始数据集
│   └── extracted/            # 解压后数据 ✅
└── implementation_plan.md     # 实现计划文档 ✅
```

## 解决方案

### 方案1: 修复依赖问题
- 降级或升级 paddleseg 到兼容版本
- 修改 work/model.py 中的导入方式

### 方案2: 独立实现
- 创建不依赖 paddleseg 的 SegFormer 实现
- 使用纯 PaddlePaddle API 实现模型

### 方案3: 简化模型
- 使用更简单的分割模型作为替代
- 确保核心功能可以正常运行

## 测试命令

### 数据处理测试
```bash
docker exec -it ocr-autoscore python3 segmentation/SegFormer4Card/data_processor.py \
  --data_root /app/segmentation/SegFormer4Card/data/data127971 \
  --extract_root /app/segmentation/SegFormer4Card/data/extracted
```

### 训练脚本测试
```bash
docker exec -it ocr-autoscore python3 segmentation/SegFormer4Card/train_demo.py \
  --train 1 --arch segformer_b2 --batchSize 2 --loadSize 256 --num_epochs 1 \
  --lr 0.001 --numOfWorkers 0 \
  --data_root /app/segmentation/SegFormer4Card/data/data127971 \
  --extract_root /app/segmentation/SegFormer4Card/data/extracted
```

## 下一步计划

1. **解决依赖问题**
   - 研究 paddleseg 版本兼容性
   - 尝试不同的安装方式或版本

2. **模型实现**
   - 如果依赖问题无法解决，创建简化的分割模型
   - 确保训练流程可以正常运行

3. **功能验证**
   - 完成端到端的训练测试
   - 验证模型推理功能
   - 实现图像掩码生成

4. **集成到 main.ipynb**
   - 将训练好的模型集成到 Jupyter Notebook
   - 提供完整的使用示例

## 技术栈

- **深度学习框架**: PaddlePaddle
- **模型架构**: SegFormer (语义分割)
- **数据处理**: OpenCV, PIL
- **容器环境**: Docker (ocr-autoscore)
- **编程语言**: Python 3.6+

## 备注

- 所有路径已配置化，避免硬编码
- 数据处理脚本支持命令行参数
- 训练脚本支持多种配置选项
- 在容器环境中测试通过数据处理功能