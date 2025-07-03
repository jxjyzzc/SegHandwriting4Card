# 容器运行故障排除指南

## 概述

本文档记录了 SegHandwriting4Card 项目在容器化部署过程中遇到的常见问题及解决方案。

## 问题分类

### 1. 数据集路径映射问题

**问题描述**: 
- 容器启动时提示 "训练数据集不存在: /data/data127971/dehw_train_dataset.zip"
- entrypoint.sh 中使用硬编码路径，未使用 docker-compose.yml 中定义的环境变量

**解决方案**:
1. 修改 `entrypoint.sh`，使用环境变量 `${DATA_ROOT}` 和 `${EXTRACT_ROOT}`
2. 在 `docker-compose.yml` 中为环境变量添加注释说明
3. 确保数据集文件放置在宿主机的 `./data` 目录下

**修复后的配置**:
```bash
# entrypoint.sh 中使用环境变量
python3 data_processor.py --data_root "${DATA_ROOT}" --extract_root "${EXTRACT_ROOT}"
```

```yaml
# docker-compose.yml 中的环境变量配置
environment:
  # 数据集压缩包所在目录（映射到宿主机 ./data 目录）
  - DATA_ROOT=/app/data
  # 数据集解压目标目录（映射到宿主机 ./work/data 目录）
  - EXTRACT_ROOT=/app/work/data
```

### 2. CUDA 兼容性问题

**问题描述**:
- Tesla M40 GPU 架构为 Maxwell (Compute Capability 5.2)
- PaddlePaddle 编译时支持的架构为 60 61 70 75 80 86，不包含 5.2
- 错误信息: "no kernel image is available for execution on the device"

**解决方案**:
1. 创建 GPU 兼容性检测脚本 `gpu_check.py`
2. 在 `entrypoint.sh` 中添加 GPU 检测步骤
3. 如果 GPU 不兼容，自动切换到 CPU 模式
4. 设置相应的环境变量强制使用 CPU

**GPU 检测逻辑**:
```python
# 检测 GPU 兼容性
try:
    paddle.set_device('gpu:0')
    test_tensor = paddle.ones([2, 2])
    result = test_tensor + 1
    return True  # GPU 兼容
except Exception:
    return False  # GPU 不兼容，切换到 CPU
```

**CPU 模式设置**:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PADDLE_USE_GPU'] = '0'
```

## 最佳实践

### 1. 环境变量使用
- 避免在脚本中硬编码路径
- 使用 docker-compose.yml 中定义的环境变量
- 为环境变量添加清晰的注释说明

### 2. GPU 兼容性处理
- 在训练开始前进行 GPU 兼容性检测
- 提供 CPU fallback 机制
- 给用户明确的错误信息和建议

### 3. 错误处理
- 使用 `set -e` 确保脚本在错误时停止
- 提供详细的错误信息和调试输出
- 使用 `|| echo "message"` 处理非关键错误

### 4. 数据集管理
- 使用相对路径和环境变量
- 确保数据集目录结构的一致性
- 提供数据集验证和自动处理功能

## 故障排除步骤

### 1. 数据集问题
1. 检查宿主机 `./data` 目录是否存在数据集文件
2. 验证 docker-compose.yml 中的卷映射配置
3. 检查容器内环境变量是否正确设置
4. 运行 `data_processor.py` 手动处理数据集

### 2. GPU 问题
1. 运行 `gpu_check.py` 检测 GPU 兼容性
2. 查看 CUDA 版本和 GPU 架构信息
3. 如需强制使用 CPU，设置环境变量 `CUDA_VISIBLE_DEVICES=""`
4. 考虑安装支持当前 GPU 架构的 PaddlePaddle 版本

### 3. 容器启动问题
1. 检查 Docker 和 nvidia-docker 安装
2. 验证 GPU 驱动和 CUDA 运行时
3. 查看容器日志获取详细错误信息
4. 使用 `docker-compose logs` 查看完整启动日志

## 更新记录

- **2024-07-03**: 初始版本，记录数据集路径和 CUDA 兼容性问题的解决方案
- 后续版本将根据新发现的问题持续更新

## 相关文件

- `entrypoint.sh`: 容器启动脚本
- `docker-compose.yml`: 容器编排配置
- `gpu_check.py`: GPU 兼容性检测脚本
- `data_processor.py`: 数据集处理脚本
- `Dockerfile.tesla-m40`: Tesla M40 优化的 Dockerfile