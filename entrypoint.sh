#!/bin/bash
set -e

# 确保数据目录存在
mkdir -p /data /work/data
chmod -R 777 /data /work/data

# 启动调试模式
echo "Starting container with entrypoint..."

# 处理数据集 (解压到/data目录)
if [ -f "data_processor.py" ]; then
    python3 data_processor.py --data_root /data/data127971 --extract_root /data || echo "Data processing failed"
else
    echo "data_processor.py not found, skipping..."
fi

# 执行训练脚本
if [ -f "train_demo.py" ]; then
    python3 train_demo.py || echo "Training failed"
else
    echo "train_demo.py not found, skipping..."
fi

# 保持容器运行
echo "Container ready for debugging"
exec tail -f /dev/null