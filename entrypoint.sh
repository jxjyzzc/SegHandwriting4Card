#!/bin/bash
# set -e  # 注释掉，允许脚本在错误时继续运行

# 确保数据目录存在
mkdir -p /data /work/data
chmod -R 777 /data /work/data

# 启动调试模式
echo "Starting container with entrypoint..."

# GPU 兼容性检测
echo "=== GPU 兼容性检测 ==="
if [ -f "gpu_check.py" ]; then
    # 运行 GPU 检测并导出环境变量
    eval $(python3 gpu_check.py 2>&1 | grep "^export " || true)
    # 如果没有设置环境变量，使用默认值
    if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
        export CUDA_VISIBLE_DEVICES=""
    fi
    if [ -z "${PADDLE_USE_GPU+x}" ]; then
        export PADDLE_USE_GPU="0"
    fi
else
    echo "gpu_check.py not found, skipping GPU check..."
    export CUDA_VISIBLE_DEVICES=""
    export PADDLE_USE_GPU="0"
fi

echo "当前环境变量:"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "PADDLE_USE_GPU: ${PADDLE_USE_GPU}"

# 处理数据集 (使用环境变量配置路径)
echo "=== 开始处理数据集 ==="
echo "DATA_ROOT: ${DATA_ROOT}"
echo "EXTRACT_ROOT: ${EXTRACT_ROOT}"
# 数据集实际位置在 data127971 子目录下
ACTUAL_DATA_ROOT="${DATA_ROOT}/data127971"
echo "ACTUAL_DATA_ROOT: ${ACTUAL_DATA_ROOT}"
if [ -f "data_processor.py" ]; then
    python3 data_processor.py --data_root "${ACTUAL_DATA_ROOT}" --extract_root "${EXTRACT_ROOT}" || echo "Data processing failed"
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