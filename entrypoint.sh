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
echo "=== 数据集处理检查 ==="
echo "DATA_ROOT: ${DATA_ROOT}"
echo "EXTRACT_ROOT: ${EXTRACT_ROOT}"
# 数据集实际位置在 data127971 子目录下
ACTUAL_DATA_ROOT="${DATA_ROOT}/data127971"
echo "ACTUAL_DATA_ROOT: ${ACTUAL_DATA_ROOT}"

# 检查数据是否已解压
if [ -d "${EXTRACT_ROOT}/train" ] && [ -d "${EXTRACT_ROOT}/test" ]; then
    echo "数据集已解压，跳过数据处理步骤"
else
    echo "数据集未解压，开始处理数据集"
    if [ -f "data_processor.py" ]; then
        python3 data_processor.py --data_root "${ACTUAL_DATA_ROOT}" --extract_root "${EXTRACT_ROOT}" || echo "Data processing failed"
    else
        echo "data_processor.py not found, skipping..."
    fi
fi

# 启动 VisualDL 服务（优先检查现有日志）
echo "=== 启动 VisualDL 服务 ==="
# 查找最新的日志目录
LOG_DIR=$(find /app/models -name "logs_*" -type d 2>/dev/null | sort | tail -1)
if [ -n "$LOG_DIR" ]; then
    echo "找到日志目录: $LOG_DIR"
    echo "启动 VisualDL 服务..."
    nohup visualdl --logdir "$LOG_DIR" --port 8040 --host 0.0.0.0 > /dev/null 2>&1 &
    sleep 3
    if pgrep -f "visualdl" > /dev/null; then
        echo "VisualDL 服务启动成功，访问地址: http://localhost:8040"
    else
        echo "VisualDL 服务启动失败"
    fi
else
    echo "未找到日志目录，VisualDL 服务将在训练开始后启动"
fi

# 检查模型是否已存在
echo "=== 训练检查 ==="
if [ -d "/app/models/segformer_b2" ] && [ "$(ls -A /app/models/segformer_b2 2>/dev/null)" ]; then
    echo "模型已存在，跳过训练步骤"
else
    echo "模型不存在，开始训练"
    if [ -f "train_demo.py" ]; then
        # 在后台启动训练
        nohup python3 train_demo.py > /app/models/training.log 2>&1 &
        TRAIN_PID=$!
        echo "训练已在后台启动，PID: $TRAIN_PID"
        
        # 等待日志目录创建
        echo "等待训练日志目录创建..."
        for i in {1..30}; do
            NEW_LOG_DIR=$(find /app/models -name "logs_*" -type d 2>/dev/null | sort | tail -1)
            if [ -n "$NEW_LOG_DIR" ] && [ "$NEW_LOG_DIR" != "$LOG_DIR" ]; then
                echo "检测到新的日志目录: $NEW_LOG_DIR"
                # 如果之前没有启动VisualDL或者有新的日志目录，启动新的VisualDL
                if ! pgrep -f "visualdl" > /dev/null || [ "$NEW_LOG_DIR" != "$LOG_DIR" ]; then
                    # 停止旧的VisualDL服务
                    pkill -f "visualdl" 2>/dev/null || true
                    sleep 2
                    # 启动新的VisualDL服务
                    echo "启动VisualDL服务监控新的训练日志..."
                    nohup visualdl --logdir "$NEW_LOG_DIR" --port 8040 --host 0.0.0.0 > /dev/null 2>&1 &
                    sleep 3
                    if pgrep -f "visualdl" > /dev/null; then
                        echo "VisualDL 服务启动成功，访问地址: http://localhost:8040"
                    else
                        echo "VisualDL 服务启动失败"
                    fi
                fi
                break
            fi
            sleep 2
        done
        
        echo "训练正在后台进行，可以通过以下命令查看日志:"
        echo "  docker exec seghandwriting-app tail -f /app/models/training.log"
    else
        echo "train_demo.py not found, skipping..."
    fi
fi

# 保持容器运行
echo "Container ready for debugging"
exec tail -f /dev/null