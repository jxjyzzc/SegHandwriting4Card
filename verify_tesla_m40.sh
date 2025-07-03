#!/bin/bash

# Tesla M40 GPU 验证脚本
# 检测 Tesla M40 显卡兼容性和 CUDA 环境

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_tesla() {
    echo -e "${PURPLE}[TESLA M40]${NC} $1"
}

# 验证开始
log_tesla "开始 Tesla M40 兼容性验证"
echo "==========================================="

# 1. 检查 NVIDIA 驱动
log_info "检查 NVIDIA 驱动..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader
    log_success "NVIDIA 驱动检测成功"
else
    log_error "NVIDIA 驱动未安装或不可用"
    exit 1
fi

echo ""

# 2. 检查 CUDA 版本
log_info "检查 CUDA 版本..."
if command -v nvcc &> /dev/null; then
    local cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    log_success "CUDA 版本: $cuda_version"
    
    # 检查是否为 CUDA 11.8
    if [[ "$cuda_version" == "11.8" ]]; then
        log_tesla "CUDA 11.8 - Tesla M40 完全兼容!"
    elif [[ "$cuda_version" =~ ^11\. ]]; then
        log_warning "CUDA $cuda_version - Tesla M40 可能兼容"
    elif [[ "$cuda_version" =~ ^12\. ]]; then
        log_error "CUDA $cuda_version - Tesla M40 不兼容!"
        log_warning "Tesla M40 最高支持 CUDA 11.8"
    else
        log_warning "CUDA $cuda_version - 兼容性未知"
    fi
else
    log_error "CUDA 编译器 (nvcc) 未找到"
    exit 1
fi

echo ""

# 3. 检查 Python 和 PaddlePaddle
log_info "检查 Python 环境..."
python3 --version
log_success "Python 检测成功"

echo ""

log_info "检查 PaddlePaddle 安装..."
if python3 -c "import paddle; print(f'PaddlePaddle 版本: {paddle.__version__}')" 2>/dev/null; then
    log_success "PaddlePaddle 安装成功"
else
    log_error "PaddlePaddle 未正确安装"
    exit 1
fi

echo ""

# 4. 测试 GPU 可用性
log_info "测试 GPU 可用性..."
if python3 -c "
import paddle
paddle.device.set_device('gpu')
print(f'GPU 设备数量: {paddle.device.cuda.device_count()}')
print(f'当前设备: {paddle.device.get_device()}')
print(f'GPU 内存信息: {paddle.device.cuda.max_memory_allocated() / 1024**3:.2f} GB')
" 2>/dev/null; then
    log_success "GPU 可用性测试通过"
else
    log_error "GPU 不可用或 PaddlePaddle GPU 版本问题"
    log_warning "可能原因: CUDA 版本不兼容或驱动问题"
    exit 1
fi

echo ""

# 5. Tesla M40 特定测试
log_tesla "执行 Tesla M40 特定测试..."
if python3 -c "
import paddle
import paddle.nn as nn

# 设置 GPU 设备
paddle.device.set_device('gpu')

# 创建简单的张量运算测试
x = paddle.randn([100, 100])
y = paddle.randn([100, 100])
z = paddle.matmul(x, y)

print(f'张量运算测试成功')
print(f'结果形状: {z.shape}')
print(f'结果类型: {z.dtype}')
print(f'设备位置: {z.place}')

# 测试简单的神经网络
model = nn.Linear(100, 50)
model.eval()
output = model(x)
print(f'神经网络测试成功')
print(f'输出形状: {output.shape}')
" 2>/dev/null; then
    log_tesla "Tesla M40 特定测试通过!"
else
    log_error "Tesla M40 特定测试失败"
    log_warning "可能需要检查 Compute Capability 5.2 支持"
    exit 1
fi

echo ""

# 6. 显示系统信息摘要
log_info "系统信息摘要:"
echo "==========================================="
echo "GPU 信息:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -1
echo ""
echo "CUDA 信息:"
nvcc --version | grep "release"
echo ""
echo "PaddlePaddle 信息:"
python3 -c "import paddle; print(f'版本: {paddle.__version__}'); print(f'CUDA 版本: {paddle.version.cuda()}'); print(f'cuDNN 版本: {paddle.version.cudnn()}')"
echo ""

# 7. Tesla M40 优化建议
log_tesla "Tesla M40 优化建议:"
echo "1. 确保使用 CUDA 11.8 或更低版本"
echo "2. 编译时使用 Compute Capability 5.2"
echo "3. 避免使用需要 CUDA 12+ 的新特性"
echo "4. 监控 GPU 内存使用 (Tesla M40 有 24GB 显存)"
echo "5. 使用混合精度训练可以提高性能"
echo ""

log_success "Tesla M40 验证完成 - 所有测试通过!"
log_tesla "您的 Tesla M40 已准备好运行 PaddlePaddle 应用"

exit 0