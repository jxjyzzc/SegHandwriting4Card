#!/bin/bash

# PaddlePaddle Docker 构建脚本 - Tesla M40 优化版
# 针对 Tesla M40 (Compute Capability 5.2) 优化
# 使用 CUDA 11.8 确保兼容性，减少网络流量消耗

set -e

# 默认配置
IMAGE_NAME="seghandwriting-app"
IMAGE_TAG="tesla-m40"
DOCKERFILE="Dockerfile.tesla-m40"
VERBOSE=false
NO_CACHE=false
CLEAN_CACHE=false
SHOW_HELP=false
VERIFY_GPU=false

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

# 显示帮助信息
show_help() {
    cat << EOF
PaddlePaddle Docker 构建脚本 - Tesla M40 优化版

专为 Tesla M40 显卡优化，解决兼容性问题并减少网络流量消耗

用法: $0 [选项]

选项:
  -h, --help              显示此帮助信息
  -v, --verbose           启用详细输出
  -t, --tag TAG           设置镜像标签 (默认: tesla-m40)
  -n, --name NAME         设置镜像名称 (默认: seghandwriting-app)
  -f, --dockerfile FILE   指定 Dockerfile (默认: Dockerfile.tesla-m40)
  --no-cache              构建时不使用缓存
  --clean-cache           构建前清理 Docker 缓存
  --verify-gpu            构建完成后验证 GPU 支持
  --original              使用原始 Dockerfile (不推荐用于 Tesla M40)

Tesla M40 优化特性:
  - 使用 CUDA 11.8 确保兼容性
  - Compute Capability 5.2 特定编译参数
  - 浅克隆减少源码下载 (节省 ~2GB)
  - 精简 CUDA 组件 (节省 ~500MB)
  - 编译后清理源码 (减少镜像大小)

网络流量对比:
  - 原始版本: ~3.5-4.5GB
  - Tesla M40 优化版: ~1.5-2GB (节省约 60%)

示例:
  $0                      # 使用 Tesla M40 优化版构建
  $0 --verbose            # 启用详细输出
  $0 --verify-gpu         # 构建后验证 GPU 支持
  $0 --clean-cache        # 清理缓存后构建

EOF
}

# 检查 Docker 是否安装
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker 未安装或不在 PATH 中"
        exit 1
    fi
}

# 验证基础镜像可用性
validate_base_image() {
    local dockerfile="$1"
    local base_image=$(grep "^FROM" "$dockerfile" | head -1 | awk '{print $2}')
    
    log_info "验证基础镜像: $base_image"
    
    # 备用镜像列表
    local fallback_images=(
        "paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6"
        "ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6"
        "registry.baidubce.com/paddlex/paddlex:paddlex3.0.0b1-paddlepaddle3.0.0b1-gpu-cuda11.8-cudnn8.6-trt8.5"
    )
    
    # 检查当前镜像
    if docker manifest inspect "$base_image" &>/dev/null; then
        log_success "基础镜像验证成功: $base_image"
        return 0
    fi
    
    log_warning "基础镜像不可用: $base_image"
    log_info "尝试备用镜像..."
    
    # 尝试备用镜像
    for fallback_image in "${fallback_images[@]}"; do
        if [[ "$fallback_image" != "$base_image" ]]; then
            log_info "检查备用镜像: $fallback_image"
            if docker manifest inspect "$fallback_image" &>/dev/null; then
                log_success "找到可用的备用镜像: $fallback_image"
                
                # 更新 Dockerfile
                sed -i "s|FROM.*|FROM $fallback_image|" "$dockerfile"
                log_info "已自动更新 Dockerfile 中的基础镜像"
                return 0
            fi
        fi
    done
    
    log_error "所有镜像地址都不可用，请检查网络连接或镜像源"
    log_info "建议的解决方案:"
    echo "  1. 检查网络连接: ping docker.io"
    echo "  2. 尝试手动拉取镜像: docker pull <image_name>"
    echo "  3. 使用代理或镜像加速器"
    echo "  4. 检查 Docker 守护进程状态"
    return 1
}

# 检查 Tesla M40 GPU
check_tesla_m40() {
    if command -v nvidia-smi &> /dev/null; then
        local gpu_info=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "")
        
        if [[ -n "$gpu_info" ]]; then
            log_tesla "检测到 NVIDIA GPU"
            echo "$gpu_info" | head -1
            
            # 检查是否为 Tesla M40 或类似的老显卡
            if echo "$gpu_info" | grep -qi "tesla\|quadro\|gtx 9\|gtx 10"; then
                log_tesla "检测到可能需要 CUDA 11.8 兼容性的显卡"
            fi
        else
            log_warning "无法获取 GPU 信息，但检测到 nvidia-smi"
        fi
    else
        log_warning "未检测到 NVIDIA GPU 或 nvidia-smi 不可用"
        log_warning "Tesla M40 需要正确安装 NVIDIA 驱动"
    fi
}

# 清理 Docker 缓存
clean_docker_cache() {
    log_info "清理 Docker 构建缓存..."
    docker builder prune -f
    docker system prune -f
    log_success "Docker 缓存清理完成"
}

# 显示网络流量预估
show_network_estimation() {
    if [[ "$DOCKERFILE" == "Dockerfile.tesla-m40" ]]; then
        log_tesla "Tesla M40 优化版网络流量预估:"
        echo "  - CUDA 11.8 工具链: ~800MB"
        echo "  - PaddlePaddle 源码 (浅克隆): ~200MB"
        echo "  - 系统包更新: ~100MB"
        echo "  - Python 依赖: ~50MB"
        echo "  - 总计预估: ~1.2GB"
        echo ""
        log_success "相比原版本节省约 60% 网络流量!"
        log_tesla "特别优化: Compute Capability 5.2 编译参数"
    else
        log_warning "网络流量预估 (原始版):"
        echo "  - CUDA 12.0 开发工具链: ~1.3GB"
        echo "  - PaddlePaddle 源码: ~2-3GB"
        echo "  - 系统包更新: ~200MB"
        echo "  - 总计预估: ~3.5-4.5GB"
        echo ""
        log_error "警告: 原始版本使用 CUDA 12.0，Tesla M40 不兼容!"
        log_warning "强烈建议使用 Tesla M40 优化版本"
    fi
    echo ""
}

# 验证 GPU 支持
verify_gpu_support() {
    local container_name="tesla-m40-verify-$(date +%s)"
    local full_image_name="${IMAGE_NAME}:${IMAGE_TAG}"
    
    log_tesla "验证 Tesla M40 GPU 支持..."
    
    if docker run --rm --gpus all --name "$container_name" "$full_image_name" /app/verify_tesla_m40.sh; then
        log_success "Tesla M40 GPU 验证成功!"
    else
        log_error "Tesla M40 GPU 验证失败"
        log_warning "请检查 NVIDIA 驱动和 Docker GPU 支持"
    fi
}

# 增强的错误处理函数
handle_build_error() {
    local exit_code=$1
    local stage="$2"
    local full_image_name="$3"
    
    log_error "构建在 $stage 阶段失败 (退出码: $exit_code)"
    
    # 分析错误类型
    case $exit_code in
        125)
            log_warning "Docker 守护进程错误，建议重启 Docker 服务"
            echo "  sudo systemctl restart docker"
            ;;
        1)
            log_warning "一般构建错误，可能的原因:"
            echo "  - Dockerfile 语法错误"
            echo "  - 网络连接问题"
            echo "  - 磁盘空间不足"
            ;;
        2)
            log_warning "文件或目录不存在"
            echo "  - 检查 Dockerfile 路径"
            echo "  - 确认构建上下文正确"
            ;;
        *)
            log_warning "未知错误 ($exit_code)"
            ;;
    esac
    
    # 保存构建日志
    local log_file="logs/build_error_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p logs
    
    # 尝试获取最后的容器日志
    local last_container=$(docker ps -lq 2>/dev/null)
    if [[ -n "$last_container" ]]; then
        docker logs "$last_container" > "$log_file" 2>&1
        log_info "构建日志已保存到: $log_file"
    fi
    
    # 提供恢复建议
    suggest_recovery_actions "$stage" "$exit_code"
    
    # 清理失败的镜像
    if docker images -q "$full_image_name" &>/dev/null; then
        log_info "清理失败的镜像..."
        docker rmi "$full_image_name" &>/dev/null || true
    fi
}

# 恢复建议函数
suggest_recovery_actions() {
    local stage="$1"
    local exit_code="$2"
    
    echo ""
    log_tesla "建议的恢复操作:"
    echo "1. 检查网络连接: ping docker.io"
    echo "2. 清理 Docker 缓存: docker system prune -f"
    echo "3. 检查磁盘空间: df -h"
    echo "4. 重新验证基础镜像: docker manifest inspect <base_image>"
    echo "5. 使用 --no-cache 重新构建"
    echo "6. 检查 Docker 守护进程: sudo systemctl status docker"
    
    if [[ "$stage" == "网络下载" ]]; then
        echo "7. 配置 Docker 镜像加速器"
        echo "8. 检查防火墙设置"
    elif [[ "$stage" == "编译" ]]; then
        echo "7. 增加 Docker 内存限制"
        echo "8. 检查 CUDA 工具链安装"
    fi
    
    echo ""
    log_info "如需详细帮助，请查看: TESLA_M40_COMPLETE_GUIDE.md"
}

# 构建 Docker 镜像
build_image() {
    local build_args=""
    
    if [[ "$NO_CACHE" == "true" ]]; then
        build_args="$build_args --no-cache"
    fi
    
    if [[ "$VERBOSE" == "true" ]]; then
        build_args="$build_args --progress=plain"
    fi
    
    local full_image_name="${IMAGE_NAME}:${IMAGE_TAG}"
    
    log_info "开始构建 Tesla M40 优化镜像: $full_image_name"
    log_info "使用 Dockerfile: $DOCKERFILE"
    
    local build_cmd="docker build $build_args -f $DOCKERFILE -t $full_image_name ."
    log_info "构建命令: $build_cmd"
    
    # 显示构建开始时间
    local start_time=$(date +%s)
    log_tesla "构建开始时间: $(date)"
    
    # 执行构建并捕获退出码
    if eval "$build_cmd"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_success "Tesla M40 优化镜像构建成功: $full_image_name"
        log_info "构建耗时: $((duration / 60))分$((duration % 60))秒"
        
        # 显示镜像信息
        local image_size=$(docker images --format "table {{.Size}}" $full_image_name | tail -n 1)
        log_info "镜像大小: $image_size"
        
        # 构建成功后的验证
        log_info "验证构建结果..."
        if docker run --rm "$full_image_name" python3 -c "import paddle; print('PaddlePaddle 导入成功')" &>/dev/null; then
            log_success "PaddlePaddle 验证通过"
        else
            log_warning "PaddlePaddle 验证失败，可能需要 GPU 环境"
        fi
        
        return 0
    else
        local exit_code=$?
        handle_build_error "$exit_code" "Docker构建" "$full_image_name"
        return 1
    fi
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -f|--dockerfile)
            DOCKERFILE="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE=true
            shift
            ;;
        --clean-cache)
            CLEAN_CACHE=true
            shift
            ;;
        --verify-gpu)
            VERIFY_GPU=true
            shift
            ;;
        --original)
            DOCKERFILE="Dockerfile"
            IMAGE_TAG="latest"
            log_warning "使用原始 Dockerfile，Tesla M40 可能不兼容"
            shift
            ;;
        *)
            log_error "未知选项: $1"
            show_help
            exit 1
            ;;
    esac
done

# 显示帮助信息
if [[ "$SHOW_HELP" == "true" ]]; then
    show_help
    exit 0
fi

# 主函数
main() {
    log_tesla "开始 Tesla M40 优化构建过程"
    
    # 检查依赖
    check_docker
    check_tesla_m40
    
    # 检查 Dockerfile 是否存在
    if [[ ! -f "$DOCKERFILE" ]]; then
        log_error "Dockerfile 不存在: $DOCKERFILE"
        exit 1
    fi
    
    # 验证基础镜像可用性
    if ! validate_base_image "$DOCKERFILE"; then
        log_error "基础镜像验证失败，无法继续构建"
        exit 1
    fi
    
    # 显示网络流量预估
    show_network_estimation
    
    # 清理缓存（如果需要）
    if [[ "$CLEAN_CACHE" == "true" ]]; then
        clean_docker_cache
    fi
    
    # 构建镜像
    if build_image; then
        log_success "Tesla M40 优化构建流程完成!"
        
        if [[ "$DOCKERFILE" == "Dockerfile.tesla-m40" ]]; then
            log_tesla "已使用 Tesla M40 专用优化版本构建"
            log_info "运行容器: docker run -it --gpus all ${IMAGE_NAME}:${IMAGE_TAG}"
            log_info "验证脚本: docker run --rm --gpus all ${IMAGE_NAME}:${IMAGE_TAG} /app/verify_tesla_m40.sh"
        fi
        
        # GPU 验证（如果需要）
        if [[ "$VERIFY_GPU" == "true" ]]; then
            verify_gpu_support
        fi
        
        exit 0
    else
        log_error "Tesla M40 优化构建流程失败!"
        exit 1
    fi
}

# 执行主函数
main "$@"