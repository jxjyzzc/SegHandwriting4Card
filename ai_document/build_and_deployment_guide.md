# 构建和部署最佳实践指南

## 概述

本文档提供了 SegHandwriting4Card 项目的构建、部署和维护最佳实践，包括 Docker 容器管理、目录结构优化和服务配置。

## 项目结构优化

### 1. 构建文件组织

所有 Docker 相关文件已移动到 `build/` 目录：

```
build/
├── Dockerfile                 # 标准版 Dockerfile
├── Dockerfile.tesla-m40       # Tesla M40 优化版
├── build_tesla_m40.sh         # Tesla M40 构建脚本
├── verify_tesla_m40.sh        # Tesla M40 验证脚本
└── .dockerignore              # Docker 忽略文件
```

**优势**：
- 清晰的文件组织
- 便于维护和版本控制
- 减少根目录文件混乱

### 2. 数据持久化目录

```
models/                        # 模型和日志持久化目录
├── logs_YYYYMMDDHHMM/        # 训练日志（按时间戳命名）
│   └── vdlrecords.*.log      # VisualDL 记录文件
└── segformer_b*/             # 训练好的模型文件
```

## Docker 配置优化

### 1. 容器挂载配置

更新后的 `docker-compose.yml` 配置：

```yaml
services:
  seghandwriting-app:
    build:
      context: .
      dockerfile: build/Dockerfile.tesla-m40  # 使用 build 目录中的 Dockerfile
    volumes:
      - ./data:/app/data                     # 数据集目录
      - ./work:/app/work                     # 工作目录
      - ./models:/app/models                 # 模型和日志目录（新增）
      - ./entrypoint.sh:/app/entrypoint.sh
      - ./gpu_check.py:/app/gpu_check.py
      - ./data_processor.py:/app/data_processor.py
      - ./train_demo.py:/app/train_demo.py
    ports:
      - "8040:8040"                         # VisualDL 服务端口
```

**关键改进**：
- ✅ 添加了 `./models:/app/models` 挂载
- ✅ 修正了 dockerfile 路径为 `build/Dockerfile.tesla-m40`
- ✅ 确保日志文件持久化到宿主机

### 2. 挂载验证

验证容器挂载是否正确：

```bash
# 检查挂载点
docker inspect seghandwriting-app --format '{{.Mounts}}'

# 应该看到包含以下挂载：
# /root/SegHandwriting4Card/models -> /app/models
```

## 服务启动和管理

### 1. 标准启动流程

```bash
# 1. 停止现有容器
docker-compose down

# 2. 启动容器
docker-compose up -d

# 3. 等待训练开始（约30秒）
sleep 30

# 4. 启动 VisualDL 服务
docker exec -d seghandwriting-app visualdl \
  --logdir /app/models/logs_$(date +%m%d%H%M) \
  --port 8040 \
  --host 0.0.0.0
```

### 2. VisualDL 服务管理

**自动启动 VisualDL**：

```bash
# 获取最新的日志目录
LOG_DIR=$(docker exec seghandwriting-app find /app/models -name "logs_*" -type d | head -1)

# 启动 VisualDL 服务
docker exec -d seghandwriting-app visualdl \
  --logdir "$LOG_DIR" \
  --port 8040 \
  --host 0.0.0.0
```

**验证服务状态**：

```bash
# 检查 VisualDL 进程
docker exec seghandwriting-app sh -c 'ps aux | grep visualdl'

# 测试端口连通性
curl -I http://localhost:8040

# 应该返回 HTTP/1.0 302 FOUND
```

### 3. 访问 VisualDL 界面

- **本地访问**: http://localhost:8040
- **重定向**: 自动重定向到 http://localhost:8040/app
- **功能**: 实时查看训练损失、学习率、模型参数等

## 构建脚本使用

### 1. Tesla M40 优化构建

```bash
# 进入构建目录
cd build/

# 使用优化脚本构建
./build_tesla_m40.sh --verbose

# 构建后验证 GPU 支持
./build_tesla_m40.sh --verify-gpu
```

### 2. 构建选项

```bash
# 显示帮助
./build_tesla_m40.sh --help

# 清理缓存后构建
./build_tesla_m40.sh --clean-cache

# 只构建不运行
./build_tesla_m40.sh --no-run

# 跳过构建只运行
./build_tesla_m40.sh --only-run
```

## 故障排除

### 1. VisualDL 无法访问

**问题**: http://localhost:8040 无法打开

**解决方案**:

```bash
# 1. 检查容器状态
docker ps | grep seghandwriting-app

# 2. 检查端口映射
docker port seghandwriting-app

# 3. 检查 VisualDL 进程
docker exec seghandwriting-app sh -c 'ps aux | grep visualdl'

# 4. 如果没有进程，手动启动
docker exec -d seghandwriting-app visualdl \
  --logdir /app/models/logs_* \
  --port 8040 \
  --host 0.0.0.0
```

### 2. 日志文件不在宿主机

**问题**: 容器内有日志，但宿主机 `models/` 目录为空

**解决方案**:

```bash
# 1. 检查挂载配置
docker inspect seghandwriting-app --format '{{.Mounts}}'

# 2. 确保 models 目录存在
mkdir -p ./models

# 3. 重启容器应用挂载
docker-compose down && docker-compose up -d
```

### 3. 构建失败

**问题**: Docker 构建过程中出错

**解决方案**:

```bash
# 1. 检查网络连接
ping docker.io

# 2. 清理 Docker 缓存
docker system prune -f

# 3. 使用详细输出重新构建
cd build/
./build_tesla_m40.sh --verbose --clean-cache
```

## 性能监控

### 1. 训练进度监控

```bash
# 查看训练日志
docker logs --tail 20 seghandwriting-app

# 实时跟踪日志
docker logs -f seghandwriting-app

# 检查资源使用
docker stats seghandwriting-app
```

### 2. 磁盘空间管理

```bash
# 检查模型目录大小
du -sh ./models/

# 清理旧的日志文件（保留最近3个）
find ./models -name "logs_*" -type d | sort | head -n -3 | xargs rm -rf

# 清理 Docker 镜像
docker image prune -f
```

## 最佳实践总结

### ✅ 推荐做法

1. **使用 build 目录**: 所有构建文件集中管理
2. **正确配置挂载**: 确保数据持久化
3. **定期清理**: 避免磁盘空间不足
4. **监控服务**: 及时发现和解决问题
5. **使用脚本**: 自动化构建和部署流程

### ❌ 避免做法

1. **不要删除挂载**: 避免数据丢失
2. **不要频繁重建**: 浪费时间和资源
3. **不要忽略日志**: 及时查看错误信息
4. **不要混用版本**: 保持 Dockerfile 和脚本一致

## 更新记录

- **2024-07-04**: 创建文档
- **2024-07-04**: 优化目录结构，修复 VisualDL 访问问题
- **2024-07-04**: 添加挂载配置和服务管理指南

---

**注意**: 本文档基于当前项目配置编写，如有更新请及时同步修改。