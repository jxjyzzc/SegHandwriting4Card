# 训练可视化功能使用指南

本指南介绍如何使用 VisualDL 进行手写笔迹擦除模型的训练可视化监控。

## 功能特性

### 1. VisualDL 集成
- **官方支持**: 使用 PaddlePaddle 官方推荐的 VisualDL 可视化工具
- **实时监控**: 训练过程中实时记录各项指标
- **丰富图表**: 支持标量、图像、直方图等多种可视化类型
- **Web界面**: 通过浏览器访问直观的可视化界面

### 2. 监控指标
- **损失曲线**: 训练损失和验证损失的实时变化
- **学习率跟踪**: 学习率调度的可视化
- **模型统计**: 梯度范数、参数范数、总参数量
- **训练效率**: 每个epoch的训练时间

### 3. 数据持久化
- VisualDL 日志文件自动保存
- 训练统计数据备份为JSON格式
- 支持训练中断后的可视化恢复

## 使用方法

### 1. 启动训练
训练脚本已自动集成 VisualDL，无需额外配置：

```bash
# 在容器中启动训练
docker exec -it seghandwriting-app python train_demo.py
```

### 2. 查看训练日志
训练开始后，会在 `/app/models/` 目录下创建带时间戳的日志目录：

```bash
# 查看日志目录
ls /app/models/
# 输出示例: logs_07031940
```

### 3. 启动 VisualDL 服务

#### 方法一：在容器内启动
```bash
# 进入容器
docker exec -it seghandwriting-app bash

# 启动 VisualDL 服务
visualdl --logdir /app/models/logs_XXXXXXXX --port 8040
```

#### 方法二：在宿主机启动（推荐）
```bash
# 确保容器端口映射
docker run -p 8040:8040 seghandwriting-app

# 或者重新启动容器并映射端口
docker stop seghandwriting-app
docker run -d --name seghandwriting-app -p 8040:8040 seghandwriting-app
```

### 4. 访问可视化界面
在浏览器中访问：`http://localhost:8040`

## 可视化内容说明

### 1. Scalar（标量图表）
- **Loss/Train**: 训练损失曲线
- **Loss/Validation**: 验证损失曲线（如果有验证集）
- **Learning_Rate**: 学习率变化
- **Model/Gradient_Norm**: 梯度范数
- **Model/Parameter_Norm**: 参数范数
- **Model/Total_Parameters**: 模型总参数量
- **Time/Epoch_Duration**: 每个epoch的训练时间

### 2. 图表功能
- **缩放**: 鼠标滚轮缩放图表
- **平移**: 拖拽移动视图
- **平滑**: 调整曲线平滑度
- **下载**: 导出图表为图片

## 监控脚本使用

### 1. 快速查看训练状态
```bash
# 运行监控脚本
python monitor_training.py
```

### 2. 持续监控模式
```bash
# 每30秒检查一次训练进度
python monitor_training.py --monitor --interval 30
```

### 3. 指定日志目录
```bash
# 监控特定的训练日志
python monitor_training.py --log_dir /app/models/logs_07031940
```

## 故障排除

### 1. VisualDL 服务无法启动
- 检查端口是否被占用：`netstat -tlnp | grep 8040`
- 尝试使用其他端口：`visualdl --logdir <path> --port 8041`
- 确保 VisualDL 已正确安装：`pip show visualdl`

### 2. 无法访问可视化界面
- 检查防火墙设置
- 确认容器端口映射正确
- 尝试使用 `127.0.0.1:8040` 而不是 `localhost:8040`

### 3. 没有训练数据显示
- 确认训练已开始并运行了至少一个epoch
- 检查日志目录是否正确
- 查看训练脚本是否有错误输出

### 4. 图表显示异常
- 刷新浏览器页面
- 清除浏览器缓存
- 重启 VisualDL 服务

## 最佳实践

### 1. 训练监控
- 定期检查损失是否下降
- 观察学习率调度是否合理
- 监控梯度范数避免梯度爆炸/消失
- 关注训练时间优化训练效率

### 2. 数据管理
- 为不同实验使用不同的日志目录
- 定期备份重要的训练日志
- 清理过期的日志文件释放存储空间

### 3. 性能优化
- 在生产环境中可以降低日志记录频率
- 使用 `--host 0.0.0.0` 允许远程访问
- 考虑使用反向代理提高访问性能

## 扩展功能

### 1. 自定义指标
可以在训练脚本中添加更多自定义指标：

```python
# 在 train_demo.py 中添加
self.visualdl_writer.add_scalar('Custom/Accuracy', accuracy, epoch)
self.visualdl_writer.add_scalar('Custom/F1_Score', f1_score, epoch)
```

### 2. 图像可视化
```python
# 记录训练样本
self.visualdl_writer.add_image('Train/Sample', image_tensor, epoch)

# 记录模型预测结果
self.visualdl_writer.add_image('Prediction/Result', pred_image, epoch)
```

### 3. 模型结构可视化
```python
# 记录模型计算图
self.visualdl_writer.add_graph(model, input_tensor)
```

## 总结

VisualDL 提供了强大而直观的训练可视化功能，能够帮助您：
- 实时监控训练进度
- 快速发现训练问题
- 优化模型性能
- 记录实验结果

通过合理使用这些可视化工具，您可以更好地理解和改进您的深度学习模型训练过程。