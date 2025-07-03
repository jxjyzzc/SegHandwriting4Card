#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控脚本
实时查看训练进度和 VisualDL 日志
"""

import os
import json
import time
import glob
import argparse
from datetime import datetime

def find_latest_log_dir(models_path="./models"):
    """查找最新的日志目录"""
    log_dirs = []
    for root, dirs, files in os.walk(models_path):
        for d in dirs:
            if d.startswith('logs_'):
                log_dirs.append(os.path.join(root, d))
    
    if not log_dirs:
        return None
    
    # 按修改时间排序，返回最新的
    log_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return log_dirs[0]

def load_training_stats(log_dir):
    """加载训练统计数据"""
    stats_file = os.path.join(log_dir, 'training_stats.json')
    if not os.path.exists(stats_file):
        return None
    
    with open(stats_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def print_training_progress(stats):
    """打印训练进度"""
    if not stats or not stats['train_losses']:
        print("暂无训练数据")
        return
    
    current_epoch = len(stats['train_losses'])
    latest_loss = stats['train_losses'][-1]
    
    print(f"\n=== 训练进度监控 ===")
    print(f"当前Epoch: {current_epoch}")
    print(f"最新训练损失: {latest_loss:.6f}")
    
    if stats['val_losses']:
        latest_val_loss = stats['val_losses'][-1]
        print(f"最新验证损失: {latest_val_loss:.6f}")
    
    if stats['learning_rates']:
        latest_lr = stats['learning_rates'][-1]
        print(f"当前学习率: {latest_lr:.8f}")
    
    if stats['epoch_times']:
        avg_time = sum(stats['epoch_times']) / len(stats['epoch_times'])
        total_time = sum(stats['epoch_times'])
        print(f"平均每轮耗时: {avg_time:.2f}秒")
        print(f"总训练时间: {total_time/60:.1f}分钟")
    
    # 显示模型参数统计
    if stats['model_stats']['grad_norm']:
        latest_grad_norm = stats['model_stats']['grad_norm'][-1]
        print(f"最新梯度范数: {latest_grad_norm:.6f}")
    
    if stats['model_stats']['param_norm']:
        latest_param_norm = stats['model_stats']['param_norm'][-1]
        print(f"最新参数范数: {latest_param_norm:.6f}")

def show_visualdl_info(log_dir):
    """显示 VisualDL 启动信息"""
    print("\n=== VisualDL 可视化服务 ===")
    print(f"日志目录: {log_dir}")
    print("\n启动 VisualDL 服务的命令:")
    print(f"visualdl --logdir {log_dir} --port 8040")
    print("\n服务启动后，在浏览器中访问: http://localhost:8040")
    print("\n注意: 如果在 Docker 容器中运行，需要映射端口:")
    print("docker run -p 8040:8040 <container_name>")
    
    # 检查是否有 VisualDL 日志文件
    vdl_files = glob.glob(os.path.join(log_dir, "vdlrecords.*"))
    if vdl_files:
        print(f"\n找到 VisualDL 日志文件: {len(vdl_files)} 个")
        for vdl_file in vdl_files[:3]:  # 只显示前3个
            print(f"  - {os.path.basename(vdl_file)}")
    else:
        print("\n未找到 VisualDL 日志文件，可能训练尚未开始或 VisualDL 未正确配置")

def monitor_training(log_dir=None, interval=30, plot_interval=300):
    """持续监控训练进程"""
    if log_dir is None:
        log_dir = find_latest_log_dir()
        if log_dir is None:
            print("未找到训练日志目录")
            return
    
    print(f"监控日志目录: {log_dir}")
    print(f"刷新间隔: {interval}秒")
    print(f"图表更新间隔: {plot_interval}秒")
    print("按 Ctrl+C 停止监控\n")
    
    last_plot_time = 0
    
    try:
        while True:
            stats = load_training_stats(log_dir)
            if stats:
                print(f"\n[{time.strftime('%H:%M:%S')}] 训练状态更新:")
                print_training_progress(stats)
                
                # 定期显示 VisualDL 信息
                current_time = time.time()
                if current_time - last_plot_time > plot_interval:
                    show_visualdl_info(log_dir)
                    last_plot_time = current_time
            else:
                print(f"\n[{time.strftime('%H:%M:%S')}] 等待训练数据...")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n监控已停止")
        # 显示 VisualDL 信息
        show_visualdl_info(log_dir)
        print("可以使用 VisualDL 查看详细的训练可视化")

def main():
    parser = argparse.ArgumentParser(description='训练监控工具')
    parser.add_argument('--log_dir', type=str, help='日志目录路径')
    parser.add_argument('--interval', type=int, default=30, help='监控刷新间隔（秒）')
    parser.add_argument('--plot_interval', type=int, default=300, help='图表更新间隔（秒）')
    parser.add_argument('--once', action='store_true', help='只运行一次，不持续监控')
    
    args = parser.parse_args()
    
    if args.once:
        # 单次运行模式
        log_dir = args.log_dir or find_latest_log_dir()
        if log_dir is None:
            print("未找到训练日志目录")
            return
        
        stats = load_training_stats(log_dir)
        if stats:
            print_training_progress(stats)
            show_visualdl_info(log_dir)
        else:
            print("未找到训练数据")
            show_visualdl_info(log_dir)
    else:
        # 持续监控模式
        monitor_training(args.log_dir, args.interval, args.plot_interval)

if __name__ == '__main__':
    main()