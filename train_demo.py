#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
试卷手写笔迹擦除模型训练脚本
基于 SegFormer 语义分割模型的手写笔迹识别与擦除

作者: OCR AutoScore Team
日期: 2024
"""

import os
import sys
import argparse
import numpy as np
import tqdm
import copy
import time
import random
import math
from data_processor import DatasetProcessor

# 添加工作目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'work'))

import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from paddle.io import DataLoader
from paddle.optimizer import Adam
from paddle.optimizer.lr import CosineAnnealingDecay
import paddle.nn.functional as F

# 导入项目模块
from dataloader import ErasingData
from utils import AverageMeter
from model import segformer_b1, segformer_b2
# from loss import MaskLoss  # 注释掉原始损失函数
import visdom
from visualdl import LogWriter
import json
from collections import defaultdict

# 可用模型字典
MODEL = {
    'segformer_b1':segformer_b1,
    'segformer_b2':segformer_b2,
}

class Session:
    def __init__(self, config):
        self.config = config
        
        # GPU 兼容性检测
        self.device = self._setup_device()
        print(f"使用设备: {self.device}")
        # 手写笔迹分割任务只需要1个类别（手写笔迹）
        self.net = MODEL[config.arch](num_classes=1, pretrained=False)
        self.best_loss = float('inf')
        self.criterion = self._create_compatible_loss()
        self.scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=self.config.lr, milestones=[100, 200], gamma=0.1)
        self.optimizer = paddle.optimizer.AdamW(learning_rate=self.scheduler, parameters=self.net.parameters(), weight_decay=self.config.wd)
        # 数据集已由entrypoint.sh中的data_processor.py处理，这里只需加载
        self.train_dataloader, self.val_dataloader = self.load_datasets()

        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)

        # 训练监控数据
        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epoch_times': [],
            'model_stats': defaultdict(list)
        }

        os.makedirs(os.path.join(self.config.modelsSavePath, self.config.arch), exist_ok=True)
        self.timestamp = time.strftime('%m%d%H%M', time.localtime(time.time()))
        
        # 创建日志目录
        self.log_dir = os.path.join(self.config.modelsSavePath, f'logs_{self.timestamp}')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化可视化组件（在log_dir创建之后）
        self.setup_visualization()

        params = np.sum([p.numel() for p in self.net.parameters()]).item() * 4 /1024/1024
        print(f"Loaded Enhance Net parameters : {params:.3e} MB")
        print(f'模型参数量: {params:.2f}MB')
        print(f'模型保存路径: {os.path.join(self.config.modelsSavePath, self.config.arch)}')
        print(f'时间戳: {self.timestamp}')
        print(f'日志保存路径: {self.log_dir}')
        
    def setup_visualization(self):
        """设置可视化组件"""
        # 设置 VisualDL 日志记录器
        self.visualdl_writer = LogWriter(logdir=self.log_dir)
        
        # 可选的 Visdom 支持
        if self.config.show_visdom:
            import visdom
            self.plotter = visdom.Visdom(env='main', port=7000)
        else:
            self.plotter = None
            
        print(f'VisualDL 日志目录: {self.log_dir}')
        print('训练完成后可使用以下命令启动 VisualDL 服务:')
        print(f'visualdl --logdir {self.log_dir} --port 8040')
        
    def log_training_stats(self, epoch, train_loss, val_loss=None, lr=None, epoch_time=None):
        """记录训练统计信息"""
        # 记录到内存
        self.training_stats['train_losses'].append(train_loss)
        if val_loss is not None:
            self.training_stats['val_losses'].append(val_loss)
        if lr is not None:
            self.training_stats['learning_rates'].append(lr)
        if epoch_time is not None:
            self.training_stats['epoch_times'].append(epoch_time)
            
        # 记录到 VisualDL
        self.visualdl_writer.add_scalar('Loss/Train', train_loss, epoch)
        if val_loss is not None:
            self.visualdl_writer.add_scalar('Loss/Validation', val_loss, epoch)
        if lr is not None:
            self.visualdl_writer.add_scalar('Learning_Rate', lr, epoch)
        if epoch_time is not None:
            self.visualdl_writer.add_scalar('Time/Epoch_Duration', epoch_time, epoch)
            
        # 记录模型参数统计
        self.log_model_stats(epoch)
        
        # 保存统计数据到文件
        stats_file = os.path.join(self.log_dir, 'training_stats.json')
        with open(stats_file, 'w', encoding='utf-8') as f:
            # 转换defaultdict为普通dict以支持JSON序列化
            serializable_stats = dict(self.training_stats)
            serializable_stats['model_stats'] = dict(serializable_stats['model_stats'])
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
            
    def log_model_stats(self, epoch):
        """记录模型参数统计信息"""
        total_params = 0
        grad_norm = 0
        param_norm = 0
        
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
            param_norm += param.norm().item() ** 2
            total_params += param.numel()
            
        grad_norm = grad_norm ** 0.5
        param_norm = param_norm ** 0.5
        
        self.training_stats['model_stats']['grad_norm'].append(grad_norm)
        self.training_stats['model_stats']['param_norm'].append(param_norm)
        self.training_stats['model_stats']['total_params'].append(total_params)
        
        # 记录到 VisualDL
        self.visualdl_writer.add_scalar('Model/Gradient_Norm', grad_norm, epoch)
        self.visualdl_writer.add_scalar('Model/Parameter_Norm', param_norm, epoch)
        if epoch == 0:  # 只在第一个epoch记录总参数量
            self.visualdl_writer.add_scalar('Model/Total_Parameters', total_params, epoch)
        
        if epoch % 10 == 0:  # 每10个epoch打印一次
            print(f'Epoch {epoch}: Grad Norm: {grad_norm:.6f}, Param Norm: {param_norm:.6f}')
            
    def plot_training_curves(self, save_path=None):
        """使用VisualDL记录训练曲线，不再使用matplotlib"""
        print(f'训练统计信息已记录到VisualDL，日志目录: {self.log_dir}')
        print('可使用以下命令查看训练曲线:')
        print(f'visualdl --logdir {self.log_dir} --port 8040')
        
    def plot_realtime_loss(self, epoch, train_loss, val_loss=None):
        """实时绘制损失曲线（使用visdom）"""
        if self.plotter is not None:
            # 绘制训练损失
            self.plotter.line(
                X=[epoch],
                Y=[train_loss],
                win='train_loss',
                name='train_loss',
                update='append' if epoch > 0 else None,
                opts=dict(title='训练损失', xlabel='Epoch', ylabel='Loss')
            )
            
            # 绘制验证损失
            if val_loss is not None:
                self.plotter.line(
                    X=[epoch],
                    Y=[val_loss],
                    win='val_loss',
                    name='val_loss',
                    update='append' if epoch > 0 else None,
                    opts=dict(title='验证损失', xlabel='Epoch', ylabel='Loss')
                )
                
    def save_training_summary(self):
        """保存训练总结报告"""
        summary = {
            'timestamp': self.timestamp,
            'model_arch': self.config.arch,
            'total_epochs': len(self.training_stats['train_losses']),
            'final_train_loss': self.training_stats['train_losses'][-1] if self.training_stats['train_losses'] else None,
            'final_val_loss': self.training_stats['val_losses'][-1] if self.training_stats['val_losses'] else None,
            'min_train_loss': min(self.training_stats['train_losses']) if self.training_stats['train_losses'] else None,
            'min_val_loss': min(self.training_stats['val_losses']) if self.training_stats['val_losses'] else None,
            'total_training_time': sum(self.training_stats['epoch_times']) if self.training_stats['epoch_times'] else None,
            'avg_epoch_time': np.mean(self.training_stats['epoch_times']) if self.training_stats['epoch_times'] else None
        }
        
        summary_file = os.path.join(self.log_dir, 'training_summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        print(f'训练总结已保存到: {summary_file}')

        # 并行训练设置
        if self.config.parallel and 'gpu' in str(self.device):
            numOfGPUs = paddle.device.cuda.device_count()
            if numOfGPUs > 1:
                self.net = paddle.DataParallel(self.net, device_ids=list(range(numOfGPUs)))
                self.criterion = paddle.DataParallel(self.criterion, device_ids=list(range(numOfGPUs)))
            else:
                self.config.parallel = False
        else:
            self.config.parallel = False
            
    def _setup_device(self):
        """
        设置计算设备，包含 GPU 兼容性检测
        """
        # 检查是否强制使用 CPU
        if os.environ.get('CUDA_VISIBLE_DEVICES') == '' or os.environ.get('PADDLE_USE_GPU') == '0':
            print("检测到 CPU 模式环境变量，使用 CPU")
            return paddle.set_device('cpu')
            
        # 检查 CUDA 是否可用
        if not paddle.is_compiled_with_cuda():
            print("PaddlePaddle 未编译 CUDA 支持，使用 CPU")
            return paddle.set_device('cpu')
            
        # 检查是否有 GPU 设备
        if paddle.device.cuda.device_count() == 0:
            print("未检测到 CUDA 设备，使用 CPU")
            return paddle.set_device('cpu')
            
        # 尝试 GPU 兼容性测试
        try:
            device = paddle.set_device('gpu:0')
            # 创建测试张量验证 GPU 兼容性
            test_tensor = paddle.ones([2, 2])
            result = test_tensor + 1
            print(f"GPU 兼容性检测通过，使用 GPU: {device}")
            return device
        except Exception as e:
            print(f"GPU 兼容性检测失败: {e}")
            print("自动切换到 CPU 模式")
            # 设置环境变量强制使用 CPU
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            os.environ['PADDLE_USE_GPU'] = '0'
            return paddle.set_device('cpu')

    def _create_compatible_loss(self):
        """
        创建兼容的损失函数，基于main.ipynb中的设计：L = L_ce + λ × L_dice，其中λ=0.5
        避免work/loss.py中的布尔索引问题
        """
        class CompatibleMaskLoss(nn.Layer):
            def __init__(self, lambda_dice=0.5):
                super(CompatibleMaskLoss, self).__init__()
                self.lambda_dice = lambda_dice  # Dice Loss权重系数
                self.ce_loss = nn.BCELoss()  # 交叉熵损失（二分类使用BCE）
            
            def dice_loss(self, input, target):
                """
                Dice Loss计算，针对细化结构进行有效惩罚
                """
                input = input.reshape((input.shape[0], -1))
                target = target.reshape((target.shape[0], -1))
                
                intersection = paddle.sum(input * target, axis=1)
                union = paddle.sum(input, axis=1) + paddle.sum(target, axis=1)
                
                # 避免除零错误
                dice_coeff = (2.0 * intersection + 1e-7) / (union + 1e-7)
                dice_loss = 1.0 - paddle.mean(dice_coeff)
                return dice_loss
            
            def forward(self, pred_mask, mask, img):
                """
                损失函数前向传播：L = L_ce + λ × L_dice
                """
                # 确保预测输出和标签形状匹配
                # 模型输出已经通过Sigmoid激活，范围在[0,1]
                # pred_mask: [batch_size, 1, height, width]
                # mask: [batch_size, 1, height, width]
                
                # 使用元素级乘法代替布尔索引，避免维度不匹配问题
                valid_region = (paddle.min(img, axis=1, keepdim=True) < 0.5).astype('float32')
                
                # 对有效区域计算损失
                pred_valid = pred_mask * valid_region
                mask_valid = mask * valid_region
                
                # 计算交叉熵损失 L_ce
                ce_loss = self.ce_loss(pred_valid, mask_valid)
                
                # 计算Dice损失 L_dice
                dice_loss = self.dice_loss(pred_valid, mask_valid)
                
                # 组合损失：L = L_ce + λ × L_dice
                total_loss = ce_loss + self.lambda_dice * dice_loss
                
                return total_loss
        
        return CompatibleMaskLoss()

    def prepare_dataset(self):
        """
        检查数据集是否已准备就绪（实际处理由entrypoint.sh完成）
        """
        print("=== 检查数据集状态 ===")
        
        # 检查数据集是否已解压
        train_data_path = os.path.join(self.config.extract_root, 'train')
        if os.path.exists(train_data_path):
            print(f"✅ 训练数据集已存在: {train_data_path}")
        else:
            print(f"❌ 训练数据集不存在: {train_data_path}")
            print("提示: 数据集应由entrypoint.sh中的data_processor.py处理")
        
        print("=== 数据集检查完成 ===")
    


    def load_datasets(self):
        """
        加载训练和验证数据集
        """
        # 确保数据路径正确设置
        if not hasattr(self.config, 'dataRoot') or not self.config.dataRoot:
            extract_root = getattr(self.config, 'extract_root', '/workspace/data/data127971')
            self.config.dataRoot = os.path.join(extract_root, 'train')
        
        train_dataset = ErasingData(self.config.dataRoot, self.config.loadSize, training=True)
        train_dataloader = paddle.io.DataLoader(
            train_dataset,
            batch_size=self.config.batchSize,
            shuffle=True,
            num_workers=0,
            use_shared_memory=False
        )
        val_dataset = ErasingData(self.config.dataRoot, self.config.loadSize, training=False)
        val_dataloader = paddle.io.DataLoader(
            val_dataset,
            batch_size=self.config.batchSize,
            shuffle=False,
            num_workers=0,
            use_shared_memory=False
        )
        return train_dataloader, val_dataloader

    def generate_mask(self, image_path, output_path=None):
        """
        生成图像的分割掩码
        """
        import cv2
        
        # 加载图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法加载图像: {image_path}")
            return None
        
        # 预处理
        image = cv2.resize(image, (self.config.loadSize, self.config.loadSize))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        # 转换为Paddle张量
        image_tensor = paddle.to_tensor(image)
        
        # 模型推理
        self.net.eval()
        with paddle.no_grad():
            output = self.net(image_tensor)
            mask = paddle.nn.functional.softmax(output, axis=1)
            mask = paddle.argmax(mask, axis=1)
            mask = mask.squeeze().numpy()
        
        # 后处理
        mask = (mask * 255).astype(np.uint8)
        
        # 保存结果
        if output_path:
            cv2.imwrite(output_path, mask)
            print(f"掩码已保存到: {output_path}")
        
        return mask
        
    def forward(self, inp):
        img = inp['img']
        mask = inp['mask']
        pred_mask = self.net(1 - img)
        loss = self.criterion(pred_mask, mask, img)
        return {
            'pred_mask':pred_mask,
            'loss':loss,
        }
        
    def train_epoch(self, epoch):
        epoch_start_time = time.time()
        self.net.train()
        total_loss = 0
        
        for i, data in enumerate(self.train_dataloader):
            data = {k:v.to(self.device) if hasattr(v, 'to') else v for k,v in data.items()}
            output = self.forward(data)
            loss = output['loss']
            self.optimizer.clear_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
            if i % 10 == 0:
                current_lr = self.optimizer.get_lr()
                print(f'Epoch [{epoch+1}/{self.config.num_epochs}], Step [{i+1}/{len(self.train_dataloader)}], Loss: {loss.item():.4f}, LR: {current_lr:.6f}')
                
                # 每10个step记录一次数据到VisualDL
                step_global = epoch * len(self.train_dataloader) + i
                self.visualdl_writer.add_scalar('Loss/Train_Step', loss.item(), step_global)
                self.visualdl_writer.add_scalar('Learning_Rate_Step', current_lr, step_global)
                self.visualdl_writer.flush()
        
        # 计算平均损失和epoch时间
        avg_train_loss = total_loss / len(self.train_dataloader)
        epoch_time = time.time() - epoch_start_time
        current_lr = self.optimizer.get_lr()
        
        print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, 耗时: {epoch_time:.2f}s")
        
        # 记录训练统计信息
        self.log_training_stats(epoch, avg_train_loss, lr=current_lr, epoch_time=epoch_time)
        
        # 强制刷新 VisualDL 缓冲区
        self.visualdl_writer.flush()
        
        # 实时可视化（如果启用visdom）
        self.plot_realtime_loss(epoch, avg_train_loss)
        
        return avg_train_loss
        
    def eval_epoch(self, epoch):
        self.net.eval()
        total_loss = 0
        with paddle.no_grad():
            for i, data in enumerate(self.val_dataloader):
                data = {k:v.to(self.device) if hasattr(v, 'to') else v for k,v in data.items()}
                output = self.forward(data)
                loss = output['loss']
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_dataloader)
        print(f"Epoch {epoch+1} Val Loss: {avg_loss:.4f}")
        return avg_loss
        
    def save_model(self, epoch, is_best=False):
        state = {
            'epoch':epoch,
            'state_dict':self.net.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'best_loss':self.best_loss
        }
        filename = os.path.join(self.config.modelsSavePath, f'model_{epoch}.pth')
        paddle.save(state, filename)
        if is_best:
            best_filename = os.path.join(self.config.modelsSavePath, 'model_best.pth')
            paddle.save(state, best_filename)
            
    def run(self):
        if self.config.train:
            print('start training')
            for epoch in range(self.config.num_epochs):
                train_loss = self.train_epoch(epoch)
                val_loss = self.eval_epoch(epoch)
                
                # 更新训练统计信息（添加验证损失）
                if len(self.training_stats['train_losses']) > epoch:
                    self.training_stats['val_losses'].append(val_loss)
                
                # 记录验证损失到 VisualDL
                self.visualdl_writer.add_scalar('Loss/Validation', val_loss, epoch)
                self.visualdl_writer.flush()
                
                # 实时可视化验证损失
                self.plot_realtime_loss(epoch, train_loss, val_loss)
                
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_model(epoch, is_best=True)
                    
                if (epoch + 1) % self.config.show_epoch == 0:
                    self.save_model(epoch)
                    # 每show_epoch个epoch生成训练曲线图
                    self.plot_training_curves()
                    
            # 训练结束后生成最终报告
            self.plot_training_curves()
            self.save_training_summary()
            # 关闭VisualDL writer
            self.visualdl_writer.close()
            print('finish training')
            print(f"训练日志和可视化文件保存在: {self.log_dir}")
        else:
            print('start predicting')
            self.generate_mask()


def parse_arguments():
    """
    解析命令行参数
    """
    # 动态获取项目根目录
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(script_path)

    parser = argparse.ArgumentParser(description='手写笔迹擦除模型训练/预测脚本')
    parser.add_argument('--train', type=int, default=1, help='训练模式 (1) 或预测模式 (0)')
    parser.add_argument('--arch', type=str, default='segformer_b2', help='模型架构')
    parser.add_argument('--pretrained', type=str, default=None, help='是否使用预训练权重')

    parser.add_argument('--modelLog', type=str, default=None, help='预训练模型路径')

    parser.add_argument('--modelsSavePath', type=str, default=os.path.join(project_root, 'models'), help='模型保存路径')

    parser.add_argument('--batchSize', type=int, default=8, help='批次大小')
    parser.add_argument('--loadSize', type=int, default=384, help='图像加载尺寸')
    
    # 数据集相关路径
    parser.add_argument('--data_root', type=str, 
                       default=os.path.join(project_root, 'data', 'data127971'),
                       help='原始数据集压缩包所在目录')
    parser.add_argument('--extract_root', type=str,
                       default=os.path.join(project_root, 'work', 'data'),
                       help='解压目标根目录')
    parser.add_argument('--dataRoot', type=str, 
                        default=os.path.join(project_root, 'work', 'data', 'train'), 
                        help='数据根目录')

    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--wd', type=float, default=1e-3, help='权重衰减')
    parser.add_argument('--numOfWorkers', type=int, default=4, help='数据加载器工作线程数')
    parser.add_argument('--parallel', type=int, default=0, help='是否并行训练')

    parser.add_argument('--show_visdom', type=int, default=0, help='是否使用Visdom可视化')

    parser.add_argument('--show_epoch', type=int, default=10, help='Visdom可视化间隔')

    return parser.parse_args()



def main():
    config = parse_arguments()
    session = Session(config)
    session.run()

if __name__ == '__main__':
    main()