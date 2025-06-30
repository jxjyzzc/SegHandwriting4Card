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
from loss import MaskLoss
import visdom

# 可用模型字典
MODEL = {
    'segformer_b1':segformer_b1,
    'segformer_b2':segformer_b2,
}

class Session:
    def __init__(self, config):
        self.config = config
        self.device = paddle.set_device('gpu' if paddle.is_compiled_with_cuda() else 'cpu')
        self.net = MODEL[config.arch]()
        self.best_loss = float('inf')
        self.criterion = MaskLoss()
        self.scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=self.config.lr, milestones=[100, 200], gamma=0.1)
        self.optimizer = paddle.optimizer.AdamW(learning_rate=self.scheduler, parameters=self.net.parameters(), weight_decay=self.config.wd)
        self.prepare_dataset()
        self.train_dataloader, self.val_dataloader = self.load_datasets()

        paddle.seed(2022)
        np.random.seed(2022)
        random.seed(2022)

        if self.config.show_visdom:
            import visdom
            self.plotter = visdom.Visdom(env='main', port=7000)

        os.makedirs(os.path.join(self.config.modelsSavePath, self.config.arch), exist_ok=True)
        self.timestamp = time.strftime('%m%d%H%M', time.localtime(time.time()))

        params = np.sum([p.numel() for p in self.net.parameters()]).item() * 4 /1024/1024
        print(f"Loaded Enhance Net parameters : {params:.3e} MB")

        if self.config.parallel:
            numOfGPUs = paddle.device.cuda.device_count()
            if numOfGPUs > 1:
                self.net = paddle.DataParallel(self.net, device_ids=list(range(numOfGPUs)))
                self.criterion = paddle.DataParallel(self.criterion, device_ids=list(range(numOfGPUs)))
            else:
                self.config.parallel = False

    def prepare_dataset(self):
        """
        准备数据集：解压和生成CSV文件
        """
        print("=== 准备数据集 ===")
        
        # 创建数据处理器并处理数据集
        processor = DatasetProcessor(self.config.data_root, self.config.extract_root)
        processor.process_all_datasets()
        
        print("=== 数据集准备完成 ===")
    


    def load_datasets(self):
        """
        加载训练和验证数据集
        """
        # 确保数据路径正确设置
        if not hasattr(self.config, 'dataRoot') or not self.config.dataRoot:
            extract_root = getattr(self.config, 'extract_root', '/opt/1panel/resource/apps/local/OCRAutoScore/segmentation/SegFormer4Card/data/extracted')
            self.config.dataRoot = os.path.join(extract_root, 'train')
        
        train_dataset = ErasingData(self.config.dataRoot, self.config.loadSize, training=True)
        train_dataloader = paddle.io.DataLoader(
            train_dataset,
            batch_size=self.config.batchSize,
            shuffle=True,
            num_workers=self.config.numOfWorkers
        )
        val_dataset = ErasingData(self.config.dataRoot, self.config.loadSize, training=False)
        val_dataloader = paddle.io.DataLoader(
            val_dataset,
            batch_size=self.config.batchSize,
            shuffle=False,
            num_workers=self.config.numOfWorkers
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
        self.net.train()
        total_loss = 0
        for i, data in enumerate(self.train_dataloader):
            data = {k:v.to(self.device) for k,v in data.items()}
            output = self.forward(data)
            loss = output['loss']
            self.optimizer.clear_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Train Loss: {total_loss / len(self.train_dataloader):.4f}")
        
    def eval_epoch(self, epoch):
        self.net.eval()
        total_loss = 0
        with paddle.no_grad():
            for i, data in enumerate(self.val_dataloader):
                data = {k:v.to(self.device) for k,v in data.items()}
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
                self.train_epoch(epoch)
                val_loss = self.eval_epoch(epoch)
                if val_loss < self.best_loss:
                    self.best_loss = val_loss
                    self.save_model(epoch, is_best=True)
                if (epoch + 1) % self.config.show_epoch == 0:
                    self.save_model(epoch)
            print('finish training')
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
                       default=os.path.join(project_root, 'data', 'extracted'),
                       help='解压目标根目录')
    parser.add_argument('--dataRoot', type=str, 
                        default=os.path.join(project_root, 'data', 'extracted', 'train'), 
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