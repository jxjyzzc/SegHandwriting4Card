#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集处理脚本 - 统一处理数据解压和CSV生成
作者: OCR AutoScore Team
日期: 2024
"""

import os
import zipfile
import csv
import argparse
from pathlib import Path

class DatasetProcessor:
    def __init__(self, data_root, extract_root):
        """
        初始化数据处理器
        
        Args:
            data_root: 原始数据集压缩包所在目录
            extract_root: 解压目标根目录
        """
        self.data_root = data_root
        self.extract_root = extract_root
        
    def extract_dataset(self, zip_path, extract_to):
        """
        解压数据集
        """
        print(f"正在解压 {zip_path} 到 {extract_to}")
        
        # 创建目标目录
        os.makedirs(extract_to, exist_ok=True)
        
        # 解压文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"解压完成: {extract_to}")
        return extract_to

    def list_zip_contents(self, zip_path, max_display=20):
        """
        列出压缩包内容
        """
        print(f"\n压缩包 {zip_path} 内容:")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            for i, file_name in enumerate(file_list[:max_display]):
                print(f"  {file_name}")
            if len(file_list) > max_display:
                print(f"  ... 还有 {len(file_list) - max_display} 个文件")
            print(f"总计: {len(file_list)} 个文件")
        return file_list

    def find_dataset_structure(self, dataset_root):
        """
        自动检测数据集目录结构
        """
        # 检查是否有嵌套的数据集目录
        nested_dirs = [d for d in os.listdir(dataset_root) 
                      if os.path.isdir(os.path.join(dataset_root, d)) 
                      and not d.startswith('__MACOSX')]
        
        if nested_dirs:
            # 使用第一个非__MACOSX目录作为实际数据集目录
            actual_dataset_dir = os.path.join(dataset_root, nested_dirs[0])
        else:
            actual_dataset_dir = dataset_root
            
        images_dir = os.path.join(actual_dataset_dir, 'images')
        gts_dir = os.path.join(actual_dataset_dir, 'gts')
        
        return actual_dataset_dir, images_dir, gts_dir

    def generate_train_csv(self, dataset_root, train_ratio=0.8):
        """
        生成训练和验证CSV文件
        
        Args:
            dataset_root: 数据集根目录
            train_ratio: 训练集比例
        """
        actual_dataset_dir, images_dir, gts_dir = self.find_dataset_structure(dataset_root)
        
        if not os.path.exists(images_dir):
            print(f"错误: 找不到图像目录 {images_dir}")
            return None, None
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(images_dir).glob(ext))
        
        image_files = sorted([str(f) for f in image_files])
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 生成CSV数据
        csv_data = []
        for img_path in image_files:
            if os.path.exists(gts_dir):
                # 训练集：包含图像和GT路径
                img_name = os.path.basename(img_path)
                gt_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
                gt_path = os.path.join(gts_dir, gt_name)
                
                if os.path.exists(gt_path):
                    csv_data.append([img_path, gt_path])
                else:
                    print(f"警告: 找不到对应的GT文件: {gt_path}")
            else:
                # 测试集：只包含图像路径
                csv_data.append([img_path])
        
        if not csv_data:
            print("错误: 没有找到有效的数据")
            return None, None
        
        # 分割训练集和验证集
        split_idx = int(len(csv_data) * train_ratio)
        train_data = csv_data[:split_idx]
        val_data = csv_data[split_idx:]
        
        # 生成CSV文件 - 在实际数据集目录中
        train_csv = os.path.join(actual_dataset_dir, 'train.csv')
        val_csv = os.path.join(actual_dataset_dir, 'test.csv')  # 使用test.csv作为验证集
        
        # 写入训练集CSV
        with open(train_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in train_data:
                if len(row) == 2:  # 训练集格式：图像路径,GT路径
                    writer.writerow([row[0]])  # 只写入图像路径，GT路径通过替换获得
                else:  # 测试集格式：只有图像路径
                    writer.writerow(row)
        
        # 写入验证集CSV
        with open(val_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in val_data:
                if len(row) == 2:  # 训练集格式
                    writer.writerow([row[0]])  # 只写入图像路径
                else:  # 测试集格式
                    writer.writerow(row)
        
        # 强制复制CSV文件到训练脚本期望的位置
        # 训练脚本期望在 dataset_root 目录下找到 train.csv 和 test.csv
        target_train_csv = os.path.join(dataset_root, 'train.csv')
        target_test_csv = os.path.join(dataset_root, 'test.csv')
        
        import shutil
        try:
            # 强制复制，覆盖已存在的文件
            shutil.copy2(train_csv, target_train_csv)
            shutil.copy2(val_csv, target_test_csv)
            print(f"✅ CSV文件已成功复制到训练脚本期望位置:")
            print(f"   训练集: {target_train_csv}")
            print(f"   验证集: {target_test_csv}")
        except Exception as e:
            print(f"❌ 复制CSV文件时出错: {e}")
        
        print(f"生成训练集CSV: {train_csv} ({len(train_data)} 个样本)")
        print(f"生成验证集CSV: {val_csv} ({len(val_data)} 个样本)")
        
        return train_csv, val_csv

    def generate_test_csv(self, dataset_root):
        """
        生成测试CSV文件
        """
        actual_dataset_dir, images_dir, _ = self.find_dataset_structure(dataset_root)
        
        if not os.path.exists(images_dir):
            print(f"错误: 找不到图像目录 {images_dir}")
            return None
        
        # 获取所有图像文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(images_dir).glob(ext))
        
        image_files = sorted([str(f) for f in image_files])
        
        # 生成测试CSV - 在实际数据集目录中
        test_csv = os.path.join(actual_dataset_dir, 'test.csv')
        # 同时在父目录（训练脚本期望的位置）生成CSV文件
        parent_test_csv = os.path.join(dataset_root, 'test.csv')
        
        for csv_path in [test_csv, parent_test_csv]:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                for img_path in image_files:
                    writer.writerow([img_path])
        
        print(f"生成测试集CSV: {test_csv} ({len(image_files)} 个样本)")
        print(f"同时复制到父目录: {parent_test_csv}")
        return test_csv

    def process_all_datasets(self):
        """
        处理所有数据集：解压和生成CSV
        """
        print("=== 开始处理数据集 ===")
        
        # 数据集文件路径
        train_zip = os.path.join(self.data_root, "dehw_train_dataset.zip")
        test_zip = os.path.join(self.data_root, "dehw_testA_dataset.zip")
        
        # 处理训练数据集
        if os.path.exists(train_zip):
            print("\n=== 处理训练数据集 ===")
            self.list_zip_contents(train_zip)
            
            train_extract_dir = os.path.join(self.extract_root, "train")
            self.extract_dataset(train_zip, train_extract_dir)
            
            # 生成训练和验证CSV
            self.generate_train_csv(train_extract_dir)
        else:
            print(f"训练数据集不存在: {train_zip}")
        
        # 处理测试数据集
        if os.path.exists(test_zip):
            print("\n=== 处理测试数据集 ===")
            self.list_zip_contents(test_zip)
            
            test_extract_dir = os.path.join(self.extract_root, "test")
            self.extract_dataset(test_zip, test_extract_dir)
            
            # 生成测试CSV
            self.generate_test_csv(test_extract_dir)
        else:
            print(f"测试数据集不存在: {test_zip}")
        
        print("\n=== 数据集处理完成 ===")

def parse_arguments():
    """
    解析命令行参数
    """
    # 动态获取项目根目录
    script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(script_path)

    # 从环境变量获取路径，默认为项目内路径
    default_data_root = os.getenv('DATA_ROOT', os.path.join(project_root, 'data', 'data127971'))
    default_extract_root = os.getenv('EXTRACT_ROOT', os.path.join(project_root, 'work', 'data'))
    
    parser = argparse.ArgumentParser(description='数据集处理脚本')
    parser.add_argument('--data_root', type=str, 
                       default=default_data_root,
                       help='原始数据集压缩包所在目录')
    parser.add_argument('--extract_root', type=str,
                       default=default_extract_root,
                       help='解压目标根目录')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='训练集比例 (默认: 0.8)')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # 创建数据处理器
    processor = DatasetProcessor(args.data_root, args.extract_root)
    
    # 处理所有数据集
    processor.process_all_datasets()

if __name__ == '__main__':
    main()
