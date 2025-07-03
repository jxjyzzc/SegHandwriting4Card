#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 兼容性检测脚本
检测当前 GPU 是否与 PaddlePaddle 兼容，如果不兼容则切换到 CPU 模式
"""

import os
import sys

def check_gpu_compatibility():
    """
    检测GPU兼容性，特别针对 Maxwell 架构
    """
    try:
        import paddle
        
        # 检查是否有可用的GPU
        if not paddle.device.is_compiled_with_cuda():
            print("PaddlePaddle未编译CUDA支持")
            return False
            
        gpu_count = paddle.device.cuda.device_count()
        if gpu_count == 0:
            print("未检测到GPU设备")
            return False
            
        print(f"检测到 {gpu_count} 个GPU设备")
        
        # 检查每个GPU的兼容性
        compatible_gpus = []
        maxwell_detected = False
        
        for i in range(gpu_count):
            try:
                # 获取GPU信息
                gpu_name = paddle.device.cuda.get_device_name(i)
                print(f"检测到GPU {i}: {gpu_name}")
                
                # 尝试在GPU上创建tensor来测试兼容性
                paddle.device.set_device(f'gpu:{i}')
                test_tensor = paddle.to_tensor([1.0], dtype='float32')
                result = test_tensor + 1.0
                
                # 检查GPU的Compute Capability来判断是否为Maxwell架构
                try:
                    # 获取GPU的Compute Capability
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                    compute_capability = f"{major}.{minor}"
                    
                    print(f"GPU {i} Compute Capability: {compute_capability}")
                    
                    # Maxwell架构的Compute Capability是5.0-5.3
                    if major == 5:
                        print(f"检测到Maxwell架构GPU {i} (CC {compute_capability})，与当前PaddlePaddle版本不兼容")
                        maxwell_detected = True
                        continue
                        
                except ImportError:
                    print("pynvml未安装，使用备用检测方法")
                    # 备用方法：检查GPU名称中是否包含已知的Maxwell GPU型号
                    maxwell_gpus = ['GTX 9', 'GTX 7', 'GTX 6', 'Tesla M', 'Quadro M']
                    if any(model in gpu_name for model in maxwell_gpus):
                        print(f"根据GPU名称检测到可能的Maxwell架构GPU {i}: {gpu_name}")
                        maxwell_detected = True
                        continue
                except Exception as e:
                    print(f"无法获取GPU {i}的Compute Capability: {e}")
                    # 如果无法获取CC信息，继续尝试运行测试
                
                print(f"GPU {i} 兼容性测试通过")
                compatible_gpus.append(i)
                
            except Exception as e:
                print(f"GPU {i} 兼容性测试失败: {e}")
                # 如果是 Maxwell 架构兼容性问题，设置 CPU fallback
                if "Maxwell" in str(e) or "Compute Capability" in str(e):
                    print("检测到 Maxwell 架构兼容性问题，设置 CPU fallback")
                    maxwell_detected = True
                continue
        
        if maxwell_detected and not compatible_gpus:
            print("检测到Maxwell架构GPU，但与PaddlePaddle不兼容，切换到CPU模式")
            print("export CUDA_VISIBLE_DEVICES=''")
            print("export PADDLE_USE_GPU='0'")
            return False
        elif compatible_gpus:
            print("GPU 兼容性检测通过")
            # 设置环境变量使用兼容的GPU
            gpu_list = ','.join(map(str, compatible_gpus))
            print(f"export CUDA_VISIBLE_DEVICES='{gpu_list}'")
            print("export PADDLE_USE_GPU='1'")
            return True
        else:
            print("所有GPU都不兼容")
            print("export CUDA_VISIBLE_DEVICES=''")
            print("export PADDLE_USE_GPU='0'")
            return False
            
    except Exception as e:
        print(f"GPU兼容性检测失败: {e}")
        # 检查是否是 Maxwell 架构问题
        if "Maxwell" in str(e) or "Compute Capability" in str(e):
            print("检测到 Maxwell 架构兼容性问题，强制使用 CPU 模式")
        print("export CUDA_VISIBLE_DEVICES=''")
        print("export PADDLE_USE_GPU='0'")
        return False

def set_cpu_mode():
    """
    设置 CPU 模式环境变量
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PADDLE_USE_GPU'] = '0'
    print("已设置为 CPU 模式")

def main():
    print("=== GPU 兼容性检测 ===")
    
    if not check_gpu_compatibility():
        print("建议: 如需使用 GPU 训练，请安装支持当前 GPU 架构的 PaddlePaddle 版本")
        print("已切换到 CPU 模式，继续执行")
        return True  # 成功切换到 CPU 模式也算成功
    
    print("GPU 兼容性检测通过")
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)