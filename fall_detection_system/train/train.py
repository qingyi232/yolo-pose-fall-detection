"""
训练脚本 - 一键启动训练
使用方法: python train.py
"""
import os
import sys
import yaml
from pathlib import Path

# 获取train目录的绝对路径
TRAIN_DIR = Path(__file__).parent.resolve()

# 切换工作目录到train文件夹
os.chdir(TRAIN_DIR)

# 添加项目根目录到路径
sys.path.insert(0, str(TRAIN_DIR))

from trainer import Trainer, check_gpu


def create_data_yaml(dataset_path):
    """
    创建正确的data.yaml配置文件
    YOLOv8要求path是绝对路径
    """
    # 使用绝对路径
    abs_path = dataset_path.resolve()
    
    config = {
        'path': str(abs_path),  # 绝对路径！
        'train': 'train/images',
        'val': 'valid/images', 
        'test': 'test/images',
        'nc': 1,
        'names': {0: 'Fall'}
    }
    
    # 写入配置文件
    yaml_path = dataset_path / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"已更新数据集配置: {yaml_path}")
    print(f"数据集绝对路径: {abs_path}")
    
    return yaml_path


def main():
    """主训练函数"""
    print("=" * 60)
    print("  YOLOv8 摔倒检测模型训练")
    print("=" * 60)
    print()
    
    # 检查GPU
    has_gpu = check_gpu()
    print()
    
    # 数据集路径
    dataset_name = 'fall-detection.v2i.yolov8'
    dataset_path = TRAIN_DIR / dataset_name
    
    # 检查数据集是否存在
    if not dataset_path.exists():
        print(f"错误: 找不到数据集文件夹: {dataset_path}")
        print("请确保 fall-detection.v2i.yolov8 文件夹在 train 目录下")
        return
    
    # 检查图片文件夹
    train_images = dataset_path / 'train' / 'images'
    val_images = dataset_path / 'valid' / 'images'
    
    if not train_images.exists():
        print(f"错误: 找不到训练图片文件夹: {train_images}")
        return
    
    # 统计图片数量
    train_count = len(list(train_images.glob('*')))
    val_count = len(list(val_images.glob('*'))) if val_images.exists() else 0
    
    print(f"工作目录: {TRAIN_DIR}")
    print(f"数据集路径: {dataset_path}")
    print(f"训练图片: {train_count} 张")
    print(f"验证图片: {val_count} 张")
    print()
    
    # 创建/更新data.yaml（使用绝对路径）
    data_yaml = create_data_yaml(dataset_path)
    print()
    
    # 训练配置
    config = {
        'model': 'yolov8n.pt',
        'data_yaml': str(data_yaml),
        'epochs': 100,
        'batch_size': 8 if has_gpu else 4,
        'img_size': 640 if has_gpu else 416,
        'device': 'cuda' if has_gpu else 'cpu',
        'save_period': 10,
    }
    
    print("训练配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()
    
    # 确认开始训练
    input("按 Enter 开始训练，按 Ctrl+C 取消...")
    print()
    
    # 创建训练器并开始训练
    trainer = Trainer(**config)
    results = trainer.train()
    
    # 绘制训练曲线
    trainer.plot_results()
    
    print()
    print("=" * 60)
    print("训练完成！")
    print(f"模型保存在: runs/train/*/weights/best.pt")
    print("=" * 60)


if __name__ == '__main__':
    main()
