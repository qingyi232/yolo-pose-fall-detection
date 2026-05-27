"""
数据集准备脚本
用于整合多个数据集到统一格式
"""
import os
import shutil
import random
from pathlib import Path
from loguru import logger


def prepare_fall_detection_dataset(source_dir: str, target_dir: str = 'datasets'):
    """
    准备 fall-detection.v2i.yolov8 数据集
    
    Args:
        source_dir: 解压后的数据集目录
        target_dir: 目标目录
    """
    source = Path(source_dir)
    target = Path(target_dir)
    
    logger.info(f"准备数据集: {source_dir}")
    
    # 创建目标目录
    (target / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (target / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (target / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (target / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # 复制训练集
    train_images = source / 'train' / 'images'
    train_labels = source / 'train' / 'labels'
    
    if train_images.exists():
        for img in train_images.glob('*'):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, target / 'images' / 'train' / img.name)
                # 复制对应的标注文件
                label_file = train_labels / (img.stem + '.txt')
                if label_file.exists():
                    shutil.copy(label_file, target / 'labels' / 'train' / label_file.name)
    
    # 复制验证集
    val_images = source / 'valid' / 'images'
    val_labels = source / 'valid' / 'labels'
    
    if val_images.exists():
        for img in val_images.glob('*'):
            if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                shutil.copy(img, target / 'images' / 'val' / img.name)
                label_file = val_labels / (img.stem + '.txt')
                if label_file.exists():
                    shutil.copy(label_file, target / 'labels' / 'val' / label_file.name)
    
    # 统计
    train_count = len(list((target / 'images' / 'train').glob('*')))
    val_count = len(list((target / 'images' / 'val').glob('*')))
    
    logger.info(f"训练集: {train_count} 张图片")
    logger.info(f"验证集: {val_count} 张图片")
    
    return train_count, val_count


def merge_datasets(dataset_dirs: list, target_dir: str = 'datasets', val_ratio: float = 0.2):
    """
    合并多个数据集
    
    Args:
        dataset_dirs: 数据集目录列表
        target_dir: 目标目录
        val_ratio: 验证集比例
    """
    target = Path(target_dir)
    
    # 创建目标目录
    (target / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (target / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (target / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (target / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    all_images = []
    
    for dataset_dir in dataset_dirs:
        source = Path(dataset_dir)
        logger.info(f"处理数据集: {dataset_dir}")
        
        # 查找所有图片
        for img_dir in ['images', 'train/images', 'valid/images', 'test/images']:
            img_path = source / img_dir
            if img_path.exists():
                for img in img_path.glob('*'):
                    if img.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        # 查找对应的标注文件
                        label_dirs = ['labels', 'train/labels', 'valid/labels', 'test/labels']
                        for label_dir in label_dirs:
                            label_file = source / label_dir / (img.stem + '.txt')
                            if label_file.exists():
                                all_images.append((img, label_file))
                                break
    
    logger.info(f"共找到 {len(all_images)} 张带标注的图片")
    
    # 随机打乱并分割
    random.shuffle(all_images)
    val_count = int(len(all_images) * val_ratio)
    
    val_set = all_images[:val_count]
    train_set = all_images[val_count:]
    
    # 复制文件
    for i, (img, label) in enumerate(train_set):
        new_name = f"train_{i:06d}{img.suffix}"
        shutil.copy(img, target / 'images' / 'train' / new_name)
        shutil.copy(label, target / 'labels' / 'train' / f"train_{i:06d}.txt")
    
    for i, (img, label) in enumerate(val_set):
        new_name = f"val_{i:06d}{img.suffix}"
        shutil.copy(img, target / 'images' / 'val' / new_name)
        shutil.copy(label, target / 'labels' / 'val' / f"val_{i:06d}.txt")
    
    logger.info(f"训练集: {len(train_set)} 张图片")
    logger.info(f"验证集: {len(val_set)} 张图片")
    
    return len(train_set), len(val_set)


def main():
    """主函数 - 交互式准备数据集"""
    print("=" * 60)
    print("  数据集准备工具")
    print("=" * 60)
    print()
    print("请选择操作:")
    print("1. 准备单个数据集 (fall-detection.v2i.yolov8)")
    print("2. 合并多个数据集")
    print()
    
    choice = input("请输入选项 (1/2): ").strip()
    
    if choice == '1':
        source = input("请输入数据集解压后的目录路径: ").strip()
        if os.path.exists(source):
            prepare_fall_detection_dataset(source)
            print("\n数据集准备完成！")
        else:
            print(f"目录不存在: {source}")
    
    elif choice == '2':
        print("请输入数据集目录（每行一个，输入空行结束）:")
        dirs = []
        while True:
            d = input().strip()
            if not d:
                break
            if os.path.exists(d):
                dirs.append(d)
            else:
                print(f"  警告: 目录不存在 {d}")
        
        if dirs:
            merge_datasets(dirs)
            print("\n数据集合并完成！")
        else:
            print("没有有效的数据集目录")
    
    else:
        print("无效选项")


if __name__ == '__main__':
    main()
