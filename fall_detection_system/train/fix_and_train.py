"""
修复数据集并训练
1. 将图片和标签重命名为简单的数字文件名
2. 开始训练
"""
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# 数据集路径
DATASET_PATH = Path(r"C:\Users\dell\Desktop\基于Yolo与Yolo-pose姿态估计的摔倒检测与区域闯入的系统设计与实现 - 副本(3)\基于Yolo与Yolo-pose姿态估计的摔倒检测与区域闯入的系统设计与实现 - 副本\fall_detection_system\train\fall-detection.v2i.yolov8")

def fix_dataset():
    """重命名数据集中的文件为简单的数字名称"""
    print("正在修复数据集文件名...")
    
    for split in ['train', 'valid', 'test']:
        images_dir = DATASET_PATH / split / 'images'
        labels_dir = DATASET_PATH / split / 'labels'
        
        if not images_dir.exists():
            print(f"跳过 {split}: 目录不存在")
            continue
        
        # 获取所有图片
        images = list(images_dir.glob('*.*'))
        print(f"{split}: 找到 {len(images)} 张图片")
        
        # 重命名
        for i, img_path in enumerate(images):
            ext = img_path.suffix.lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue
            
            # 新文件名
            new_name = f"{split}_{i:05d}"
            new_img = images_dir / f"{new_name}{ext}"
            
            # 对应的标签文件
            old_label = labels_dir / f"{img_path.stem}.txt"
            new_label = labels_dir / f"{new_name}.txt"
            
            # 重命名图片
            if img_path != new_img and not new_img.exists():
                try:
                    img_path.rename(new_img)
                except Exception as e:
                    print(f"  重命名失败: {img_path.name} -> {e}")
                    continue
            
            # 重命名标签
            if old_label.exists() and old_label != new_label and not new_label.exists():
                try:
                    old_label.rename(new_label)
                except:
                    pass
    
    print("数据集修复完成！")
    print()

def create_yaml():
    """创建data.yaml"""
    yaml_content = f"""path: {DATASET_PATH}
train: train/images
val: train/images
nc: 1
names:
  0: Fall
"""
    yaml_path = DATASET_PATH / 'data.yaml'
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)
    print(f"配置文件已创建: {yaml_path}")
    return str(yaml_path)

def train(yaml_path):
    """开始训练"""
    print()
    print("=" * 50)
    print("开始训练...")
    print("=" * 50)
    
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=yaml_path,
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        project='runs/train',
        name='fall_model',
        exist_ok=True,
        patience=20,
        save=True,
        save_period=5,
        workers=0,  # Windows下设为0避免多进程问题
    )
    
    print()
    print("=" * 50)
    print("训练完成！")
    print("模型位置: runs/train/fall_model/weights/best.pt")
    print("=" * 50)

def main():
    print("=" * 60)
    print("  YOLOv8 摔倒检测模型训练")
    print("=" * 60)
    print()
    
    # 1. 修复数据集
    fix_dataset()
    
    # 2. 创建配置
    yaml_path = create_yaml()
    
    # 3. 确认
    input("按 Enter 开始训练...")
    
    # 4. 训练
    train(yaml_path)

if __name__ == '__main__':
    main()
