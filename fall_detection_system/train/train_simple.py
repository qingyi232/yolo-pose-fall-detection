"""
简化训练脚本 - 使用硬编码的绝对路径
"""
from ultralytics import YOLO

# 客户电脑上数据集的绝对路径
DATASET_PATH = r"C:\Users\dell\Desktop\基于Yolo与Yolo-pose姿态估计的摔倒检测与区域闯入的系统设计与实现 - 副本(3)\基于Yolo与Yolo-pose姿态估计的摔倒检测与区域闯入的系统设计与实现 - 副本\fall_detection_system\train\fall-detection.v2i.yolov8"

def main():
    print("=" * 60)
    print("  YOLOv8 摔倒检测模型训练")
    print("=" * 60)
    print()
    print(f"数据集路径: {DATASET_PATH}")
    print()
    
    # 创建data.yaml内容 - 使用绝对路径
    data_yaml = f"""
path: {DATASET_PATH}
train: train/images
val: train/images
nc: 1
names:
  0: Fall
"""
    
    # 保存到数据集目录
    yaml_path = DATASET_PATH + r"\data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(data_yaml)
    
    print(f"配置文件已更新: {yaml_path}")
    print()
    
    input("按 Enter 开始训练，按 Ctrl+C 取消...")
    print()
    
    # 加载预训练模型
    model = YOLO('yolov8n.pt')
    
    # 开始训练
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
    )
    
    print()
    print("=" * 60)
    print("训练完成！")
    print("模型位置: runs/train/fall_model/weights/best.pt")
    print("=" * 60)

if __name__ == '__main__':
    main()
