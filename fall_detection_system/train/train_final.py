"""
最终训练脚本
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # 解决OpenMP警告

from ultralytics import YOLO

# 数据集路径
DATASET = r"C:\Users\dell\Desktop\1\train\fall-detection.v2i.yolov8"

def main():
    print("=" * 50)
    print("YOLOv8 摔倒检测训练")
    print("=" * 50)
    print(f"数据集: {DATASET}")
    
    # 创建配置
    yaml = f"""path: {DATASET}
train: train/images
val: train/images
nc: 1
names:
  0: Fall
"""
    yaml_path = DATASET + r"\data.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml)
    
    print("配置已创建")
    input("按Enter开始训练...")
    
    try:
        model = YOLO('yolov8n.pt')
        
        results = model.train(
            data=yaml_path,
            epochs=50,
            imgsz=640,
            batch=8,
            device=0,
            project='runs/train',
            name='fall',
            exist_ok=True,
            workers=0,
            verbose=True,
        )
        
        print()
        print("=" * 50)
        print("训练完成!")
        print("模型位置: runs/train/fall/weights/best.pt")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n按Enter退出...")

if __name__ == '__main__':
    main()
