"""
训练模块 - YOLOv8模型训练
包含完整的训练流程、EMA、学习率调度和可视化
"""
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import yaml
import json
from loguru import logger


class Trainer:
    """YOLO模型训练器"""
    
    def __init__(
        self,
        model='yolov8n-pose.pt',
        data_yaml='datasets/dataset.yaml',
        project_name='runs/train',
        epochs=100,
        batch_size=16,
        img_size=640,
        device='cuda',
        use_ema=True,
        optimizer_type='Auto',
        scheduler_type='cosine',
        save_period=10,
    ):
        """
        Args:
            model: YOLO模型实例或模型路径
            data_yaml: 数据集配置文件路径
            project_name: 项目输出目录
            epochs: 训练轮数
            batch_size: 批次大小
            img_size: 图像大小
            device: 设备 ('cuda' 或 'cpu')
            use_ema: 是否使用EMA
            optimizer_type: 优化器类型
            scheduler_type: 学习率调度器类型
            save_period: 保存周期
        """
        self.model_path = model
        self.data_yaml = data_yaml
        self.project_name = project_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.device = device
        self.use_ema = use_ema
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.save_period = save_period
        
        # 创建输出目录
        self.output_dir = Path(project_name) / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史记录
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'mAP50': [],
            'mAP50_95': [],
        }
        
        logger.info(f"训练器初始化完成")
        logger.info(f"模型: {model}")
        logger.info(f"数据集: {data_yaml}")
        logger.info(f"输出目录: {self.output_dir}")
    
    def train(self):
        """开始训练"""
        logger.info("=" * 50)
        logger.info("开始训练...")
        logger.info(f"Epochs: {self.epochs}")
        logger.info(f"Batch Size: {self.batch_size}")
        logger.info(f"Image Size: {self.img_size}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 50)
        
        # 检查CUDA
        if self.device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA不可用，切换到CPU")
            self.device = 'cpu'
        
        if self.device == 'cuda':
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # 加载模型
        logger.info(f"加载模型: {self.model_path}")
        model = YOLO(self.model_path)
        
        # 处理数据集路径 - 确保使用绝对路径
        data_yaml_path = Path(self.data_yaml).resolve()
        logger.info(f"数据集配置: {data_yaml_path}")
        
        # 开始训练
        results = model.train(
            data=str(data_yaml_path),
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            device=self.device,
            project=str(self.output_dir.parent),
            name=self.output_dir.name,
            exist_ok=True,
            pretrained=True,
            optimizer=self.optimizer_type,
            cos_lr=(self.scheduler_type == 'cosine'),
            save_period=self.save_period,
            plots=True,
            verbose=True,
        )
        
        # 保存训练结果
        self._save_results(results)
        
        logger.info("=" * 50)
        logger.info("训练完成!")
        logger.info(f"最佳模型保存在: {self.output_dir / 'weights' / 'best.pt'}")
        logger.info("=" * 50)
        
        return results

    
    def _save_results(self, results):
        """保存训练结果"""
        # 保存训练配置
        config = {
            'model': self.model_path,
            'data_yaml': self.data_yaml,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'img_size': self.img_size,
            'device': self.device,
            'optimizer': self.optimizer_type,
            'scheduler': self.scheduler_type,
        }
        
        config_path = self.output_dir / 'train_config.json'
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练配置已保存: {config_path}")
    
    def plot_results(self):
        """绘制训练结果图表"""
        results_csv = self.output_dir / 'results.csv'
        if not results_csv.exists():
            logger.warning("未找到results.csv文件")
            return
        
        import pandas as pd
        df = pd.read_csv(results_csv)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss曲线
        if 'train/box_loss' in df.columns:
            axes[0, 0].plot(df['train/box_loss'], label='Box Loss')
            axes[0, 0].plot(df['train/cls_loss'], label='Cls Loss')
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # mAP曲线
        if 'metrics/mAP50(B)' in df.columns:
            axes[0, 1].plot(df['metrics/mAP50(B)'], label='mAP50')
            axes[0, 1].plot(df['metrics/mAP50-95(B)'], label='mAP50-95')
            axes[0, 1].set_title('mAP Metrics')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('mAP')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Precision/Recall
        if 'metrics/precision(B)' in df.columns:
            axes[1, 0].plot(df['metrics/precision(B)'], label='Precision')
            axes[1, 0].plot(df['metrics/recall(B)'], label='Recall')
            axes[1, 0].set_title('Precision & Recall')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 学习率
        if 'lr/pg0' in df.columns:
            axes[1, 1].plot(df['lr/pg0'], label='LR')
            axes[1, 1].set_title('Learning Rate')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('LR')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        logger.info(f"训练曲线已保存: {plot_path}")


def check_gpu():
    """检查GPU状态"""
    print("=" * 50)
    print("GPU 检查")
    print("=" * 50)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        print(f"  显存: {props.total_memory / 1024**3:.1f} GB")
        print(f"  CUDA版本: {torch.version.cuda}")
    else:
        print("✗ CUDA 不可用，将使用CPU训练（速度很慢）")
    
    print("=" * 50)
    return torch.cuda.is_available()
