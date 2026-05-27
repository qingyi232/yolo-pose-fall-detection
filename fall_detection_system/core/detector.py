"""
模块1: 多目标检测与人体姿态估计模块
"""
import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger
from .config_manager import config

class PersonDetector:
    """人体检测器 - 使用YOLOv8"""
    
    def __init__(self):
        model_path = config.get('model.yolo_detect', 'yolov8n.pt')
        self.device = config.get('model.device', 'cpu')
        self.confidence = config.get('model.confidence', 0.5)
        self.iou_threshold = config.get('model.iou_threshold', 0.45)
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"目标检测模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"目标检测模型加载失败: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> list:
        """
        检测画面中的人体
        返回: [(x1, y1, x2, y2, confidence), ...]
        """
        results = self.model(
            frame, 
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            classes=[0],  # 只检测人
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    detections.append((int(x1), int(y1), int(x2), int(y2), conf))
        
        return detections


class PoseEstimator:
    """人体姿态估计器 - 使用YOLOv8-Pose"""
    
    # 关键点索引
    KEYPOINTS = {
        'nose': 0, 'left_eye': 1, 'right_eye': 2,
        'left_ear': 3, 'right_ear': 4, 'left_shoulder': 5,
        'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11,
        'right_hip': 12, 'left_knee': 13, 'right_knee': 14,
        'left_ankle': 15, 'right_ankle': 16
    }
    
    def __init__(self):
        model_path = config.get('model.yolo_pose', 'yolov8n-pose.pt')
        self.device = config.get('model.device', 'cpu')
        self.confidence = config.get('model.confidence', 0.5)
        self.keypoint_conf = config.get('fall_detection.keypoint_confidence', 0.3)
        
        try:
            self.model = YOLO(model_path)
            logger.info(f"姿态估计模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"姿态估计模型加载失败: {e}")
            raise
    
    def estimate(self, frame: np.ndarray) -> list:
        """
        估计画面中人体的姿态
        返回: [{'bbox': (x1,y1,x2,y2), 'keypoints': np.array, 'confidence': float}, ...]
        """
        results = self.model(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )
        
        poses = []
        for result in results:
            if result.keypoints is not None:
                keypoints = result.keypoints.data.cpu().numpy()
                boxes = result.boxes
                
                for i, kpts in enumerate(keypoints):
                    if boxes is not None and i < len(boxes):
                        box = boxes[i]
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        
                        poses.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'keypoints': kpts,  # shape: (17, 3) - x, y, confidence
                            'confidence': conf
                        })
        
        return poses
    
    def get_keypoint(self, keypoints: np.ndarray, name: str) -> tuple:
        """获取指定关键点坐标"""
        idx = self.KEYPOINTS.get(name)
        if idx is not None and idx < len(keypoints):
            x, y, conf = keypoints[idx]
            if conf >= self.keypoint_conf:
                return (int(x), int(y), conf)
        return None
    
    def calculate_angle(self, p1: tuple, p2: tuple, p3: tuple) -> float:
        """计算三点形成的角度"""
        if None in [p1, p2, p3]:
            return None
        
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        return np.degrees(angle)
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray, color=(0, 255, 0)):
        """在画面上绘制骨架"""
        # 骨架连接关系
        skeleton = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
            (5, 11), (6, 12), (11, 12),  # 躯干
            (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
        ]
        
        # 绘制关键点
        for i, (x, y, conf) in enumerate(keypoints):
            if conf >= self.keypoint_conf:
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        
        # 绘制骨架线
        for i, j in skeleton:
            if (keypoints[i][2] >= self.keypoint_conf and 
                keypoints[j][2] >= self.keypoint_conf):
                pt1 = (int(keypoints[i][0]), int(keypoints[i][1]))
                pt2 = (int(keypoints[j][0]), int(keypoints[j][1]))
                cv2.line(frame, pt1, pt2, color, 2)
        
        return frame
