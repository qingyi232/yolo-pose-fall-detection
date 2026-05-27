"""
模块4: 摔倒检测算法模块
"""
import numpy as np
from collections import deque
from loguru import logger
from .config_manager import config
from .detector import PoseEstimator

class FallDetector:
    """摔倒检测器"""
    
    def __init__(self):
        self.pose_estimator = PoseEstimator()
        self.angle_threshold = config.get('fall_detection.angle_threshold', 45)
        self.height_ratio_threshold = config.get('fall_detection.height_ratio_threshold', 1.0)
        self.confirm_frames = config.get('fall_detection.confirm_frames', 5)
        
        # 每个人的状态追踪 {person_id: deque([is_falling, ...])}
        self.person_states = {}
        self.person_id_counter = 0
    
    def detect(self, frame: np.ndarray) -> list:
        """
        检测摔倒
        返回: [{'bbox': tuple, 'is_falling': bool, 'confidence': float, 'keypoints': array}, ...]
        """
        poses = self.pose_estimator.estimate(frame)
        results = []
        
        for pose in poses:
            bbox = pose['bbox']
            keypoints = pose['keypoints']
            
            # 多种摔倒判断方法
            fall_score = 0
            checks = 0
            
            # 方法1: 身体高宽比
            ratio_fall = self._check_height_ratio(bbox)
            if ratio_fall is not None:
                fall_score += ratio_fall
                checks += 1
            
            # 方法2: 躯干角度
            trunk_fall = self._check_trunk_angle(keypoints)
            if trunk_fall is not None:
                fall_score += trunk_fall
                checks += 1
            
            # 方法3: 头部位置
            head_fall = self._check_head_position(keypoints, bbox)
            if head_fall is not None:
                fall_score += head_fall
                checks += 1
            
            # 方法4: 关键点分布
            dist_fall = self._check_keypoint_distribution(keypoints)
            if dist_fall is not None:
                fall_score += dist_fall
                checks += 1
            
            # 综合判断
            is_falling = False
            confidence = 0
            if checks > 0:
                confidence = fall_score / checks
                is_falling = confidence > 0.5
            
            results.append({
                'bbox': bbox,
                'is_falling': is_falling,
                'confidence': confidence,
                'keypoints': keypoints
            })
        
        return results
    
    def _check_height_ratio(self, bbox: tuple) -> float:
        """检查身体高宽比"""
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        
        if width == 0:
            return None
        
        # 如果检测框太小（只有头部），不判断
        if height < 100 or width < 50:
            return None
        
        ratio = height / width
        # 正常站立时高度应该大于宽度
        # 摔倒时宽度可能大于高度
        if ratio < self.height_ratio_threshold:
            return 1.0  # 可能摔倒
        return 0.0
    
    def _check_trunk_angle(self, keypoints: np.ndarray) -> float:
        """检查躯干角度"""
        # 获取肩部和髋部中点
        left_shoulder = self.pose_estimator.get_keypoint(keypoints, 'left_shoulder')
        right_shoulder = self.pose_estimator.get_keypoint(keypoints, 'right_shoulder')
        left_hip = self.pose_estimator.get_keypoint(keypoints, 'left_hip')
        right_hip = self.pose_estimator.get_keypoint(keypoints, 'right_hip')
        
        if None in [left_shoulder, right_shoulder, left_hip, right_hip]:
            return None
        
        # 计算肩部和髋部中点
        shoulder_mid = ((left_shoulder[0] + right_shoulder[0]) / 2,
                       (left_shoulder[1] + right_shoulder[1]) / 2)
        hip_mid = ((left_hip[0] + right_hip[0]) / 2,
                  (left_hip[1] + right_hip[1]) / 2)
        
        # 计算躯干与垂直方向的角度
        dx = shoulder_mid[0] - hip_mid[0]
        dy = shoulder_mid[1] - hip_mid[1]
        
        # 躯干角度（与垂直方向的夹角）
        angle = np.degrees(np.arctan2(abs(dx), abs(dy)))
        
        if angle > self.angle_threshold:
            return min(angle / 90, 1.0)  # 归一化到0-1
        return 0.0
    
    def _check_head_position(self, keypoints: np.ndarray, bbox: tuple) -> float:
        """检查头部位置是否异常低"""
        nose = self.pose_estimator.get_keypoint(keypoints, 'nose')
        left_hip = self.pose_estimator.get_keypoint(keypoints, 'left_hip')
        right_hip = self.pose_estimator.get_keypoint(keypoints, 'right_hip')
        
        if nose is None or (left_hip is None and right_hip is None):
            return None
        
        # 计算髋部高度
        if left_hip and right_hip:
            hip_y = (left_hip[1] + right_hip[1]) / 2
        else:
            hip_y = (left_hip or right_hip)[1]
        
        # 正常情况下头部应该在髋部上方
        # 如果头部接近或低于髋部，可能是摔倒
        if nose[1] > hip_y - 20:  # 头部低于髋部
            return 1.0
        return 0.0
    
    def _check_keypoint_distribution(self, keypoints: np.ndarray) -> float:
        """检查关键点的水平分布"""
        valid_points = []
        for kpt in keypoints:
            if kpt[2] > 0.3:  # 置信度阈值
                valid_points.append((kpt[0], kpt[1]))
        
        # 至少需要8个有效关键点才能判断（需要看到躯干）
        if len(valid_points) < 8:
            return None
        
        points = np.array(valid_points)
        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        
        if y_range == 0:
            return 1.0
        
        # 水平分布大于垂直分布时可能是躺倒
        ratio = x_range / y_range
        if ratio > 1.5:
            return min(ratio / 3, 1.0)
        return 0.0
    
    def update_tracking(self, person_id: int, is_falling: bool) -> bool:
        """
        更新追踪状态，返回是否确认摔倒
        使用连续帧确认机制减少误报
        """
        if person_id not in self.person_states:
            self.person_states[person_id] = deque(maxlen=self.confirm_frames)
        
        self.person_states[person_id].append(is_falling)
        
        # 检查是否连续多帧都检测到摔倒
        if len(self.person_states[person_id]) >= self.confirm_frames:
            fall_count = sum(self.person_states[person_id])
            if fall_count >= self.confirm_frames * 0.8:  # 80%的帧检测到摔倒
                return True
        
        return False
    
    def reset_tracking(self, person_id: int = None):
        """重置追踪状态"""
        if person_id:
            self.person_states.pop(person_id, None)
        else:
            self.person_states.clear()
