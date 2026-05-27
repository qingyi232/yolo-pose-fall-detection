"""
模块5: 区域闯入检测模块
"""
import cv2
import numpy as np
from loguru import logger
from .config_manager import config

class Zone:
    """检测区域"""
    
    def __init__(self, zone_id: int, name: str, points: list, color: list = [255, 0, 0]):
        self.id = zone_id
        self.name = name
        self.points = np.array(points, dtype=np.int32)
        self.color = tuple(color)
    
    def contains_point(self, point: tuple) -> bool:
        """检查点是否在区域内"""
        result = cv2.pointPolygonTest(self.points, point, False)
        return result >= 0
    
    def contains_bbox(self, bbox: tuple, threshold: float = 0.3) -> bool:
        """
        检查边界框是否在区域内
        threshold: 边界框与区域重叠的比例阈值
        """
        x1, y1, x2, y2 = bbox
        
        # 检查中心点
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        if self.contains_point(center):
            return True
        
        # 检查底部中心点（人的脚部位置）
        bottom_center = ((x1 + x2) // 2, y2)
        if self.contains_point(bottom_center):
            return True
        
        # 检查四个角点
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        inside_count = sum(1 for c in corners if self.contains_point(c))
        
        return inside_count / 4 >= threshold
    
    def draw(self, frame: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        """在画面上绘制区域"""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.points], self.color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.polylines(frame, [self.points], True, self.color, 2)
        
        # 绘制区域名称在框内左上角
        x_min = int(np.min(self.points[:, 0]))
        y_min = int(np.min(self.points[:, 1]))
        cv2.putText(frame, f"Zone-{self.id}", (x_min + 10, y_min + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame


class IntrusionDetector:
    """区域闯入检测器"""
    
    def __init__(self):
        self.zones = []
        self.load_zones_from_config()
        self.intrusion_history = {}  # {zone_id: {person_id: timestamp}}
    
    def load_zones_from_config(self):
        """从配置加载区域"""
        zones_config = config.get('intrusion.zones', [])
        self.zones = []
        for z in zones_config:
            zone = Zone(
                zone_id=z['id'],
                name=z['name'],
                points=z['points'],
                color=z.get('color', [255, 0, 0])
            )
            self.zones.append(zone)
        logger.info(f"加载了 {len(self.zones)} 个检测区域")
    
    def add_zone(self, name: str, points: list, color: list = [255, 0, 0]) -> int:
        """添加新区域"""
        zone_id = config.add_zone(name, points, color)
        zone = Zone(zone_id, name, points, color)
        self.zones.append(zone)
        logger.info(f"添加区域: {name}")
        return zone_id
    
    def remove_zone(self, zone_id: int):
        """移除区域"""
        self.zones = [z for z in self.zones if z.id != zone_id]
        config.delete_zone(zone_id)
        logger.info(f"移除区域: {zone_id}")
    
    def update_zone(self, zone_id: int, points: list):
        """更新区域"""
        for zone in self.zones:
            if zone.id == zone_id:
                zone.points = np.array(points, dtype=np.int32)
                config.update_zone(zone_id, points)
                break
    
    def detect(self, detections: list) -> list:
        """
        检测闯入
        detections: [(x1, y1, x2, y2, confidence), ...] 或 [{'bbox': tuple}, ...]
        返回: [{'zone': Zone, 'bbox': tuple, 'intrusion': bool}, ...]
        """
        results = []
        
        for det in detections:
            if isinstance(det, dict):
                bbox = det.get('bbox')
            else:
                bbox = det[:4]
            
            for zone in self.zones:
                is_intrusion = zone.contains_bbox(bbox)
                if is_intrusion:
                    results.append({
                        'zone': zone,
                        'zone_id': zone.id,
                        'zone_name': zone.name,
                        'bbox': bbox,
                        'intrusion': True
                    })
        
        return results
    
    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """绘制所有区域"""
        for zone in self.zones:
            frame = zone.draw(frame)
        return frame
    
    def draw_intrusions(self, frame: np.ndarray, intrusions: list) -> np.ndarray:
        """绘制闯入警告"""
        for intrusion in intrusions:
            bbox = intrusion['bbox']
            zone = intrusion['zone']
            
            # 绘制红色边界框
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         (0, 0, 255), 3)
            
            # 绘制警告文字
            text = f"INTRUSION: {zone.name}"
            cv2.putText(frame, text, (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def get_zones_info(self) -> list:
        """获取所有区域信息"""
        return [{
            'id': z.id,
            'name': z.name,
            'points': z.points.tolist(),
            'color': list(z.color)
        } for z in self.zones]


class ZoneEditor:
    """区域编辑器 - 用于交互式绘制区域"""
    
    def __init__(self, window_name: str = "Zone Editor"):
        self.window_name = window_name
        self.points = []
        self.drawing = False
        self.current_frame = None
    
    def mouse_callback(self, event, x, y, flags, param):
        """鼠标回调函数"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append([x, y])
            self._draw_points()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键完成绘制
            self.drawing = False
    
    def _draw_points(self):
        """绘制当前点"""
        if self.current_frame is None:
            return
        
        frame = self.current_frame.copy()
        
        # 绘制点
        for i, pt in enumerate(self.points):
            cv2.circle(frame, tuple(pt), 5, (0, 255, 0), -1)
            if i > 0:
                cv2.line(frame, tuple(self.points[i-1]), tuple(pt), (0, 255, 0), 2)
        
        # 如果有3个以上的点，绘制闭合多边形
        if len(self.points) >= 3:
            cv2.line(frame, tuple(self.points[-1]), tuple(self.points[0]), (0, 255, 0), 2)
        
        cv2.imshow(self.window_name, frame)
    
    def edit(self, frame: np.ndarray) -> list:
        """
        交互式编辑区域
        左键添加点，右键完成
        返回: 点列表
        """
        self.current_frame = frame.copy()
        self.points = []
        self.drawing = True
        
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.imshow(self.window_name, frame)
        
        print("左键点击添加顶点，右键完成绘制，按ESC取消")
        
        while self.drawing:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                self.points = []
                break
            elif key == 13:  # Enter
                break
        
        cv2.destroyWindow(self.window_name)
        return self.points
