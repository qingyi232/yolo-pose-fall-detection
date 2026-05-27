"""
摔倒检测与区域闯入系统 - 主程序
基于YOLO与YOLO-Pose姿态估计
"""
import cv2
import time
import threading
import argparse
from pathlib import Path
from loguru import logger

# 配置日志
logger.add("logs/system_{time}.log", rotation="10 MB", retention="7 days")

from core.config_manager import config
from core.video_stream import VideoStreamManager
from core.detector import PersonDetector, PoseEstimator
from core.fall_detector import FallDetector
from core.intrusion_detector import IntrusionDetector
from core.alert_manager import AlertManager, Event
from core.monitor import SystemMonitor, ServiceHealth
from core.database import DatabaseManager
from api.routes import app, socketio, init_api, broadcast_event


class FallDetectionSystem:
    """摔倒检测系统主类"""
    
    def __init__(self):
        self.config = config
        self.running = False
        self.processing_thread = None
        
        # 初始化各模块
        logger.info("正在初始化系统模块...")
        
        # 视频流管理
        self.video_manager = VideoStreamManager()
        self.video_manager.init_from_config()
        
        # 检测器
        self.person_detector = PersonDetector()
        self.fall_detector = FallDetector()
        self.intrusion_detector = IntrusionDetector()
        
        # 预警管理
        self.alert_manager = AlertManager()
        self.alert_manager.register_callback(broadcast_event)
        
        # 系统监控
        self.system_monitor = SystemMonitor()
        self.service_health = ServiceHealth()
        self._register_health_checks()
        
        # 处理后的帧缓存
        self.processed_frames = {}
        
        logger.info("系统初始化完成")
    
    def _register_health_checks(self):
        """注册健康检查"""
        self.service_health.register_service(
            'video_streams',
            lambda: any(s.is_running() for s in self.video_manager.streams.values())
        )
        self.service_health.register_service(
            'detection_system',
            lambda: self.running
        )
    
    def start(self):
        """启动系统"""
        if self.running:
            logger.warning("系统已在运行中")
            return
        
        logger.info("正在启动系统...")
        self.running = True
        
        # 启动视频流
        self.video_manager.start_all()
        
        # 启动系统监控
        self.system_monitor.start()
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        logger.info("系统启动成功")
    
    def stop(self):
        """停止系统"""
        logger.info("正在停止系统...")
        self.running = False
        
        # 停止视频流
        self.video_manager.stop_all()
        
        # 停止系统监控
        self.system_monitor.stop()
        
        # 等待处理线程结束
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        
        logger.info("系统已停止")
    
    def _processing_loop(self):
        """主处理循环"""
        while self.running:
            for camera_id, stream in self.video_manager.streams.items():
                if not stream.is_running():
                    continue
                
                frame = stream.get_latest_frame()
                if frame is None:
                    continue
                
                try:
                    # 处理帧
                    processed_frame = self._process_frame(frame, camera_id)
                    self.processed_frames[camera_id] = processed_frame
                except Exception as e:
                    logger.error(f"处理帧失败: {e}")
            
            time.sleep(0.03)  # ~30 FPS
    
    def _process_frame(self, frame, camera_id: int):
        """处理单帧"""
        result_frame = frame.copy()
        
        # 1. 摔倒检测
        if config.get('fall_detection.enabled', True):
            fall_results = self.fall_detector.detect(frame)
            
            for i, result in enumerate(fall_results):
                bbox = result['bbox']
                is_falling = result['is_falling']
                keypoints = result['keypoints']
                
                # 绘制骨架
                self.fall_detector.pose_estimator.draw_skeleton(
                    result_frame, keypoints,
                    color=(0, 0, 255) if is_falling else (0, 255, 0)
                )
                
                # 绘制边界框
                color = (0, 0, 255) if is_falling else (0, 255, 0)
                cv2.rectangle(result_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # 摔倒预警
                if is_falling:
                    cv2.putText(result_frame, "FALL DETECTED!", (bbox[0], bbox[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # 触发事件
                    event = Event(
                        event_type=Event.EVENT_FALL,
                        camera_id=camera_id,
                        details={'confidence': result['confidence'], 'bbox': bbox}
                    )
                    self.alert_manager.trigger_event(event)
        
        # 2. 区域闯入检测
        if config.get('intrusion.enabled', True):
            # 绘制检测区域
            result_frame = self.intrusion_detector.draw_zones(result_frame)
            
            # 使用人体检测结果
            detections = self.person_detector.detect(frame)
            intrusions = self.intrusion_detector.detect(detections)
            
            # 绘制闯入警告
            result_frame = self.intrusion_detector.draw_intrusions(result_frame, intrusions)
            
            # 触发闯入事件
            for intrusion in intrusions:
                event = Event(
                    event_type=Event.EVENT_INTRUSION,
                    camera_id=camera_id,
                    details={
                        'zone_id': intrusion['zone_id'],
                        'zone_name': intrusion['zone_name'],
                        'bbox': intrusion['bbox']
                    }
                )
                self.alert_manager.trigger_event(event)
        
        # 绘制信息
        self._draw_info(result_frame, camera_id)
        
        return result_frame
    
    def _draw_info(self, frame, camera_id: int):
        """绘制信息叠加层"""
        # 时间戳
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 摄像头ID
        cv2.putText(frame, f"Camera: {camera_id}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def get_processed_frame(self, camera_id: int):
        """获取处理后的帧"""
        return self.processed_frames.get(camera_id)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='摔倒检测与区域闯入系统')
    parser.add_argument('--host', default='0.0.0.0', help='API服务地址')
    parser.add_argument('--port', type=int, default=5000, help='API服务端口')
    parser.add_argument('--no-gui', action='store_true', help='无GUI模式')
    args = parser.parse_args()
    
    # 创建必要目录
    Path('logs').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    Path('snapshots').mkdir(exist_ok=True)
    
    # 初始化系统
    system = FallDetectionSystem()
    
    # 注入到API
    init_api(system)
    
    # 启动系统
    system.start()
    
    try:
        # 启动API服务（使用eventlet或gevent异步模式）
        logger.info(f"API服务启动: http://{args.host}:{args.port}")
        socketio.run(app, host=args.host, port=args.port, debug=False, 
                    allow_unsafe_werkzeug=True, use_reloader=False)
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    finally:
        system.stop()


if __name__ == '__main__':
    main()
