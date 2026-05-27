"""
模块2: 视频流处理与数据采集模块
"""
import cv2
import threading
import queue
import time
import numpy as np
from loguru import logger
from .config_manager import config

class VideoStream:
    """视频流处理类"""
    
    def __init__(self, source_id: int, source_url, name: str = "Camera"):
        self.source_id = source_id
        self.source_url = source_url
        self.name = name
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=10)
        self.running = False
        self.thread = None
        self.fps = config.get('video.fps', 30)
        self.frame_width = config.get('video.frame_width', 1280)
        self.frame_height = config.get('video.frame_height', 720)
        self.last_frame = None
        self.frame_count = 0
    
    def start(self):
        """启动视频流"""
        try:
            self.cap = cv2.VideoCapture(self.source_url)
            if not self.cap.isOpened():
                logger.error(f"无法打开视频源: {self.source_url}")
                return False
            
            # 设置分辨率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            logger.info(f"视频流启动成功: {self.name}")
            return True
        except Exception as e:
            logger.error(f"视频流启动失败: {e}")
            return False
    
    def stop(self):
        """停止视频流"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        logger.info(f"视频流已停止: {self.name}")
    
    def _capture_loop(self):
        """视频采集循环"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning(f"读取帧失败: {self.name}")
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            self.last_frame = frame
            
            # 非阻塞放入队列
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame)
                except:
                    pass
    
    def get_frame(self) -> np.ndarray:
        """获取最新帧"""
        try:
            return self.frame_queue.get(timeout=1)
        except queue.Empty:
            return self.last_frame
    
    def get_latest_frame(self) -> np.ndarray:
        """获取最新帧（非阻塞）"""
        return self.last_frame
    
    def is_running(self) -> bool:
        return self.running and self.cap is not None and self.cap.isOpened()


class VideoStreamManager:
    """视频流管理器"""
    
    def __init__(self):
        self.streams = {}
    
    def init_from_config(self):
        """从配置初始化所有视频流"""
        sources = config.get('video.sources', [])
        for source in sources:
            if source.get('enabled', True):
                self.add_stream(
                    source['id'],
                    source['url'],
                    source.get('name', f"Camera_{source['id']}")
                )
    
    def add_stream(self, source_id: int, source_url, name: str = None):
        """添加视频流"""
        if source_id in self.streams:
            logger.warning(f"视频流已存在: {source_id}")
            return
        
        stream = VideoStream(source_id, source_url, name or f"Camera_{source_id}")
        self.streams[source_id] = stream
        logger.info(f"添加视频流: {name}")
    
    def start_all(self):
        """启动所有视频流"""
        for stream in self.streams.values():
            stream.start()
    
    def stop_all(self):
        """停止所有视频流"""
        for stream in self.streams.values():
            stream.stop()
    
    def get_stream(self, source_id: int) -> VideoStream:
        """获取指定视频流"""
        return self.streams.get(source_id)
    
    def get_all_streams(self) -> dict:
        """获取所有视频流"""
        return self.streams
    
    def get_status(self) -> list:
        """获取所有视频流状态"""
        status = []
        for sid, stream in self.streams.items():
            status.append({
                'id': sid,
                'name': stream.name,
                'running': stream.is_running(),
                'frame_count': stream.frame_count
            })
        return status


class VideoRecorder:
    """视频录制器"""
    
    def __init__(self, output_path: str, fps: int = 30, resolution: tuple = (1280, 720)):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self.recording = False
    
    def start(self):
        """开始录制"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.output_path, fourcc, self.fps, self.resolution
        )
        self.recording = True
        logger.info(f"开始录制: {self.output_path}")
    
    def write_frame(self, frame: np.ndarray):
        """写入帧"""
        if self.recording and self.writer:
            # 调整分辨率
            if frame.shape[:2][::-1] != self.resolution:
                frame = cv2.resize(frame, self.resolution)
            self.writer.write(frame)
    
    def stop(self):
        """停止录制"""
        if self.writer:
            self.writer.release()
        self.recording = False
        logger.info(f"录制完成: {self.output_path}")
