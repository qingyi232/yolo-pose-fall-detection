"""
演示测试脚本 - 使用本地摄像头测试系统
"""
import cv2
import time
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.config_manager import config
from core.detector import PersonDetector, PoseEstimator
from core.fall_detector import FallDetector
from core.intrusion_detector import IntrusionDetector

def test_with_camera():
    """使用摄像头测试"""
    print("正在初始化检测器...")
    
    # 初始化检测器
    fall_detector = FallDetector()
    intrusion_detector = IntrusionDetector()
    
    # 添加测试区域
    intrusion_detector.add_zone("测试区域", [[100, 100], [400, 100], [400, 400], [100, 400]])
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return
    
    print("按 'q' 退出, 按 's' 截图")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 摔倒检测
        fall_results = fall_detector.detect(frame)
        
        for result in fall_results:
            bbox = result['bbox']
            is_falling = result['is_falling']
            keypoints = result['keypoints']
            
            # 绘制骨架
            color = (0, 0, 255) if is_falling else (0, 255, 0)
            fall_detector.pose_estimator.draw_skeleton(frame, keypoints, color)
            
            # 绘制边界框
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            if is_falling:
                cv2.putText(frame, "FALL!", (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 绘制检测区域
        frame = intrusion_detector.draw_zones(frame)
        
        # 区域闯入检测
        if fall_results:
            intrusions = intrusion_detector.detect(fall_results)
            frame = intrusion_detector.draw_intrusions(frame, intrusions)
        
        # 显示FPS
        cv2.putText(frame, f"Press 'q' to quit", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('Fall Detection Demo', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f'snapshot_{int(time.time())}.jpg', frame)
            print("截图已保存")
    
    cap.release()
    cv2.destroyAllWindows()


def test_with_image(image_path: str):
    """使用图片测试"""
    print(f"测试图片: {image_path}")
    
    fall_detector = FallDetector()
    
    frame = cv2.imread(image_path)
    if frame is None:
        print("无法读取图片")
        return
    
    # 检测
    results = fall_detector.detect(frame)
    
    for result in results:
        bbox = result['bbox']
        is_falling = result['is_falling']
        confidence = result['confidence']
        keypoints = result['keypoints']
        
        print(f"检测结果: 摔倒={is_falling}, 置信度={confidence:.2f}")
        
        # 绘制
        color = (0, 0, 255) if is_falling else (0, 255, 0)
        fall_detector.pose_estimator.draw_skeleton(frame, keypoints, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        label = f"{'FALL' if is_falling else 'Normal'}: {confidence:.2f}"
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow('Result', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        # 测试图片
        test_with_image(sys.argv[1])
    else:
        # 测试摄像头
        test_with_camera()
