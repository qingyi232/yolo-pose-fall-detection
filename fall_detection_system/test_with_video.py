"""
使用视频文件测试摔倒检测系统
无需摄像头，可用于演示和测试
"""
import cv2
import sys
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from core.fall_detector import FallDetector
from core.intrusion_detector import IntrusionDetector

def test_with_video(video_path: str):
    """使用视频文件测试"""
    print(f"正在加载视频: {video_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {video_path}")
        return
    
    print("正在初始化检测器（首次运行会下载模型）...")
    fall_detector = FallDetector()
    intrusion_detector = IntrusionDetector()
    
    # 添加测试检测区域
    intrusion_detector.add_zone("禁止区域", [[50, 50], [300, 50], [300, 300], [50, 300]])
    
    print("开始检测... 按 'q' 退出, 按 'p' 暂停")
    
    frame_count = 0
    fall_count = 0
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("视频播放完毕，重新开始...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame_count += 1
            
            # 摔倒检测
            results = fall_detector.detect(frame)
            
            for result in results:
                bbox = result['bbox']
                is_falling = result['is_falling']
                confidence = result['confidence']
                keypoints = result['keypoints']
                
                # 绘制骨架
                color = (0, 0, 255) if is_falling else (0, 255, 0)
                fall_detector.pose_estimator.draw_skeleton(frame, keypoints, color)
                
                # 绘制边界框
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # 显示状态
                status = f"FALL! ({confidence:.2f})" if is_falling else f"Normal ({confidence:.2f})"
                cv2.putText(frame, status, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if is_falling:
                    fall_count += 1
            
            # 绘制检测区域
            frame = intrusion_detector.draw_zones(frame)
            
            # 区域闯入检测
            if results:
                intrusions = intrusion_detector.detect(results)
                frame = intrusion_detector.draw_intrusions(frame, intrusions)
            
            # 显示信息
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Falls detected: {fall_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 'p' to pause", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Fall Detection - Video Test', frame)
        
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("暂停" if paused else "继续")
        elif key == ord('s'):
            # 保存截图
            filename = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"截图已保存: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n检测完成! 共检测到 {fall_count} 次摔倒事件")


def test_with_image(image_path: str):
    """使用单张图片测试"""
    print(f"正在加载图片: {image_path}")
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"错误: 无法读取图片 {image_path}")
        return
    
    print("正在初始化检测器...")
    fall_detector = FallDetector()
    
    # 检测
    results = fall_detector.detect(frame)
    
    print(f"\n检测到 {len(results)} 个人:")
    for i, result in enumerate(results):
        bbox = result['bbox']
        is_falling = result['is_falling']
        confidence = result['confidence']
        keypoints = result['keypoints']
        
        print(f"  人物{i+1}: 摔倒={is_falling}, 置信度={confidence:.2f}")
        
        # 绘制
        color = (0, 0, 255) if is_falling else (0, 255, 0)
        fall_detector.pose_estimator.draw_skeleton(frame, keypoints, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        label = f"{'FALL' if is_falling else 'OK'}: {confidence:.2f}"
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.imshow('Detection Result', frame)
    print("\n按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_with_camera(camera_id: int = 0):
    """使用摄像头测试"""
    print(f"正在打开摄像头 {camera_id}...")
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"错误: 无法打开摄像头 {camera_id}")
        print("提示: 请检查摄像头是否连接，或尝试其他ID (0, 1, 2...)")
        return
    
    print("正在初始化检测器...")
    fall_detector = FallDetector()
    intrusion_detector = IntrusionDetector()
    
    # 不再重复添加测试区域，使用配置文件中的区域
    
    print("检测中... 按 'q' 退出, 按 'f' 全屏")
    
    # 创建可调整大小的窗口
    cv2.namedWindow('Fall Detection - Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Fall Detection - Camera', 1280, 720)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测
        results = fall_detector.detect(frame)
        
        for result in results:
            bbox = result['bbox']
            is_falling = result['is_falling']
            keypoints = result['keypoints']
            
            color = (0, 0, 255) if is_falling else (0, 255, 0)
            fall_detector.pose_estimator.draw_skeleton(frame, keypoints, color)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            if is_falling:
                cv2.putText(frame, "FALL DETECTED!", (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # 绘制区域
        frame = intrusion_detector.draw_zones(frame)
        
        cv2.imshow('Fall Detection - Camera', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('f'):
            cv2.setWindowProperty('Fall Detection - Camera', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("=" * 50)
    print("摔倒检测系统 - 测试工具")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if input_path.isdigit():
            # 摄像头ID
            test_with_camera(int(input_path))
        elif input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            # 视频文件
            test_with_video(input_path)
        elif input_path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            # 图片文件
            test_with_image(input_path)
        else:
            print(f"不支持的文件格式: {input_path}")
    else:
        print("\n使用方法:")
        print("  python test_with_video.py <视频文件>    # 测试视频")
        print("  python test_with_video.py <图片文件>    # 测试图片")
        print("  python test_with_video.py 0            # 测试摄像头0")
        print("\n示例:")
        print("  python test_with_video.py fall_video.mp4")
        print("  python test_with_video.py test.jpg")
        print("  python test_with_video.py 0")
        print("\n没有测试素材? 可以从网上下载摔倒检测数据集视频")
        print("或者直接运行摄像头测试:")
        
        choice = input("\n是否使用摄像头测试? (y/n): ").strip().lower()
        if choice == 'y':
            test_with_camera(0)
