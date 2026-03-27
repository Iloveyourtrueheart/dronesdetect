import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QLabel, QFileDialog,
                               QMessageBox, QLineEdit)
from PySide6 import QtCore, QtGui
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QMouseEvent
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
import time
import os

# ==================== 配置 ====================
DEFAULT_CONFIDENCE = 0.85
DEFAULT_ALARM_COOLDOWN = 5000  # 毫秒
DEFAULT_AUDIO_FILE = "jing.mp3"

class DrawArea(QLabel):
    """自定义绘制区域"""
    def __init__(self):
        super().__init__()
        self.points = []
        self.setMinimumSize(640, 480)
        self.setStyleSheet("border: 1px solid black;")
        
    def mousePressEvent(self, event):
        if len(self.points) < 4:
            x = event.pos().x()
            y = event.pos().y()
            # 存储相对坐标（0-1范围）
            if self.pixmap():
                pix_rect = self.pixmap().rect()
                if pix_rect.width() > 0:
                    rel_x = x / self.width()
                    rel_y = y / self.height()
                    self.points.append([rel_x, rel_y])
                    self.update()
            
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.pixmap():
            painter = QPainter(self)
            painter.drawPixmap(self.rect(), self.pixmap())
            
            if self.points:
                # 绘制点
                painter.setPen(QPen(QColor(0, 255, 0), 3))
                for i, (rel_x, rel_y) in enumerate(self.points):
                    x = int(rel_x * self.width())
                    y = int(rel_y * self.height())
                    painter.drawPoint(x, y)
                    painter.drawText(x + 5, y - 5, str(i+1))
                
                # 绘制四边形
                if len(self.points) == 4:
                    painter.setPen(QPen(QColor(255, 0, 0), 2))
                    points = []
                    for rel_x, rel_y in self.points:
                        x = int(rel_x * self.width())
                        y = int(rel_y * self.height())
                        points.append(QtCore.QPoint(x, y))
                    painter.drawPolygon(QtGui.QPolygon(points))
    
    def clear_points(self):
        self.points = []
        self.update()
    
    def get_roi_points(self, frame_width, frame_height):
        """获取相对于实际帧的ROI点坐标"""
        if len(self.points) != 4:
            return None
        roi = []
        for rel_x, rel_y in self.points:
            x = int(rel_x * frame_width)
            y = int(rel_y * frame_height)
            roi.append([x, y])
        return np.array(roi, dtype=np.int32)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 区域入侵检测")
        self.setGeometry(100, 100, 1200, 800)
        
        # 初始化变量
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.model = None
        self.detection_enabled = False
        self.alarm_playing = False
        self.last_alarm_time = 0
        
        # 音频播放器
        self.audio_player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.audio_player.setAudioOutput(self.audio_output)
        if os.path.exists(DEFAULT_AUDIO_FILE):
            self.audio_player.setSource(QUrl.fromLocalFile(DEFAULT_AUDIO_FILE))
        
        self.setup_ui()
        self.load_model()
    
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # ========== 左侧视频区域 ==========
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.video_label = DrawArea()
        left_layout.addWidget(self.video_label)
        
        main_layout.addWidget(left_widget, 4)
        
        # ========== 右侧控制面板 ==========
        right_widget = QWidget()
        right_widget.setMaximumWidth(300)
        right_layout = QVBoxLayout(right_widget)
        
        # 标题
        title_label = QLabel("区域入侵检测控制")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        right_layout.addWidget(title_label)
        
        # 视频源
        right_layout.addWidget(QLabel("视频源:"))
        
        self.rtmp_edit = QLineEdit()
        self.rtmp_edit.setPlaceholderText("rtmp://example.com/live/stream")
        right_layout.addWidget(self.rtmp_edit)
        
        btn_layout = QHBoxLayout()
        self.open_btn = QPushButton("打开文件")
        self.open_btn.clicked.connect(self.open_video)
        btn_layout.addWidget(self.open_btn)
        
        self.rtmp_btn = QPushButton("连接RTMP")
        self.rtmp_btn.clicked.connect(self.connect_rtmp)
        btn_layout.addWidget(self.rtmp_btn)
        right_layout.addLayout(btn_layout)
        
        # 播放控制
        play_layout = QHBoxLayout()
        self.play_btn = QPushButton("播放")
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        play_layout.addWidget(self.play_btn)
        
        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)
        play_layout.addWidget(self.stop_btn)
        right_layout.addLayout(play_layout)
        
        # 检测控制
        right_layout.addSpacing(20)
        right_layout.addWidget(QLabel("检测控制:"))
        
        detect_layout = QHBoxLayout()
        self.start_detect_btn = QPushButton("开始检测")
        self.start_detect_btn.clicked.connect(self.start_detection)
        self.start_detect_btn.setEnabled(False)
        detect_layout.addWidget(self.start_detect_btn)
        
        self.stop_detect_btn = QPushButton("停止检测")
        self.stop_detect_btn.clicked.connect(self.stop_detection)
        self.stop_detect_btn.setEnabled(False)
        detect_layout.addWidget(self.stop_detect_btn)
        right_layout.addLayout(detect_layout)
        
        # 区域控制
        self.clear_btn = QPushButton("清除区域")
        self.clear_btn.clicked.connect(self.clear_roi)
        self.clear_btn.setEnabled(False)
        right_layout.addWidget(self.clear_btn)
        
        # 状态显示
        right_layout.addSpacing(20)
        right_layout.addWidget(QLabel("状态信息:"))
        
        self.status_label = QLabel("状态: 未连接")
        self.status_label.setStyleSheet("background-color: #e0e0e0; padding: 5px;")
        right_layout.addWidget(self.status_label)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("background-color: #e0e0e0; padding: 5px;")
        right_layout.addWidget(self.fps_label)
        
        self.car_count_label = QLabel("区域内车辆: 0")
        self.car_count_label.setStyleSheet("background-color: #e0e0e0; padding: 5px;")
        right_layout.addWidget(self.car_count_label)
        
        self.alarm_label = QLabel("报警状态: 正常")
        self.alarm_label.setStyleSheet("background-color: #00ff00; padding: 5px;")
        right_layout.addWidget(self.alarm_label)
        
        right_layout.addStretch()
        main_layout.addWidget(right_widget, 1)
    
    def load_model(self):
        """加载YOLOv8模型"""
        try:
            from ultralytics import YOLO
            model_path = 'best.pt'
            if not os.path.exists(model_path):
                model_path = 'yolov8n.pt'
            self.model = YOLO(model_path)
            self.status_label.setText("状态: 模型加载成功")
        except Exception as e:
            self.status_label.setText(f"状态: 模型加载失败")
    
    def create_roi_mask(self, frame, roi_points):
        """创建ROI掩码"""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [roi_points], 255)
        return mask
    
    def detect_in_roi(self, frame, roi_points):
        """只在ROI区域内检测"""
        # 创建掩码
        mask = self.create_roi_mask(frame, roi_points)
        
        # 应用掩码到原图
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # 在掩码图上检测
        results = self.model(masked_frame, conf=DEFAULT_CONFIDENCE)
        
        detected_cars = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = box.conf[0].cpu().numpy()
                    
                    # 验证检测框中心点确实在ROI内（双重保障）
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    if cv2.pointPolygonTest(roi_points, (center_x, center_y), False) >= 0:
                        detected_cars.append({
                            'bbox': [x1, y1, x2, y2],
                            'center': [center_x, center_y],
                            'conf': conf
                        })
        
        return detected_cars, masked_frame
    
    def update_frame(self):
        """更新视频帧"""
        if not self.cap or not self.cap.isOpened():
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return
        
        frame_height, frame_width = frame.shape[:2]
        display_frame = frame.copy()
        
        # 获取ROI点
        roi_points = self.video_label.get_roi_points(frame_width, frame_height)
        
        # 绘制ROI区域（用于显示）
        if roi_points is not None:
            cv2.polylines(display_frame, [roi_points], True, (255, 0, 0), 2)
        
        # 检测
        detected_cars = []
        if self.detection_enabled and roi_points is not None and self.model:
            detected_cars, masked = self.detect_in_roi(frame, roi_points)
            
            # 绘制检测框
            for car in detected_cars:
                x1, y1, x2, y2 = car['bbox']
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(display_frame, f"Car {car['conf']:.2f}", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # 更新计数
            self.car_count_label.setText(f"区域内车辆: {len(detected_cars)}")
            
            # 报警逻辑
            current_time = int(time.time() * 1000)
            if len(detected_cars) > 0:
                if current_time - self.last_alarm_time > DEFAULT_ALARM_COOLDOWN:
                    if not self.alarm_playing:
                        self.audio_player.play()
                        self.alarm_playing = True
                        self.alarm_label.setText("报警状态: 报警中!!!")
                        self.alarm_label.setStyleSheet("background-color: #ff0000; padding: 5px;")
                    self.last_alarm_time = current_time
            else:
                if self.alarm_playing:
                    self.audio_player.stop()
                    self.alarm_playing = False
                    self.alarm_label.setText("报警状态: 正常")
                    self.alarm_label.setStyleSheet("background-color: #00ff00; padding: 5px;")
        
        # 显示
        self.display_frame(display_frame)
    
    def display_frame(self, frame):
        """显示帧"""
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        qt_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), 
                                     Qt.KeepAspectRatio, 
                                     Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
    
    def open_video(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "", "Video files (*.mp4 *.avi *.mov)")
        if file_path:
            if self.cap:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            self.setup_video()
    
    def connect_rtmp(self):
        rtmp_url = self.rtmp_edit.text().strip()
        if not rtmp_url:
            QMessageBox.warning(self, "警告", "请输入RTMP地址")
            return
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(rtmp_url)
        self.setup_video()
    
    def setup_video(self):
        if self.cap and self.cap.isOpened():
            self.play_btn.setEnabled(True)
            self.stop_btn.setEnabled(True)
            self.start_detect_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.status_label.setText("状态: 视频已连接")
        else:
            QMessageBox.warning(self, "警告", "无法打开视频源")
    
    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("播放")
            self.status_label.setText("状态: 已暂停")
        else:
            self.timer.start(30)
            self.play_btn.setText("暂停")
            self.status_label.setText("状态: 播放中")
    
    def stop_video(self):
        self.timer.stop()
        self.play_btn.setText("播放")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.status_label.setText("状态: 已停止")
        self.video_label.clear()
    
    def start_detection(self):
        if len(self.video_label.points) != 4:
            QMessageBox.warning(self, "警告", "请先绘制4个点确定检测区域")
            return
        self.detection_enabled = True
        self.start_detect_btn.setEnabled(False)
        self.stop_detect_btn.setEnabled(True)
        self.status_label.setText("状态: 检测中")
        if not self.timer.isActive():
            self.toggle_play()
    
    def stop_detection(self):
        self.detection_enabled = False
        self.start_detect_btn.setEnabled(True)
        self.stop_detect_btn.setEnabled(False)
        self.status_label.setText("状态: 播放中")
        if self.alarm_playing:
            self.audio_player.stop()
            self.alarm_playing = False
            self.alarm_label.setText("报警状态: 正常")
            self.alarm_label.setStyleSheet("background-color: #00ff00; padding: 5px;")
        self.car_count_label.setText("区域内车辆: 0")
    
    def clear_roi(self):
        self.video_label.clear_points()
        self.detection_enabled = False
        self.start_detect_btn.setEnabled(True)
        self.stop_detect_btn.setEnabled(False)
    
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        self.timer.stop()
        if self.alarm_playing:
            self.audio_player.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())