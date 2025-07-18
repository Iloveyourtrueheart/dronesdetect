from ultralytics import YOLO
import os

def train_yolov8():
    # 1. 加载预训练模型
    model = YOLO("D:\yolov8\YOLOv11-test\yolov8s.yaml").load("D:\\yolov8\\runs\\train\\yolov8-nepu6\\weights\\last.pt").to(device=0) # build from YAML and transfer weights   可以选择 yolov8n/s/m/l/x.ptD:\yolov8\YOLOv11-test\yolov8s.pt
    
    # 2. 训练配置
    results = model.train(
        data='D:\yolov8\YOLOv11-test\yolov8-NEPUvehicle\data.yaml',       # 数据配置文件路径
        epochs=600,             # 训练轮次
        patience=600,            # 早停耐心值(轮次)
        batch=8,               # 批量大小
        imgsz=640,              # 输入图像尺寸
        save=True,              # 保存训练结果
        save_period=100,         # 每多少轮保存一次模型
        cache=False,            # 是否缓存数据集
        device=0,          # 使用GPU (如 '0' 或 '0,1,2,3')
        workers=8,              # 数据加载线程数
        project='runs/train',   # 结果保存目录'.]=/
        
        name='yolov8-nepu-again',             # 实验名称
        pretrained=True,        # 使用预训练权重
        optimizer='auto',       # 优化器 (SGD, Adam, AdamW, etc.)
        seed=42,                # 随机种子
        deterministic=True,     # 确定性模式
        close_mosaic=200        # 最后多少轮关闭马赛克增强
    )

if __name__ == '__main__':
    train_yolov8()