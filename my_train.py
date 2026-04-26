
from ultralytics import YOLO



if __name__ == '__main__':
    model = YOLO(r'F:/Knee/ultralytics-main/ultralytics/cfg/models/11/my_yolo11_knee.yaml').load(r'F:/Knee/ultralytics-main/yolo11n-cls.pt')  # 直接加载预训练模型
    # model = YOLO(r'F:/yolo/ultralytics-main/ultralytics-main/ultralytics/cfg/models/v8/yolov8.yaml').load(r'F:/yolo/ultralytics-main/ultralytics-main/weights/detection/yolov8n.pt')  # 直接加载预训练模型
    # results = model.train(data='F:/yolo/ultralytics-main/ultralytics-main/ultralytics/cfg/models/v8/my_yolov8_LWN.yaml',
    #                       epochs=100, imgsz=320, batch=64)

    model.train(
        data="F:/Knee/ultralytics-main/MedicalExpert-I",
        epochs=150,
        imgsz=416,
        batch=32,
        optimizer='SGD',  # 使用随机梯度下降
        lr0=0.01,  # 初始学习率
        # patience=20,  # 早停容忍轮数
        cos_lr=True  # 可选，使用余弦退火策略（如需关闭则设为 False）

    )
