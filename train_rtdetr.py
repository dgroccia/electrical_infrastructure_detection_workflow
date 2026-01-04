from ultralytics import RTDETR
from datetime import datetime

model = RTDETR('rtdetr-l.pt')
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

results = model.train(
    data='data/ttpla_yolo/data.yaml',
    epochs=100,
    batch=8,
    imgsz=640,
    patience=50,
    project='outputs/yolo_runs',
    name=f'rtdetr_l_{timestamp}',
    exist_ok=True,
)
