from ultralytics import YOLO


model = YOLO('/data/workspace/lhr/CMFANet/CMFANet/weights/best.pt',task='multi')  # Validate the model
model.predict(source='test.jpg', imgsz=(640,640), device=[0],name='pre', save=True, conf=0.25, iou=0.45, show_labels=False)
