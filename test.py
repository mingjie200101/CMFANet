import sys
sys.path.insert(0, "/home/jiayuan/ultralytics-main/ultralytics")

from ultralytics import YOLO


number = 3 #input how many tasks in your work
model = YOLO('/media/lwb/92781CDD781CC1C1/lhr/yolomv1/runs/multi/lm-yolo-manet+s-s/weights/best.pt')  # Validate the model
model.predict(source='/media/lwb/92781CDD781CC1C1/lhr/yolomv1/test2', imgsz=(384,672), device=[0],name='vis', save=True, conf=0.25, iou=0.45, show_labels=False)
