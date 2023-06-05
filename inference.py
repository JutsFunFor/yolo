import json
import numpy as np
import cv2 as cv
from PIL import Image
from datetime import datetime
from ultralytics import YOLO
import os


def run_yolov5(cap_address, model_path, image_size, order_number=None, order_id=None, menu_item_id=None):
    # Format to datetime
    f = '%Y-%m-%d %H:%M:%S'
    res_line = {"Detection": {"Nozzle0": {}, "Nozzle1": {}, "Syrup": {}, "HasCup": {}, "NoCup": {}},
                "OrderId": f"{order_id}",
                "OrderNumber": f"{order_number}",
                "MenuItemId": f"{menu_item_id}",
                "DateTime": f"{datetime.now().strftime(f)}"}

    model = YOLO(model_path)
    cap = cv.VideoCapture(cap_address)

    ret, frame = cap.read()
    if ret:
        results = model.predict(frame, imgsz=image_size, conf=0.6, stream=False)
        for r in results:
            boxes = r.boxes.cpu().numpy()  # get boxes on cpu in numpy
            for box in boxes:  # iterate boxes
                cls_name = r.names[int(box.cls[0])]
                coords = box.xyxy[0].astype(int)
                res_line["Detection"][cls_name]['Xmin'] = f"{coords[0]}"
                res_line["Detection"][cls_name]['Ymin'] = f"{coords[1]}"
                res_line["Detection"][cls_name]['Xmax'] = f"{coords[2]}"
                res_line["Detection"][cls_name]['Ymax'] = f"{coords[3]}"
                res_line["Detection"][cls_name]['Conf'] = f"{box.conf[0]:.2f}"

    else:
        print("Image was not captured!")


    return res_line

start_time = datetime.now()
result = run_yolov5('rtsp://admin:pipipi@192.168.1.22', 'best.pt', 480)
end_time = datetime.now()

print(f"Total Inference time: {end_time-start_time}")
print(result)