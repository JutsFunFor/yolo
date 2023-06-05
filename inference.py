import cv2 as cv
from datetime import datetime
from ultralytics import YOLO
import torch
import os

def run_yolov8(model_path, cap_address, image_size, conf=0.6):
    # Format to datetime
    f = '%Y-%m-%d %H:%M:%S'
    # res_line = {"Detection": {"Nozzle0": {}, "Nozzle1": {}, "Syrup": {}, "HasCup": {}, "NoCup": {}, "Warning":{}},
    #             "DateTime": f"{datetime.now().strftime(f)}"}
    res_line = {}


    model = YOLO(model_path)
    cap = cv.VideoCapture(cap_address)

    ret, frame = cap.read()

    if ret:
        results = model.predict(frame, imgsz=image_size, conf=conf, stream=False, save=True)
        for r in results:
            # boxes = r.boxes.cpu().numpy()  # get boxes on cpu in numpy
            for idx, box in enumerate(r.boxes):  # iterate boxes
                cls_name = f"{r.names[int(box.cls[0])]}_{idx}"
                coords = box.xyxy[0].type(torch.int)
                res_line[cls_name] = {}
                res_line[cls_name]['Xmin'] = f"{coords[0]}"
                res_line[cls_name]['Ymin'] = f"{coords[1]}"
                res_line[cls_name]['Xmax'] = f"{coords[2]}"
                res_line[cls_name]['Ymax'] = f"{coords[3]}"
                res_line[cls_name]['Conf'] = f"{box.conf[0]:.2f}"

    else:
        print("Image was not captured!")
    return res_line
if __name__ == "__main__":

    start_time = datetime.now()
    result = run_yolov8('best_cm.pt', 'rtsp://admin:pipipi@192.168.1.22',  480)
    end_time = datetime.now()

    print(f"Total Inference time: {end_time-start_time}")
    print(result)