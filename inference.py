import cv2 as cv
from datetime import datetime
from ultralytics import YOLO
import torch


def run_yolov8(model, cap_address, image_size, conf=0.6):
    res_line = {}
    print(f"[INFO run_yolov8() Time:{datetime.now()}] initializing VideoCapture object")
    cap = cv.VideoCapture(cap_address)
    print(f"[INFO run_yolov8() Time:{datetime.now()}] capture frame")
    ret, frame = cap.read()

    if ret:
        print(f"[INFO run_yolov8() Time:{datetime.now()}] frame was successfully captured, preforming predictions")
        results = model.predict(frame, imgsz=image_size, conf=conf, stream=False, save=False)
        print(f"[INFO run_yolov8() Time:{datetime.now()}] predictions were made")
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
        print(f"[INFO run_yolov8() Time:{datetime.now()}] creating reply")
    else:
        print(f"[INFO run_yolov8() Time:{datetime.now()}] image was not captured!")
    return res_line


if __name__ == "__main__":

    start_time = datetime.now()
    result = run_yolov8('best_cm.pt', 'rtsp://admin:pipipi@192.168.1.22',  480)
    end_time = datetime.now()

    print(f"Total Inference time: {end_time-start_time}")
    print(result)