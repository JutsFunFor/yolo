from datetime import datetime
import nats
from nats.aio.errors import ErrTimeout, ErrNoServers
import json
from ultralytics import YOLO
import torch
import cv2 as cv
import threading
import sys
import signal


class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        self.lock = threading.Lock()  # Lock for thread safety
        self.is_running = True  # Flag to signal thread termination
        super(CameraBufferCleanerThread, self).__init__(name=name)

    def run(self):
        while self.is_running:
            ret, frame = self.camera.read()
            with self.lock:
                self.last_frame = frame

    def stop(self):
        self.is_running = False

class NatsClient:

    def __init__(self, path):
        with open(path, 'r') as file:
            config = json.load(file)

        self.model_path = config["inference"]["modelPath"]
        self.model = YOLO(self.model_path)
        self.conf = config["inference"]["threshold"]
        self.rtsp_address = config["inference"]["rtspAddress"]
        self._url = config["inference"]["natsUrl"]
        self.send_topic = config["inference"]["sendResultsTopic"]
        self._size = (config["inference"]['tensor_size'], config["inference"]['tensor_size'])  # input tensor shape
        self._action_completed_topic = 'complexos.bus.actionCompleted' # complexos.bus.checkpoint
        self.actions = config["inference"]["actions"]
        self._connect_timeout = config["inference"]["connectTimeout"]
        self.cap = cv.VideoCapture(self.rtsp_address)
        self.cam_cleaner = CameraBufferCleanerThread(self.cap)
        print(f"[INFO NatsClinet __init__() Time: {datetime.now()}] config successfully loaded!")

    def run_yolov8(self):
        res_line = {}

        def signal_handler(sig, frame):
            print('Ctrl+C pressed. Cleaning up...')
            self.cam_cleaner.stop()
            self.cam_cleaner.join()
            self.cap.release()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        with self.cam_cleaner.lock:
            if self.cam_cleaner.last_frame is not None:

                print(f"[INFO run_yolov8() Time:{datetime.now()}] preforming predictions")
                results = self.model.predict(self.cam_cleaner.last_frame, imgsz=self._size, conf=self.conf, stream=False, save=False)
                print(f"[INFO run_yolov8() Time:{datetime.now()}] predictions were made")
                for r in results:
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

        return res_line

    async def receive_msg(self):
        """Receive message from _actionCompleted_topic"""
        try:
            connect_t = datetime.now()
            print(f"[INFO receive_msg() Time: {connect_t}] trying to connect {self._url}")
            self._nc = await nats.connect(servers=[self._url], connect_timeout=self._connect_timeout)
            connected_t = datetime.now()
            print(f"[INFO receive_msg() Time: {connected_t}] successfully connected to {self._url}")

        except (ErrNoServers, ErrTimeout) as err:
            print(f'[Exception receive_msg() Time: {datetime.now()}] {err}')

        # Init yolov8 and publish reply
        async def _receive_callback(msg):

            with open("/yolo_cm/time_stats.csv", "a") as f:
                start_t = datetime.now()
                print(f"[INFO _receive_callback() Time: {start_t}] reading messages")
                data = json.loads(msg.data.decode())
                receive_msg_t = datetime.now()
                print(f"[INFO _receive_callback() Time: {receive_msg_t}] receive msg: {data}")

                if data['action']['action'] in self.actions:

                    receive_action_t = datetime.now()
                    print(f"[INFO _receive_callback() Time: {receive_action_t}] start capturing action: {data['action']['action']}")
                    self.reply = self.run_yolov8()
                    self.reply['OrderId'] = data['action']['orderId']
                    self.reply['OrderNumber'] = data['meta']['orderNumber']
                    self.reply['MenuItemId'] = data['order']['menuItemId']
                    end_t = datetime.now()
                    print(f"[INFO _receive_callback() Time: {end_t}] sending predictions to: {self.send_topic}")
                    elapsed_t = end_t - start_t
                    print(f"[INFO _receive_callback() Elapsed Time: {elapsed_t}]")
                    print(self.reply)
                    print("-------------------------------------------------------------------------------------------------------------------------------------")
                    f.write(f"{connect_t},{connected_t},{start_t},{receive_msg_t},{receive_action_t},{end_t},{elapsed_t}\n")

                await self._nc.publish(self.send_topic, json.dumps(self.reply).encode())
        await self._nc.subscribe(self._action_completed_topic, cb=_receive_callback)