from datetime import datetime
import nats
from nats.aio.errors import ErrTimeout, ErrNoServers
import json
from ultralytics import YOLO
import torch
import cv2 as cv

class NatsClient:
    def __init__(self):
        with open("/yolo_cm/config.json", 'r') as file:
            config = json.load(file)

        self.model_path = config["inference"]["modelPath"]
        self.model = YOLO(self.model_path)
        self.conf = config["inference"]["threshold"]
        self.rstp_address = config["inference"]["rstpAddress"]
        self._url = config["inference"]["natsUrl"]
        self.send_topic = config["inference"]["sendResultsTopic"]
        self._size = (config["inference"]['tensor_size'], config["inference"]['tensor_size'])  # input tensor shape
        self._action_completed_topic = 'complexos.bus.actionCompleted' # complexos.bus.checkpoint
        self.actions = config["inference"]["actions"]
        self.cap = cv.VideoCapture(self.rstp_address)
        self._connect_timeout = config["inference"]["connectTimeout"]

        print(f"[INFO NatsClinet __init__() Time: {datetime.now()}] config successfully loaded!")

    def run_yolov8(self, model):
        res_line = {}

        print(f"[INFO run_yolov8() Time:{datetime.now()}] capture frame")
        ret, frame = self.cap.read()

        if ret:
            print(f"[INFO run_yolov8() Time:{datetime.now()}] frame was successfully captured, preforming predictions")
            results = model.predict(frame, imgsz=self._size, conf=self.conf, stream=False, save=False)
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
                print(f"[INFO _receive_callback() Time: {start_t}] start reading messages")
                data = json.loads(msg.data.decode())
                receive_msg_t = datetime.now()
                print(f"[INFO _receive_callback() Time: {receive_msg_t}] receive msg: {data}")

                if data['action']['name'] in self.actions:
                    receive_action_t = datetime.now()
                    print(f"[INFO _receive_callback() Time: {receive_action_t}] start capturing action: {data['action']['name']}")
                    self.reply = self.run_yolov8(self.model)
                    self.reply['OrderId'] = data['action']['orderId']
                    self.reply['OrderNumber'] = data['meta']['orderNumber']
                    self.reply['MenuItemId'] = data['order']['menuItemId']
                    end_t = datetime.now()
                    print(f"[INFO _receive_callback() Time: {end_t}] sending predictions to: {self.send_topic}")
                    elapsed_t = end_t - start_t
                    print(f"[INFO _receive_callback() Elapsed Time: {elapsed_t}]")
                    print("-------------------------------------------------------------------------------------------------------------------------------------")
                    f.write(f"{connect_t},{connected_t},{start_t},{receive_msg_t},{receive_action_t},{end_t},{elapsed_t}\n")
                await self._nc.publish(self.send_topic, json.dumps(self.reply).encode())
        await self._nc.subscribe(self._action_completed_topic, cb=_receive_callback)
