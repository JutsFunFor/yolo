from datetime import datetime
import nats
from nats.aio.errors import ErrTimeout, ErrNoServers
import json
from inference import run_yolov8


class NatsClient:
    def __init__(self):
        with open("/yolo_cm/config.json", 'r') as file:
            config = json.load(file)

        self.model = config["inference"]["modelPath"]
        self.conf = config["inference"]["threshold"]
        self.rstp_address = config["inference"]["rstpAddress"]
        self._url = config["inference"]["natsUrl"]
        self.send_topic = config["inference"]["sendResultsTopic"]
        self._size = (config["inference"]['tensor_size'], config["inference"]['tensor_size'])  # input tensor shape
        self._action_completed_topic = 'complexos.bus.actionCompleted' # complexos.bus.checkpoint
        self.actions = config["inference"]["actions"]
        print(f"[INFO NatsClinet __init__() Time: {datetime.now()}] config successfully loaded!")

    async def receive_msg(self):
        """Receive message from _actionCompleted_topic"""
        try:
            print(f"[INFO receive_msg() Time: {datetime.now()}] trying to connect {self._url}")
            self._nc = await nats.connect(servers=[self._url])
            print(f"[INFO receive_msg() Time: {datetime.now()}] successfully connected to {self._url}")
        except (ErrNoServers, ErrTimeout) as err:
            print(f'[Exception receive_msg() Time: {datetime.now()}] {err}')

        # Init yolov8 and publish reply
        async def _receive_callback(msg):
            print(f"[INFO _receive_callback() Time: {datetime.now()}] start reading messages")
            data = json.loads(msg.data.decode())
            print(f"[INFO _receive_callback() Time: {datetime.now()}] receive msg: {data}")

            if data['action']['name'] in self.actions:
                print(f"[INFO _receive_callback() Time: {datetime.now()}] start capturing action: {data['action']['name']}")
                reply = run_yolov8(self.model, self.rstp_address, self._size,  self.conf)
                reply['OrderId'] = data['action']['orderId']
                reply['OrderNumber'] = data['meta']['orderNumber']
                reply['MenuItemId'] = data['order']['menuItemId']
                print(f"[INFO _receive_callback() Time: {datetime.now()}] sending predictions to: {self.send_topic}")

                await self._nc.publish(self.send_topic, json.dumps(reply).encode())
        await self._nc.subscribe(self._action_completed_topic, cb=_receive_callback)
