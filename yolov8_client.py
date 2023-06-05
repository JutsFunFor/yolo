import datetime
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrTimeout, ErrNoServers
import json
from inference import run_yolov8


class NatsClient:
    def __init__(self):
        with open("config.json", 'r') as file:
            config = json.load(file)

        self.model = config["inference"]["modelPath"]
        self.conf = config["inference"]["threshold"]
        self.rstp_address = config["inference"]["rstpAddress"]
        self._url = config["inference"]["natsUrl"]
        self.send_topic = config["inference"]["sendResultsTopic"]
        self._nc = NATS()
        self._size = (config["inference"]['tensor_size'], config["inference"]['tensor_size'])  # input tensor shape
        self._actionCompleted_topic = 'complexos.bus.actionCompleted' # complexos.bus.checkpoint
        print("Config loaded")

    async def receive_msg(self, event_loop):
        """Receive message from _actionCompleted_topic"""
        try:
            print(f"[INFO receive_msg()] trying to connect {self._url}")
            await self._nc.connect(servers=[self._url], loop=event_loop)
            print(f"[INFO receive_msg()] succsessfully connected to {self._url}")
        except (ErrNoServers, ErrTimeout) as err:
            print(err)

        # Init yolov8 and publish reply
        async def _receive_callback(msg):
            print("[INFO _receive_callback()] starting read income message")
            data = json.loads(msg.data.decode())
            print(data['action']['name'])
            if data['action']['name'] == 'take free cup and make a coffee':
                print(data)
                print(datetime.datetime.now())
                reply = run_yolov8(self.model, self.rstp_address, self._size,  self.conf)
                reply['OrderId'] = data['action']['orderId']
                reply['OrderNumber'] = data['meta']['orderNumber']
                reply['MenuItemId'] = data['order']['menuItemId']

                await self._nc.publish(self.send_topic, json.dumps(reply).encode())
        await self._nc.subscribe(self._actionCompleted_topic, cb=_receive_callback)
