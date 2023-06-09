import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrNoServers, ErrTimeout
import json


async def receive_msg(event_loop):
    """Receive message from topic"""

    nc = NATS()
    # url = 'nats://complexos.local:4222'
    url = "nats://demo.nats.io:4222"
    topic1 = 'complexos.bus.actionCompleted'
    topic2 = 'complexos.bus.orderReady'
    topic3 = "complexos.bus.actionCompleted"
    topic4 = 'complexos.order.*' # from api.js

    try:
        await nc.connect(servers=[url], loop=event_loop)
        data = {"action":{"name":"take free cup and make a coffee"}}
        await nc.publish(topic3, json.dumps(data).encode())
    except (ErrNoServers, ErrTimeout) as err:
        print(err)

    # MAIN PART
    async def _receive_callback(msg):

        data = json.loads(msg.data.decode())
        print(type(data), data)
        # print(data.keys())

        with open('data.txt', "a") as f:
            f.write('\n')
            f.write(str(data))

    await nc.subscribe(topic3, cb=_receive_callback)

loop = asyncio.get_event_loop()
loop.create_task(receive_msg(loop))

try:
    loop.run_forever()
except Exception as err:
    print(err)
finally:
    loop.close()