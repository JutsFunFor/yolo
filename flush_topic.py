import asyncio
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrNoServers, ErrTimeout
import json


async def receive_msg(event_loop):
    """Receive message from topic"""

    nc = NATS()
    url = 'nats://complexos.local:4222'
    # url = "nats://demo.nats.io:4222"
    topic1 = 'complexos.bus.actionCompleted'

    try:
        await nc.connect(servers=[url], loop=event_loop, connect_timeout=20)
        # create data to imitate complexos query
        data = {"action": {"name": "take free cup and make a coffee", "orderId":"123"},"meta": {"orderNumber":"900"}, "order": {"menuItemId":"2313"}}
        # send query to topic1
        await nc.publish(topic1, json.dumps(data).encode())
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
    await nc.subscribe(topic1, cb=_receive_callback)


loop = asyncio.get_event_loop()
loop.create_task(receive_msg(loop))

try:
    loop.run_forever()
except Exception as err:
    print(err)
finally:
    loop.close()