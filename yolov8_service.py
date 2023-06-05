import asyncio
from yolov8_client import NatsClient

if __name__ == '__main__':
    client = NatsClient()

    try:
        asyncio.run(client.receive_msg())
    except Exception as err:
        print(err)