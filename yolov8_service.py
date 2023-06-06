import asyncio
from yolov8_client import NatsClient

if __name__ == '__main__':
        client = NatsClient()
        loop = asyncio.get_event_loop()
        loop.create_task(client.receive_msg())

        try:
            loop.run_forever()
        except Exception as err:
            print(err)
        finally:
            loop.close()