import asyncio
from yolov8_client import NatsClient
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    args = parser.parse_args()
    config_path = args.config_path
    client = NatsClient(config_path)

    loop = asyncio.get_event_loop()
    loop.create_task(client.receive_msg())

    try:
        loop.run_forever()
    except Exception as err:
        print(err)
    finally:
        loop.close()