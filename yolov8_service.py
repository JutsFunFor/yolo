import asyncio
from yolov8_client import NatsClient

if __name__ == '__main__':
    config_path = "/yolo_cm/config.json"
    # client = NatsClient(config_path)
    loop = asyncio.get_event_loop()
    loop.create_task(NatsClient(config_path).receive_msg())

    try:
        loop.run_forever()
    except Exception as err:
        print(err)
    finally:
        loop.close()