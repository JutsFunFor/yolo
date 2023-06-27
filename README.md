# How to

`git clone https://github.com/JutsFunFor/yolo.git && cd yolo`

## Build docker image (for server only)

`sudo docker build .`

`sudo docker tag "YOUR_IMAGE_ID YOUR_DOCKER_HUB/IMAGE_NAME:TAG"`
### Kiosk inference docker image available on docker hub `alekseyml/yolov8:nuke`

# Change `config.json` file

![image](https://github.com/JutsFunFor/yolo/assets/43553016/7cda425a-2218-40f0-b708-f0be69e6698e)

`modelPath` - change for best_cm/best_right/best_left models

`rtspAddress` - change according camera IP. Default CM IP 192.168.1.22

`threshold` - thresh for detection confidence level. Depends on light conditions in complex and should be tuned

`natsUrl` - default nsta url on port 4222

`connectTimeout` - timeout in seconds to connect to nats server

`sendResultsTopic` - sepcified topic for results 

`actionCompletedTopic` - specified topic for receiving signals (actions)

`tensorSize` - default image size for model input

`actions` - list of actions to perform inference (add actions if neccessary)

# Run docker image
You have to mount volume that contains inference scripts into docker container.

Do not forget to change `modelPath` prefix in config to `/yolo_cm/best_cm.pt` or `/yolo_left/best_left.pt` or `/yolo_right/best_right.pt` in order to mount this path like below:

`export YOLO_CM="/yolo_cm/yolov8_service.py /yolo_cm/config.json"`


`export YOLO_LEFT="/yolo_left/yolov8_service.py /yolo_left/config.json"`


`export YOLO_RIGHT="/yolo_right/yolov8_service.py /yolo_right/config.json"`

Kiosk inference cm model example:

`sudo docker run -it -v /home/foodtronics/yolo:/yolo_cm  --privileged alekseyml/yolov8:nuke python3 $YOLO_CM`

Kiosk inference left model example (do not forget to change `modelPath` from `/yolo_cm/best_cm.pt` to  `/yolo_left/best_left.pt` and camera IP):

`sudo docker run -it -v /home/foodtronics/yolo:/yolo_left  --privileged alekseyml/yolov8:nuke python3 $YOLO_LEFT`
