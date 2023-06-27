# How to

`git clone https://github.com/JutsFunFor/yolo.git && cd yolo`

## Build docker image (for server only)
### Kiosk inference docker image available on docker hub `alekseyml/yolov8:nuke`

`sudo docker build .`

`sudo docker tag "YOUR_IMAGE_ID YOUR_DOCKER_HUB/IMAGE_NAME:TAG"`

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

`actions` - list of actions to perform inference

# Run docker image
You have to mount volume that contains scripts into docker container 
 
`sudo docker run -it -v /home/...YOUR_PATH.../yolo:/yolo_cm  --privileged alekseyml/yolov8:nuke python3 /yolo_cm/yolov8_service.py`
