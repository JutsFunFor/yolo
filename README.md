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

Do not forget to change `modelPath` prefix in config to `/yolo_cm/best_cm.pt` or `/yolo_left/best_left.pt` or `/yolo_right/best_right.pt` in order to mount this path:

`export YOLO_CM="/yolo_cm/yolov8_service.py /yolo_cm/config.json"`


`export YOLO_LEFT="/yolo_left/yolov8_service.py /yolo_left/config.json"`


`export YOLO_RIGHT="/yolo_right/yolov8_service.py /yolo_right/config.json"`

Kiosk inference cm model example:

`sudo docker run -it -v /home/foodtronics/yolo:/yolo_cm  --privileged alekseyml/yolov8:nuke python3 $YOLO_CM`

Kiosk left model example (do not forget to change `modelPath` from `/yolo_cm/best_cm.pt` to  `/yolo_left/best_left.pt` and camera IP):

`sudo docker run -it -v /home/foodtronics/yolo:/yolo_left  --privileged alekseyml/yolov8:nuke python3 $YOLO_LEFT`

Kiosk right model example:

`sudo docker run -it -v /home/foodtronics/yolo:/yolo_right  --privileged alekseyml/yolov8:nuke python3 $YOLO_RIGHT`

# Inference result line
Multiclass detection assumes that there are several objects on image that can be refered to the same class. So, result line (results after detection that are sent to `sendResultsTopic` in json format) contains class name + `_idx` postfix. 
This postfix refers to object index on image. For example result class name can be represented as `Warning_0`, `Warning_1`, ... `Warnign_N`. 

## Base class names for CM model:

`Nozzle0`, `Nozzle1`, `Warning`, `HasCup`, `NoCup`, `Flood`

## Base class names for RIGHT model:

`Buffer0` - `Buffer6`, `Flood`, `LiftCup2`, `LiftCup3`, `LiftEmpty2`, `LiftEmpty3`, `NoCup1`, `NoCup2`, `Warning`, `hasCup1`, `hasCup2`

## Base class names for LEFT model:

`Buffer0` - `Buffer6`,  `Flood`, `HasCup1`, `hasCup2`, `LiftCup1`, `LiftCup2`, `LiftEmpty1`, `LiftEmpty2`, `NoCup1`, `NoCup2`, `NozzleCup0`, `NozzleCup1`, `NozzleEmpty0`, `NozzleEmpty1`, `Warning`

For each class name there are `Xmin`, `Ymin`, `Xmax`, `Ymax` and `Conf` parameters (parse as string).

There are also `OrderId`, `OrderNumber`, `MenuItemId` parameters wich are parsed from topic message. You can manually disable this behaviour commenting 124-126 line `yolov8_client.py`

# Dataset & weights

Dataset available at `roboflow` https://app.roboflow.com/coffee200all

For new cases of detector usage you have to update weights. 

1) Upload new cases (video or images) to `roboflow`
2) Label that samples according existing dataset (check for all already labeled images)
3) Export dataset with yolov8 format from roboflow
4) Traing model on new dataset
5) Choose best weights and load it into git repo
   
Workflow described at https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/.

