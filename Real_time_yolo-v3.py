# import library
import cv2
import numpy as np
import utilities

wdt = 320
nms_threshold = 0.3
confTh = 0.5

modelConfiguration = './YOLO-v3/yolov3.cfg.txt'
modelWeight = './YOLO-v3/yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)    ## deactivate to run model on GPU

# get classes
classes = utilities.loadClasses('coco.names.txt')

# Real-time object detection
cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    blob = cv2.dnn.blobFromImage(img, 1/255, (wdt, wdt), [0, 0, 0], crop = False)
    net.setInput(blob)
    LayerNames = net.getLayerNames()
    OutputNames = [LayerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(OutputNames)
    utilities.getboundingbox(outputs, img, th =confTh, nms_threshold = nms_threshold, classes = classes )
    
    cv2.imshow("window", img)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
cv2.destroyAllWindows()
        
