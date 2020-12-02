import numpy as np
import cv2

def loadClasses(filename):
    classes = []
    with open('./YOLO-v3/coco.names.txt', 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    return classes


def getboundingbox(outputs, img, th, nms_threshold, classes):
    ht, wt, ct = img.shape
    bounding_box = []
    classids = []
    prob = []
    for output in outputs:
        for box in output:
            scores = box[5:]
            classid = np.argmax(scores)
            confs = scores[classid]
            if confs>th:
                w, h = int(box[2]*wt), int(box[3]*ht)
                x, y = int(box[0]*wt-w/2), int(box[1]*ht-h/2)
                bounding_box.append([x, y, w, h])
                classids.append(classid)
                prob.append(float(confs))
#     Non-max supression
    idxs = cv2.dnn.NMSBoxes(bounding_box, prob, th, nms_threshold)
    for i in idxs:
        i = i[0]
        bbox = bounding_box[i]
        x, y,w, h = bbox[0], bbox[1], bbox[2], bbox[3]
#        draw bounding box
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
#        write labels
        cv2.putText(img, f"{classes[classids[i]].upper()}:{int(prob[i]*100)}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
