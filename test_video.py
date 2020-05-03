import numpy as np
import cv2
import os

model_directory = "yolov3"
model_name      = "yolov3.weights"
cfg_name        = "yolov3.cfg"
label_names     = "coco.names"
confidence_val  = 0.1
thresh_val      = 0.5

LABELS = open(os.path.join(model_directory, label_names)).read().strip().split("\n")
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
# font 
font = cv2.FONT_HERSHEY_SIMPLEX 

weightsPath = os.path.sep.join([ model_directory, model_name])
configPath = os.path.sep.join([ model_directory, cfg_name])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ ln[i[0] - 1] for i in net.getUnconnectedOutLayers() ]

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes       = []
    confidences = []
    classIDs    = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_val:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_val,thresh_val)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            color = [int(c) for c in COLORS[classIDs[i]]]
            frame = cv2.rectangle(frame, (x,y), (x+w, y+h),color)
            text = f"{LABELS[classIDs[i]]}  : {confidences[i]}"
            frame = cv2.putText(frame, text, (x,y-3), font,  
                            0.5, color, 1, cv2.LINE_AA)
            


    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()