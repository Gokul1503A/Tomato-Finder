import cv2 as cv
from ultralytics import YOLO
import numpy as np

video = cv.VideoCapture(0)

model = YOLO("best.pt")




while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)
    result = results[0]

    bb = np.array(result.boxes.xyxy.cpu(),dtype='int')
    classes =np.array(result.boxes.cls.cpu(),dtype='int')
    #print(classes)

    for box, cls in zip(bb, classes):
        (x,y,x2,y2) = box
        if (cls>=1 and cls<=3):
            s = "ripened"
            cv.rectangle(frame,(x,y),(x2,y2),(0,255,0),2)
        else:
            s = "unripined"
            cv.rectangle(frame,(x,y),(x2,y2),(255,0,0),2)
        cv.putText(frame, str(s), (x, y - 5), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)

    cv.imshow('video',frame)
    key = cv.waitKey(1)
    if key == 27:
        break

video.release()
cv.destroyAllWindows()
    
