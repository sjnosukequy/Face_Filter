import cv2
import numpy as np
import dlib, copy
import Face_Filter_cam
import traceback

webcam = True
cap = cv2.VideoCapture(0)

try:
    while webcam:
        success, img = cap.read()
        img = cv2.resize(img, (0, 0), None, 1, 1)
        imgOriginal = copy.deepcopy(img)
        try:
            imfilter = Face_Filter_cam.Bat_Filter(imgOriginal)
        except:
            imfilter = imgOriginal
        imfilter = np.array(imfilter)
        cv2.imshow('Web cam filter', imfilter)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Web cam filter', cv2.WND_PROP_VISIBLE) < 1: 
            break
except Exception:
    traceback.print_exc()
    while webcam:
        img = cv2.imread('./Filter_image/nocam.jpg')
        cv2.imshow('Web cam filter', img)
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Web cam filter', cv2.WND_PROP_VISIBLE) < 1: 
            break

# After the loop release the cap object 
cap.release() 
# Destroy all the windows 
cv2.destroyAllWindows() 
