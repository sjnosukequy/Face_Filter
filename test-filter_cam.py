import cv2
import numpy as np
import dlib, copy
import Face_Filter_cam
import traceback, threading, sys, time

webcam = True
cap = cv2.VideoCapture(0)
test = [None]
thread = [None]
heightResize = 200
# used to record the time when we processed last frame 
prev_frame_time = 0
  
# used to record the time at which we processed current frame 
new_frame_time = 0

if (cap.isOpened() == False):
    print("Unable to connect to camera, kindly check your web camera.")
    sys.exit()

success, img = cap.read()
if success == True:
    height = img.shape[0]
    # calculate resize scale
    frame_resize_scale = float(height)/heightResize
    size = img.shape[0:2]

try:
    while webcam:
        success, img = cap.read()
        new_frame_time = time.time() 
        imgOriginal = copy.deepcopy(img)
        imgOriginal= cv2.resize(imgOriginal, None, fx = 1.0/frame_resize_scale, fy = 1.0/frame_resize_scale, interpolation = cv2.INTER_LINEAR)
        try:
            thread[0]= threading.Thread(target=Face_Filter_cam.Clown, args=(imgOriginal, test))
            thread[0].run()
            # if(thread[0] == None):
            #     thread[0]= threading.Thread(target=Face_Filter_cam.SunGlass_filter, args=(imgOriginal, test))
            #     thread[0].run()
            # else:
            #     if(thread[0].is_alive() == False):
            #         thread[0]= threading.Thread(target=Face_Filter_cam.SunGlass_filter, args=(imgOriginal, test))
            #         thread[0].run()
            if(test[0] != None):
                imfilter = test[0]
            else:
                imfilter = imgOriginal
            # imfilter = Face_Filter_cam.Bat_Filter(imgOriginal, test)
        except:
            imfilter = imgOriginal
        imfilter = np.array(imfilter)
        imfilter = cv2.resize(imfilter, None, fx = frame_resize_scale, fy = frame_resize_scale, interpolation = cv2.INTER_LINEAR)

        fps = 1/(new_frame_time-prev_frame_time) 
        prev_frame_time = new_frame_time 
        # converting the fps into integer 
        fps = int(fps)
        cv2.putText(imfilter, "{0:.2f}-framePerSecond".format(fps), (50, size[0]-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
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
