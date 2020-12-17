import time
import picamera
import numpy as np
import cv2
with picamera.PiCamera(resolution=(640,480)) as camera:
    for num in range(30):
        camera.start_preview()
        time.sleep(5)
        fname="calib_640_480/calib_"+str(num)+".png"
        camera.capture(fname)