import time
import picamera import PiCamera
import numpy as np
import cv2
from picamera.array import PiYUVArray, PiRGBArray
from argparse import ArgumentParser
import matplotlib.pyplot as plt
RUN_TIMER = 5
resolution=(1280,760)
camera = PiCamera()
camera.sensor_mode = 7
camera.resolution = res
camera.framerate = 20
rawCapture = PiYUVArray(camera, size=res)
stream = camera.capture_continuous(rawCapture, format="yuv", use_video_port=True)
t = time.time()
first_frame = True
for f in stream:
    if first_frame:
        first_frame = False
        # Reset the buffer for the next image
        rawCapture.truncate(0)
        continue
    
    # Stop after RUN_TIMER seconds
    if (time.time() - t) > RUN_TIMER:
        break
    
    I = f.array[:, :, 0]
    
    # Reset the buffer for the next image
    rawCapture.truncate(0)