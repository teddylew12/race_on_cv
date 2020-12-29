import numpy as np
from pupil_apriltags import Detector
from stream import Stream
from tag import Tag
import cv2
from time import sleep, time
from picamera import PiCamera
from argparse import ArgumentParser
from  pdb import set_trace

def undistort(img,m1,m2):
        return cv2.remap(img, m1,m2, interpolation=cv2.INTER_LINEAR)
    
camera_info={}
camera_info["res"] = (640,480)
camera_info["K"] = np.array([[314.22174729465604, 0.0, 337.0278425306902],
                             [0.0, 311.4202447283487, 238.99954338265644],
                             [0.0, 0.0, 1.0]])
camera_info["D"] = np.array([[-0.03953861358665185],
                             [0.014918638704331555],
                             [-0.022402610396196412],
                             [0.00863418416543917]])

# Camera Intrinsic Matrix (3x3)
kold = np.array([[313.11130800756115, 0.0, 336.11351317641487],
                 [0.0, 310.34427179740504, 239.24222723346466],
                 [0.0, 0.0, 1.0]])

# Fisheye Camera Distortion Matrix
dold = np.array([[-0.03574382363559852],
                 [0.0028133336786254765],
                 [-0.007814648102960479],
                 [0.003381442340208307]])

new1, new2 = cv2.fisheye.initUndistortRectifyMap(camera_info["K"], camera_info["D"],
                                                                                 np.eye(3), camera_info["K"],
                                                                                 camera_info["res"], cv2.CV_16SC2)

old1, old2 = cv2.fisheye.initUndistortRectifyMap(kold, dold,np.eye(3), kold,camera_info["res"], cv2.CV_16SC2)

img = cv2.imread("../Desktop/reproject.png")

photo1 = undistort(img, new1, new2)
photo2 = undistort(img, old1, old2)

cv2.imshow("New", photo1)
cv2.imshow("old", photo2)
cv2.imshow("orig", img)

cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
 



