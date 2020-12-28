import numpy as np
from pupil_apriltags import Detector
from stream import Stream
from tag import Tag
import cv2
from time import sleep, time
from picamera import PiCamera

RES = (640, 480)

TAG_SIZE=.123
FAMILIES = "tagStandard41h12"
tags=Tag(TAG_SIZE,FAMILIES)
tags.locations[1] = np.array([[0.], [0.], [0.]])
tags.orientations[1] = tags.eulerAnglesToRotationMatrix([0., 0., 0.])
tags.locations[2] = np.array([[0.], [0.], [0.]])
tags.orientations[2] = tags.eulerAnglesToRotationMatrix([0., 0., 0.])

detector = Detector(families=tags.family,nthreads=4)

camera_info = {}
#Camera Resolution
camera_info["res"] = RES
#Camera Intrinsic Matrix (3x3)
camera_info["K"] = np.array([[313.11130800756115, 0.0, 336.11351317641487], 
                             [0.0, 310.34427179740504, 239.24222723346466], 
                             [0.0, 0.0, 1.0]])
#The non-default elements of the K array, in the AprilTag specification
camera_info["params"] = [313.111, 310.344, 336.114, 239.242]
#Fisheye Camera Distortion Matrix
camera_info["D"] = np.array([[-0.03574382363559852], 
                             [0.0028133336786254765], 
                             [-0.007814648102960479], 
                             [0.003381442340208307]])
#Fisheye flag
camera_info["fisheye"] = True
camera_info["map_1"],camera_info["map_2"]  = cv2.fisheye.initUndistortRectifyMap(camera_info["K"], camera_info["D"], 
                                                                  np.eye(3), camera_info["K"], 
                                                                  camera_info["res"], cv2.CV_16SC2)



stream1 = open('image.data', 'w+b')
# Capture the image in YUV format
with PiCamera() as camera:
    camera.resolution = RES
    camera.start_preview()
    print("2 Seconds")
    sleep(2)
    camera.capture(stream1, 'yuv')
# Rewind the stream for reading
stream1.seek(0)
# Calculate the actual image size in the stream (accounting for rounding
# of the resolution)
fwidth = (RES[0] + 31) // 32 * 32
fheight = (RES[1] + 15) // 16 * 16
# Load the Y (luminance) data from the stream
I1_distorted = np.fromfile(stream1, dtype=np.uint8, count=fwidth*fheight).reshape((fheight, fwidth))
#I2_distorted = np.fromfile(stream2, dtype=np.uint8, count=fwidth*fheight).reshape((fheight, fwidth))
I1 = cv2.remap(I1_distorted, camera_info["map_1"], camera_info["map_2"], interpolation=cv2.INTER_LINEAR)
#I2 = cv2.remap(I2_distorted, camera_info["map_1"], camera_info["map_2"], interpolation=cv2.INTER_LINEAR)
detected_tags1 = detector.detect(I1, estimate_tag_pose=True, camera_params=camera_info["params"],
                                                     tag_size=tags.size)
#detected_tags2 = detector.detect(I2, estimate_tag_pose=True, camera_params=camera_info["params"],
#                                                     tag_size=tags.size)

#for t in detected_tags1: 
    #print(t.tag_id,f"{ (t.pose_R @ (tags.tag_corr @ tags.locations[t.tag_id]) + t. pose_t)}")
for t in detected_tags1:
    print(f"{t.tag_id}")
    print(tags.transform_to_global_frame(t.tag_id, t.pose_R,t.pose_t)):

import pdb;pdb.set_trace()
x=1
