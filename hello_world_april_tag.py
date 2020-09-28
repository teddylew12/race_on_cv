import numpy as np
import cv2
from pupil_apriltags import Detector
img = cv2.imread("pic_0.jpg",0)
print(img.shape)
at_detector = Detector(families='tagStandard41h12')
resize=.1
width = int(img.shape[1] * resize)
height = int(img.shape[0] * resize)
dim = (width, height)
print(dim)
new_img=cv2.resize(img,dim)
tags = at_detector.detect(new_img, estimate_tag_pose=False, camera_params=None, tag_size=None)
if tags:
    print(f"Tag ID: {tags[0].tag_id}")
else:
    print("No Tags Found")

