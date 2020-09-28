
import cv2
from pupil_apriltags import Detector
#For Ted's Iphone
TAG_SIZE = .14*.14
#From Matlab Camera Calibration
CAMERA_PARAMS=[3101.04,3095.80,1909.71,1507.12]
img = cv2.imread("standard_tag_1_14cm.jpg",0)
at_detector = Detector(families='tagStandard41h12')
tags = at_detector.detect(img, estimate_tag_pose=True, camera_params=CAMERA_PARAMS, tag_size=TAG_SIZE)
if tags:
    print(f"Tag ID: {tags[0].tag_id}")
    import pdb;pdb.set_trace()
else:
    print("No Tags Found")