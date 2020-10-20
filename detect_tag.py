# Detect a single april-tags in a real time streaming video using webcam on laptop

from pupil_apriltags import Detector
import cv2
import numpy

TAG_FAMILY = 'tagStandard41h12'
window_size = (640, 480)
TAG_SIZE = .5*.5
# CAMERA_PARAMS=[]
cap = cv2.VideoCapture(0)

cap.set(3, window_size[0])  # set width
cap.set(4, window_size[1])  # set height

at_detector = Detector(families=TAG_FAMILY,
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect tags
    tags = at_detector.detect(gray, estimate_tag_pose=False, tag_size=TAG_SIZE)

    # show video in realtime
    cv2.imshow('frame', gray)

    # print the detected result on terminal
    if tags:
        print(f"Tag ID: {tags[0].tag_id}")
    else:
        print("no tag detected")

    k = cv2.waitKey(1)
    if k == 27:  # 27 == ESC
        # out = cv2.imwrite('capture5.jpg', frame)
        # print(result)
        break

cap.release()
cv2.destroyAllWindows()