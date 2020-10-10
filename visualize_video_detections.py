import cv2
from pupil_apriltags import Detector
import numpy as np
from argparse import  ArgumentParser
from pathlib import Path
def find_top_left_corner(corners):
    min_idx=0
    min_val=100000
    max_idx=0
    max_val=-100000
    #Get the minimum pair of coordinates, which is the top left
    #and the maximum, which is the bottom right
    for idx,(x,y) in enumerate(corners):
        if x+y<min_val:
            min_idx=idx
            min_val=x+y
        if x+y>max_val:
            max_idx=idx
            max_val=x+y

    left= (int(corners[min_idx][0]), int(corners[min_idx][1]))
    right= (int(corners[max_idx][0]), int(corners[max_idx][1]))
    return left,right
#Load the image
parser = ArgumentParser(description='FilePath')
parser.add_argument('-f','--filename',required=True,help="path to your file within the Videos folder")
args=parser.parse_args()
filename=Path(f"Videos/{args.filename}")
video = cv2.VideoCapture("Videos/la_video_2.MOV")
# Start the detector
detector = Detector(families='tagStandard41h12')
while True:
    ret, frame = video.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect Tags

    tags = detector.detect(gray, estimate_tag_pose=False)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for tag in sorted(tags, key=lambda x: x.tag_id):
        # Get corners of the tag in the image
        left, right = find_top_left_corner(tag.corners)
        area =(np.abs(left[0]-right[0]))^2
        # Add a box around the tag
        clr = (255, 255, 0)
        img = cv2.rectangle(img, left, right, clr, thickness=10)
        # Add some text with the tag number
        text_loc = (left[0], left[1]-area)
        img = cv2.putText(img, f"Tag #{tag.tag_id}", text_loc, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.05*area,
                          color=(255, 0, 0), thickness=5)
        # Rescale the image for easy viewing
    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('frame', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()
