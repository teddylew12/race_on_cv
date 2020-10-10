import cv2
from pupil_apriltags import Detector
import numpy as np
def find_top_left_corner(corners):
    min_idx=0
    min_val=100000
    max_idx=0
    max_val=-100000
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
img = cv2.imread("Images/la_photo_3.jpg", cv2.IMREAD_GRAYSCALE)
# Start the detector
detector = Detector(families='tagStandard41h12')
#Detect Tags
tags = detector.detect(img, estimate_tag_pose=False)
#Convert back to color to add boxes and text
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#Go through each tag, starting with the tag with the lowest tag id
for tag in sorted(tags,key=lambda x:x.tag_id):
    #Get corners of the tag in the image
    left,right = find_top_left_corner(tag.corners)
    #Add a box around the tag
    area = (np.abs(left[0] - right[0])) ^ 2
    # Add a box around the tag
    clr = (255, 255, 0)
    img = cv2.rectangle(img, left, right, clr, thickness=10)
    img = cv2.circle(img, left, 10, color=(0, 0, 255), thickness=10)
    img = cv2.circle(img, right, 10, color=(255, 0, 255), thickness=10)
    # Add some text with the tag number
    text_loc = (left[0], left[1]- area)
    img = cv2.putText(img, f"Tag #{tag.tag_id}", text_loc, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=.05 * area,
                      color=(255, 0, 0), thickness=5)

#Rescale the image for easy viewing
scale_percent = 20  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#View the image
cv2.imshow("output", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
