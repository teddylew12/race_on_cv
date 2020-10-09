import cv2
from pupil_apriltags import Detector

# Size of the tags
tag_size_dict = {}
tag_size_dict["boulder"] = .14
tag_size_dict["los_angeles"] = .055
#Parameters for my camera
CAMERA_PARAMS = [3101.04, 3095.80, 1909.71, 1507.12]
#Load the image
img = cv2.imread("Images/la_photo_3.jpg", cv2.IMREAD_GRAYSCALE)
# Start the detector
detector = Detector(families='tagStandard41h12')
#Detect Tags
tags = detector.detect(img, estimate_tag_pose=True, camera_params=CAMERA_PARAMS, tag_size=tag_size_dict["los_angeles"])
#Convert back to color to add boxes and text
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#Go through each tag, starting with the tag with the lowest tag id
for tag in sorted(tags,key=lambda x:x.tag_id):
    #Get corners of the tag in the image
    left_corner = (int(tag.corners[0][0]), int(tag.corners[0][1]))
    right_corner = (int(tag.corners[2][0]), int(tag.corners[2][1]))
    #Add a box around the tag
    clr = (255, 255, 0)
    img = cv2.rectangle(img, left_corner, right_corner, clr, thickness=10)
    #Add some text with the tag number
    text_loc = (int(tag.corners[2][0] + 200), int(tag.corners[2][1]))
    img = cv2.putText(img, f"Tag #{tag.tag_id}", text_loc, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=5,
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
