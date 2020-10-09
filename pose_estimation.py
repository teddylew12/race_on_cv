import cv2
from pupil_apriltags import Detector
import numpy as np

# Size of the tags
tag_size = {}
tag_size["los_angeles"] = .055
# Locations of the tags: x,y,z
tag_locations = {}
tag_locations[0] = np.array([0, 0, 0])
tag_locations[2] = np.array([0, 0, .195])
camera_location = np.array([-.085, 1.28, 0])
# Parameters for my camera
CAMERA_PARAMS = [3101.04, 3095.80, 1909.71, 1507.12]
# Load the image
img = cv2.imread("Images/la_photo_4.jpg", cv2.IMREAD_GRAYSCALE)
# Start the detector
detector = Detector(families='tagStandard41h12')
# Detect Tags
tags = detector.detect(img, estimate_tag_pose=True, camera_params=CAMERA_PARAMS, tag_size=tag_size["los_angeles"])
# Go through each tag, starting with the tag with the lowest tag id
estimated_camera_locations = []
for tag in sorted(tags, key=lambda x: x.tag_id):
    translation =np.transpose(tag.pose_t)
    estimated_camera_locations.append(tag_locations[tag.tag_id] + translation)
import pdb;

pdb.set_trace()
