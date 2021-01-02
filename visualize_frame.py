import numpy as np
from pupil_apriltags import Detector
from tag import Tag
import cv2
from time import sleep, time
from picamera import PiCamera

RES = (640, 480)


def visualize_frame(img, tags):
    color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for tag in tags:
        # Add bounding rectangle
        for idx in range(len(tag.corners)):
            cv2.line(color_img, tuple(tag.corners[idx - 1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)),
                     (0, 255, 0), thickness=3)
        # Add Tag ID text
        cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int) + 10, tag.corners[0, 1].astype(int) + 10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255),
                    thickness=3)
        # Add Tag Corner
        cv2.circle(color_img, tuple(tag.corners[0].astype(int)), 2, color=(255, 0, 255), thickness=3)
    return color_img


# Create a Tag object
TAG_SIZE = .123
FAMILIES = "tagStandard41h12"
tags = Tag(TAG_SIZE, FAMILIES)

# Add information about tag locations
# Function Arguments are id,x,y,z,theta_x,theta_y,theta_z
tags.add_tag(1, 115., 31.5, 0., 0., 0., 0.)
tags.add_tag(2, 95.75, 50., 0., 0., 0., 0.)

# Create Detector
detector = Detector(families=tags.family, nthreads=4)

camera_info = {}
# Camera Resolution
camera_info["res"] = RES
# Camera Intrinsic Matrix (3x3)
camera_info["K"] = np.array([[313.11130800756115, 0.0, 336.11351317641487],
                             [0.0, 310.34427179740504, 239.24222723346466],
                             [0.0, 0.0, 1.0]])
# The non-default elements of the K array, in the AprilTag specification
camera_info["params"] = [313.111, 310.344, 336.114, 239.242]
# Fisheye Camera Distortion Matrix
camera_info["D"] = np.array([[-0.03574382363559852],
                             [0.0028133336786254765],
                             [-0.007814648102960479],
                             [0.003381442340208307]])
# Fisheye flag
camera_info["fisheye"] = True
camera_info["map_1"], camera_info["map_2"] = cv2.fisheye.initUndistortRectifyMap(camera_info["K"], camera_info["D"],
                                                                                 np.eye(3), camera_info["K"],
                                                                                 camera_info["res"], cv2.CV_16SC2)

stream = open('image.data', 'w+b')
# Capture the image in YUV format
with PiCamera() as camera:
    camera.resolution = RES
    camera.start_preview()
    print("2 Seconds")
    sleep(2)
    camera.capture(stream, 'yuv')
# Rewind the stream for reading
stream.seek(0)
# Calculate the actual image size in the stream (accounting for rounding
# of the resolution)
fwidth = (RES[0] + 31) // 32 * 32
fheight = (RES[1] + 15) // 16 * 16
# Load the Y (luminance) data from the stream
I_distorted = np.fromfile(stream, dtype=np.uint8, count=fwidth * fheight).reshape((fheight, fwidth))
I = cv2.remap(I_distorted, camera_info["map_1"], camera_info["map_2"], interpolation=cv2.INTER_LINEAR)

detected_tags = detector.detect(I, estimate_tag_pose=True, camera_params=camera_info["params"],
                                tag_size=tags.size)
# Estimate Camera Pose
for t in detected_tags:
    print(f"{t.tag_id}")
    print(tags.estimate_pose(t.tag_id, t.pose_R, t.pose_t))
# Visualize The Frame
cv2.imshow("Visualized Tags", visualize_frame(I, detected_tags))
cv2.waitKey(0)  # waits until a key is pressed
cv2.destroyAllWindows()  # destroys the window showing image
