import numpy as np
from pupil_apriltags import Detector
from stream import Stream
from tag import Tag
import cv2
from time import sleep, time
from picamera import PiCamera
from argparse import ArgumentParser
from pdb import set_trace

# Get time of stream and name for saving outputs
parser = ArgumentParser()
parser.add_argument("-rn", "--run_name", type=str, required=True)
parser.add_argument("-t", "--run_time", type=int)
args = parser.parse_args()
if args.run_time:
    MAX_TIME = args.run_time
else:
    MAX_TIME = 4
# Camera information
FPS = 50
RES = (640, 480)
camera_info = {}
# Camera Resolution
camera_info["res"] = RES
camera_info["K"] = np.array([[314.22174729465604, 0.0, 337.0278425306902],
                             [0.0, 311.4202447283487, 238.99954338265644],
                             [0.0, 0.0, 1.0]])
camera_info["D"] = np.array([[-0.03953861358665185],
                             [0.014918638704331555],
                             [-0.022402610396196412],
                             [0.00863418416543917]])

# Camera Intrinsic Matrix (3x3)
# The non-default elements of the K array, in the AprilTag specification
camera_info["params"] = [314.222, 311.420, 337.028, 239.]
# Fisheye Camera Distortion Matrix

# Fisheye flag
camera_info["fisheye"] = True
camera_info["map_1"], camera_info["map_2"] = cv2.fisheye.initUndistortRectifyMap(camera_info["K"], camera_info["D"],
                                                                                 np.eye(3), camera_info["K"],
                                                                                 camera_info["res"], cv2.CV_16SC2)

# Tag information
TAG_SIZE = .123
FAMILIES = "tagStandard41h12"
tags = Tag(TAG_SIZE, FAMILIES)
# Add information about tag locations
# Function Arguments are id,x,y,z,theta_x,theta_y,theta_z
tags.add_tag(0, 76.25, 30.5, 0., 0., 0., 0.)
tags.add_tag(1, 115., 31.5, 0., 0., 0., 0.)
tags.add_tag(2, 95.75, 50., 0., 0., 0., 0.)
tags.add_tag(3, 0., 41., 38.75, 0., -np.pi / 2, 0.)
tags.add_tag(4, 0., 54., 19.25, 0., -np.pi / 2, 0.)

starting_position = np.array([[.635], [1.0668], [2.7432]])

# Create Apriltags Detector
detector = Detector(families=tags.family, nthreads=4)

# Create the camera object
with PiCamera() as camera:
    camera.resolution = RES
    camera.framerate = FPS
    # Reduce the shutter speed to reduce blur, given in microseconds
    camera.shutter_speed = int(1000000 / (3 * FPS))
    # Create the stream object
    stream = Stream(detector, camera_info, tags, starting_position, args.run_name)
    sleep(1)
    print("Starting")
    try:
        # Start recording frames to the stream object
        camera.start_recording(stream, format='yuv')
        t0 = time()

        while True:
            camera.wait_recording(1)
            # If the time limit is reached, end the recording
            if (time() - t0) > MAX_TIME:
                camera.stop_recording()
                break

    except Exception as e:
        print(e)
        camera.close()
    # Print Timing Information
    stream.print_statistics(MAX_TIME)
    # Save camera frames with detections overlaid
    stream.save_video(MAX_TIME)
    # Save position numpy arrays
    stream.save_positions()
    # Create scatter plot of estimated positions
    stream.scatter_position()
    # Animate the stream to show the camera moving in real time
    # stream.animate_position(MAX_TIME)

    set_trace()
    camera.close()
