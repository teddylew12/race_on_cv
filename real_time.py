import numpy as np
from pupil_apriltags import Detector
from stream import Stream
from tag import Tag
import cv2
from time import sleep, time
from picamera import PiCamera
import matplotlib.pyplot as plt
from argparse import ArgumentParser
parser=ArgumentParser()
parser.add_argument("-rn","--run_name",type=str,required=True)
parser.add_argument("-t","--run_time",type=int)
args=parser.parse_args()
vid_name=f"{args.run_name}.avi"
ani_name=f"{args.run_name}_ani.avi"
pos_name=f"{args.run_name}.npy
if args.run_time:
    MAX_TIME=args.run_time
else:
    MAX_TIME = 4
FPS = 50
RES = (640, 480)

TAG_SIZE=.123
FAMILIES = "tagStandard41h12"
tags=Tag(TAG_SIZE,FAMILIES)
tags.locations[0]=np.array([[.10795],[.7493],[0]])
tags.locations[1]=np.array([[1.1303],[.762],[0]])
tags.locations[2]=np.array([[.4953],[1.35255],[0]])
tags.locations[3]=np.array([[2.74955],[1.55321],[2.08915]])
tags.locations[4]=np.array([[2.74955],[1.58115],[2.71145]])

tags.orientations[0]=tags.eulerAnglesToRotationMatrix([0.,0.,0.])
tags.orientations[1]=tags.eulerAnglesToRotationMatrix([0.,0.,0.])
tags.orientations[2]=tags.eulerAnglesToRotationMatrix([0.,0.,0.])
tags.orientations[3]=tags.eulerAnglesToRotationMatrix([0.,np.pi/2.,0.])
tags.orientations[4]=tags.eulerAnglesToRotationMatrix([0.,np.pi/2.,0.])


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


with PiCamera() as camera:

    camera.resolution = RES
    camera.framerate = FPS
    
    stream = Stream(detector,camera_info,tags)
    
    try:
        camera.start_recording(stream, format='yuv')
        t0 = time()
        
        while True:
            camera.wait_recording(1)
            #The signal code is not stopping the recording, so I added this
            if (time()-t0) > MAX_TIME:
                camera.stop_recording()
                break
                
    except Exception as e:
        print(e)
        camera.close()
    stream.print_statistics(MAX_TIME)
    stream.save_video(vid_name,MAX_TIME)
    stream.save_positions(
    stream.animate_position(ani_name)
    import pdb;pdb.set_trace()
    camera.close()