import numpy as np
import cv2
from time import time
class Stream():
    
    def __init__(self,detector, camera_info,tags):
        self.detector = detector
        self.camera_info=camera_info
        self.res=camera_info["res"]
        self.tags=tags
        # Timing arrays
        self.read = []
        self.undist = []
        self.detect = []
        self.end = []
        self.positions = np.zeros(shape=(3, 1))
        self.raw_frames = []
        self.frames = []

    def undistort(self, img):
        return cv2.remap(img, self.camera_info["map_1"], self.camera_info["map_2"], interpolation=cv2.INTER_NEAREST) 
      
    # Called when new image is available
    def write(self, data):
        t1 = time()
        
        # Get the Y component which is the gray image
        I_distorted = np.frombuffer(data,
                                    dtype=np.uint8,
                                    count=self.res[0] * self.res[1]).reshape(self.res[1],self.res[0])
        self.raw_frames.append(I_distorted)
        self.read.append(time() - t1)
    
        I = self.undistort(I_distorted)
        self.frames.append(I)
        self.undist.append(time() - t1)
        
        detected_tags = self.detector.detect(I, estimate_tag_pose=True, camera_params=self.camera_info["params"],
                                                     tag_size=self.tags.size)
        self.detect.append(time()-t1)
        tmp_poses=[]
        for tag in detected_tags:
            tmp_poses.append(np.matmul(self.tag_orientations[tag.tag_id],tag.pose_t) + self.tag_locations[tag.tag_id])
        if tmp_poses:
            avg = np.mean(np.concatenate(tmp_poses, axis=1), axis=1, keepdims=True)
            self.positions = np.hstack((self.positions, avg))

            # If no new tag detections in this frame, assume (naively) that the camera stayed in the same position
            # as the previous frame
        else:
            self.positions = np.hstack((self.positions, self.positions[:, -1:]))
            pass
        
        self.end.append(time() - t1)
        
    def print_statistics(self):

        print(f"Time to Read:{np.mean(self.read)}")
        print(f"Time to Undistort:{np.mean(self.undist)-np.mean(self.read)}")
        print(f"Time to Detect:{np.mean(self.detect)-np.mean(self.undist)}")
        print(f"Time to End of Loop:{np.mean(self.end)-np.mean(self.detect)}")
        print(f"TRUE FPS: {1/self.positions.shape[1]}")