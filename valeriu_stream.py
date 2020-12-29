import picamera
import signal
import numpy as np

stop_process = False


def signal_handler(signal, frame):
    global stop_process
    stop_process = True


signal.signal(signal.SIGINT, signal_handler)

RES = (640, 480)
FPS = 30


# Class to process camera messages
class Stream():

    def __init__(self, map1, map2):

        # Save maps to use for undistortion
        self.map1 = map1
        self.map2 = map2

        # Timing arrays
        self.read = []
        self.undist = []
        self.detect = []
        self.end = []
        self.full = []

    def undistort(self, I):
        return cv2.remap(I, self.map1, self.map2, interpolation=cv2.INTER_NEAREST)

    # Automatically called by the camera once a new frame is available
    def write(self, data):

        t1 = time()

        # Get the Y component (gray image) as a np.array
        # and discard the U and V color components
        I_distorted = np.frombuffer(data, dtype=np.uint8, count=RES[0] * RES[1]).reshape(RES)
        self.read.append(time() - t1)

        I = self.undistort(I_distorted)
        self.undist.append(time() - t1)

        detected_tags = detector.detect(I, estimate_tag_pose=True, camera_params=camera_info["params"],
                                        tag_size=tag_size)
        self.detect.append(time() - t1)

        tmp_poses = [tag.pose_t for tag in detected_tags]
        if tmp_poses:
            avg = np.mean(np.concatenate(tmp_poses, axis=1), axis=1, keepdims=True)
            self.positions = np.hstack((positions, avg))

            # printing here might actually slow the loop
            print(f"Time:{time() - t0:.3f}, X:{avg[0,0]:.3f},Y:{avg[1,0]:.3f},X:{avg[2,0]:.3f}")
        else:
            # If no new tag detections in this frame, assume (naively) that the camera stayed in the same position
            # as the previous frame
            self.positions = np.hstack((positions, positions[:, -1:]))
            print("No detections")

        self.end.append(time() - t1)
        self.full.append(time() - t0)

        # Stop if we processed 100 frames
        # Will still record for at most 1 second, check logic below
        if len(self.read) > 100:
            stop_process = True

    def print_statistics(self):

        print(f"Time to Read:{np.mean(self.read)}")
        print(f"Time to Undistort:{np.mean(self.undist)-np.mean(self.read)}")
        print(f"Time to Detect:{np.mean(self.detect)-np.mean(self.undist)}")
        print(f"Time to End of Loop:{np.mean(self.end)-np.mean(self.detect)}")
        print(f"Total Time:{np.mean(self.end)}")
        print(f"Estimated FPS:{1/np.mean(self.end)}")


# Start capturing camera images
with picamera.PiCamera() as camera:
    camera.resolution = RES
    camera.framerate = FPS

    s = Stream(camera_info["map_1"], camera_info["map_2"])

    try:
        # Camera will call the write method of the stream object for each frame
        camera.start_recording(s, format='yuv')

        t0 = time()

        # Record for a second and then check if we need to stop or not
        while not stop_process:
            camera.wait_recording(1)


    except:
        pass

    s.print_statistics()

    camera.close()