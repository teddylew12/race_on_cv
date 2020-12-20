import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import os
import shutil
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
        self.found_tags = []

    def undistort(self, img):
        return cv2.remap(img, self.camera_info["map_1"], self.camera_info["map_2"], interpolation=cv2.INTER_LINEAR) 
      
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
        self.found_tags.append(detected_tags)
        self.detect.append(time()-t1)
        tmp_poses=[]
        for tag in detected_tags:
            tmp_poses.append(np.matmul(self.tags.orientations[tag.tag_id],tag.pose_t) + self.tags.locations[tag.tag_id])
        
        if tmp_poses:
            avg = np.mean(np.concatenate(tmp_poses, axis=1), axis=1, keepdims=True)
            self.positions = np.hstack((self.positions, avg))
            
        # If no new tag detections in this frame, assume (naively) that the camera stayed in the same position
        # as the previous frame
        else:
            self.positions = np.hstack((self.positions, self.positions[:, -1:]))
            
        self.end.append(time() - t1)
    def save_positions(self,fname):
        np.save(fname,self.positions)
    def load_positions(self,fname):
        self.positions=np.load(fname)
    def print_statistics(self,max_time):

        print(f"Time to Read:{np.mean(self.read)}")
        print(f"Time to Undistort:{np.mean(self.undist)-np.mean(self.read)}")
        print(f"Time to Detect:{np.mean(self.detect)-np.mean(self.undist)}")
        print(f"Time to End of Loop:{np.mean(self.end)-np.mean(self.detect)}")
        print(f"TRUE FPS: {self.positions.shape[1]/max_time}")
    def visualize_frame(self,img,tags):
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        for tag in tags:
            for idx in range(len(tag.corners)):
                cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0),thickness=3)

            cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255),
                    thickness=3)
        return color_img
        
    def save_video(self,fname,max_time):
        
        writer = cv2.VideoWriter(fname,
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         np.floor(self.positions.shape[1]/max_time), self.res)
        for i,frame in enumerate(self.frames):
            writer.write(self.visualize_frame(frame,self.found_tags[i]))
        writer.release()
    def animate_position(self,savename=None):
        plt.ion()
        positions=self.positions[:,1:]
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.view_init(25, 10)
        if savename is not None:
            saving=True
            os.mkdir("tmp")
        plt.draw()
        for i in range(positions.shape[1]-1):
            plt.cla()
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')

            ax.set_title('Positions')
            
            ax.scatter(positions[0,:i], positions[1,:i], positions[2,:i], color="b")
            ax.scatter(positions[0,i], positions[1,i], positions[2,i], color="r")
            fig.canvas.draw_idle()
            if saving:
                fig.savefig(f"tmp/{i}.png")
            plt.pause(0.1)
        plt.show()
        if saving:
            image_folder="tmp"
            images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
            frame = cv2.imread(os.path.join(image_folder, images[0]))
            height, width, layers = frame.shape
            writer = cv2.VideoWriter(savename, 0, 1, (width,height))
            for image in sorted(images,key=lambda x:int(x.split(".")[0])):
                writer.write(cv2.imread(os.path.join(image_folder, image)))
            writer.release()
            cv2.destroyAllWindows()
            try:
                shutil.rmtree(image_folder)
            except OSError as e:
                print("Error: %s : %s" % (dir_path, e.strerror))
