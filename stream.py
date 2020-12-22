import numpy as np
import cv2
from time import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import os
import shutil


class Stream():

    def __init__(self, detector, camera_info, tags, initial_pose, run_name):
        self.detector = detector
        self.camera_info = camera_info
        self.res = camera_info["res"]
        self.tags = tags
        self.run_name = run_name

        # Timing arrays
        self.undist = []
        self.detect = []
        self.end = []
        self.dts = []
        self.positions = initial_pose
        self.raw_positions = initial_pose
        self.frames = []
        self.found_tags = []

    def write(self, data):
        '''
        Called when new frame is available
        '''
        # Start timer
        t1 = time()

        # Get the Y component which is the gray image
        I_raw = np.frombuffer(data,dtype=np.uint8,count=self.res[0] * self.res[1]).reshape(self.res[1], self.res[0])

        # Remove fisheye distortion
        I = self.undistort(I_raw)
        self.frames.append(I)
        self.undist.append(time() - t1)

        # Detect tags on undistorted image
        detected_tags = self.detector.detect(I, estimate_tag_pose=True, camera_params=self.camera_info["params"],
                                             tag_size=self.tags.size)
        self.found_tags.append(detected_tags)
        self.detect.append(time() - t1)

        # Raw pose estimation from detected tags
        tmp_poses= [self.tags.transform_to_global_frame(t.tag_id,t.pose_t) for t in detected_tags]

        # Apply filtering to smooth results
        smoothed_position = self.ghfilter(tmp_poses, time() - t1)
        self.positions = np.hstack((self.positions, smoothed_position))
        self.end.append(time() - t1)

    def ghfilter(self, measurements, time_step_est):
        '''
        GH filter implementation with H=0, current only works with straight line in -Z direction
        :param measurements: Apriltag raw pose estimates from time step k
        :param time_step_est: Estimate of time elapsed since step k-1
        :return: Smoothed pose estimated
        '''
        # Add correction for time step due to code outside of write function
        dt = time_step_est + .00475
        self.dts.append(dt)
        g = .3
        v = np.array([[0.], [0.], [-.182]])
        if measurements:
            # Average measurements
            avg = np.mean(np.concatenate(measurements, axis=1), axis=1, keepdims=True)
            # Save unfiltered position
            self.raw_positions = np.hstack((self.raw_positions, avg))
            # Propagate state prediction
            x_pred = self.positions[:, -1:] + dt * v
            # Apply correction from measurements
            residual = avg - x_pred
            return x_pred + g * residual
        else:
            # No new measurements, just propagate known velocity
            self.raw_positions = np.hstack((self.raw_positions, self.raw_positions[:, -1:]))
            return self.positions[:, -1:] + dt * v

    def visualize_frame(self, img, tags):
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
        return color_img

    def scatter_position(self):
        savename = self.run_name + ".png"
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.view_init(25, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        phi = np.linspace(0, 2 * np.pi, self.positions.shape[1])
        rgb_cycle = np.vstack((
            .5 * (1. + np.cos(phi)),
            .5 * (1. + np.cos(phi + 2 * np.pi / 3)),
            .5 * (1. + np.cos(phi - 2 * np.pi / 3)))).T
        ax.scatter(self.positions[0, :], self.positions[2, :], self.positions[1, :], c=rgb_cycle)
        plt.show()
        fig.savefig(savename)

    # Animation Loop
    def animate_position(self, max_time):
        savename = self.run_name + "_ani.avi"
        os.mkdir("tmp")
        plt.ion()
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.view_init(25, 10)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_zlabel('Y')
        ax.set_title('Positions')

        plt.draw()
        for i in range(self.positions.shape[1]):
            plt.cla()
            ax.scatter(self.positions[0, :i], self.positions[2, :i], self.positions[1, :i], color="b")
            ax.scatter(self.positions[0, i], self.positions[2, i], self.positions[1, i], color="r")
            fig.canvas.draw_idle()
            fig.savefig(f"tmp/{i}.png")
            plt.pause(0.1)
        plt.show()
        image_folder = "tmp"
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        writer = cv2.VideoWriter(savename,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 np.floor(self.positions.shape[1] / max_time), self.res)
        for image in sorted(images, key=lambda x: int(x.split(".")[0])):
            writer.write(cv2.imread(os.path.join(image_folder, image)))
        writer.release()
        cv2.destroyAllWindows()
        try:
            shutil.rmtree(image_folder)
        except OSError as e:
            print("Error: %s : %s" % (image_folder, e.strerror))

    def save_positions(self):
        # Save smoothed positions
        fname = self.run_name + ".npy"
        np.save(fname, self.positions)

        # Save unsmoothed positions
        raw_fname = self.run_name + "_raw.npy"
        np.save(raw_fname, self.raw_positions)

    def load_positions(self, fname):
        self.positions = np.load(fname)

    def print_statistics(self, max_time):

        print(f"Time to Undistort:{np.mean(self.undist)}")
        print(f"Time to Detect:{np.mean(self.detect)-np.mean(self.undist)}")
        print(f"Time to End of Loop:{np.mean(self.end)-np.mean(self.detect)}")
        print(f"TRUE FPS: {self.positions.shape[1]/max_time}")
        print(f"EST FPS from DT: {1/np.mean(self.dts)}")

    def save_video(self, max_time):
        fname = self.run_name + ".avi"
        writer = cv2.VideoWriter(fname,
                                 cv2.VideoWriter_fourcc(*'MJPG'),
                                 np.floor(self.positions.shape[1] / max_time), self.res)
        for i, frame in enumerate(self.frames):
            writer.write(self.visualize_frame(frame, self.found_tags[i]))
        writer.release()

    def undistort(self, img):
        return cv2.remap(img, self.camera_info["map_1"], self.camera_info["map_2"], interpolation=cv2.INTER_LINEAR)
