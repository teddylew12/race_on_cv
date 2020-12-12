import cv2
from pupil_apriltags import Detector
import numpy as np
import matplotlib.pyplot as plt


class ATDetector:
    def __init__(self, families, tag_size, camera):
        self.detector = Detector(families=families)
        self.camera = camera
        self.tag_size = tag_size

    def estimate_video_pose(self, fname, tag_locations=None, show_animation=True):
        video = cv2.VideoCapture(fname)
        poses = np.zeros(shape=(3, 1))
        while True:
            ret, color_raw = video.read()
            if not ret:
                break
            gray_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2GRAY)
            gray = self.undistort(gray_raw)
            detected_tags = self.detector.detect(gray, estimate_tag_pose=True, camera_params=self.camera["params"],
                                                 tag_size=self.tag_size)
            tmp_poses = []
            for tag in detected_tags:
                if self.camera["flipped"]:
                    pose = np.matmul(self.camera["flip_correction"], tag.pose_t)
                else:
                    pose = tag.pose_t
                if tag_locations:
                    pose = pose + tag_locations[tag.tag_id]
                tmp_poses.append(pose)
            if tmp_poses:
                poses=np.concatenate((poses, np.mean(np.concatenate(tmp_poses, axis=1), axis=1, keepdims=True)), axis=1)
            if show_animation:
                plt.cla()
                # for stopping simulation with the esc key.
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(0) if event.key == 'escape' else None])
                plt.scatter(poses[0, :], poses[2, :])
                plt.axis("equal")
                plt.grid(True)
                plt.pause(0.001)
        return poses

    def estimate_image_pose(self, fname, tag_locations=None):
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if self.camera["fisheye"]:
            img = self.undistort(img)
        detected_tags = self.detector.detect(img, estimate_tag_pose=True, camera_params=self.camera["params"],
                                             tag_size=self.tag_size)
        for tag in detected_tags:
            pose=tag.pose_t
            import pdb;pdb.set_trace()
            if self.camera["flipped"]:
                pose = np.matmul(self.camera["flip_correction"], tag.pose_t)
            tag_id=tag.tag_id
            if tag_locations:
                pose = pose + tag_locations[tag_id]
            print(f"Tag {tag_id} estimates the camera at:\nX:{pose[0,0]}\n"f"Y:{pose[1,0]}"
                  f"\nZ:{pose[2,0]}")
    def visualize_frame(self,img,scale):
        if self.camera["fisheye"]:
            img = self.undistort(img)
        detected_tags = self.detector.detect(img, estimate_tag_pose=False)
        if not detected_tags:
            print("Warning: No Tag Found in Frame")
        color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for tag in sorted(detected_tags, key=lambda x: x.tag_id):
            # Get corners of the tag in the image
            left, right = self.find_top_left_corner(tag.corners)
            # Add a box around the tag
            area = (np.abs(left[0] - right[0])) ^ 2
            # Add a box around the tag
            color = cv2.rectangle(color, left, right, color=(255, 255, 0), thickness=10)
            # Add marks for 0th and 1st corner to check orientation
            color = cv2.circle(color, (int(tag.corners[0][0]), int(tag.corners[0][1])), 10, color=(255, 0, 255),
                               thickness=10)
            color = cv2.circle(color, (int(tag.corners[1][0]), int(tag.corners[1][1])), 10, color=(128, 128, 255),
                               thickness=10)
            # Add some text with the tag number
            text_loc = (left[0], left[1] - area)
            color = cv2.putText(color, f"Tag #{tag.tag_id}", text_loc, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=.05 * area, color=(255, 0, 0), thickness=5)
        return self.scale(color, scale)


    def visualize_image_detections(self, fname, scale=.3):
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        color=self.visualize_frame(img,scale)
        cv2.imshow("Image With Detections", color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def visualize_video_detections(self, fname, scale=.3):
        video = cv2.VideoCapture(fname)
        while True:
            ret, color_raw = video.read()
            if not ret:
                break
            gray_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2GRAY)
            color=self.visualize_frame(gray_raw,scale)
            cv2.imshow(str(fname), color)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    def estimate_image_orientation(self,fname):
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if self.camera["fisheye"]:
            img = self.undistort(img)
        detected_tags = self.detector.detect(img, estimate_tag_pose=True, camera_params=self.camera["params"],
                                             tag_size=self.tag_size)
        for tag in detected_tags:
            R=tag.pose_R
            if self.camera["flipped"]:
                R = np.matmul(self.camera["flip_correction"],R)
            print(R)
            y_rot = np.arcsin(R[2][0])
            x_rot = np.arccos(R[2][2] / np.cos(y_rot))
            z_rot = np.arccos(R[0][0] / np.cos(y_rot))
            theta_y = y_rot * (180 / np.pi)
            theta_x = x_rot * (180 / np.pi)
            theta_z = z_rot * (180 / np.pi)
            print(f"Tag {tag.tag_id} estimates the camera pose with:\nTheta_x:{theta_x}\n"f"Theta_y:{theta_y}"
                  f"\nTheta_Z:{theta_z}")
    def find_top_left_corner(self, corners):
        min_idx = 0
        min_val = 100000
        max_idx = 0
        max_val = -100000
        for idx, (x, y) in enumerate(corners):
            if x + y < min_val:
                min_idx = idx
                min_val = x + y
            if x + y > max_val:
                max_idx = idx
                max_val = x + y

        left = (int(corners[min_idx][0]), int(corners[min_idx][1]))
        right = (int(corners[max_idx][0]), int(corners[max_idx][1]))
        return left, right

    def scale(self, img, scale_factor):
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)
        # resize image
        return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    def undistort(self, img):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(self.camera["K"], self.camera["D"], np.eye(3),
                                                         self.camera["K"], self.camera["res"], cv2.CV_16SC2)
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
