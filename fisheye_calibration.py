import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import time
import picamera
import os
from argparse import ArgumentParser

def undistort_current_image(I, map1, map2):
    undistorted_img = cv2.remap(I, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

parser=ArgumentParser()
parser.add_argument("-num","--num_photos",default=30,type=int)
parser.add_argument("-resx",default=640,type=int)
parser.add_argument("-resy",default=480,type=int)
parser.add_argument("-f","--folder_name",type=str)
args=parser.parse_args()
if args.folder_name:
    folder_name = args.folder_name + "/"
else:
    folder_name= f"calibration_{args.resx}_{args.resy}/"
os.makedirs(folder_name,exist_ok=True)
# Function to undistort captured images
RES = (args.resx,args.resy)
#Initialize Camera Object
with picamera.PiCamera(resolution=RES) as camera:
    #Take and save pictures
    for num in range(args.num_photos):
        camera.start_preview()
        time.sleep(3)
        fname=folder_name+str(num)+".png"
        camera.capture(fname)


img_folder=glob.glob(folder_name+ "*.png")
# Initialize properly sized arrays and some other openCV preparation steps
CHECKERBOARD = (7, 9)  # this tuple must contain the number of color changes per column and line respectively
# (which equals number of squares minus 1)

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_CHECK_COND

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
for fname in img_folder:
    img=cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(img,
                                             CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(img, corners, (3, 3), (-1, -1), subpix_criteria)
        imgpoints.append(corners)
    else:
        print(f'Corners not detected for {fname}\n')
N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        img.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(img.shape[::-1]))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")

DIM = img.shape[::-1]
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2
.BORDER_CONSTANT)
import pdb;pdb.set_trace()
scale_percent = 20  # percent of original size
width = int(undistorted_img.shape[1] * scale_percent / 100)
height = int(undistorted_img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
undistorted_img = cv2.resize(undistorted_img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("undistorted", undistorted_img)
cv2.imshow("distorted",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Save map1 and map2 somewhere so you don't need to recalibrate your camera everytime
