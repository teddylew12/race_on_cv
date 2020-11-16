import cv2
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob

# Function to undistort captured images
def undistort_current_image(I, map1, map2 ):
    undistorted_img = cv2.remap(I, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img



img_folder=glob.glob("Images/calib4/*.jpg")
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
with open("camera-parameters.pkl", "wb") as f:
    pickle.dump([map1, map2], f)