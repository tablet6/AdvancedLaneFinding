import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


def find_obj_imgpoints(images):
    # num corners
    nx = 9
    ny = 6

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    points3d = []  # 3d points in real world space
    points2d = []  # 2d points in image plane.
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, draw corners
        if ret == True:
            points3d.append(objp)
            points2d.append(corners)

    return points3d, points2d


def undistort_img(fname, points3d, points2d):
    img = cv2.imread(fname)
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points3d, points2d, img_size, None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    return dst

def display(orig, dst):
    src = cv2.imread(orig)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(src)
    ax1.set_title('Original Image', fontsize=30)
    assert isinstance(dst, object)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Arrays to store object points and image points from all the images.
objpoints, imgpoints = find_obj_imgpoints(images)

for fname in images:
    undistort = undistort_img(fname, objpoints, imgpoints)
    display(fname, undistort)

with open('wide_dist_pickle.p', 'wb') as f:
    pickle.dump(objpoints, f)
    pickle.dump(imgpoints, f)
