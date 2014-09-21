from os.path import isfile, join
from os import listdir
import itertools
import sys
import numpy as np
import cv2
import glob

board_w = 6 # ours is 5
board_h = 7 # ours is 8
square = 26

#Clear all CV windows
cv2.destroyAllWindows()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# The OpenCV test images are a 7x6 grid with 30mm squares
# Our test images are a 5x8 grid with 26mm squares
objp = np.zeros((board_w * board_h, 3), np.float32)
objp[:, :2] = np.mgrid[0:board_h, 0:board_w].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

chessImages = glob.glob('*.jpg')

for image in chessImages:
    img = cv2.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,6), corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(1000)

# ret =>
# mtx = camera matrix as a list of 3 rows
#       [Fx     0       Cx]
#       [0      Fy      Cy]
#       [0      0       1 ]
# dist = list of distortion coefficients [K1 K2 P1 P2 K3]
# rvecs = rotation vectors
# tvecs = translation vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

#ret = ret * square
#mtx = mtx * square
#dist = dist * square
#rvecs = rvecs * square
#tvecs = tvecs * square

# read in an immage to undistort
img = cv2.imread('left05.jpg')
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult.jpg',dst)


'''Re-projection error gives a good estimation of just how exact the found
parameters are. This should be as close to zero as possible. Given the
intrinsic, distortion, rotation and translation matrices, we first
transform the object point to image point using cv2.projectPoints(). Then
we calculate the absolute norm between what we got with our transformation
and the corner finding algorithm. To find the average error we calculate
the arithmetical mean of the errors calculate for all the calibration
images.'''
# calculate Re-Projection error
mean_error = 0.0
tot_error = 0.0
for i in xrange(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    tot_error += error

print "total error: ", mean_error/len(objpoints)

cv2.destroyAllWindows()
