from os.path import isfile, join
import freenect
from os import listdir
import itertools
import sys
import numpy as np
import cv2
import cv2.cv as cv
import glob

# destroy all windows ???
cv2.destroyAllWindows()

def depthImgConvert(img):
    #convert from 16bit to 8bit
    return img

def convertDepthToMeters(img):

    

    return #an integer in mm




def undistortImage():
    return 1

'''
To convert to Camera co-ordinates from image pixels the following must be applied:
P_screen = I * P_world

| x_screen | = I * | x_world |
| y_screen |       | y_world |
|    1     |       | z_world |
                   |    1    |
where

I = | f_x    0    c_x    0 |    =   Mat_K
    |  0    f_y   c_y    0 |
    |  0     0     1     0 |
is the 3x4 intrinsics matrix, f being the focal point and c the center of projection.

If you solve the system above, you get:
x_screen = (x_world/z_world)*f_x + c_x
y_screen = (y_world/z_world)*f_y + c_y

But, you want to do the reverse, so your answer is:
x_world = (x_screen - c_x) * z_world / f_x
y_world = (y_screen - c_y) * z_world / f_y'''

def convertToWorldCoords(coordsList, Mat_K, distCoefs):

    worldCoords = []

    fx = Mat_K[0][0]
    fy = Mat_K[1][1]
    cx = Mat_K[0][2]
    cy = Mat_K[1][2]
    
    for cup in coordsList:
        cupx = cup[0]
        cupy = cup[1]
        cupd = cup[2]

        x_world = (cupx - cx) * cupd / fx
        y_world = (cupy - cy) * cupd / fy
        
        worldCoords.append([x_world, y_world, cupd])
    
    return worldCoords


def convertToSuryaCoords(worldCoordsList, origin):

    return 1


def findCameraCoords(point):#M_K, depth_map, point):
    #extract point depth from map
    #transform the point from distorted to undistorted x,y coordinate
    #
    path = '../Git/METR4202_A2/'
    dist = np.load(path+'CalibrationImages/Caliboutput/dist.npy')
    M_K = np.load(path+'CalibrationImages/Caliboutput/mtx1.npy')
    print M_K
    print dist
    print point
    a = cv2.undistortPoints(point, M_K, dist)
    #cv2.undistortPoints(
    print a
                        
    return (x,y,d)


if __name__ == '__main__':
    print 'asdasda'
    a= np.array([[1,2]])
    print type(a)
    findCameraCoords(a)                       
