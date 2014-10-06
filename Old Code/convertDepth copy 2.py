from os.path import isfile, join
import freenect
from os import listdir
import itertools
import sys
import numpy as np
import cv2
import cv2.cv as cv

# destroy all windows ???
cv2.destroyAllWindows()

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

def convertToWorldCoords(coordsList):

    dist = np.load('CalibrationImages/Caliboutput/dist.npy')
    Mat_K = np.load('CalibrationImages/Caliboutput/mtx1.npy')

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
        
        worldCoords.append([int(round(x_world, 0)), int(round(y_world, 0)), cupd])

    return worldCoords

'''
    T = [Cos{}  -Sin{}  Tx] = Affine Transformation Matrix
        [Sin{}  Cos{}   Ty]
        [ 0      0      1 ]
The matrix T will be referred to as a homogeneous transformation
matrix. It is important to remember that $ T$ represents a rotation
followed by a translation (not the other way around).'''
def convertToSuryaCoords(worldCoordsList, origin):
    
    return 1                    
