from os.path import isfile, join
import freenect
from os import listdir
import itertools
import sys
import numpy as np
import cv2
import cv2.cv as cv
from MatchingFunctions import drawKeyPoints

'''
To convert to Camera co-ordinates from image pixels the following must be applied:
P_screen = I * P_world

| x_screen | = I * | x_world |
| y_screen |       | y_world |
|    1     |       | z_world |
                   |    1    |
where

I = | f_x    0    c_x    0 |    =   Cam_Mat
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
    Cam_Mat = np.load('CalibrationImages/Caliboutput/mtx1.npy')

    worldCoords = []

    fx = Cam_Mat[0][0]
    fy = Cam_Mat[1][1]
    cx = Cam_Mat[0][2]
    cy = Cam_Mat[1][2]       
    
    for cup in coordsList:
        cupx = cup[0]
        cupy = cup[1]
        cupd = cup[2]
        
        x_world = (cupx - cx) * cupd / fx
        y_world = (cupy - cy) * cupd / fy
        
        worldCoords.append([int(round(x_world, 0)), int(round(y_world, 0)), cupd])

    return worldCoords


def convertToSuryaCoordsSimple(worldCoordsList):

    simpleSuryaCoords = []
    matSize = 384       # mm
    minDepthRange = 600 # mm
    kinectHeight = 175  # mm
    
    for cup in worldCoordsList:
        cupx = cup[0]
        cupy = cup[1]
        cupd = cup[2]
        
        x_Surya = -(cupd - minDepthRange - matSize)
        y_Surya =  cupx - matSize/2
        z_Surya = -(cupy - kinectHeight)
        
        simpleSuryaCoords.append([int(round(x_Surya, 0)), int(round(y_Surya, 0)), int(round(z_Surya, 0))])
    
    return simpleSuryaCoords                    


def findFiducial():

    # load fiducial with greyscale flag
    tag = cv2.imread('Frame.png', 0)
    # tag = cv2.imread('Tags/Tag36h9.png', 0)
    real_width = 51.5   # mm (measure these!!!!)
    h, w = tag.shape    # h and w of fiducial

    # load in one of our "distorted" images with greyscale flag
    img = cv2.imread('CalibrationImages/Fiducial/Fiducial1.jpg')
    #img = cv2.imread('photo.jpg') # trainImage
    #img, timestamp = freenect.sync_get_video()     #(possibly take an image)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # Load previously saved data about the camera - generated with the chessboard photos
    dist = np.load('CalibrationImages/Caliboutput/dist.npy')
    Cam_Mat = np.load('CalibrationImages/Caliboutput/mtx1.npy')

    '''
    # undistort the image!
    ph, pw = img.shape[:2]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(Cam_Mat, dist, (pw, ph), 1, (pw, ph))
    img = cv2.undistort(img, Cam_Mat, dist, None, newcameramtx)
    '''

    # Initiate ORB detector
    orb = cv2.ORB()

    # find the keypoints and descriptors with ORB:
    # fiducial
    tag_kp = orb.detect(tag, None)
    tag_kp, tag_des = orb.compute(tag, tag_kp)
    # photo
    img_kp = orb.detect(img, None)
    img_kp, img_des = orb.compute(img, img_kp)

    # find the fiducial
    print("finding fiducial")
    M = find(tag_des, tag_kp, img_des, img_kp)
    draw_outline(M, h, w, img)

    # write out full size image
    cv2.imwrite('found.jpg', img)

    # show the image
    cv2.imshow('found', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    


# does the match, if it's good returns the homography transform
def find(des, kp, img_des, img_kp):

    MIN_MATCH_COUNT = 50

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors.
    matches = bf.match(des, img_des)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    print "Matches found -> %d/%d" % (len(matches), MIN_MATCH_COUNT)
    
    if len(matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp[m.queryIdx].pt for m in matches[:MIN_MATCH_COUNT]]).reshape(-1,1,2)
        dst_pts = np.float32([img_kp[m.trainIdx].pt for m in matches[:MIN_MATCH_COUNT]]).reshape(-1,1,2)

        # get the transformation between the flat fiducial and the found fiducial in the photo
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        print M
        np.save('CalibrationImages/Caliboutput/HomoMat', M)
        return M    # return the transform

    else:
        print "Not enough matches are found - %d/%d" % (len(matches), MIN_MATCH_COUNT)


#draws a box round the fiducial
def draw_outline(M, h, w, img):
    #array containing co-ords of the fiducial
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #transform the coords of the fiducial onto the picture
    dst = cv2.perspectiveTransform(pts, M)
    #draw a box around the fiducial
    cv2.polylines(img,[np.int32(dst)],True,(255,0,0),5, cv2.CV_AA)


def convertToSuryaCoords(worldCoordsList):
    
    HomoMat = np.load('CalibrationImages/Caliboutput/HomoMat.npy')
    Cam_Mat = np.load('CalibrationImages/Caliboutput/mtx1.npy')

    
    for cup in worldCoordsList:
        cupnp = np.array(cup)
        print cupnp.shape

if __name__== '__main__':
    findFiducial()















