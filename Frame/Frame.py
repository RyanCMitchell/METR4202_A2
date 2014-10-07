import cv2
import numpy as np
import glob
import freenect

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def drawCube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)

    # draw ground floor in green
    img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

    # draw top layer in red color
    img = cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

    return img

def PointFind(point, img, depth, col = (255,0,0)):
    point = point[0,:]
    point = tuple(point)
    cv2.circle(img, point, 3, col, -1)
    return [point[0], point[1], depth[point[1],point[0]]]
    
    

def FrameFind():
    # Load previously saved data
    mtx = np.load('CalibrationImages/Caliboutput/Old/mtx1.npy')
    dist = np.load('CalibrationImages/Caliboutput/Old/dist.npy')

    board_w = 5 # ours is 5
    board_h = 8 # ours is 8

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_h, 0:board_w].T.reshape(-1, 2)


    img, timestamp = freenect.sync_get_video()
    depth, timestamp = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
    #img = cv2.imread('CalibrationImages/Frame/Frame2.jpg')
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (board_h, board_w), None)

    """
    col = round(255/len(corners),0)
    for j in xrange(len(corners)):
        i = corners[j]
        i = i[0,:]
        i = tuple(i)
        cv2.circle(img, i, 3, (0,int(j*col),0), -1)
    """
    TopLeft = PointFind(corners[7], img, depth, col = (255,0,0))
    TopRight = PointFind(corners[-1], img, depth, col = (0,0,255))
    BottomLeft = PointFind(corners[0], img, depth, col = (255,255,0))
    BottomRight = PointFind(corners[-8], img, depth, col = (0,255,255))

    cv2.imshow('Frame',img)
    cv2.waitKey(0)
    return 



"""
while 1<2:
    FrameFind()
"""

FrameFind()

    
