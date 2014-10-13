import numpy as np
from math import sqrt
import freenect
import cv2
import cv2.cv as cv
from MatchingFunctions import drawKeyPoints


dist = np.load('CalibrationImages/Caliboutput/dist.npy')
Cam_Mat = np.load('CalibrationImages/Caliboutput/mtx1.npy')


def convertToWorldCoords(coordsList):
    """
    To convert to Camera co-ordinates from image pixels the following must be
    applied:

    P_screen = I * P_world

    | x_screen | = I * | x_world |
    | y_screen |       | y_world |
    |    1     |       | z_world |
                       |    1    |
    where,

    I = | f_x    0    c_x    0 |    =   Cam_Mat
        |  0    f_y   c_y    0 |
        |  0     0     1     0 |
    is the 3x4 intrinsics matrix, f being the focal point and c the center
    of projection.

    If you solve the system above, you get:
    x_screen = (x_world/z_world)*f_x + c_x
    y_screen = (y_world/z_world)*f_y + c_y

    But, you want to do the reverse, so the answer is:
    x_world = (x_screen - c_x) * z_world / f_x
    y_world = (y_screen - c_y) * z_world / f_y"""

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
    '''If you know the environment and where you will be placing the Kinect
    you can use this function as a simple transform.'''
    simpleSuryaCoords = []
    matSize = 384       # mm
    minDepthRange = 600 # mm
    kinectHeight = 180  # mm

    for cup in worldCoordsList:
        cupx = cup[0]
        cupy = cup[1]
        cupd = cup[2]
        
        x_Surya = -(cupd - minDepthRange - matSize)
        y_Surya =  cupx - matSize/2
        z_Surya = -(cupy - kinectHeight)
        
        simpleSuryaCoords.append([int(round(x_Surya, 0)), int(round(y_Surya, 0)), int(round(z_Surya, 0))])
    
    return simpleSuryaCoords


def PointFind(point, img, depth, col = (255,0,0)):
    point = point[0,:]
    point = tuple(point)
    cv2.circle(img, point, 3, col, -1)
    return [point[0], point[1], depth[point[1],point[0]]]



def FrameFind():
    '''Find the Chess Board and use it as a reference to transform the Coord
    system. This function only needs to be run once.
    FrameFind() -> Numpy array of corners as
    ([TopLeft, BottomLeft, BottomRight, TopRight])'''

    board_w = 5 # ours is 5
    board_h = 8 # ours is 8
    square = 26 # mm

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((board_w * board_h, 3), np.float32)
    objp[:,:2] = np.mgrid[0:board_h*square:square, 0:board_w*square:square].T.reshape(-1, 2)


    img, timestamp = freenect.sync_get_video()
    depth, timestamp = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
    # img = cv2.imread('CalibrationImages/Frame/Frame2.jpg')
    
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

    Corners = convertToWorldCoords([TopLeft, BottomLeft, BottomRight, TopRight])
    PixCorners = np.array([TopLeft, BottomLeft, BottomRight, TopRight])

    Corners = np.array(Corners)
    np.save('CalibrationImages/Caliboutput/corners.npy',Corners)
    np.save('CalibrationImages/Caliboutput/PixCorners.npy',PixCorners)
    """
    cv2.line(img, tuple(BottomLeft[:2]), tuple(TopLeft[:2]), (255,0,0),3)
    cv2.line(img, tuple(BottomLeft[:2]), tuple(BottomRight[:2]), (0,0,255),3)
    cv2.line(img, tuple(PixCorners[1][:2]), tuple(PixCorners[0][:2]), (255,0,0),3)
    cv2.line(img, tuple(PixCorners[1][:2]), tuple(PixCorners[2][:2]), (0,0,255),3)
    cv2.imshow('Frame',img)
    cv2.imwrite('CoordTransform.png', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """
    return Corners


def transformCoords(coords, Corners):
    """
    Calculates the coordinate of points in coords to the True World Coordinates
    where origin0, x0, y0, and z0 are the coordinates of the origin, and x, y
    and z unit vectors of the True World Coordinates relative to the camera.
    
    transformCoords(tuple,tuple,tuple,tuple, list(np vector)) -> list([tuple])
    """
    
    #Load the frame corners
    #Corners = np.load('CalibrationImages/Caliboutput/corners.npy')
    
    #Find y and z unit vectors
    origin0 = Corners[1]
    y = Corners[2]-origin0
    y0 = y/(np.linalg.norm(y))
    z = Corners[0]-origin0
    z0 = z/(np.linalg.norm(z))

    #Find x-unit vector
    x0 = np.cross(y0, z0)

    #Construct rotation matrix
    R = np.zeros([4,4])
    R[:3,0] = x0
    R[:3,1] = y0
    R[:3,2] = z0
    R[3,3] = 1

    #R = rotMatrix(origin0, x0, y0, z0)
    coordTWC = []
    for coord in coords:
        coord = np.asarray(coord) #Convert to numpy vector
        coordTWC.append(np.rint(transformPoint(R, origin0, coord)).astype(int).tolist())
        #print "Camera: ", coord, "-> World: ", transformPoint(R, origin0, coord)
    return coordTWC


def transformPoint(R, origin, pt):
    rotCoord    = (R.T).dot(np.append(pt, 1).T)
    shiftOrigin = (R.T).dot(np.append(origin, 1).T)
    return (rotCoord - shiftOrigin)[:3]


if __name__ == '__main__':

    FrameFind()

    Corners = np.load('CalibrationImages/Caliboutput/corners.npy')
    U = Corners[0]
    V = Corners[1]
    W = Corners[2]
    X = Corners[3]
    
    a = transformCoords([U, V, W, X])
