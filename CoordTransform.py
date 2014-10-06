import numpy as np
from math import sqrt
import freenect
import cv2
from convertDepth import convertToWorldCoords

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

    Corners = convertToWorldCoords([TopLeft,TopRight,BottomLeft,BottomRight])

    Corners = np.array(Corners)
    np.save('CalibrationImages/Caliboutput/corners.npy',Corners)
    
    cv2.imshow('Frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def transformCoords(coords):
    """
    Calculates the coordinate of points in coords to the True World Coordinates
    where origin0, x0, y0, and z0 are the coordinates of the origin, and x,y
    and z unit vectors of the True World Coordinates relative to the camera.
    
    transformCoords(tuple,tuple,tuple,tuple, list(np vector)) -> list(np vector)
    """

    #Load the frame corners
    Corners = np.load('CalibrationImages/Caliboutput/corners.npy')
    print Corners
    
    #Find y and z unit vectors
    origin0 = Corners[2]
    y = Corners[3]-Corners[2]
    y0 = y/(np.linalg.norm(y))+origin0
    z = Corners[0]-Corners[2]
    z0 = z/(np.linalg.norm(z))+origin0

    #Find x-unit vector
    x0 = origin0 + np.cross(y0-origin0, z0-origin0)
    
    #Construct rotation matrix
    R = np.zeros([4,4])
    R[:3,0] = x0-origin0
    R[:3,1] = y0-origin0
    R[:3,2] = z0-origin0
    R[3,3] = 1

    #R = rotMatrix(origin0,x0,y0,z0)
    coordTWC = []
    for coord in coords:
        coord = np.asarray(coord) #Convert to numpy vector
        coordTWC.append(transformPoint(R,origin0,coord))
        print "Camera: ",coord,"-> World: ", transformPoint(R,origin0,coord)
    return coordTWC

def transformPoint(R,origin,pt):
    rotCoord    =(R.T).dot(np.append(pt,1).T)
    shiftOrigin =(R.T).dot(np.append(origin,1).T)

    return (rotCoord - shiftOrigin)[:3]


#x0,y0,z0,origin0 must be numpy vectors
def rotMatrix(origin0, x0,y0,z0):
    R = np.zeros([4,4])
    #C = np.eye(4)
    R[:3,0] = x0-origin0
    R[:3,1] = y0-origin0
    R[:3,2] = z0-origin0
    R[3,3] = 1
    #C[:3,3] = -origin0
    return R
    


if __name__ == '__main__':

    #FrameFind()

    Corners = np.load('CalibrationImages/Caliboutput/corners.npy')
    x = np.array([3,1,1])
    y = Corners[3]
    z = Corners[0]
    o = np.array([3,1,0])
    cam = np.array([0,0,0])
    #R = rotMatrix(o,x,y,z)

    '''
    print "Point in Camera Space"
    print o
    print "R"
    print R
    print "C"
    print C

    rotatedAxes =(R.T).dot(np.transpose(np.append(x,1)))
    point = (rotatedAxes - rotatedOffset)[:3]
    print "Point in World Space"
    print point
    '''
    testPt = np.array([10,40,500])
    
    testy = np.array([3-(1/sqrt(2)), 1+(1/sqrt(2)),0])
    testz = np.array([3-(1/sqrt(2)), 1-(1/sqrt(2)),0])
    a = transformCoords([o,x,y,z,testPt,cam])
    print np.asarray(a)

    

    #print R
    #print C
