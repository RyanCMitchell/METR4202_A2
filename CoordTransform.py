import numpy as np
from math import sqrt


def transformCoords(origin0,y0,z0, coords):
    """
    Calculates the coordinate of points in coords to the True World Coordinates
    where origin0, x0, y0, and z0 are the coordinates of the origin, and x,y
    and z unit vectors of the True World Coordinates relative to the camera.
    
    transformCoords(tuple,tuple,tuple,tuple, list(np vector)) -> list(np vector)
    """
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
    x = np.array([3,1,1])
    y = np.array([2,1,0])
    z = np.array([3,0,0])
    o = np.array([3,1,0])
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
    testPt = np.array([0,1,-1])
    
    testy = np.array([3-(1/sqrt(2)), 1+(1/sqrt(2)),0])
    testz = np.array([3-(1/sqrt(2)), 1-(1/sqrt(2)),0])
    a = transformCoords(o,x,testy,testz,[o,x,y,z,testPt])
    print np.asarray(a)

    

    #print R
    #print C
    
