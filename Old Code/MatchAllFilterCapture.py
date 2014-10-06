from os.path import isfile, join
from os import listdir
import cv2
import numpy as np
import itertools
import sys
from MatchingFunctions import findKeyPoints, drawKeyPoints, match, findKeyPointsDist, drawImageMappedPoints, saveImageMappedPoints, MatchAllCapture, Cluster
from matplotlib import pyplot as plt
from math import sqrt
from convertDepth import convertToWorldCoords

def MatchAllCluster(save, maxdist=200, groups=3, filtparam=2.0):
    PointsList, DisList, img, depth = MatchAllCapture(0,maxdist)
    PointsClusterList = []
    for i in xrange(len(PointsList)):
        if depth[PointsList[i].pt[1], PointsList[i].pt[0]] <> 0 and depth[PointsList[i].pt[1], PointsList[i].pt[0]] < 1000:
            PointsClusterList.append([PointsList[i].pt[0], PointsList[i].pt[1]])

    Z = np.array(PointsClusterList)
     
    # convert to np.float32
    Z = np.float32(Z)

    segregated, centers, distFromCenter, distFromCenterAve = Cluster(Z, 3)

    #Create List for reduced points
    segregatedF = []
    for i in xrange(groups):
        segregatedF.append([])
    
    #Remove points which are not close to centroid
    for j in xrange(groups):
        for i in range(len(segregated[j])):
            if distFromCenter[j][i] < filtparam*distFromCenterAve[j]:
                segregatedF[j].append(segregated[j][i])
    for j in xrange(groups):
        segregatedF[j] = np.array(segregatedF[j])

    # Create a centriod depth list
    FinalCenters = []
    for j in xrange(groups):
        FinalCenters.append([centers[j][0],centers[j][1],depth[centers[j][1],centers[j][0]]])

    # Convert to world coordinates
    FinalCentersWC = convertToWorldCoords(FinalCenters)
    
    return segregatedF, centers, img, depth, FinalCenters, FinalCentersWC

def DrawMatchAllCluster(save, maxdist=200, groups=3, filtparam=2.0):
    segregated, centers, img, depth, FinalCenters, FinalCentersWC = MatchAllCluster(save, maxdist, groups, filtparam)

    # Round and print coordinates
    print FinalCenters
    print FinalCentersWC
    
    # Draw the groups
    colourList=[(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    for j in xrange(groups):
        centerst = tuple(np.array(centers[j])+np.array([0,50]))
        cv2.putText(img,str(FinalCentersWC[j]), centerst, cv2.FONT_HERSHEY_SIMPLEX, 0.3, colourList[j])
        cv2.circle(img, centers[j], 10, colourList[j], -1)
        cv2.circle(img, centers[j], 2, (0,0,0), -1)
        for i in range(len(segregated[j])):
            pt_a = (int(segregated[j][i,0]), int(segregated[j][i,1]))
            cv2.circle(img, pt_a, 3, colourList[j])
        rpt1 = tuple(segregated[j].min(axis=0))
        rpt2 = tuple(segregated[j].max(axis=0))
        cv2.rectangle(img, rpt1, rpt2, colourList[j])
    
    if save == 1:
        cv2.imwrite('ProcessedImages/ProcessedCluster'+str(ImageNo)+'.jpg', img)

    cv2.imshow("Cups Stream", img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    """
    # Black out 0 depth
    for i in xrange(depth.shape[0]):
        for j in xrange(depth.shape[1]):
            if depth[i,j] == 0 or depth[i,j]>1000:
                img[i,j] = [0,0,0]

    cv2.imshow("Cups Stream", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

if __name__== '__main__':
    
    
    #DrawMatchAllCluster(0,60,3,2)

    cv2.destroyAllWindows()
    while 1<2:
        DrawMatchAllCluster(0,100,3,2)
        cv2.waitKey(50)
    
        

    
