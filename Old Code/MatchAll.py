from os.path import isfile, join
from os import listdir
import cv2
import numpy as np
import itertools
import sys
from MatchingFunctions import findKeyPoints, drawKeyPoints, match, findKeyPointsDist, drawImageMappedPoints, saveImageMappedPoints, MatchAll 
from matplotlib import pyplot as plt
from math import sqrt

def MatchAllCluster(ImageNo, save, maxdist=200, groups=3):
    PointsList, DisList, img = MatchAll(ImageNo,0,maxdist)
    PointsClusterList = []
    for i in xrange(len(PointsList)):
        PointsClusterList.append([PointsList[i].pt[0], PointsList[i].pt[1]])

    Z = np.array(PointsClusterList)
     
    # convert to np.float32
    Z = np.float32(Z)
     
    # define criteria and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center = cv2.kmeans(Z,groups,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now separate the data, Note the flatten()
    segregated = []
    centers = []
    distFromCenter = []
    segregatedF = []
    for i in xrange(groups):
        segregated.append(Z[label.flatten()==i])
        distFromCenter.append([])
        segregatedF.append([])
        centers.append((int(center[i][0]), int(center[i][1])))

    # Create a distance from centroid list
    for j in xrange(groups):
        x1 = centers[j][0]
        y1 = centers[j][1]
        for i in range(len(segregated[j])):
            x2 = segregated[j][i][0]
            y2 = segregated[j][i][1]
            distFromCenter[j].append( sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))

    # Create an average distance from centroid list
    distFromCenterAve = []
    for j in xrange(groups):
        distFromCenterAve.append(sum(distFromCenter[j])/len(distFromCenter[j]))

    #Remove points which are not close to centroid
    filtparam = 2.0
    for j in xrange(groups):
        for i in range(len(segregated[j])):
            if distFromCenter[j][i] < filtparam*distFromCenterAve[j]:
                segregatedF[j].append(segregated[j][i])
    for j in xrange(groups):
        segregatedF[j] = np.array(segregatedF[j])
    
    return segregatedF, centers, img

def DrawMatchAllCluster(ImageNo, save, maxdist=200, groups=3):
    segregated, centers, img = MatchAllCluster(ImageNo, save, maxdist, groups)
    
    # Draw the groups
    colourList=[(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    for j in xrange(groups):
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

if __name__== '__main__':
    #DrawMatchAllCluster(1,1,100)
    
    for i in xrange(8):
        DrawMatchAllCluster(i,1,100,3)
    
