# -*- coding: utf-8 -*-
from os.path import isfile, join
from os import listdir
import numpy as np
import freenect, itertools, sys, time, cv2
from MatchingFunctions import findKeyPoints, drawKeyPoints, match, findKeyPointsDist, drawImageMappedPoints, saveImageMappedPoints, MatchAllCapture, Cluster, fit_ellipses
from matplotlib import pyplot as plt
from math import sqrt
from convertDepth import convertToWorldCoords


def MatchAllCluster(save, maxdist=200, filtparam=2.0):
    
    PointsList, DisList, img, depth = MatchAllCapture(0,maxdist)
    
    PointsClusterList = []
    for i in xrange(len(PointsList)):
        if depth[PointsList[i].pt[1], PointsList[i].pt[0]] <> 0 and depth[PointsList[i].pt[1], PointsList[i].pt[0]] < 1000:
            PointsClusterList.append([PointsList[i].pt[0], PointsList[i].pt[1]])

    Z = np.array(PointsClusterList)

    if len(Z) < 20:
        print "No Cups"
        cv2.imshow("Cups Stream", img)
        return
        
     
    # convert to np.float32
    Z = np.float32(Z)

    # Determine how many cups there are
    segregated, centers, distFromCenter, distFromCenterAve1 = Cluster(Z, 1)
    segregated, centers, distFromCenter, distFromCenterAve2 = Cluster(Z, 2)
    segregated, centers, distFromCenter, distFromCenterAve3 = Cluster(Z, 3)
    segregated, centers, distFromCenter, distFromCenterAve4 = Cluster(Z, 4)
    segregated, centers, distFromCenter, distFromCenterAve5 = Cluster(Z, 5)

    distFromCenterAveList = [(sum(distFromCenterAve1)/len(distFromCenterAve1))*1.0,
    (sum(distFromCenterAve2)/len(distFromCenterAve2))*2.0,
    (sum(distFromCenterAve3)/len(distFromCenterAve3))*3.0,
    (sum(distFromCenterAve4)/len(distFromCenterAve4))*4.0,
    (sum(distFromCenterAve5)/len(distFromCenterAve5))*5.0]

    groups = distFromCenterAveList.index(min(distFromCenterAveList))+1

    segregated, centers, distFromCenter, distFromCenterAve = Cluster(Z, groups)

    #Create List for reduced points
    segregatedF = []
    for i in xrange(groups):
        segregatedF.append([])

    #Remove points which are not close to centroid
    for j in xrange(groups):
        for i in range(len(segregated[j])):
            if distFromCenter[j][i] < filtparam*distFromCenterAve[j]:
                segregatedF[j].append(segregated[j][i])


    #remove clusters that are >= 3 points or all superimposed
    i = 0
    while i < groups:
        if len(segregatedF[i]) <= 5 or np.isnan(np.std(segregatedF[i])):
            del segregatedF[i], centers[i], distFromCenter[i], distFromCenterAve[i]
            groups -= 1
        i += 1

    for j in xrange(groups):
        segregatedF[j] = np.array(segregatedF[j])

    # Create a centriod depth list
    FinalCenters = []
    for j in xrange(groups):
        FinalCenters.append([centers[j][0],centers[j][1],depth[centers[j][1],centers[j][0]]])

    # Convert to world coordinates
    FinalCentersWC = convertToWorldCoords(FinalCenters)
    
    segregated = segregatedF
    FC = FinalCenters
    colourList=[(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 255, 0), (255, 0, 255)]

    # Seperate the top of each cup in pixel space
    depthimg = img.copy()
    depthmask = depth.copy()

    # Start Cup Classification loop
    for j in xrange(groups):
        centx = FC[j][0]
        centy = FC[j][1]
        centdepth = FC[j][2]
        
        # Choose pixel area likley to contain a cup
        w = -0.08811*centdepth+103.0837
        h = -0.13216*centdepth+154.6256
        h = h
        cup1 = depthimg[(centy-h):(centy), (centx-w):(centx+w)]
        cup11 = np.copy(cup1)
        cupDepth1 = depthmask[(centy-h):(FC[j][1]), (centx-w):(centx+w)]

        # Create blank binary images to fill with depth thresholds
        shape1 = np.zeros(cupDepth1.shape,dtype=np.uint8)

        
        # Fill with threshold depths
        upper = centdepth+100
        lower = centdepth-50
        depthRange = []
        depthRangePos = []
        midDepthRange = []
        for i in xrange(cupDepth1.shape[0]):
            for k in xrange(cupDepth1.shape[1]):
                if lower<cupDepth1[i,k]<upper:
                    shape1[i,k] = cupDepth1[i,k]
                    depthRange.append(cupDepth1[i,k])
                    depthRangePos.append([k,i])
                    if i == cupDepth1.shape[0]-1:
                        midDepthRange.append(k)

        if len(midDepthRange) < 3:
            continue

        cv2.imshow('Depth',cup11)
        
        shape1gray = cv2.cvtColor(cup11,cv2.COLOR_BGR2GRAY)

        cv2.imshow('gray',shape1gray)
        

        
        # Apply a median filter to the depth thresholds        
        shape1blur = cv2.equalizeHist(shape1gray)
        cv2.imshow('blur',shape1blur)

        #cv2.imshow('cnt',cup2)
        #cv2.waitKey(0)

        # Determine the bouding rectangle of the largest contour in the top area
        thresh1 = 40
        thresh2 = 200
        edges = cv2.Canny(shape1gray,thresh1,thresh2)
        cv2.imshow('edges',edges)
        contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cont = np.vstack(contours)

        for cnt in contours:
            if len(cnt)>10:
                cv2.drawContours(cup1,[cnt],0,colourList[j],2)
                ellipse = cv2.fitEllipse(cnt)
                print ellipse 
                cv2.ellipse(cup1, ellipse, colourList[j+2], 2)

        cv2.imshow('ellipse',cup1)

        Maxpos = depthRangePos[depthRange.index(max(depthRange))]
        Minpos = depthRangePos[depthRange.index(min(depthRange))]
        print Maxpos
        print Minpos
        midMin = min(midDepthRange)
        midMax = max(midDepthRange)
        s3 = [(centx-w)+midMin,centy,centdepth]
        s4 = [(centx-w)+midMax,centy,centdepth]
        mid = [s3,s4]
        midWorld = convertToWorldCoords(mid)
        CupMidWidth = midWorld[1][0]-midWorld[0][0]
        CupTopWidth = max(depthRange)-min(depthRange)

        cv2.circle(cup11, tuple(Maxpos), 3, colourList[j+1])
        cv2.circle(cup11, tuple(Minpos), 3, colourList[j+1])
        cv2.line(cup11, tuple(Maxpos), tuple(Minpos), colourList[j+1])

        cv2.imshow('MaxMin',cup11)
        cv2.waitKey(0)

        

        print "Mid Width",CupMidWidth
        print "Top Width",CupTopWidth
        if CupMidWidth > CupTopWidth:
            
            cupOrientation = "Upsidedown"
            cupFill = "Empty"
            
            if CupTopWidth > 100:
                cupType = "Not a Cup"
            elif CupTopWidth > 58:
                cupType = "Large"
            elif CupTopWidth > 49:
                cupType = "Medium"
            elif CupTopWidth > 20:
                cupType = "Small"
            else:
                cupType = "Not a Cup"

        else:

            cupOrientation = "Upright"
            cupFill = "Unsure"
            
            if CupTopWidth > 100:
                cupType = "Not a Cup"
            elif CupTopWidth > 81:
                cupType = "Large"
            elif CupTopWidth > 69:
                cupType = "Medium"
            elif CupTopWidth > 30:
                cupType = "Small"
            else:
                cupType = "Not a Cup"

        
        #Draw the top of the bounding rectangle
        if cupType <> "Not a Cup":
            FinalCentersWC[j].append(cupType)
            FinalCentersWC[j].append(cupOrientation)
            FinalCentersWC[j].append(cupFill)
        
    
    # Draw the groups
    deleteList = []
    for j in xrange(groups):
        if len(FinalCentersWC[j]) > 3:
            centerst = tuple(np.array(centers[j])+np.array([0,50]))
            cv2.putText(img,str(FinalCentersWC[j]), centerst, cv2.FONT_HERSHEY_SIMPLEX, 0.3, colourList[j])
            cv2.circle(img, centers[j], 10, colourList[j], -1)
            cv2.circle(img, centers[j], 2, (0,0,0), -1)
            for i in range(len(segregated[j])):
                pt_a = (int(segregated[j][i,0]), int(segregated[j][i,1]))
                cv2.circle(img, pt_a, 3, colourList[j])
                cv2.line(img, pt_a, centers[j], colourList[j])
        else:
            deleteList.append(j)
            
    FinalFinalCentersWC = [i for j, i in enumerate(FinalCentersWC) if j not in deleteList]
    
    if save == 1:
        cv2.imwrite('ProcessedImages/ProcessedCluster'+str(ImageNo)+'.jpg', img)

    if len(FinalFinalCentersWC)<>0:
        print FinalFinalCentersWC
        cv2.imshow("Cups Stream", img)


if __name__== '__main__':
    
    
    cv2.destroyAllWindows()
    while 1<2:
        MatchAllCluster(0,80,2)
        cv2.waitKey(10)

    
    
        

    
