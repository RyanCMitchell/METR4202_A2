# -*- coding: utf-8 -*-
from os.path import isfile, join
from os import listdir
import numpy as np
import freenect, itertools, sys, time, cv2
from MatchingFunctions import findKeyPoints, drawKeyPoints, match, findKeyPointsDist, drawImageMappedPoints, saveImageMappedPoints, MatchAllCaptureGlass, Cluster, fit_ellipses
from matplotlib import pyplot as plt
from math import sqrt
from CoordTransform import convertToWorldCoords


def GlassFind(img0,depth0):
    import cv2
    import freenect
    import numpy as np
    
    img = np.copy(img0)
    depth = np.copy(depth0)
    
    #Blur and threshold depth
    depth = cv2.blur(depth,(10,10))
    mask = np.zeros(img.shape,dtype=np.uint8)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            if depth[i,j] == 0 and i>50 and j>30 and j<595:
                mask[i,j] = 255

    #Edge detect and find contours
    thresh1 = 200
    thresh2 = 300
    edges = cv2.Canny(mask,thresh1,thresh2)
    contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    #Exclude contours with inappropriate depth
    GlassCenters = []
    GlassContours = []
    GlassBox = []
    for cnt in contours:
        if len(cnt) > 10:
            cnt = cv2.convexHull(cnt)
            area = cv2.contourArea(cnt)
            if 2000<area<10000:
                M = cv2.moments(cnt)
                if M['m00'] <> 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    x,y,w,h = cv2.boundingRect(cnt)
                    box = [x,y,w,h]
                    if h>w:
                        GlassCenters.append([cx,cy])
                        GlassContours.append(cnt)
                        GlassBox.append(box)
                                        
    return GlassCenters, GlassContours, GlassBox


def MatchAllClusterGlass(save, maxdist=100, filtparam=2.0):
    
    PointsList, DisList, img, depth = MatchAllCaptureGlass(0,maxdist)
    
    PointsClusterList = []
    for i in xrange(len(PointsList)):
        if True:
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

    splitTend = 0.2
    distFromCenterAveList = [(sum(distFromCenterAve1)/len(distFromCenterAve1))*1.0*splitTend,
    (sum(distFromCenterAve2)/len(distFromCenterAve2))*2.0*splitTend,
    (sum(distFromCenterAve3)/len(distFromCenterAve3))*3.0*splitTend]

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
        if len(segregatedF[i]) <= 25 or np.isnan(np.std(segregatedF[i])):
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
    print FC
    
    """
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
        shape2 = np.zeros(cupDepth1.shape,dtype=np.uint8)

        
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

        Maxpos = depthRangePos[depthRange.index(max(depthRange))]
        Minpos = depthRangePos[depthRange.index(min(depthRange))]
        Cutoff = max(Maxpos[1],Minpos[1])
        
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
        #cv2.imshow('MaxMin',cup11)

        for i in xrange(Cutoff):
            for k in xrange(cupDepth1.shape[1]):
                if lower<cupDepth1[i,k]<upper:
                    shape2[i,k] = 255
        #cv2.imshow('thresh',shape2)

        blurThresh = cv2.blur(shape2,(5,5))
        #cv2.imshow('blur',blurThresh)

        thresh1 = 200
        thresh2 = 300
        edges = cv2.Canny(blurThresh,thresh1,thresh2)
        #cv2.imshow('edges',edges)

        contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)==0:
            continue
        cont = np.vstack(contours)
        hull = cv2.convexHull(cont)

        cupGray = cv2.cvtColor(cup1,cv2.COLOR_BGR2GRAY)
        ColourAverage = np.array([0,0,0])
        count = 0
        for i in xrange(Cutoff):
            for k in xrange(cupDepth1.shape[1]):
                if cv2.pointPolygonTest(hull,(k,i), False)>0:
                    ColourAverage += np.array(cupGray[i,k])
                    count += 1
        ColourAverageF = ColourAverage/count
        whiteColour = max(cupGray[Maxpos[1],Maxpos[0]], cupGray[Minpos[1],Minpos[0]])
        fillRatio = float(ColourAverageF[0])/whiteColour
        
        #cv2.drawContours(cup1,[hull],0,colourList[j],2)
        #cv2.imshow('hull',cup1)
        #cv2.waitKey(0)
        
        
        
        print "mid width",CupMidWidth
        print "top width",CupTopWidth

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
            if fillRatio < 0.8:
                cupFill = "Full"
            else:
                cupFill = "Empty"
                
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
    """
        
    
    # Draw the groups
    deleteList = []
    for j in xrange(groups):
        if len(FinalCentersWC[j]) > 1:
            centerst = tuple(np.array(centers[j])+np.array([0,50]))
            cv2.putText(img,str(FinalCentersWC[j]), centerst, cv2.FONT_HERSHEY_SIMPLEX, 0.3, colourList[j])
            cv2.circle(img, centers[j], 10, colourList[j], -1)
            cv2.circle(img, centers[j], 2, (0,0,0), -1)
            for i in range(len(segregated[j])):
                pt_a = (int(segregated[j][i,0]), int(segregated[j][i,1]))
                cv2.circle(img, pt_a, 3, colourList[j])
                cv2.line(img, pt_a, centers[j], colourList[j])
                rpt1 = tuple(segregated[j].min(axis=0))
                rpt2 = tuple(segregated[j].max(axis=0))
                cv2.rectangle(img, rpt1, rpt2, colourList[j])
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
        MatchAllClusterGlass(0,60,0.6)
        cv2.waitKey(10)

    
    
        

    
